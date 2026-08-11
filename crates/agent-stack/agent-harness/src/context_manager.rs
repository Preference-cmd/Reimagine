//! Conversation context management for the agent loop.
//!
//! [`ContextManager`] owns the per-session conversation history and
//! applies a rolling-window strategy: when the stored history exceeds
//! `recent_turns * 2` messages, the oldest messages are dropped and the
//! remainder is re-attached to a fresh system prompt on each provider
//! call. V1 does not summarize dropped messages.
//!
//! Token counting is heuristic, without external dependencies. CJK
//! characters (code point above `0x2E80`) average one token per two
//! characters; all other characters average one token per four
//! characters. A fixed per-message overhead of 4 tokens is added.
//!
//! Estimation is behind the [`TokenEstimator`] seam: the heuristic is
//! the default implementation, and an exact counter (e.g. tiktoken-rs,
//! deferred to V3) can be swapped in without touching callers.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::provider::{ContentBlock, Message};

/// Pre-budget token estimate for a single image file block. This is a
/// planning-time figure only — the provider's reported `Usage` is the
/// source of truth for actual consumption.
const IMAGE_TOKENS_PER_BLOCK: usize = 1024;

/// Pre-budget fallback for non-image file blocks (audio, video, ...).
const OTHER_FILE_TOKENS_PER_BLOCK: usize = 256;

/// Default soft-trigger cushion (CM-V2a, decision table M1): compaction
/// is triggered once the stored history exceeds
/// `max_tokens - reserved_tokens`; the hard limit remains `max_tokens`.
const DEFAULT_RESERVED_TOKENS: usize = 16_000;

/// Token estimation seam (CM-V2a, decision table M1). Callers may
/// install an exact counter later without changing `ContextManager`
/// internals.
///
/// Contract: `estimate_message` returns the full message cost including
/// any fixed per-message overhead (the default heuristic adds 4 per
/// text block); `estimate_text` returns the raw text cost and is used
/// for plain text outside a message envelope. Custom estimators should
/// mirror these semantics.
pub trait TokenEstimator: Send + Sync {
    /// Estimate the tokens of a full message (content blocks plus
    /// fixed overhead).
    fn estimate_message(&self, message: &Message) -> usize;
    /// Estimate the tokens of plain text (no message envelope
    /// overhead).
    fn estimate_text(&self, text: &str) -> usize;
}

/// Default heuristic estimator: CJK 2 chars/token, other 4 chars/token,
/// +4 per text block, fixed budgets for file blocks.
pub struct HeuristicEstimator;

impl TokenEstimator for HeuristicEstimator {
    fn estimate_message(&self, message: &Message) -> usize {
        estimate_message_tokens(message)
    }

    fn estimate_text(&self, text: &str) -> usize {
        estimate_tokens(text)
    }
}

/// Serialized snapshot of a session's context written by
/// [`ContextManager::persist`] and read back by
/// [`ContextManager::load`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PersistedSession {
    session_id: String,
    created_at: String,
    history: Vec<Message>,
    compaction_summary: Option<String>,
    total_tokens: usize,
}

/// Persistence failure returned by [`ContextManager::persist`] and
/// [`ContextManager::load`].
#[derive(Debug)]
pub enum ContextError {
    Io(std::io::Error),
    Json(serde_json::Error),
    NotFound(String),
}

impl std::fmt::Display for ContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "context persistence io error: {err}"),
            Self::Json(err) => write!(f, "context persistence json error: {err}"),
            Self::NotFound(session_id) => {
                write!(f, "no persisted context for session {session_id:?}")
            }
        }
    }
}

impl std::error::Error for ContextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Json(err) => Some(err),
            Self::NotFound(_) => None,
        }
    }
}

/// Rolling-window context configuration.
pub struct ContextConfig {
    /// Token hard limit for the stored history.
    pub max_tokens: usize,
    /// Rolling window: keep the most recent `recent_turns * 2` messages.
    pub recent_turns: usize,
    /// Session persistence directory. Stored now for AS-03 session
    /// persistence; not used by the V1 context manager.
    pub session_dir: PathBuf,
    /// Soft-trigger cushion (CM-V2a, decision table M1):
    /// `needs_compaction()` fires once the history exceeds
    /// `max_tokens - reserved_tokens`; `is_over_hard_limit()` fires at
    /// `max_tokens`.
    pub reserved_tokens: usize,
}

impl ContextConfig {
    /// V1-compatible constructor: keeps old struct-literal call sites
    /// readable while defaulting the soft-trigger cushion.
    pub fn new(max_tokens: usize, recent_turns: usize, session_dir: PathBuf) -> Self {
        Self {
            max_tokens,
            recent_turns,
            session_dir,
            reserved_tokens: DEFAULT_RESERVED_TOKENS,
        }
    }
}

/// Decomposition of the per-call token budget
/// `max_tokens = system + summary + windowed_history + input`
/// (CM-V2a, decision table M1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetSnapshot {
    pub system_tokens: usize,
    pub summary_tokens: usize,
    pub history_tokens: usize,
    pub input_tokens: usize,
    pub reserved_tokens: usize,
    pub max_tokens: usize,
}

impl BudgetSnapshot {
    /// Sum of every component except the reserved cushion.
    pub fn windowed_total(&self) -> usize {
        self.system_tokens + self.summary_tokens + self.history_tokens + self.input_tokens
    }

    /// `true` when the windowed total exceeds the soft trigger line
    /// (`max_tokens - reserved_tokens`).
    pub fn is_over_budget(&self) -> bool {
        self.windowed_total() > self.max_tokens.saturating_sub(self.reserved_tokens)
    }

    /// `true` when the windowed total exceeds the hard limit.
    pub fn is_over_hard_limit(&self) -> bool {
        self.windowed_total() > self.max_tokens
    }
}

/// Manages conversation history for a single agent session.
pub struct ContextManager {
    config: ContextConfig,
    history: Vec<Message>,
    estimator: Box<dyn TokenEstimator>,
}

impl ContextManager {
    pub fn new(config: ContextConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            estimator: Box::new(HeuristicEstimator),
        }
    }

    /// Install a custom estimator (exact tokenizer, calibrated model).
    /// The default [`HeuristicEstimator`] remains when unset.
    pub fn with_estimator(mut self, estimator: Box<dyn TokenEstimator>) -> Self {
        self.estimator = estimator;
        self
    }

    /// Build the message vector for a provider call:
    /// `[system, ...compacted_history, ...input]`.
    ///
    /// The stored history is truncated to the most recent
    /// `recent_turns * 2` messages before the vector is built. The
    /// system prompt is re-attached on every call and is never stored.
    pub fn prepare_messages(&mut self, system: &str, input: &[Message]) -> Vec<Message> {
        let window = self.config.recent_turns.saturating_mul(2);
        let start = self.history.len().saturating_sub(window);
        self.history = self.history.split_off(start);

        let mut messages = Vec::with_capacity(self.history.len() + 1 + input.len());
        messages.push(Message::system(system));
        messages.extend(self.history.iter().cloned());
        messages.extend(input.iter().cloned());
        messages
    }

    /// Append a completed turn (user input + assistant output) to the
    /// conversation history.
    pub fn commit_turn(&mut self, messages: &[Message]) {
        self.history.extend(messages.iter().cloned());
    }

    /// Estimated total tokens of the stored history.
    pub fn token_count(&self) -> usize {
        self.history
            .iter()
            .map(|m| self.estimator.estimate_message(m))
            .sum()
    }

    /// Estimated tokens of the history message at `index`. Returns
    /// `None` when `index` is outside the stored history.
    pub fn message_tokens(&self, index: usize) -> Option<usize> {
        self.history.get(index).map(|m| self.estimator.estimate_message(m))
    }

    /// Budget decomposition of the next provider call: `system`,
    /// `input` and the reserved cushion are supplied by the caller's
    /// perspective (`system_tokens` / `input_tokens`), while
    /// `history_tokens` and `summary_tokens` come from the stored
    /// state (summary is always 0 until CM-V2c fills it).
    pub fn budget_snapshot(&self, system_tokens: usize, input_tokens: usize) -> BudgetSnapshot {
        BudgetSnapshot {
            system_tokens,
            summary_tokens: 0,
            history_tokens: self.token_count(),
            input_tokens,
            reserved_tokens: self.config.reserved_tokens,
            max_tokens: self.config.max_tokens,
        }
    }

    /// `true` when the stored history exceeds the soft trigger line
    /// (`max_tokens - reserved_tokens`); the turn should compact
    /// before sending.
    pub fn needs_compaction(&self) -> bool {
        self.token_count() > self.config.max_tokens.saturating_sub(self.config.reserved_tokens)
    }

    /// `true` when the stored history exceeds `max_tokens`; the turn
    /// must compact (or drop) before sending.
    pub fn is_over_hard_limit(&self) -> bool {
        self.token_count() > self.config.max_tokens
    }

    /// Persist this session's context to
    /// `{session_dir}/{session_id}.json`, creating the session
    /// directory if it does not exist.
    pub fn persist(&self, session_id: &str) -> Result<(), ContextError> {
        let state = PersistedSession {
            session_id: session_id.to_owned(),
            created_at: created_at(),
            history: self.history.clone(),
            compaction_summary: None,
            total_tokens: self.token_count(),
        };
        std::fs::create_dir_all(&self.config.session_dir).map_err(ContextError::Io)?;
        let path = self.config.session_dir.join(format!("{session_id}.json"));
        let json = serde_json::to_string_pretty(&state).map_err(ContextError::Json)?;
        std::fs::write(path, json).map_err(ContextError::Io)
    }

    /// Load a session's context previously written by
    /// [`ContextManager::persist`]. The returned manager uses `config`
    /// for the window and token threshold; only the history is restored
    /// (a custom estimator installed via [`ContextManager::with_estimator`]
    /// must be re-applied by the caller after loading).
    pub fn load(session_id: &str, config: ContextConfig) -> Result<Self, ContextError> {
        let path = config.session_dir.join(format!("{session_id}.json"));
        let json = std::fs::read(&path).map_err(|err| {
            if err.kind() == std::io::ErrorKind::NotFound {
                ContextError::NotFound(session_id.to_owned())
            } else {
                ContextError::Io(err)
            }
        })?;
        let state: PersistedSession = serde_json::from_slice(&json).map_err(ContextError::Json)?;
        Ok(Self {
            config,
            history: state.history,
            estimator: Box::new(HeuristicEstimator),
        })
    }
}

/// Unix epoch seconds used as the `created_at` persisted field.
fn created_at() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs()
        .to_string()
}

/// Estimate the token count of a message from its content blocks: text
/// blocks use the character heuristic below, file blocks use fixed
/// per-block budgets ([`IMAGE_TOKENS_PER_BLOCK`] for images,
/// [`OTHER_FILE_TOKENS_PER_BLOCK`] otherwise).
fn estimate_message_tokens(message: &Message) -> usize {
    message
        .blocks()
        .iter()
        .map(|block| match block {
            ContentBlock::Text(text) => estimate_tokens(text),
            ContentBlock::File(file) if file.media_type().starts_with("image/") => {
                IMAGE_TOKENS_PER_BLOCK
            }
            ContentBlock::File(_) => OTHER_FILE_TOKENS_PER_BLOCK,
        })
        .sum()
}

/// Estimate the token count of `text`.
///
/// Counting is character-based (`chars().count()`), so multi-byte
/// non-CJK characters such as Cyrillic, Arabic, and accented Latin are
/// handled correctly and fall into the "other" bucket.
pub(crate) fn estimate_tokens(text: &str) -> usize {
    let chars = text.chars().count();
    let cjk = text.chars().filter(|c| *c as u32 > 0x2E80).count();
    let other = chars - cjk;
    (cjk / 2) + (other / 4) + 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use crate::provider::{ContentBlock, FileContentBlock, ToolCall, ToolCallId};

    fn config(recent_turns: usize) -> ContextConfig {
        ContextConfig::new(10_000, recent_turns, PathBuf::from("/tmp/reimagine-test"))
    }

    #[test]
    fn prepare_keeps_full_history_within_window() {
        let mut manager = ContextManager::new(config(2));
        manager.commit_turn(&[Message::user("u1"), Message::assistant("a1")]);
        manager.commit_turn(&[Message::user("u2"), Message::assistant("a2")]);

        let messages = manager.prepare_messages("sys", &[Message::user("u3")]);
        assert_eq!(messages.len(), 6);
        assert_eq!(messages[0].role(), "system");
        assert_eq!(messages[0].content(), "sys");
        assert_eq!(messages[1].content(), "u1");
        assert_eq!(messages[2].content(), "a1");
        assert_eq!(messages[3].content(), "u2");
        assert_eq!(messages[4].content(), "a2");
        assert_eq!(messages[5].content(), "u3");
    }

    #[test]
    fn prepare_drops_oldest_messages_beyond_window() {
        let mut manager = ContextManager::new(config(2));
        for i in 0..3 {
            manager.commit_turn(&[
                Message::user(format!("u{i}")),
                Message::assistant(format!("a{i}")),
            ]);
        }

        // Window is 4; the oldest 2 of 6 history messages are dropped.
        let messages = manager.prepare_messages("sys", &[Message::user("u3")]);
        assert_eq!(messages.len(), 6);
        assert_eq!(messages[0].content(), "sys");
        assert_eq!(messages[1].content(), "u1");
        assert_eq!(messages[2].content(), "a1");
        assert_eq!(messages[3].content(), "u2");
        assert_eq!(messages[4].content(), "a2");
        assert_eq!(messages[5].content(), "u3");
    }

    #[test]
    fn prepare_with_empty_input_still_attaches_system() {
        let mut manager = ContextManager::new(config(2));
        let messages = manager.prepare_messages("sys", &[]);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role(), "system");
        assert_eq!(messages[0].content(), "sys");
    }

    #[test]
    fn commit_turn_accumulates_history() {
        let mut manager = ContextManager::new(config(2));
        assert_eq!(manager.token_count(), 0);
        manager.commit_turn(&[Message::user("u1"), Message::assistant("a1")]);
        manager.commit_turn(&[Message::user("u2")]);

        // A single character counts the 4-token per-message overhead.
        assert_eq!(manager.token_count(), 12);
    }

    #[test]
    fn token_count_sums_history_content() {
        let mut manager = ContextManager::new(config(2));
        manager.commit_turn(&[Message::user("a"), Message::assistant("b")]);
        assert_eq!(manager.token_count(), 8);
    }

    #[test]
    fn needs_compaction_flips_when_tokens_exceed_soft_trigger() {
        // max_tokens 100, reserved 90 -> soft line at 10.
        let mut manager = ContextManager::new(ContextConfig {
            max_tokens: 100,
            recent_turns: 2,
            session_dir: PathBuf::from("/tmp/reimagine-test"),
            reserved_tokens: 90,
        });
        assert!(!manager.needs_compaction());
        manager.commit_turn(&[Message::user("a"), Message::assistant("b")]);
        // 4 + 4 = 8 <= 10: under the soft line.
        assert!(!manager.needs_compaction());
        manager.commit_turn(&[Message::user("c"), Message::assistant("d")]);
        // 16 > 10: soft trigger fires.
        assert!(manager.needs_compaction());
    }

    #[test]
    fn is_over_hard_limit_flips_at_max_tokens() {
        let mut manager = ContextManager::new(ContextConfig::new(
            20,
            2,
            PathBuf::from("/tmp/reimagine-test"),
        ));
        manager.commit_turn(&[Message::user("a"), Message::assistant("b")]);
        // 8 <= 20; soft line (20 - 16k -> 0) already fired, hard has not.
        assert!(manager.needs_compaction());
        assert!(!manager.is_over_hard_limit());
        manager.commit_turn(&[Message::user("c"), Message::assistant("d")]);
        // 16 <= 20: still under the hard limit.
        assert!(!manager.is_over_hard_limit());
        manager.commit_turn(&[Message::user("e"), Message::assistant("f")]);
        // 24 > 20: hard limit fires.
        assert!(manager.is_over_hard_limit());
    }

    #[test]
    fn soft_trigger_precedes_hard_limit() {
        let mut manager = ContextManager::new(ContextConfig {
            max_tokens: 100,
            recent_turns: 2,
            session_dir: PathBuf::from("/tmp/reimagine-test"),
            reserved_tokens: 40,
        });
        manager.commit_turn(&[Message::user("a"), Message::assistant("b")]);
        // 8 < 60: neither fires.
        assert!(!manager.needs_compaction());
        assert!(!manager.is_over_hard_limit());
        for i in 0..6 {
            manager.commit_turn(&[
                Message::user(format!("u{i}")),
                Message::assistant(format!("a{i}")),
            ]);
        }
        // 7 turns = 56 tokens: soft (60) not yet fired, hard (100) not.
        assert!(!manager.needs_compaction());
        manager.commit_turn(&[Message::user("u7"), Message::assistant("a7")]);
        // 64 > 60: soft fires; 64 < 100: hard still quiet.
        assert!(manager.needs_compaction());
        assert!(!manager.is_over_hard_limit());
    }

    #[test]
    fn new_config_defaults_reserved_cushion() {
        let config = ContextConfig::new(64_000, 20, PathBuf::from("/tmp/reimagine-test"));
        assert_eq!(config.reserved_tokens, 16_000);
    }

    #[test]
    fn budget_snapshot_decomposes_call_budget() {
        let mut manager = ContextManager::new(ContextConfig {
            max_tokens: 100,
            recent_turns: 2,
            session_dir: PathBuf::from("/tmp/reimagine-test"),
            reserved_tokens: 40,
        });
        manager.commit_turn(&[Message::user("a"), Message::assistant("b")]);
        let snapshot = manager.budget_snapshot(/* system */ 20, /* input */ 12);
        assert_eq!(snapshot.system_tokens, 20);
        assert_eq!(snapshot.summary_tokens, 0);
        assert_eq!(snapshot.history_tokens, 8);
        assert_eq!(snapshot.input_tokens, 12);
        assert_eq!(snapshot.reserved_tokens, 40);
        assert_eq!(snapshot.max_tokens, 100);
        assert_eq!(snapshot.windowed_total(), 40);
        // Soft line = 100 - 40 = 60; hard line = 100.
        assert!(!snapshot.is_over_budget());
        assert!(!snapshot.is_over_hard_limit());

        manager.commit_turn(&[Message::user("u1"), Message::assistant("a1")]);
        let snapshot = manager.budget_snapshot(20, 12);
        assert_eq!(snapshot.windowed_total(), 48);
        assert!(!snapshot.is_over_budget());

        manager.commit_turn(&[Message::user("u2"), Message::assistant("a2")]);
        let snapshot = manager.budget_snapshot(20, 12);
        // system 20 + history 24 (3 turns) + input 12 = 56 <= 60: quiet.
        assert_eq!(snapshot.windowed_total(), 56);
        assert!(!snapshot.is_over_budget());
    }

    #[test]
    fn budget_snapshot_flips_over_budget_with_large_history() {
        let mut manager = ContextManager::new(ContextConfig {
            max_tokens: 100,
            recent_turns: 2,
            session_dir: PathBuf::from("/tmp/reimagine-test"),
            reserved_tokens: 40,
        });
        for i in 0..10 {
            manager.commit_turn(&[
                Message::user(format!("u{i}")),
                Message::assistant(format!("a{i}")),
            ]);
        }
        // 20 messages * 4 tokens = 80. With system 20 + input 0: 100 > 60 soft.
        let snapshot = manager.budget_snapshot(20, 0);
        assert_eq!(snapshot.history_tokens, 80);
        assert_eq!(snapshot.windowed_total(), 100);
        assert!(snapshot.is_over_budget());
        // 100 > 100 is false: exactly at the hard line.
        assert!(!snapshot.is_over_hard_limit());

        // 20 system + 80 history + 1 input message = 104 > 100 hard.
        let snapshot = manager.budget_snapshot(20, 4);
        assert!(snapshot.is_over_hard_limit());
    }

    #[test]
    fn message_tokens_attributes_per_message() {
        let mut manager = ContextManager::new(ContextConfig::new(
            10_000,
            2,
            PathBuf::from("/tmp/reimagine-test"),
        ));
        manager.commit_turn(&[Message::user("abc"), Message::assistant("defg")]);
        // "abc" = 0 (3 chars / 4) + 4 overhead; "defg" = 1 + 4.
        assert_eq!(manager.message_tokens(0), Some(4));
        assert_eq!(manager.message_tokens(1), Some(5));
        assert_eq!(manager.message_tokens(2), None);
    }

    struct FlatEstimator(usize);

    impl TokenEstimator for FlatEstimator {
        fn estimate_message(&self, _message: &Message) -> usize {
            self.0
        }

        fn estimate_text(&self, _text: &str) -> usize {
            self.0
        }
    }

    #[test]
    fn custom_estimator_replaces_heuristic() {
        let mut manager = ContextManager::new(ContextConfig::new(
            10_000,
            2,
            PathBuf::from("/tmp/reimagine-test"),
        ))
        .with_estimator(Box::new(FlatEstimator(7)));
        manager.commit_turn(&[Message::user("a"), Message::assistant("b")]);
        assert_eq!(manager.token_count(), 14);
        assert_eq!(manager.message_tokens(0), Some(7));
        assert_eq!(
            manager.budget_snapshot(7, 7).history_tokens,
            14
        );
    }

    #[test]
    fn empty_string_counts_overhead_only() {
        assert_eq!(estimate_tokens(""), 4);
    }

    #[test]
    fn pure_english_is_four_chars_per_token() {
        let text = "a".repeat(1000);
        assert_eq!(estimate_tokens(&text), 254);
    }

    #[test]
    fn pure_cjk_is_two_chars_per_token() {
        let text = "界".repeat(1000);
        assert_eq!(estimate_tokens(&text), 504);
    }

    #[test]
    fn mixed_combines_both_estimators() {
        let mut text = "a".repeat(400);
        text.push_str(&"界".repeat(600));
        assert_eq!(estimate_tokens(&text), 404);
    }

    #[test]
    fn whitespace_heavy_counts_as_other() {
        let text = " ".repeat(1000);
        assert_eq!(estimate_tokens(&text), 254);
    }

    #[test]
    fn cyrillic_counts_as_other() {
        assert_eq!(estimate_tokens("Привет мир"), 6);
    }

    #[test]
    fn image_block_counts_fixed_budget() {
        let mut manager = ContextManager::new(config(2));
        manager.commit_turn(&[Message::user_with_blocks(vec![
            ContentBlock::Text("look".into()),
            ContentBlock::File(FileContentBlock::data("image/png", "AAAA")),
        ])]);
        // "look": 4 chars -> 1 + 4 overhead; plus the image budget.
        assert_eq!(manager.token_count(), 5 + IMAGE_TOKENS_PER_BLOCK);
    }

    #[test]
    fn non_image_file_block_counts_fallback_budget() {
        let mut manager = ContextManager::new(config(2));
        manager.commit_turn(&[Message::user_with_blocks(vec![ContentBlock::File(
            FileContentBlock::data("audio/mpeg", "AAAA"),
        )])]);
        // No text block -> no character estimate; only the fallback.
        assert_eq!(manager.token_count(), OTHER_FILE_TOKENS_PER_BLOCK);
    }

    #[test]
    fn text_only_history_is_unchanged_by_blocks_path() {
        let mut manager = ContextManager::new(config(2));
        manager.commit_turn(&[Message::user("a"), Message::assistant("b")]);
        assert_eq!(manager.token_count(), 8);
    }

    #[test]
    fn arabic_counts_as_other() {
        assert_eq!(estimate_tokens("مرحبا بالعالم"), 7);
    }

    fn temp_session_dir(prefix: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("reimagine-agent-context-{prefix}-{nonce}"))
    }

    #[test]
    fn persist_load_round_trips_history() {
        let dir = temp_session_dir("round-trip");
        let cfg = ContextConfig::new(10_000, 2, dir.clone());
        let mut manager = ContextManager::new(cfg);
        manager.commit_turn(&[
            Message::user("u1"),
            Message::assistant("a1"),
            Message::user("u2"),
            Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(
                    ToolCallId::new("c1"),
                    "echo",
                    json!({"x": 1}),
                )],
            ),
        ]);
        manager.persist("sess-1").expect("persist failed");

        let loaded = ContextManager::load(
            "sess-1",
            ContextConfig::new(10_000, 2, dir),
        )
        .expect("load failed");
        assert_eq!(loaded.token_count(), manager.token_count());
        assert_eq!(loaded.history, manager.history);
    }

    #[test]
    fn persist_round_trips_file_blocks() {
        let dir = temp_session_dir("file-blocks");
        let cfg = ContextConfig::new(10_000, 2, dir.clone());
        let mut manager = ContextManager::new(cfg);
        manager.commit_turn(&[Message::user_with_blocks(vec![
            ContentBlock::Text("describe".into()),
            ContentBlock::File(FileContentBlock::data("image/png", "AAAA")),
        ])]);
        manager.persist("sess-1").expect("persist failed");

        let loaded = ContextManager::load(
            "sess-1",
            ContextConfig::new(10_000, 2, dir),
        )
        .expect("load failed");
        assert_eq!(loaded.history, manager.history);
        assert_eq!(loaded.token_count(), manager.token_count());
    }

    #[test]
    fn load_old_string_content_format_fails() {
        // Development-period format change: files written with the V1
        // `content` string fail to load and must be skipped by callers
        // (the daemon's resume path already logs and skips).
        let dir = temp_session_dir("old-format");
        std::fs::create_dir_all(&dir).expect("mkdir failed");
        std::fs::write(
            dir.join("sess-1.json"),
            r#"{"session_id":"sess-1","created_at":"0","history":[{"role":"user","content":"hi"}],"compaction_summary":null,"total_tokens":4}"#,
        )
        .expect("write failed");
        let result = ContextManager::load(
            "sess-1",
            ContextConfig::new(10_000, 2, dir),
        );
        assert!(result.is_err());
    }

    #[test]
    fn persist_creates_session_dir() {
        let dir = temp_session_dir("mkdir");
        let manager = ContextManager::new(ContextConfig::new(10_000, 2, dir.clone()));
        manager.persist("sess-1").expect("persist failed");
        assert!(dir.join("sess-1.json").is_file());
    }

    #[test]
    fn load_missing_session_returns_err() {
        let dir = temp_session_dir("missing");
        let result = ContextManager::load(
            "nope",
            ContextConfig::new(10_000, 2, dir),
        );
        assert!(result.is_err());
    }

    #[test]
    fn persist_writes_compaction_summary_none() {
        let dir = temp_session_dir("summary");
        let manager = ContextManager::new(ContextConfig::new(10_000, 2, dir.clone()));
        manager.persist("sess-1").expect("persist failed");

        let json = std::fs::read_to_string(dir.join("sess-1.json")).expect("read failed");
        let state: serde_json::Value = serde_json::from_str(&json).expect("json parse failed");
        assert_eq!(state["session_id"], "sess-1");
        assert_eq!(state["history"], serde_json::Value::Array(vec![]));
        assert!(state["compaction_summary"].is_null());
        assert_eq!(state["total_tokens"], 0);
    }
}
