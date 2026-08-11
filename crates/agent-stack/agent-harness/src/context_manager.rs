//! Conversation context management for the agent loop.
//!
//! [`ContextManager`] owns the per-session conversation history and
//! applies a budget-driven rolling window (CM-V2b, decision table M2):
//! the most recent `tail_turns` complete turns are kept verbatim, the
//! older turns are kept back to the soft budget line, and everything
//! older than that is evicted in whole turn units — never splitting a
//! user↔assistant turn or a tool-call↔tool-result pair. V1 did not
//! summarize dropped messages; V2 keeps the same drop semantics until
//! the summarization module (CM-V2c) lands.
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

/// Default verbatim tail length (CM-V2b, decision table M2): the most
/// recent `tail_turns` complete turns are kept word-for-word.
const DEFAULT_TAIL_TURNS: usize = 2;

/// Default compaction target ratio (CM-V2b, decision table M3): a
/// compaction pass aims to leave the history at this fraction of the
/// window budget. Consumed by the summarization module (CM-V2c).
const DEFAULT_COMPACT_TO_RATIO: f32 = 0.6;

/// Default prune threshold (CM-V2b, decision table M2): once old tool
/// outputs accumulate past this many tokens, the prune stage replaces
/// them with placeholders until `prune_target_tokens` is reached.
const DEFAULT_PRUNE_THRESHOLD_TOKENS: usize = 40_000;

/// Default prune target (CM-V2b, decision table M2).
const DEFAULT_PRUNE_TARGET_TOKENS: usize = 20_000;

/// Default per-record truncation limit (CM-V2b, decision table M2):
/// tool results are truncated to this many estimated tokens when
/// committed.
const DEFAULT_RECORD_TRUNCATE_TOKENS: usize = 2_000;

/// Verbatim tail budget, as a fraction of the available window
/// (`max_tokens - reserved_tokens`), clamped to [`TAIL_BUDGET_MIN`] /
/// [`TAIL_BUDGET_MAX`] (CM-V2b, decision table M2).
const TAIL_BUDGET_RATIO: f32 = 0.25;
const TAIL_BUDGET_MIN: usize = 2_000;
const TAIL_BUDGET_MAX: usize = 8_000;

/// Placeholder text that replaces pruned tool outputs.
const TOOL_OUTPUT_PRUNE_PLACEHOLDER: &str = "[tool output pruned]";

/// Suffix appended to tool results truncated at commit time.
const TOOL_OUTPUT_TRUNCATED_SUFFIX: &str = "\n[output truncated]";

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
    /// Verbatim tail length (CM-V2b): the most recent `tail_turns`
    /// complete turns are kept word-for-word.
    pub tail_turns: usize,
    /// Compaction target ratio (CM-V2b, consumed by CM-V2c): a
    /// compaction pass aims to leave the history at this fraction of
    /// the window budget.
    pub compact_to_ratio: f32,
    /// Prune threshold (CM-V2b): tool outputs older than the budget
    /// line are replaced with placeholders once their accumulated
    /// tokens exceed this.
    pub prune_threshold_tokens: usize,
    /// Prune target (CM-V2b): the prune stage stops once remaining tool
    /// output tokens are at or below this.
    pub prune_target_tokens: usize,
    /// Per-record truncation limit (CM-V2b): tool results are truncated
    /// to this many estimated tokens when committed.
    pub record_truncate_tokens: usize,
}

impl ContextConfig {
    /// V1-compatible constructor: keeps old struct-literal call sites
    /// readable while defaulting every window-policy parameter.
    pub fn new(max_tokens: usize, recent_turns: usize, session_dir: PathBuf) -> Self {
        Self {
            max_tokens,
            recent_turns,
            session_dir,
            reserved_tokens: DEFAULT_RESERVED_TOKENS,
            tail_turns: DEFAULT_TAIL_TURNS,
            compact_to_ratio: DEFAULT_COMPACT_TO_RATIO,
            prune_threshold_tokens: DEFAULT_PRUNE_THRESHOLD_TOKENS,
            prune_target_tokens: DEFAULT_PRUNE_TARGET_TOKENS,
            record_truncate_tokens: DEFAULT_RECORD_TRUNCATE_TOKENS,
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
    /// Number of messages at the front of `history` that the window
    /// never evicts (pinned goal/constraints and the sticky summary,
    /// CM-V2b decision table M2). Managed by the summarization module
    /// (CM-V2c); public accessors keep the window honest.
    sticky_count: usize,
}

impl ContextManager {
    pub fn new(config: ContextConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            estimator: Box::new(HeuristicEstimator),
            sticky_count: 0,
        }
    }

    /// Install a custom estimator (exact tokenizer, calibrated model).
    /// The default [`HeuristicEstimator`] remains when unset.
    pub fn with_estimator(mut self, estimator: Box<dyn TokenEstimator>) -> Self {
        self.estimator = estimator;
        self
    }

    /// Number of sticky front messages the window never evicts.
    pub fn sticky_count(&self) -> usize {
        self.sticky_count
    }

    /// Build the message vector for a provider call:
    /// `[system, ...sticky, ...windowed_history, ...input]`.
    ///
    /// The stored history is windowed before the vector is built
    /// (CM-V2b): the most recent `tail_turns` complete turns are kept
    /// verbatim, older turns are kept back to the soft budget line
    /// (`max_tokens - reserved_tokens`), and older messages are evicted
    /// in whole turn units. The sticky prefix (summary / pinned blocks)
    /// is never evicted. The system prompt is re-attached on every call
    /// and is never stored.
    pub fn prepare_messages(&mut self, system: &str, input: &[Message]) -> Vec<Message> {
        self.apply_window();

        let mut messages = Vec::with_capacity(self.history.len() + 1 + input.len());
        messages.push(Message::system(system));
        messages.extend(self.history.iter().cloned());
        messages.extend(input.iter().cloned());
        messages
    }

    /// Append a completed turn (user input + assistant output) to the
    /// conversation history. Tool results are truncated to
    /// `record_truncate_tokens` at commit time (CM-V2b).
    pub fn commit_turn(&mut self, messages: &[Message]) {
        for message in messages {
            if message.role() == "tool" {
                self.history.push(truncate_tool_result(message, self.config.record_truncate_tokens));
            } else {
                self.history.push(message.clone());
            }
        }
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
            sticky_count: 0,
        })
    }

    /// Window the stored history in place (CM-V2b, decision table M2):
    ///
    /// 1. **Prune stage**: tool outputs older than the verbatim tail
    ///    are replaced with placeholders once their accumulated tokens
    ///    exceed `prune_threshold_tokens`, stopping at
    ///    `prune_target_tokens`.
    /// 2. **Verbatim tail**: the most recent `tail_turns` complete turns
    ///    are kept word-for-word, bounded by the tail budget
    ///    (`25%` of the available window, 2k-8k clamp).
    /// 3. **Middle window**: turns between the sticky prefix and the
    ///    tail are kept from newest to oldest until the soft budget
    ///    line (`max_tokens - reserved_tokens`) is filled; the rest is
    ///    evicted in whole turn units (a user↔assistant turn and its
    ///    tool-call↔tool-result pairs are never split). A single turn
    ///    larger than the window budget is cut at its first assistant
    ///    message, keeping the user message.
    fn apply_window(&mut self) {
        self.prune_old_tool_outputs();
        let sticky_end = self.sticky_count.min(self.history.len());
        let turns = turn_spans(&self.history[sticky_end..]);

        let window_budget = self
            .config
            .max_tokens
            .saturating_sub(self.config.reserved_tokens);
        let tail_budget = tail_budget(window_budget);
        let sticky_tokens: usize = (0..sticky_end)
            .map(|i| self.message_tokens(i).unwrap_or(0))
            .sum();

        // Verbatim tail: the most recent `tail_turns` turn spans; drop
        // older tail turns while the tail exceeds its budget (never
        // splits a turn).
        let mut tail_start = turns.len().saturating_sub(self.config.tail_turns.max(1));
        let mut tail_tokens: usize = turns[tail_start..]
            .iter()
            .map(|span| self.span_tokens(sticky_end, span))
            .sum();
        while tail_start + 1 < turns.len() && tail_tokens > tail_budget {
            tail_tokens = tail_tokens.saturating_sub(self.span_tokens(sticky_end, &turns[tail_start]));
            tail_start += 1;
        }

        // Middle window: newest-to-oldest turns between the sticky
        // prefix and the tail, filling the remaining budget.
        let mut middle_budget = window_budget.saturating_sub(sticky_tokens + tail_tokens);
        let mut keep_from = tail_start;
        while keep_from > 0 {
            let span = turns[keep_from - 1];
            let span_tokens = self.span_tokens(sticky_end, &span);
            if span_tokens > middle_budget {
                break;
            }
            middle_budget -= span_tokens;
            keep_from -= 1;
        }

        // Oversized turn: the first turn that did not fit is cut at its
        // first assistant message when it alone exceeds the window
        // budget, so the user message is still kept. Skipped when the
        // remaining middle budget is exhausted (a user prefix would not
        // fit either — plain eviction is the honest choice).
        let mut oversized_cut: Option<(usize, usize)> = None;
        if keep_from > 0 && middle_budget > 0 {
            let span = turns[keep_from - 1];
            let span_tokens = self.span_tokens(sticky_end, &span);
            if span_tokens > window_budget
                && let Some(off) = first_assistant_offset(&self.history[sticky_end..], span)
            {
                let abs_start = sticky_end + span.0;
                oversized_cut = Some((abs_start + off, abs_start + span.1));
                keep_from -= 1;
            }
        }

        // Apply the oversized cut first (it lies inside the keep
        // region), then evict everything before the keep region.
        if let Some((cut_start, cut_end)) = oversized_cut {
            self.history.drain(cut_start..cut_end);
        }
        let keep_abs_start = if keep_from >= turns.len() {
            self.history.len().min(sticky_end + turns.last().map(|s| s.0 + s.1).unwrap_or(0))
        } else {
            sticky_end + turns[keep_from].0
        };
        if keep_abs_start > sticky_end {
            self.history.drain(sticky_end..keep_abs_start);
        }
    }

    /// Estimated tokens of a turn span `(start, len)` whose indexes
    /// are relative to the sticky-offset slice, resolved to absolute
    /// history indexes via `sticky_end`.
    fn span_tokens(&self, sticky_end: usize, span: &(usize, usize)) -> usize {
        (span.0..span.0 + span.1)
            .map(|i| self.message_tokens(sticky_end + i).unwrap_or(0))
            .sum()
    }

    /// Prune stage (CM-V2b): once tool outputs outside the verbatim
    /// tail accumulate past `prune_threshold_tokens`, replace the
    /// oldest of them with placeholders until `prune_target_tokens`
    /// is reached. Replaced outputs keep their `tool_call_id` so
    /// tool-call↔result pairing is preserved.
    fn prune_old_tool_outputs(&mut self) {
        let tool_tokens: usize = self
            .history
            .iter()
            .map(|m| {
                if m.role() == "tool" {
                    self.estimator.estimate_message(m)
                } else {
                    0
                }
            })
            .sum();
        if tool_tokens <= self.config.prune_threshold_tokens {
            return;
        }

        let sticky_end = self.sticky_count.min(self.history.len());
        let turns = turn_spans(&self.history[sticky_end..]);
        let tail_start = turns
            .len()
            .saturating_sub(self.config.tail_turns.max(1));

        let placeholder = Message::tool_result(
            crate::provider::ToolCallId::new("pruned"),
            TOOL_OUTPUT_PRUNE_PLACEHOLDER,
        );
        let placeholder_tokens = self.estimator.estimate_message(&placeholder);
        let mut remaining = tool_tokens;
        'outer: for span in &turns[..tail_start] {
            for idx in span.0..span.0 + span.1 {
                let absolute = sticky_end + idx;
                if self.history[absolute].role() != "tool" {
                    continue;
                }
                let tokens = self
                    .estimator
                    .estimate_message(&self.history[absolute]);
                let id = self.history[absolute]
                    .tool_call_id()
                    .cloned()
                    .unwrap_or_else(|| crate::provider::ToolCallId::new("pruned"));
                self.history[absolute] = Message::tool_result(id, TOOL_OUTPUT_PRUNE_PLACEHOLDER);
                remaining = remaining
                    .saturating_sub(tokens)
                    .saturating_add(placeholder_tokens);
                if remaining <= self.config.prune_target_tokens {
                    break 'outer;
                }
            }
        }
    }
}

/// Split `messages` into complete turn spans. A turn starts at a
/// `user` message and runs to the message before the next `user`
/// message; leading non-`user` messages (resume artifacts) join the
/// first turn. Returns `(start_index, len)` pairs.
fn turn_spans(messages: &[Message]) -> Vec<(usize, usize)> {
    if messages.is_empty() {
        return Vec::new();
    }
    let mut spans: Vec<(usize, usize)> = vec![(0, 0)];
    let mut seen_user = messages[0].role() == "user";
    for (idx, message) in messages.iter().enumerate().skip(1) {
        if message.role() == "user" {
            if seen_user {
                let last = spans.last_mut().expect("non-empty");
                last.1 = idx - last.0;
                spans.push((idx, 0));
            } else {
                seen_user = true;
            }
        }
    }
    let last = spans.last_mut().expect("non-empty");
    last.1 = messages.len() - last.0;
    spans
}

/// Offset of the first `assistant` message inside the turn slice, if
/// any — the cut point for a turn that alone exceeds the window
/// budget.
fn first_assistant_offset(messages: &[Message], span: (usize, usize)) -> Option<usize> {
    (0..span.1)
        .find(|offset| messages[span.0 + offset].role() == "assistant")
}

/// Verbatim tail budget: `25%` of the available window, clamped to
/// [`TAIL_BUDGET_MIN`] / [`TAIL_BUDGET_MAX`] (CM-V2b).
fn tail_budget(window_budget: usize) -> usize {
    let ratio = (window_budget as f32 * TAIL_BUDGET_RATIO) as usize;
    ratio.clamp(TAIL_BUDGET_MIN, TAIL_BUDGET_MAX)
}

/// Truncate a `tool` result message to `record_truncate_tokens`
/// estimated tokens at commit time (CM-V2b). The truncation is
/// character-based (an ASCII approximation of the token budget) with
/// the truncated marker appended; messages already within budget are
/// returned unchanged.
fn truncate_tool_result(message: &Message, record_truncate_tokens: usize) -> Message {
    if message.role() != "tool" {
        return message.clone();
    }
    let content = message.content();
    let text = match message.blocks().first() {
        Some(ContentBlock::Text(text)) => text.as_str(),
        _ => content,
    };
    let limit_chars = record_truncate_tokens.saturating_mul(4);
    if text.chars().count() <= limit_chars {
        return message.clone();
    }
    let truncated: String = text.chars().take(limit_chars).collect();
    Message::tool_result(
        message.tool_call_id().cloned().unwrap_or_else(|| crate::provider::ToolCallId::new("truncated")),
        format!("{truncated}{TOOL_OUTPUT_TRUNCATED_SUFFIX}"),
    )
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
            reserved_tokens: 90,
            ..ContextConfig::new(100, 2, PathBuf::from("/tmp/reimagine-test"))
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
            reserved_tokens: 40,
            ..ContextConfig::new(100, 2, PathBuf::from("/tmp/reimagine-test"))
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
            reserved_tokens: 40,
            ..ContextConfig::new(100, 2, PathBuf::from("/tmp/reimagine-test"))
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
            reserved_tokens: 40,
            ..ContextConfig::new(100, 2, PathBuf::from("/tmp/reimagine-test"))
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

    fn window_config(max_tokens: usize, reserved_tokens: usize, tail_turns: usize) -> ContextConfig {
        ContextConfig {
            tail_turns,
            ..ContextConfig::new(max_tokens, 20, PathBuf::from("/tmp/reimagine-test"))
        }
        .with_reserved(reserved_tokens)
    }

    impl ContextConfig {
        fn with_reserved(mut self, reserved_tokens: usize) -> Self {
            self.reserved_tokens = reserved_tokens;
            self
        }
    }

    fn turn(i: usize, chars: usize) -> Vec<Message> {
        vec![
            Message::user(format!("u{i}:{}", "a".repeat(chars))),
            Message::assistant(format!("a{i}:{}", "b".repeat(chars))),
        ]
    }

    fn tool_turn(i: usize, chars: usize) -> Vec<Message> {
        vec![
            Message::user(format!("u{i}:{}", "a".repeat(chars))),
            Message::assistant_with_tool_calls(
                format!("call{i}"),
                vec![ToolCall::new(
                    ToolCallId::new(format!("call-{i}")),
                    "echo",
                    json!({"x": i}),
                )],
            ),
            Message::tool_result(ToolCallId::new(format!("call-{i}")), "x".repeat(chars)),
            Message::assistant(format!("a{i}:{}", "b".repeat(chars))),
        ]
    }

    fn text_of(message: &Message) -> String {
        message.content().to_owned()
    }

    #[test]
    fn window_keeps_verbatim_tail_and_drops_oldest() {
        // Window budget 8k, tail budget 2k (clamped), tail_turns 2.
        // A turn of 2x400 chars is ~208 tokens; 60 turns ~12.5k tokens.
        let mut manager = ContextManager::new(window_config(10_000, 2_000, 2));
        for i in 0..60 {
            manager.commit_turn(&turn(i, 400));
        }

        let messages = manager.prepare_messages("sys", &[]);
        let history: Vec<Message> = messages[1..].to_vec();
        // The 2 newest turns survive verbatim.
        let last = history.last().expect("non-empty");
        assert_eq!(text_of(last), format!("a59:{}", "b".repeat(400)));
        assert!(history.iter().any(|m| text_of(m).starts_with("u59")));
        assert!(history.iter().any(|m| text_of(m).starts_with("u58")));
        // Older turns are evicted; nothing above the window budget.
        assert!(!history.iter().any(|m| text_of(m).starts_with("u0")));
        assert!(!history.iter().any(|m| text_of(m).starts_with("u21")));
        assert!(history.iter().any(|m| text_of(m).starts_with("u22")));
        assert!(manager.token_count() <= 10_000);
    }

    #[test]
    fn window_never_splits_tool_pairs() {
        // Window budget 100 (10k - 9.9k): only the 2-turn verbatim
        // tail survives; the middle cannot fit a single turn.
        let mut manager = ContextManager::new(window_config(10_000, 9_900, 2));
        for i in 0..10 {
            manager.commit_turn(&tool_turn(i, 100));
        }

        manager.prepare_messages("sys", &[]);
        let history = manager.history.clone();
        // The tail is the last 2 complete turns, tool pairs intact.
        assert_eq!(history.len(), 8);
        assert_eq!(history[0].role(), "user");
        assert_eq!(history[0].content(), format!("u8:{}", "a".repeat(100)));
        assert_eq!(history[4].role(), "user");
        // No orphaned tool results: every tool message has a matching
        // assistant tool call within the kept window.
        for message in &history {
            if message.role() == "tool" {
                let id = message.tool_call_id().expect("tool id");
                let matched = history.iter().any(|m| {
                    m.role() == "assistant"
                        && m.tool_calls().iter().any(|call| call.id() == id)
                });
                assert!(matched, "orphan tool result {id}");
            }
        }
    }

    #[test]
    fn window_clamps_tail_by_token_budget() {
        // 6 turns of ~1010 tokens each; window budget 2k. The tail
        // (4 turns) exceeds the 2k tail budget and shrinks to the
        // newest turn; the middle cannot fit another turn, so only the
        // newest turn survives.
        let mut manager = ContextManager::new(window_config(10_000, 8_000, 4));
        for i in 0..6 {
            manager.commit_turn(&turn(i, 2_000));
        }

        let messages = manager.prepare_messages("sys", &[]);
        let history: Vec<Message> = messages[1..].to_vec();
        assert!(history.iter().any(|m| text_of(m).starts_with("u5")));
        assert!(!history.iter().any(|m| text_of(m).starts_with("u4")));
        assert!(!history.iter().any(|m| text_of(m).starts_with("u0")));
    }

    #[test]
    fn sticky_prefix_is_never_evicted() {
        // Window budget 500: only sticky + tail fit.
        let mut manager = ContextManager::new(window_config(10_000, 9_500, 2));
        manager.sticky_count = 2;
        manager.commit_turn(&[Message::user("pinned-goal"), Message::assistant("pinned-plan")]);
        for i in 0..10 {
            manager.commit_turn(&turn(i, 100));
        }

        let messages = manager.prepare_messages("sys", &[]);
        let history: Vec<Message> = messages[1..].to_vec();
        assert_eq!(history[0].content(), "pinned-goal");
        assert_eq!(history[1].content(), "pinned-plan");
        assert!(history.iter().any(|m| text_of(m).starts_with("u9")));
        assert!(history.iter().any(|m| text_of(m).starts_with("u8")));
        assert!(!history.iter().any(|m| text_of(m).starts_with("u0")));
    }

    #[test]
    fn prune_replaces_old_tool_outputs_keeping_ids() {
        // threshold 300 < 5 tool outputs (~395 tokens) -> prune to ~100.
        let mut manager = ContextManager::new(ContextConfig {
            prune_threshold_tokens: 300,
            prune_target_tokens: 100,
            ..window_config(10_000, 2_000, 2)
        });
        for i in 0..5 {
            manager.commit_turn(&tool_turn(i, 300));
        }

        manager.prepare_messages("sys", &[]);
        let history = manager.history.clone();
        // The last 2 turns keep their raw outputs; older ones are
        // placeholders with their tool_call_id preserved.
        let last_turn: Vec<&Message> = history.iter().filter(|m| m.content().starts_with("a4")).collect();
        assert!(!last_turn.is_empty());
        let pruned: Vec<&Message> = history
            .iter()
            .filter(|m| m.role() == "tool" && m.content() == TOOL_OUTPUT_PRUNE_PLACEHOLDER)
            .collect();
        assert!(!pruned.is_empty(), "no tool outputs were pruned");
        assert!(pruned.iter().all(|m| m.tool_call_id().is_some()));
        assert!(history
            .iter()
            .filter(|m| m.role() == "tool" && m.content().starts_with("xxxxx"))
            .count()
            >= 2);
    }

    #[test]
    fn commit_truncates_oversized_tool_results() {
        let mut manager = ContextManager::new(ContextConfig {
            record_truncate_tokens: 10,
            ..window_config(10_000, 2_000, 2)
        });
        manager.commit_turn(&[Message::tool_result(ToolCallId::new("c1"), "x".repeat(100))]);

        let stored = &manager.history[0];
        assert_eq!(stored.role(), "tool");
        assert!(stored.content().ends_with(TOOL_OUTPUT_TRUNCATED_SUFFIX));
        assert_eq!(stored.tool_call_id().map(|id| id.as_str()), Some("c1"));
        assert!(stored.content().chars().count() <= 40 + TOOL_OUTPUT_TRUNCATED_SUFFIX.chars().count());
    }

    #[test]
    fn oversized_turn_is_cut_at_first_assistant_keeping_user() {
        // A single turn larger than the whole window budget (~8k): its
        // assistant half is cut, the user message survives.
        let mut manager = ContextManager::new(window_config(10_000, 2_000, 2));
        manager.commit_turn(&[Message::user("huge:".to_owned() + &"a".repeat(16_000)), Message::assistant("b".repeat(16_000))]);
        manager.commit_turn(&turn(1, 100));
        manager.commit_turn(&turn(2, 100));

        let messages = manager.prepare_messages("sys", &[]);
        let history: Vec<Message> = messages[1..].to_vec();
        assert!(history.iter().any(|m| m.content().starts_with("huge:")));
        assert!(!history.iter().any(|m| m.content() == "b".repeat(16_000)));
    }

    #[test]
    fn turn_spans_group_by_user_messages() {
        let messages = vec![
            Message::user("u1"),
            Message::assistant("a1"),
            Message::user("u2"),
            Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(ToolCallId::new("c1"), "echo", json!({}))],
            ),
            Message::tool_result(ToolCallId::new("c1"), "r"),
            Message::assistant("a2"),
        ];
        let spans = turn_spans(&messages);
        assert_eq!(spans, vec![(0, 2), (2, 4)]);
    }

    #[test]
    fn turn_spans_merge_leading_non_user_messages() {
        let messages = vec![
            Message::assistant("a0"),
            Message::user("u1"),
            Message::assistant("a1"),
        ];
        let spans = turn_spans(&messages);
        assert_eq!(spans, vec![(0, 3)]);
    }

    #[test]
    fn empty_history_windows_to_empty_output() {
        let mut manager = ContextManager::new(window_config(10_000, 9_500, 2));
        let messages = manager.prepare_messages("sys", &[]);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role(), "system");
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
