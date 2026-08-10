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

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::provider::Message;

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
    /// Token threshold that triggers [`ContextManager::needs_compaction`].
    pub max_tokens: usize,
    /// Rolling window: keep the most recent `recent_turns * 2` messages.
    pub recent_turns: usize,
    /// Session persistence directory. Stored now for AS-03 session
    /// persistence; not used by the V1 context manager.
    pub session_dir: PathBuf,
}

/// Manages conversation history for a single agent session.
pub struct ContextManager {
    config: ContextConfig,
    history: Vec<Message>,
}

impl ContextManager {
    pub fn new(config: ContextConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
        }
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
            .map(|message| estimate_tokens(message.content()))
            .sum()
    }

    /// `true` when the stored history exceeds `max_tokens`.
    pub fn needs_compaction(&self) -> bool {
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
    /// for the window and token threshold; only the history is restored.
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

    use crate::provider::{ToolCall, ToolCallId};

    fn config(recent_turns: usize) -> ContextConfig {
        ContextConfig {
            max_tokens: 10_000,
            recent_turns,
            session_dir: PathBuf::from("/tmp/reimagine-test"),
        }
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
    fn needs_compaction_flips_when_tokens_exceed_threshold() {
        let mut manager = ContextManager::new(ContextConfig {
            max_tokens: 8,
            recent_turns: 2,
            session_dir: PathBuf::from("/tmp/reimagine-test"),
        });
        manager.commit_turn(&[Message::user("a"), Message::assistant("b")]);
        assert!(!manager.needs_compaction());
        manager.commit_turn(&[Message::user("c"), Message::assistant("d")]);
        assert!(manager.needs_compaction());
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
        let cfg = ContextConfig {
            max_tokens: 10_000,
            recent_turns: 2,
            session_dir: dir.clone(),
        };
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
            ContextConfig {
                max_tokens: 10_000,
                recent_turns: 2,
                session_dir: dir,
            },
        )
        .expect("load failed");
        assert_eq!(loaded.token_count(), manager.token_count());
        assert_eq!(loaded.history, manager.history);
    }

    #[test]
    fn persist_creates_session_dir() {
        let dir = temp_session_dir("mkdir");
        let manager = ContextManager::new(ContextConfig {
            max_tokens: 10_000,
            recent_turns: 2,
            session_dir: dir.clone(),
        });
        manager.persist("sess-1").expect("persist failed");
        assert!(dir.join("sess-1.json").is_file());
    }

    #[test]
    fn load_missing_session_returns_err() {
        let dir = temp_session_dir("missing");
        let result = ContextManager::load(
            "nope",
            ContextConfig {
                max_tokens: 10_000,
                recent_turns: 2,
                session_dir: dir,
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn persist_writes_compaction_summary_none() {
        let dir = temp_session_dir("summary");
        let manager = ContextManager::new(ContextConfig {
            max_tokens: 10_000,
            recent_turns: 2,
            session_dir: dir.clone(),
        });
        manager.persist("sess-1").expect("persist failed");

        let json = std::fs::read_to_string(dir.join("sess-1.json")).expect("read failed");
        let state: serde_json::Value = serde_json::from_str(&json).expect("json parse failed");
        assert_eq!(state["session_id"], "sess-1");
        assert_eq!(state["history"], serde_json::Value::Array(vec![]));
        assert!(state["compaction_summary"].is_null());
        assert_eq!(state["total_tokens"], 0);
    }
}
