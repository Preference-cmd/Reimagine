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

use crate::provider::Message;

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
}
