//! Heuristic token counting for agent messages.
//!
//! Estimates token count from message content without external
//! dependencies. CJK characters (code point above `0x2E80`) average one
//! token per two characters; all other characters average one token per
//! four characters. A fixed per-message overhead of 4 tokens is added.

/// Estimate the token count of `text`.
///
/// Counting is character-based (`chars().count()`), so multi-byte
/// non-CJK characters such as Cyrillic, Arabic, and accented Latin are
/// handled correctly and fall into the "other" bucket.
#[allow(dead_code)] // consumed by the context manager in a later ticket.
pub(crate) fn estimate_tokens(text: &str) -> usize {
    let chars = text.chars().count();
    let cjk = text.chars().filter(|c| *c as u32 > 0x2E80).count();
    let other = chars - cjk;
    (cjk / 2) + (other / 4) + 4
}

#[cfg(test)]
mod tests {
    use super::*;

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
