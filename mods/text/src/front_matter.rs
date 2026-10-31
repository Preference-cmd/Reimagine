//! Flat front matter: a leading `---` block of `key: value` string pairs.
//!
//! Only that one shape is supported; there is no nested YAML. Blank lines and
//! `#` comment lines inside the block are ignored, and a block that is never
//! closed with `---` is an error.

use std::collections::BTreeMap;

/// A document split into optional front matter and body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedDocument<'a> {
    /// Present when the document started with a `---` block.
    pub front_matter: Option<BTreeMap<String, String>>,
    /// Everything after the block, or the whole document when there is none.
    pub body: &'a str,
}

/// Why a front matter block could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrontMatterError {
    /// The opening `---` has no closing `---`.
    Unclosed,
    /// A line inside the block is not `key: value`.
    InvalidLine { number: usize, line: String },
}

impl FrontMatterError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Unclosed => "UNCLOSED_FRONT_MATTER",
            Self::InvalidLine { .. } => "INVALID_FRONT_MATTER",
        }
    }

    pub fn message(&self) -> String {
        match self {
            Self::Unclosed => {
                "front matter starts with `---` but has no closing `---` line".to_owned()
            }
            Self::InvalidLine { number, line } => {
                format!("front matter line {number} is not a `key: value` pair: `{line}`")
            }
        }
    }
}

/// Split `document` into front matter and body.
pub fn parse(document: &str) -> Result<ParsedDocument<'_>, FrontMatterError> {
    let first_end = document.find('\n').unwrap_or(document.len());
    let first = document[..first_end].trim_end_matches('\r');
    if !is_delimiter(first) {
        return Ok(ParsedDocument {
            front_matter: None,
            body: document,
        });
    }

    let mut consumed = (first_end + 1).min(document.len());
    let mut lines = Vec::new();
    let mut closed = false;
    let mut number = 1;
    while consumed < document.len() {
        let rest = &document[consumed..];
        let end = rest.find('\n').unwrap_or(rest.len());
        let raw = rest[..end].trim_end_matches('\r');
        number += 1;
        consumed += end;
        let has_newline = end < rest.len();
        if has_newline {
            consumed += 1;
        }
        if is_delimiter(raw) {
            closed = true;
            break;
        }
        lines.push((number, raw.to_owned()));
    }
    // An unclosed block is one error regardless of what its lines say.
    if !closed {
        return Err(FrontMatterError::Unclosed);
    }

    let mut fields = BTreeMap::new();
    for (number, raw) in lines {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            return Err(FrontMatterError::InvalidLine { number, line: raw });
        };
        let key = key.trim();
        if key.is_empty() {
            return Err(FrontMatterError::InvalidLine { number, line: raw });
        }
        fields.insert(key.to_owned(), value.trim().to_owned());
    }
    Ok(ParsedDocument {
        front_matter: Some(fields),
        body: &document[consumed..],
    })
}

fn is_delimiter(line: &str) -> bool {
    line.trim_end() == "---"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_document_without_front_matter_is_all_body() {
        let parsed = parse("# Title\n\nBody").expect("parse");
        assert_eq!(parsed.front_matter, None);
        assert_eq!(parsed.body, "# Title\n\nBody");
    }

    #[test]
    fn a_flat_block_is_parsed_and_removed_from_the_body() {
        let parsed = parse("---\ntitle: Interview\nauthor: A\n---\nBody line\n").expect("parse");
        let fields = parsed.front_matter.expect("front matter");
        assert_eq!(fields["title"], "Interview");
        assert_eq!(fields["author"], "A");
        assert_eq!(parsed.body, "Body line\n");
    }

    #[test]
    fn crlf_blank_lines_comments_and_colons_in_values_are_handled() {
        let parsed =
            parse("---\r\n# note\r\n\r\nurl: https://example.com/a\r\n---\r\nBody").expect("parse");
        let fields = parsed.front_matter.expect("front matter");
        assert_eq!(fields["url"], "https://example.com/a");
        assert_eq!(fields.len(), 1);
        assert_eq!(parsed.body, "Body");
    }

    #[test]
    fn an_empty_block_is_some_empty_map() {
        let parsed = parse("---\n---\nBody").expect("parse");
        assert_eq!(parsed.front_matter, Some(BTreeMap::new()));
        assert_eq!(parsed.body, "Body");
    }

    #[test]
    fn an_unclosed_block_is_an_error() {
        let error = parse("---\ntitle: T\nBody").expect_err("unclosed");
        assert_eq!(error.code(), "UNCLOSED_FRONT_MATTER");

        let error = parse("---").expect_err("a lone opening line is unclosed");
        assert_eq!(error.code(), "UNCLOSED_FRONT_MATTER");
    }

    #[test]
    fn a_line_without_a_key_is_an_error() {
        let error = parse("---\ntitle: T\nnot a pair\n---\nBody").expect_err("invalid line");
        assert_eq!(error.code(), "INVALID_FRONT_MATTER");
        assert!(error.message().contains("line 3"));
    }

    #[test]
    fn a_line_without_a_value_is_an_empty_string() {
        let parsed = parse("---\ntitle:\n---\nBody").expect("parse");
        assert_eq!(parsed.front_matter.expect("front matter")["title"], "");
    }
}
