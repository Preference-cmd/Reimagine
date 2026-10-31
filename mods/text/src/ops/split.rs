//! `text.split`: cut a document at a heading level or a literal separator.
//!
//! Asset input produces `docs` (files written to the output directory);
//! inline input produces `pieces`. Empty pieces are dropped. A front matter
//! block describes the whole document, so it is stripped before splitting.

use std::path::Path;

use mod_sdk::diagnostic::{Diagnostic, caller_error};
use mod_sdk::serde_json::{Value, json};

use crate::ID;
use crate::document::{self, Document, Form};
use crate::front_matter;
use crate::markdown;

pub fn run(input: &Value, output_dir: &Path) -> Result<Value, Diagnostic> {
    let document = Document::load(input)?;
    let parsed = front_matter::parse(&document.text)
        .map_err(|error| caller_error(ID, error.code(), error.message()))?;
    let by = input
        .get("by")
        .and_then(Value::as_object)
        .ok_or_else(|| caller_error(ID, "INVALID_BY", "`by` must be an object with a `kind`"))?;
    let kind = by.get("kind").and_then(Value::as_str).ok_or_else(|| {
        caller_error(
            ID,
            "INVALID_BY",
            "`by.kind` must be `heading` or `separator`",
        )
    })?;

    let pieces = match kind {
        "heading" => {
            let level = by
                .get("level")
                .and_then(Value::as_u64)
                .filter(|level| (1..=6).contains(level))
                .ok_or_else(|| {
                    caller_error(
                        ID,
                        "INVALID_BY",
                        "`by.level` must be an integer from 1 to 6",
                    )
                })? as u8;
            let sections = markdown::sections(parsed.body, level);
            if sections.is_empty() {
                return Err(caller_error(
                    ID,
                    "NO_HEADINGS",
                    format!("the document has no level-{level} ATX headings to split on"),
                ));
            }
            sections
                .into_iter()
                .map(|section| {
                    let title = if section.heading.text.trim().is_empty() {
                        first_non_empty_line(section.text).unwrap_or_default()
                    } else {
                        section.heading.text
                    };
                    Piece::new(section.text.to_owned(), title)
                })
                .collect()
        }
        "separator" => {
            let separator = by
                .get("separator")
                .and_then(Value::as_str)
                .ok_or_else(|| caller_error(ID, "INVALID_BY", "`by.separator` must be a string"))?;
            if separator.is_empty() {
                return Err(caller_error(
                    ID,
                    "EMPTY_SEPARATOR",
                    "`by.separator` must not be empty",
                ));
            }
            parsed
                .body
                .split(separator)
                .map(str::trim)
                .filter(|piece| !piece.is_empty())
                .map(|piece| {
                    let title = first_non_empty_line(piece).unwrap_or_default();
                    Piece::new(piece.to_owned(), title)
                })
                .collect()
        }
        other => {
            return Err(caller_error(
                ID,
                "INVALID_BY",
                format!("unknown split kind `{other}`; use heading or separator"),
            ));
        }
    };

    output(document.form, output_dir, pieces)
}

struct Piece {
    text: String,
    title: String,
    chars: usize,
}

impl Piece {
    fn new(text: String, title: String) -> Self {
        let chars = text.chars().count();
        Self { text, title, chars }
    }
}

fn first_non_empty_line(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(str::to_owned)
}

fn output(form: Form, output_dir: &Path, pieces: Vec<Piece>) -> Result<Value, Diagnostic> {
    let mut values = Vec::with_capacity(pieces.len());
    match form {
        Form::Asset => {
            for (index, piece) in pieces.into_iter().enumerate() {
                let name = format!("piece-{}.md", index + 1);
                document::write_asset(output_dir, &name, &piece.text)?;
                values.push(json!({
                    "doc": name,
                    "title": piece.title,
                    "chars": piece.chars,
                }));
            }
            Ok(json!({ "docs": values }))
        }
        Form::Inline => {
            for piece in pieces {
                document::check_inline(&piece.text)?;
                values.push(json!({
                    "text": piece.text,
                    "title": piece.title,
                    "chars": piece.chars,
                }));
            }
            Ok(json!({ "pieces": values }))
        }
    }
}

#[cfg(test)]
mod tests {
    use mod_sdk::serde_json::json;
    use tempfile::tempdir;

    use super::*;

    fn split_inline(text: &str, by: Value) -> Result<Value, Diagnostic> {
        let temp = tempdir().expect("temp dir");
        run(&json!({ "text": text, "by": by }), temp.path())
    }

    #[test]
    fn heading_split_keeps_the_heading_line_and_nested_sections() {
        let result = split_inline(
            "intro\n\n## One\n\nfirst\n\n### Sub\n\nsub text\n\n## Two\n\nsecond\n",
            json!({ "kind": "heading", "level": 2 }),
        )
        .expect("split");
        let pieces = result["pieces"].as_array().expect("pieces");
        assert_eq!(pieces.len(), 2);
        assert_eq!(pieces[0]["title"], "One");
        assert_eq!(
            pieces[0]["text"],
            "## One\n\nfirst\n\n### Sub\n\nsub text\n\n"
        );
        assert_eq!(
            pieces[0]["chars"],
            pieces[0]["text"].as_str().unwrap().chars().count()
        );
        assert_eq!(pieces[1]["title"], "Two");
        assert_eq!(pieces[1]["text"], "## Two\n\nsecond\n");
    }

    #[test]
    fn heading_split_ignores_fenced_headings_and_requires_a_match() {
        let result = split_inline(
            "```\n## not a heading\n```\n\n## Real\n\nbody\n",
            json!({ "kind": "heading", "level": 2 }),
        )
        .expect("split");
        assert_eq!(result["pieces"].as_array().expect("pieces").len(), 1);

        let error = split_inline(
            "plain text without headings",
            json!({ "kind": "heading", "level": 1 }),
        )
        .expect_err("no headings");
        assert_eq!(error.code().as_str(), "MOD/CALLER_NO_HEADINGS");
    }

    #[test]
    fn separator_split_is_literal_and_drops_empty_pieces() {
        let result = split_inline(
            "one\n\n---\n\ntwo\n\n---\n\n\n\n---\n\n",
            json!({ "kind": "separator", "separator": "\n---\n" }),
        )
        .expect("split");
        assert_eq!(
            result["pieces"],
            json!([
                { "text": "one", "title": "one", "chars": 3 },
                { "text": "two", "title": "two", "chars": 3 }
            ])
        );
    }

    #[test]
    fn separator_title_is_the_first_non_empty_line() {
        let result = split_inline(
            "\n\n# A title\n\nbody\nSEP\nsecond piece",
            json!({ "kind": "separator", "separator": "SEP" }),
        )
        .expect("split");
        assert_eq!(result["pieces"][0]["title"], "# A title");
        assert_eq!(result["pieces"][0]["text"], "# A title\n\nbody");
        assert_eq!(result["pieces"][1]["title"], "second piece");
    }

    #[test]
    fn an_empty_separator_is_a_diagnostic() {
        let error = split_inline("text", json!({ "kind": "separator", "separator": "" }))
            .expect_err("empty separator");
        assert_eq!(error.code().as_str(), "MOD/CALLER_EMPTY_SEPARATOR");
    }

    #[test]
    fn invalid_by_shapes_are_diagnostics() {
        for by in [
            json!(null),
            json!({}),
            json!({ "kind": "paragraph" }),
            json!({ "kind": "heading" }),
            json!({ "kind": "heading", "level": 0 }),
            json!({ "kind": "heading", "level": 7 }),
            json!({ "kind": "separator" }),
        ] {
            let error = split_inline("## A\n", by.clone()).expect_err("invalid by");
            assert!(
                error.code().as_str().starts_with("MOD/CALLER_"),
                "{by}: unexpected code {}",
                error.code()
            );
        }
    }

    #[test]
    fn asset_input_writes_one_md_file_per_piece() {
        let temp = tempdir().expect("temp dir");
        let input_path = temp.path().join("input.md");
        std::fs::write(&input_path, "## A\n\nfirst\n\n## B\n\nsecond\n").expect("write");
        let output_dir = temp.path().join("out");
        std::fs::create_dir_all(&output_dir).expect("out dir");
        let result = run(
            &json!({
                "doc": input_path.to_str().expect("utf8 path"),
                "by": { "kind": "heading", "level": 2 }
            }),
            &output_dir,
        )
        .expect("split");
        let docs = result["docs"].as_array().expect("docs");
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0]["doc"], "piece-1.md");
        assert_eq!(docs[1]["doc"], "piece-2.md");
        assert_eq!(
            std::fs::read_to_string(output_dir.join("piece-1.md")).expect("read"),
            "## A\n\nfirst\n\n"
        );
        assert!(result.get("pieces").is_none());
    }
}
