//! `text.split`: cut a document at a heading level or a literal separator.
//!
//! Asset input produces `docs` (files written to the output directory);
//! inline input produces `pieces`. Whitespace-only pieces are dropped. A front
//! matter block describes the whole document, so it is stripped before
//! splitting: the heading scan, the piece texts and `chars` all describe the
//! body that remains, never the metadata block.

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
            let level = heading_level(by.get("level")).ok_or_else(|| {
                caller_error(
                    ID,
                    "INVALID_BY",
                    "`by.level` must be an integer from 1 to 6",
                )
            })?;
            let regions = markdown::regions(parsed.body, level);
            if regions.is_empty() {
                return Err(caller_error(
                    ID,
                    "NO_HEADINGS",
                    format!(
                        "the document has no ATX headings at level {level} or above to split on"
                    ),
                ));
            }
            regions
                .into_iter()
                .filter_map(|region| match region.heading {
                    Some(heading) => {
                        let title = if heading.text.trim().is_empty() {
                            first_non_empty_line(region.text).unwrap_or_default()
                        } else {
                            heading.text
                        };
                        Some(Piece::new(region.text.to_owned(), title))
                    }
                    // The preamble carries content outside any target-level
                    // section; keep it unless it is only whitespace.
                    None if region.text.trim().is_empty() => None,
                    None => Some(Piece::new(
                        region.text.to_owned(),
                        first_non_empty_line(region.text).unwrap_or_default(),
                    )),
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
                // Pieces are kept verbatim: trimming would change the text a
                // later `join` has to reproduce. Only whitespace-only pieces
                // are dropped, and the title is read without mutating them.
                .filter(|piece| !piece.trim().is_empty())
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

/// Read `by.level`. JSON Schema `integer` also accepts an integral number
/// written with a fractional part (`2.0`), so both forms are accepted here;
/// fractional and out-of-range values are rejected.
fn heading_level(value: Option<&Value>) -> Option<u8> {
    let number = value?.as_f64()?;
    if !(1.0..=6.0).contains(&number) || number.fract() != 0.0 {
        return None;
    }
    Some(number as u8)
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
        assert_eq!(pieces.len(), 3);
        assert_eq!(pieces[0]["title"], "intro");
        assert_eq!(pieces[0]["text"], "intro\n\n");
        assert_eq!(pieces[1]["title"], "One");
        assert_eq!(
            pieces[1]["text"],
            "## One\n\nfirst\n\n### Sub\n\nsub text\n\n"
        );
        assert_eq!(
            pieces[1]["chars"],
            pieces[1]["text"].as_str().unwrap().chars().count()
        );
        assert_eq!(pieces[2]["title"], "Two");
        assert_eq!(pieces[2]["text"], "## Two\n\nsecond\n");
    }

    #[test]
    fn heading_split_preserves_the_preamble_and_higher_level_sections() {
        let source = "Preamble line\n\nmore\n\n## One\n\nfirst\n\n# Part\n\nunder part\n\n## Two\n\nsecond\n";
        let result = split_inline(source, json!({ "kind": "heading", "level": 2 })).expect("split");
        let pieces = result["pieces"].as_array().expect("pieces");
        let titles: Vec<&str> = pieces
            .iter()
            .map(|piece| piece["title"].as_str().expect("title"))
            .collect();
        assert_eq!(titles, ["Preamble line", "One", "Part", "Two"]);
        assert_eq!(pieces[0]["text"], "Preamble line\n\nmore\n\n");
        assert_eq!(pieces[2]["text"], "# Part\n\nunder part\n\n");
    }

    #[test]
    fn heading_split_loses_no_non_whitespace_text() {
        for source in [
            "Preamble line\n\nmore\n\n## One\n\nfirst\n",
            "Preamble\n\n## One\n\nfirst\n\n# Part\n\nunder part\n\n## Two\n\nsecond\n",
            "# Top\n\nintro\n\n## A\n\n### B\n\nbody\n",
        ] {
            let result =
                split_inline(source, json!({ "kind": "heading", "level": 2 })).expect("split");
            let joined: String = result["pieces"]
                .as_array()
                .expect("pieces")
                .iter()
                .map(|piece| piece["text"].as_str().expect("text"))
                .collect();
            assert_eq!(joined, source, "every byte is in exactly one piece");
        }

        // A whitespace-only preamble is dropped, but nothing else is.
        let result = split_inline(
            "\n\n## One\n\nfirst\n",
            json!({ "kind": "heading", "level": 2 }),
        )
        .expect("split");
        let joined: String = result["pieces"]
            .as_array()
            .expect("pieces")
            .iter()
            .map(|piece| piece["text"].as_str().expect("text"))
            .collect();
        assert_eq!(joined, "## One\n\nfirst\n");
    }

    #[test]
    fn heading_split_ignores_fenced_headings_and_requires_a_match() {
        let result = split_inline(
            "intro\n\n```\n## not a heading\n```\n\n## Real\n\nbody\n",
            json!({ "kind": "heading", "level": 2 }),
        )
        .expect("split");
        let pieces = result["pieces"].as_array().expect("pieces");
        assert_eq!(pieces.len(), 2);
        assert_eq!(pieces[0]["title"], "intro");
        assert_eq!(pieces[1]["title"], "Real");

        let error = split_inline(
            "plain text without headings",
            json!({ "kind": "heading", "level": 1 }),
        )
        .expect_err("no headings");
        assert_eq!(error.code().as_str(), "MOD/CALLER_NO_HEADINGS");

        let error = split_inline(
            "### Deep only\n\nbody\n",
            json!({ "kind": "heading", "level": 2 }),
        )
        .expect_err("no boundary headings");
        assert_eq!(error.code().as_str(), "MOD/CALLER_NO_HEADINGS");
    }

    #[test]
    fn an_integral_level_is_accepted_with_or_without_a_fraction() {
        for level in [json!(2), json!(2.0), json!(6.0)] {
            let result = split_inline(
                "## A\n\nbody\n",
                json!({ "kind": "heading", "level": level }),
            )
            .expect("integral level");
            assert_eq!(result["pieces"].as_array().expect("pieces").len(), 1);
        }
        let error = split_inline("## A\n\nbody\n", json!({ "kind": "heading", "level": 6.5 }))
            .expect_err("fractional level");
        assert_eq!(error.code().as_str(), "MOD/CALLER_INVALID_BY");
        let error = split_inline("## A\n\nbody\n", json!({ "kind": "heading", "level": 0.0 }))
            .expect_err("zero level");
        assert_eq!(error.code().as_str(), "MOD/CALLER_INVALID_BY");
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
                { "text": "one\n", "title": "one", "chars": 4 },
                { "text": "\ntwo\n", "title": "two", "chars": 5 }
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
        assert_eq!(result["pieces"][0]["text"], "\n\n# A title\n\nbody\n");
        assert_eq!(result["pieces"][1]["title"], "second piece");
        assert_eq!(result["pieces"][1]["text"], "\nsecond piece");
    }

    #[test]
    fn separator_split_then_join_is_byte_exact() {
        let source = "alpha\nSEP\n\nbeta\nSEP\ngamma";
        let result = split_inline(source, json!({ "kind": "separator", "separator": "SEP" }))
            .expect("split");
        let texts: Vec<Value> = result["pieces"]
            .as_array()
            .expect("pieces")
            .iter()
            .map(|piece| piece["text"].clone())
            .collect();
        let temp = tempdir().expect("temp dir");
        let joined =
            crate::ops::join::run(&json!({ "texts": texts, "separator": "SEP" }), temp.path())
                .expect("join");
        assert_eq!(joined["text"], source);
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
