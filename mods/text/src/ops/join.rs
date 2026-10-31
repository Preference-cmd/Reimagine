//! `text.join`: concatenate documents in order, with a literal separator or
//! a template rendered per piece.
//!
//! Input is exactly one of `docs` (asset paths) and `texts` (inline strings),
//! plus exactly one of `separator` and `template`. The result mirrors the
//! input form. Separator mode keeps every piece verbatim; template mode
//! resolves placeholders against each piece's front matter and body.

use std::path::Path;

use mod_sdk::diagnostic::{Diagnostic, caller_error};
use mod_sdk::serde_json::Value;

use crate::ID;
use crate::document::{self, Form};
use crate::front_matter;
use crate::template;

pub fn run(input: &Value, output_dir: &Path) -> Result<Value, Diagnostic> {
    let docs = input.get("docs").filter(|value| !value.is_null());
    let texts = input.get("texts").filter(|value| !value.is_null());
    let (pieces, form) = match (docs, texts) {
        (Some(docs), None) => (load_docs(docs)?, Form::Asset),
        (None, Some(texts)) => (load_texts(texts)?, Form::Inline),
        (Some(_), Some(_)) => {
            return Err(caller_error(
                ID,
                "INVALID_JOIN_INPUT",
                "give exactly one of `docs` and `texts`; both were provided",
            ));
        }
        (None, None) => {
            return Err(caller_error(
                ID,
                "INVALID_JOIN_INPUT",
                "give exactly one of `docs` and `texts`; neither was provided",
            ));
        }
    };

    let separator = input.get("separator").filter(|value| !value.is_null());
    let source_template = input.get("template").filter(|value| !value.is_null());
    let joined = match (separator, source_template) {
        (Some(separator), None) => {
            let separator = separator.as_str().ok_or_else(|| {
                caller_error(ID, "INVALID_JOIN_INPUT", "`separator` must be a string")
            })?;
            pieces.join(separator)
        }
        (None, Some(source)) => {
            let source = source.as_str().ok_or_else(|| {
                caller_error(ID, "INVALID_JOIN_INPUT", "`template` must be a string")
            })?;
            let mut joined = String::new();
            for piece in &pieces {
                let parsed = front_matter::parse(piece)
                    .map_err(|error| caller_error(ID, error.code(), error.message()))?;
                let rendered =
                    template::render(source, None, parsed.front_matter.as_ref(), parsed.body)
                        .map_err(|error| caller_error(ID, error.code(), error.message()))?;
                joined.push_str(&rendered);
            }
            joined
        }
        (Some(_), Some(_)) => {
            return Err(caller_error(
                ID,
                "INVALID_JOIN_INPUT",
                "give exactly one of `separator` and `template`; both were provided",
            ));
        }
        (None, None) => {
            return Err(caller_error(
                ID,
                "INVALID_JOIN_INPUT",
                "give exactly one of `separator` and `template`; neither was provided",
            ));
        }
    };

    document::result(form, output_dir, "joined.md", &joined)
}

fn load_docs(value: &Value) -> Result<Vec<String>, Diagnostic> {
    let docs = value.as_array().ok_or_else(|| {
        caller_error(
            ID,
            "INVALID_JOIN_INPUT",
            "`docs` must be an array of `{doc, title?}` items",
        )
    })?;
    let mut pieces = Vec::with_capacity(docs.len());
    for (index, item) in docs.iter().enumerate() {
        let path = item.get("doc").and_then(Value::as_str).ok_or_else(|| {
            caller_error(
                ID,
                "INVALID_JOIN_INPUT",
                format!("`docs[{index}].doc` must be a string asset path"),
            )
        })?;
        let bytes = std::fs::read(path).map_err(|error| {
            caller_error(
                ID,
                "READ_FAILED",
                format!("failed to read the document at `{path}`: {error}"),
            )
        })?;
        let text = String::from_utf8(bytes).map_err(|_| {
            caller_error(
                ID,
                "INVALID_UTF8",
                format!("the document at `{path}` is not valid UTF-8"),
            )
        })?;
        pieces.push(text);
    }
    Ok(pieces)
}

fn load_texts(value: &Value) -> Result<Vec<String>, Diagnostic> {
    let texts = value.as_array().ok_or_else(|| {
        caller_error(
            ID,
            "INVALID_JOIN_INPUT",
            "`texts` must be an array of strings",
        )
    })?;
    let mut pieces = Vec::with_capacity(texts.len());
    for (index, text) in texts.iter().enumerate() {
        let text = text.as_str().ok_or_else(|| {
            caller_error(
                ID,
                "INVALID_JOIN_INPUT",
                format!("`texts[{index}]` must be a string"),
            )
        })?;
        document::check_inline(text)?;
        pieces.push(text.to_owned());
    }
    Ok(pieces)
}

#[cfg(test)]
mod tests {
    use mod_sdk::serde_json::json;
    use tempfile::tempdir;

    use super::*;

    fn out_dir(temp: &tempfile::TempDir) -> std::path::PathBuf {
        let dir = temp.path().join("out");
        std::fs::create_dir_all(&dir).expect("out dir");
        dir
    }

    #[test]
    fn texts_join_with_a_literal_separator() {
        let temp = tempdir().expect("temp dir");
        let result = run(
            &json!({ "texts": ["one", "two", "three"], "separator": "\n---\n" }),
            &out_dir(&temp),
        )
        .expect("join");
        assert_eq!(result, json!({ "text": "one\n---\ntwo\n---\nthree" }));
    }

    #[test]
    fn texts_join_with_a_template_rendered_per_piece() {
        let temp = tempdir().expect("temp dir");
        let result = run(
            &json!({
                "texts": ["---\ntitle: A\n---\nfirst", "second"],
                "template": "<h2>{{body}}</h2>"
            }),
            &out_dir(&temp),
        )
        .expect("join");
        assert_eq!(result, json!({ "text": "<h2>first</h2><h2>second</h2>" }));
    }

    #[test]
    fn docs_join_in_input_order_and_produce_a_file() {
        let temp = tempdir().expect("temp dir");
        let one = temp.path().join("one.md");
        let two = temp.path().join("two.md");
        std::fs::write(&one, "first").expect("write");
        std::fs::write(&two, "second").expect("write");
        let dir = out_dir(&temp);
        let result = run(
            &json!({
                "docs": [
                    { "doc": one.to_str().expect("utf8 path"), "title": "ignored" },
                    { "doc": two.to_str().expect("utf8 path") }
                ],
                "separator": "\n\n"
            }),
            &dir,
        )
        .expect("join");
        assert_eq!(result, json!({ "doc": "joined.md" }));
        assert_eq!(
            std::fs::read_to_string(dir.join("joined.md")).expect("read"),
            "first\n\nsecond"
        );
        assert!(result.get("text").is_none());
    }

    #[test]
    fn docs_join_with_a_template_uses_each_pieces_front_matter() {
        let temp = tempdir().expect("temp dir");
        let result = run(
            &json!({
                "texts": ["---\ntitle: First\n---\nBody one"],
                "template": "# {{title}}\n\n{{body}}\n"
            }),
            &out_dir(&temp),
        )
        .expect("join");
        assert_eq!(result, json!({ "text": "# First\n\nBody one\n" }));
    }

    #[test]
    fn exactly_one_input_form_and_one_combiner_are_required() {
        let temp = tempdir().expect("temp dir");
        let dir = out_dir(&temp);
        for input in [
            json!({ "separator": "x" }),
            json!({ "docs": [], "texts": [], "separator": "x" }),
            json!({ "texts": ["a"] }),
            json!({ "texts": ["a"], "separator": "x", "template": "{{body}}" }),
        ] {
            let error = run(&input, &dir).expect_err("invalid join input");
            assert_eq!(error.code().as_str(), "MOD/CALLER_INVALID_JOIN_INPUT");
        }
    }

    #[test]
    fn invalid_items_are_diagnostics() {
        let temp = tempdir().expect("temp dir");
        let dir = out_dir(&temp);
        for input in [
            json!({ "texts": ["a", 7], "separator": "" }),
            json!({ "docs": [{ "title": "no doc" }], "separator": "" }),
            json!({ "texts": "not an array", "separator": "" }),
        ] {
            let error = run(&input, &dir).expect_err("invalid items");
            assert_eq!(error.code().as_str(), "MOD/CALLER_INVALID_JOIN_INPUT");
        }
    }

    #[test]
    fn a_template_missing_a_placeholder_is_a_diagnostic() {
        let temp = tempdir().expect("temp dir");
        let error = run(
            &json!({ "texts": ["a"], "template": "{{nope}}" }),
            &out_dir(&temp),
        )
        .expect_err("missing placeholder");
        assert_eq!(error.code().as_str(), "MOD/CALLER_MISSING_PLACEHOLDER");
    }

    #[test]
    fn a_non_utf8_doc_is_a_diagnostic() {
        let temp = tempdir().expect("temp dir");
        let path = temp.path().join("bad.md");
        std::fs::write(&path, [0xff, 0x00]).expect("write");
        let error = run(
            &json!({
                "docs": [{ "doc": path.to_str().expect("utf8 path") }],
                "separator": ""
            }),
            &out_dir(&temp),
        )
        .expect_err("invalid utf8");
        assert_eq!(error.code().as_str(), "MOD/CALLER_INVALID_UTF8");
    }

    #[test]
    fn inline_join_output_over_the_cap_is_a_diagnostic() {
        let temp = tempdir().expect("temp dir");
        let piece = "x".repeat(document::MAX_INLINE_BYTES);
        let error = run(
            &json!({ "texts": [piece, piece], "separator": "" }),
            &out_dir(&temp),
        )
        .expect_err("output over the cap");
        assert_eq!(error.code().as_str(), "MOD/CALLER_INLINE_TOO_LARGE");

        // The same two pieces as assets join fine.
        let one = temp.path().join("one.md");
        let two = temp.path().join("two.md");
        std::fs::write(&one, &piece).expect("write");
        std::fs::write(&two, &piece).expect("write");
        run(
            &json!({
                "docs": [
                    { "doc": one.to_str().expect("utf8 path") },
                    { "doc": two.to_str().expect("utf8 path") }
                ],
                "separator": ""
            }),
            &out_dir(&temp),
        )
        .expect("asset join");
    }

    #[test]
    fn an_empty_array_joins_to_empty_text() {
        let temp = tempdir().expect("temp dir");
        let result = run(&json!({ "texts": [], "separator": "x" }), &out_dir(&temp)).expect("join");
        assert_eq!(result, json!({ "text": "" }));
    }
}
