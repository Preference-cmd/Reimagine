//! `text.inspect`: title, ATX headings, flat front matter and stats.

use std::path::Path;

use mod_sdk::diagnostic::{Diagnostic, caller_error};
use mod_sdk::serde_json::{Map, Value, json};

use crate::document::Document;
use crate::front_matter;
use crate::markdown;

pub fn run(input: &Value, _output_dir: &Path) -> Result<Value, Diagnostic> {
    let document = Document::load(input)?;
    let parsed = front_matter::parse(&document.text)
        .map_err(|error| caller_error(crate::ID, error.code(), error.message()))?;
    let headings = markdown::headings(parsed.body);

    let title = parsed
        .front_matter
        .as_ref()
        .and_then(|fields| fields.get("title").cloned())
        .or_else(|| {
            headings
                .iter()
                .find(|heading| heading.level == 1)
                .map(|heading| heading.text.clone())
        })
        .unwrap_or_default();

    let front_matter = match &parsed.front_matter {
        Some(fields) => {
            let mut map = Map::new();
            for (key, value) in fields {
                map.insert(key.clone(), Value::String(value.clone()));
            }
            Value::Object(map)
        }
        None => Value::Null,
    };
    let heading_values: Vec<Value> = headings
        .iter()
        .map(|heading| json!({ "level": heading.level, "text": heading.text }))
        .collect();

    Ok(json!({
        "title": title,
        "headings": heading_values,
        "front_matter": front_matter,
        "stats": stats(parsed.body),
    }))
}

fn stats(body: &str) -> Value {
    // Stats describe the content; a front matter block is metadata and is
    // reported separately as `front_matter`.
    let mut paragraphs = 0;
    let mut in_paragraph = false;
    for line in body.lines() {
        if line.trim().is_empty() {
            in_paragraph = false;
        } else if !in_paragraph {
            paragraphs += 1;
            in_paragraph = true;
        }
    }
    json!({
        "chars": body.chars().count(),
        "lines": body.lines().count(),
        "paragraphs": paragraphs,
    })
}

#[cfg(test)]
mod tests {
    use mod_sdk::serde_json::json;
    use tempfile::tempdir;

    use super::*;

    fn inspect(text: &str) -> Value {
        let temp = tempdir().expect("temp dir");
        run(&json!({ "text": text }), temp.path()).expect("inspect")
    }

    #[test]
    fn title_comes_from_front_matter_then_the_first_h1() {
        let with_front_matter = inspect("---\ntitle: From front matter\n---\n# Heading\n\nBody");
        assert_eq!(with_front_matter["title"], "From front matter");

        let with_h1 = inspect("## Sub\n\n# The H1\n\n# Later");
        assert_eq!(with_h1["title"], "The H1");

        let without_title = inspect("## Only a subheading\n\nBody");
        assert_eq!(without_title["title"], "");
    }

    #[test]
    fn headings_skip_fenced_code_and_front_matter_is_flat() {
        let result = inspect(
            "---\ntitle: T\nauthor: A\n---\n# One\n\n```rust\n# not a heading\n```\n\n## Two ##\n\n### Three\n",
        );
        assert_eq!(
            result["headings"],
            json!([
                { "level": 1, "text": "One" },
                { "level": 2, "text": "Two" },
                { "level": 3, "text": "Three" }
            ])
        );
        assert_eq!(
            result["front_matter"],
            json!({ "title": "T", "author": "A" })
        );
    }

    #[test]
    fn front_matter_is_null_when_absent() {
        let result = inspect("# Title\n\nBody");
        assert_eq!(result["front_matter"], Value::Null);
    }

    #[test]
    fn stats_count_chars_lines_and_paragraphs_of_the_body() {
        let body = "one\n\n2\nthree\n\n\nfour";
        let result = inspect(body);
        assert_eq!(
            result["stats"],
            json!({
                "chars": body.chars().count(),
                "lines": 7,
                "paragraphs": 3,
            })
        );
    }

    #[test]
    fn stats_exclude_the_front_matter_block() {
        let result = inspect("---\ntitle: T\n---\nBody\n");
        assert_eq!(result["stats"]["lines"], 1);
        assert_eq!(result["stats"]["chars"], 5);
        assert_eq!(result["stats"]["paragraphs"], 1);
    }

    #[test]
    fn an_unclosed_front_matter_block_is_a_diagnostic() {
        let temp = tempdir().expect("temp dir");
        let error = run(&json!({ "text": "---\ntitle: T" }), temp.path())
            .expect_err("unclosed front matter");
        assert_eq!(error.code().as_str(), "MOD/CALLER_UNCLOSED_FRONT_MATTER");
    }
}
