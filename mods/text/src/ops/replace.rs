//! `text.replace`: literal, whole-word or regular-expression replacement
//! with a per-pair count.

use std::path::Path;

use mod_sdk::diagnostic::{Diagnostic, caller_error};
use mod_sdk::serde_json::{Map, Value, json};
use regex::{NoExpand, Regex, RegexBuilder};

use crate::ID;
use crate::document::{self, Document};

pub fn run(input: &Value, output_dir: &Path) -> Result<Value, Diagnostic> {
    let document = Document::load(input)?;
    let pairs = input
        .get("pairs")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            caller_error(
                ID,
                "INVALID_PAIR",
                "`pairs` must be an array of replacement pairs",
            )
        })?;

    let mut text = document.text;
    let mut counts = Vec::with_capacity(pairs.len());
    for (index, pair) in pairs.iter().enumerate() {
        let pair = Pair::parse(pair, index)?;
        counts.push(pair.regex.find_iter(&text).count() as u64);
        text = if pair.expand {
            pair.regex
                .replace_all(&text, pair.replace.as_str())
                .into_owned()
        } else {
            pair.regex
                .replace_all(&text, NoExpand(pair.replace.as_str()))
                .into_owned()
        };
    }

    let mut result = Map::new();
    document::put(&mut result, document.form, output_dir, "output.md", &text)?;
    result.insert("counts".to_owned(), json!(counts));
    Ok(Value::Object(result))
}

struct Pair {
    regex: Regex,
    replace: String,
    /// Whether the replacement string expands `$1` capture references.
    expand: bool,
}

impl Pair {
    fn parse(value: &Value, index: usize) -> Result<Self, Diagnostic> {
        let object = value.as_object().ok_or_else(|| {
            caller_error(
                ID,
                "INVALID_PAIR",
                format!("pair {index} must be an object"),
            )
        })?;
        let find = object.get("find").and_then(Value::as_str).ok_or_else(|| {
            caller_error(
                ID,
                "INVALID_PAIR",
                format!("pair {index} needs a string `find`"),
            )
        })?;
        if find.is_empty() {
            return Err(caller_error(
                ID,
                "EMPTY_FIND",
                format!("pair {index} has an empty `find`"),
            ));
        }
        let replace = object
            .get("replace")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                caller_error(
                    ID,
                    "INVALID_PAIR",
                    format!("pair {index} needs a string `replace`"),
                )
            })?;
        let mode = object.get("mode").and_then(Value::as_str).ok_or_else(|| {
            caller_error(
                ID,
                "INVALID_PAIR",
                format!("pair {index} needs a string `mode`"),
            )
        })?;
        let case_sensitive = match object.get("case_sensitive") {
            None | Some(Value::Null) => true,
            Some(Value::Bool(value)) => *value,
            Some(_) => {
                return Err(caller_error(
                    ID,
                    "INVALID_PAIR",
                    format!("pair {index} `case_sensitive` must be a boolean"),
                ));
            }
        };

        let (pattern, expand) = match mode {
            "literal" => (regex::escape(find), false),
            "word" => (format!(r"\b{}\b", regex::escape(find)), false),
            "regex" => (find.to_owned(), true),
            other => {
                return Err(caller_error(
                    ID,
                    "INVALID_PAIR",
                    format!("pair {index} has unknown mode `{other}`; use literal, word or regex"),
                ));
            }
        };
        let regex = RegexBuilder::new(&pattern)
            .case_insensitive(!case_sensitive)
            .build()
            .map_err(|error| {
                caller_error(
                    ID,
                    "INVALID_REGEX",
                    format!("pair {index} regex `{find}` is invalid: {error}"),
                )
            })?;
        Ok(Self {
            regex,
            replace: replace.to_owned(),
            expand,
        })
    }
}

#[cfg(test)]
mod tests {
    use mod_sdk::serde_json::json;
    use tempfile::tempdir;

    use super::*;

    fn replace(text: &str, pairs: Value) -> (String, Vec<u64>) {
        let temp = tempdir().expect("temp dir");
        let result = run(&json!({ "text": text, "pairs": pairs }), temp.path()).expect("replace");
        let counts: Vec<u64> = result["counts"]
            .as_array()
            .expect("counts")
            .iter()
            .map(|count| count.as_u64().expect("count"))
            .collect();
        (
            result["text"].as_str().expect("text output").to_owned(),
            counts,
        )
    }

    #[test]
    fn word_literal_dollar_and_regex_capture_modes_run_in_order() {
        let (text, counts) = replace(
            "cat catalog\ncat.\nAlice@example.com",
            json!([
                { "find": "cat", "replace": "dog", "mode": "word" },
                { "find": "@example.com", "replace": "$1", "mode": "literal" },
                { "find": r"(\w+)\$1", "replace": "$1 (at)", "mode": "regex" }
            ]),
        );
        assert_eq!(text, "dog catalog\ndog.\nAlice (at)");
        assert_eq!(counts, [2, 1, 1]);
    }

    #[test]
    fn literal_and_word_replacements_do_not_expand_dollars() {
        let (text, counts) = replace(
            "price",
            json!([{ "find": "price", "replace": "$0 $1 $name", "mode": "literal" }]),
        );
        assert_eq!(text, "$0 $1 $name");
        assert_eq!(counts, [1]);
    }

    #[test]
    fn case_sensitivity_defaults_to_true_and_can_be_turned_off() {
        let (text, counts) = replace(
            "Cat cat CAT",
            json!([
                { "find": "cat", "replace": "x", "mode": "literal" },
                { "find": "cat", "replace": "y", "mode": "literal", "case_sensitive": false }
            ]),
        );
        // Pair 1 replaces only the lowercase `cat`; pair 2 then sees `Cat`
        // and `CAT` around the already replaced `x`, so it replaces two.
        assert_eq!(text, "y x y");
        assert_eq!(counts, [1, 2]);
    }

    #[test]
    fn word_mode_uses_unicode_boundaries() {
        let (text, counts) = replace(
            "猫 猫咪 猫",
            json!([{ "find": "猫", "replace": "dog", "mode": "word" }]),
        );
        // CJK characters are word characters, so `猫咪` has no boundary
        // between 猫 and 咪: only the two standalone 猫 match.
        assert_eq!(text, "dog 猫咪 dog");
        assert_eq!(counts, [2]);
    }

    #[test]
    fn an_empty_find_is_a_diagnostic() {
        let temp = tempdir().expect("temp dir");
        let error = run(
            &json!({ "text": "x", "pairs": [{ "find": "", "replace": "y", "mode": "literal" }] }),
            temp.path(),
        )
        .expect_err("empty find");
        assert_eq!(error.code().as_str(), "MOD/CALLER_EMPTY_FIND");
    }

    #[test]
    fn a_pair_without_a_mode_is_a_diagnostic() {
        let temp = tempdir().expect("temp dir");
        let error = run(
            &json!({ "text": "x", "pairs": [{ "find": "x", "replace": "y" }] }),
            temp.path(),
        )
        .expect_err("missing mode");
        assert_eq!(error.code().as_str(), "MOD/CALLER_INVALID_PAIR");
        assert!(error.message().contains("`mode`"));
    }

    #[test]
    fn an_invalid_regex_is_a_diagnostic() {
        let temp = tempdir().expect("temp dir");
        let error = run(
            &json!({ "text": "x", "pairs": [{ "find": "(", "replace": "y", "mode": "regex" }] }),
            temp.path(),
        )
        .expect_err("invalid regex");
        assert_eq!(error.code().as_str(), "MOD/CALLER_INVALID_REGEX");
    }

    #[test]
    fn invalid_pairs_are_diagnosed() {
        let temp = tempdir().expect("temp dir");
        for pairs in [
            json!(null),
            json!([7]),
            json!([{ "replace": "y", "mode": "literal" }]),
            json!([{ "find": "x", "replace": "y" }]),
            json!([{ "find": "x", "mode": "literal" }]),
            json!([{ "find": "x", "replace": "y", "mode": "glob" }]),
            json!([{ "find": "x", "replace": "y", "mode": "literal", "case_sensitive": "no" }]),
        ] {
            let result = run(&json!({ "text": "x", "pairs": pairs }), temp.path());
            let error = result.expect_err("invalid pair");
            assert!(
                error.code().as_str().starts_with("MOD/CALLER_INVALID_PAIR"),
                "unexpected code {}",
                error.code()
            );
        }
    }

    #[test]
    fn an_empty_pair_list_leaves_the_text_alone() {
        let (text, counts) = replace("same", json!([]));
        assert_eq!(text, "same");
        assert!(counts.is_empty());
    }
}
