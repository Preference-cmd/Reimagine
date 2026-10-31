//! `text.normalize`: whitespace, full-width punctuation, line endings and
//! Unicode form.

use std::path::Path;

use mod_sdk::diagnostic::Diagnostic;
use mod_sdk::serde_json::Value;
use unicode_normalization::UnicodeNormalization;

use super::{boolean, choice};
use crate::document::{self, Document};

pub fn run(input: &Value, output_dir: &Path) -> Result<Value, Diagnostic> {
    let document = Document::load(input)?;
    let whitespace = boolean(input, "whitespace", true)?;
    let punctuation = choice(input, "punctuation", &["none", "halfwidth"], "none")?;
    let newline = boolean(input, "newline", true)?;
    let unicode = choice(input, "unicode", &["NFC", "none"], "NFC")?;

    let mut text = document.text;
    if newline {
        text = unify_newlines(&text);
    }
    if whitespace {
        text = collapse_whitespace(&text);
    }
    if punctuation == "halfwidth" {
        text = halfwidth_punctuation(&text);
    }
    if unicode == "NFC" {
        text = text.nfc().collect();
    }
    document::result(document.form, output_dir, "output.md", &text)
}

fn unify_newlines(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

/// Collapse runs of spaces and tabs into one space, trim line ends and keep
/// blank lines.
fn collapse_whitespace(text: &str) -> String {
    let mut collapsed = String::with_capacity(text.len());
    for (index, line) in text.split('\n').enumerate() {
        if index > 0 {
            collapsed.push('\n');
        }
        let trimmed = line.trim_end_matches([' ', '\t']);
        let mut in_run = false;
        for character in trimmed.chars() {
            if character == ' ' || character == '\t' {
                if !in_run {
                    collapsed.push(' ');
                    in_run = true;
                }
            } else {
                collapsed.push(character);
                in_run = false;
            }
        }
    }
    collapsed
}

/// Map full-width ASCII punctuation (U+FF01..U+FF5E) to its ASCII
/// counterpart. Full-width digits and letters are left alone: this rule is
/// about punctuation.
fn halfwidth_punctuation(text: &str) -> String {
    text.chars()
        .map(|character| {
            let code = character as u32;
            if (0xFF01..=0xFF5E).contains(&code) {
                let ascii = (code - 0xFEE0) as u8;
                if ascii.is_ascii_punctuation() {
                    return ascii as char;
                }
            }
            character
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use mod_sdk::serde_json::json;
    use tempfile::tempdir;

    use super::*;

    fn normalize(text: &str, options: Value) -> String {
        let temp = tempdir().expect("temp dir");
        let mut input = options;
        input["text"] = Value::String(text.to_owned());
        let result = run(&input, temp.path()).expect("normalize");
        result["text"].as_str().expect("text output").to_owned()
    }

    #[test]
    fn defaults_normalize_whitespace_newlines_punctuation_form_and_are_idempotent() {
        let input = "  a  \tb \r\n\r\nwide，punctuation！\r\n";
        let once = normalize(input, json!({}));
        assert_eq!(once, " a b\n\nwide，punctuation！\n");
        // Default punctuation is `none`; ask for halfwidth explicitly.
        let once = normalize(input, json!({ "punctuation": "halfwidth" }));
        assert_eq!(once, " a b\n\nwide,punctuation!\n");
        assert_eq!(
            normalize(&once, json!({ "punctuation": "halfwidth" })),
            once,
            "normalize must be idempotent"
        );
    }

    #[test]
    fn each_rule_can_be_switched_off() {
        let source = "a  b\r\nc\r\n";
        assert_eq!(
            normalize(
                source,
                json!({ "whitespace": false, "newline": false, "unicode": "none" })
            ),
            source
        );
        assert_eq!(
            normalize(source, json!({ "whitespace": false, "unicode": "none" })),
            "a  b\nc\n"
        );
    }

    #[test]
    fn halfwidth_maps_punctuation_but_not_letters_or_digits() {
        let full = "！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～";
        let half = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
        assert_eq!(halfwidth_punctuation(full), half);
        assert_eq!(halfwidth_punctuation("Ａ１Ｂ２　x"), "Ａ１Ｂ２　x");
    }

    #[test]
    fn nfc_composes_decomposed_text() {
        let decomposed = "e\u{301}";
        assert_eq!(normalize(decomposed, json!({})), "é");
        assert_eq!(
            normalize(decomposed, json!({ "unicode": "none" })),
            decomposed
        );
    }

    #[test]
    fn unknown_options_are_diagnostics() {
        let temp = tempdir().expect("temp dir");
        for input in [
            json!({ "text": "x", "whitespace": "yes" }),
            json!({ "text": "x", "punctuation": "full" }),
            json!({ "text": "x", "newline": 1 }),
            json!({ "text": "x", "unicode": "NFKC" }),
        ] {
            let error = run(&input, temp.path()).expect_err("invalid option");
            assert_eq!(error.code().as_str(), "MOD/CALLER_INVALID_OPTION");
        }
    }

    #[test]
    fn asset_input_writes_output_md() {
        let temp = tempdir().expect("temp dir");
        let input_path = temp.path().join("input.md");
        std::fs::write(&input_path, " a  b \n").expect("write");
        let output_dir = temp.path().join("out");
        std::fs::create_dir_all(&output_dir).expect("out dir");
        let result = run(
            &json!({ "doc": input_path.to_str().expect("utf8 path") }),
            &output_dir,
        )
        .expect("normalize");
        assert_eq!(result, json!({ "doc": "output.md" }));
        assert_eq!(
            std::fs::read_to_string(output_dir.join("output.md")).expect("read"),
            " a b\n"
        );
    }
}
