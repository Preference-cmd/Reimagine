//! `text.template`: render `{{name}}` placeholders from values, front matter
//! and the body.

use std::path::Path;

use mod_sdk::diagnostic::{Diagnostic, caller_error};
use mod_sdk::serde_json::Value;

use crate::ID;
use crate::document::{self, Document};
use crate::front_matter;
use crate::template;

pub fn run(input: &Value, output_dir: &Path) -> Result<Value, Diagnostic> {
    let document = Document::load(input)?;
    let source = input
        .get("template")
        .and_then(Value::as_str)
        .ok_or_else(|| caller_error(ID, "INVALID_TEMPLATE", "`template` must be a string"))?;
    let values = match input.get("values") {
        None | Some(Value::Null) => None,
        Some(Value::Object(values)) => {
            template::validate_values(values)
                .map_err(|error| caller_error(ID, error.code(), error.message()))?;
            Some(values)
        }
        Some(_) => {
            return Err(caller_error(
                ID,
                "INVALID_VALUES",
                "`values` must be an object of strings, numbers and booleans",
            ));
        }
    };

    let parsed = front_matter::parse(&document.text)
        .map_err(|error| caller_error(ID, error.code(), error.message()))?;
    let rendered = template::render(source, values, parsed.front_matter.as_ref(), parsed.body)
        .map_err(|error| caller_error(ID, error.code(), error.message()))?;
    document::result(document.form, output_dir, "output.md", &rendered)
}

#[cfg(test)]
mod tests {
    use mod_sdk::serde_json::json;
    use tempfile::tempdir;

    use super::*;

    fn run_inline(input: Value) -> Result<Value, Diagnostic> {
        let temp = tempdir().expect("temp dir");
        run(&input, temp.path())
    }

    #[test]
    fn values_beat_front_matter_which_beats_body() {
        let result = run_inline(json!({
            "text": "---\ntitle: From front matter\nauthor: A\n---\nBody text",
            "template": "{{title}}|{{author}}|{{count}}|{{body}}",
            "values": { "title": "From values", "count": 3 }
        }))
        .expect("template");
        assert_eq!(result, json!({ "text": "From values|A|3|Body text" }));
    }

    #[test]
    fn a_document_without_front_matter_still_has_a_body() {
        let result = run_inline(json!({
            "text": "Just body",
            "template": "<{{body}}>"
        }))
        .expect("template");
        assert_eq!(result["text"], "<Just body>");
    }

    #[test]
    fn a_missing_placeholder_is_a_diagnostic() {
        let error = run_inline(json!({
            "text": "Body",
            "template": "{{nope}}"
        }))
        .expect_err("missing placeholder");
        assert_eq!(error.code().as_str(), "MOD/CALLER_MISSING_PLACEHOLDER");
    }

    #[test]
    fn non_scalar_values_are_a_diagnostic_even_when_unused() {
        let error = run_inline(json!({
            "text": "Body",
            "template": "{{body}}",
            "values": { "nested": { "a": 1 } }
        }))
        .expect_err("non-scalar value");
        assert_eq!(error.code().as_str(), "MOD/CALLER_NON_SCALAR_VALUE");
    }

    #[test]
    fn an_unclosed_placeholder_is_a_diagnostic() {
        let error = run_inline(json!({
            "text": "Body",
            "template": "{{body"
        }))
        .expect_err("unclosed placeholder");
        assert_eq!(error.code().as_str(), "MOD/CALLER_INVALID_TEMPLATE");
    }

    #[test]
    fn asset_input_writes_the_rendered_document() {
        let temp = tempdir().expect("temp dir");
        let input_path = temp.path().join("input.md");
        std::fs::write(&input_path, "# Title\n\nBody").expect("write");
        let output_dir = temp.path().join("out");
        std::fs::create_dir_all(&output_dir).expect("out dir");
        let result = run(
            &json!({
                "doc": input_path.to_str().expect("utf8 path"),
                "template": "{{body}}\n\n-- end"
            }),
            &output_dir,
        )
        .expect("template");
        assert_eq!(result, json!({ "doc": "output.md" }));
        assert_eq!(
            std::fs::read_to_string(output_dir.join("output.md")).expect("read"),
            "# Title\n\nBody\n\n-- end"
        );
    }

    #[test]
    fn template_must_be_a_string() {
        let error =
            run_inline(json!({ "text": "Body", "template": 7 })).expect_err("non-string template");
        assert_eq!(error.code().as_str(), "MOD/CALLER_INVALID_TEMPLATE");
    }
}
