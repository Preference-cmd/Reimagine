//! `{{name}}` placeholder rendering. There are no loops, conditionals or
//! escaping: a placeholder is looked up in `values`, then the document's
//! front matter, then the body, and a missing name is an error.

use std::collections::BTreeMap;

use mod_sdk::serde_json::{Map, Value};

/// Why a template could not be rendered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderError {
    /// No value, front matter field or `body` matches the placeholder.
    Missing(String),
    /// A `{{` is never closed with `}}`.
    Unclosed,
    /// A placeholder has no name.
    Empty,
    /// A `values` entry is an object, array or null.
    NonScalar(String),
}

impl RenderError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Missing(_) => "MISSING_PLACEHOLDER",
            Self::Unclosed | Self::Empty => "INVALID_TEMPLATE",
            Self::NonScalar(_) => "NON_SCALAR_VALUE",
        }
    }

    pub fn message(&self) -> String {
        match self {
            Self::Missing(name) => format!(
                "the template uses `{{{{{name}}}}}`, but no value, front matter field or `body` provides it"
            ),
            Self::Unclosed => "the template has a `{{` without a closing `}}`".to_owned(),
            Self::Empty => "the template has an empty placeholder `{{}}`".to_owned(),
            Self::NonScalar(name) => format!("`values.{name}` is not a string, number or boolean"),
        }
    }
}

/// Render `template`, resolving every `{{name}}` from `values`, then the
/// document's front matter, then `body`.
pub fn render(
    template: &str,
    values: Option<&Map<String, Value>>,
    front_matter: Option<&BTreeMap<String, String>>,
    body: &str,
) -> Result<String, RenderError> {
    let mut rendered = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find("{{") {
        rendered.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let Some(end) = after.find("}}") else {
            return Err(RenderError::Unclosed);
        };
        let name = after[..end].trim();
        if name.is_empty() {
            return Err(RenderError::Empty);
        }
        let replacement = if let Some(value) = values.and_then(|values| values.get(name)) {
            scalar(value).ok_or_else(|| RenderError::NonScalar(name.to_owned()))?
        } else if let Some(value) = front_matter.and_then(|fields| fields.get(name)) {
            value.clone()
        } else if name == "body" {
            body.to_owned()
        } else {
            return Err(RenderError::Missing(name.to_owned()));
        };
        rendered.push_str(&replacement);
        rest = &after[end + 2..];
    }
    rendered.push_str(rest);
    Ok(rendered)
}

/// Reject a `values` object with a value that has no string form. Checking
/// every entry up front means an unused bad value still fails the call.
pub fn validate_values(values: &Map<String, Value>) -> Result<(), RenderError> {
    for (name, value) in values {
        if scalar(value).is_none() {
            return Err(RenderError::NonScalar(name.clone()));
        }
    }
    Ok(())
}

fn scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(flag) => Some(flag.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use mod_sdk::serde_json::json;

    use super::*;

    fn fields(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[test]
    fn values_beat_front_matter_which_beats_body() {
        let values = json!({ "title": "From values", "count": 3 })
            .as_object()
            .expect("object")
            .clone();
        let front_matter = fields(&[("title", "From front matter"), ("author", "A")]);
        let rendered = render(
            "{{title}}|{{author}}|{{count}}|{{body}}",
            Some(&values),
            Some(&front_matter),
            "The body",
        )
        .expect("render");
        assert_eq!(rendered, "From values|A|3|The body");
    }

    #[test]
    fn body_is_available_without_front_matter() {
        let rendered = render("<{{body}}>", None, None, "text").expect("render");
        assert_eq!(rendered, "<text>");
    }

    #[test]
    fn booleans_and_numbers_render_as_json_scalars() {
        let values = json!({ "flag": true, "n": 1.5 })
            .as_object()
            .expect("object")
            .clone();
        let rendered = render("{{flag}} {{n}}", Some(&values), None, "").expect("render");
        assert_eq!(rendered, "true 1.5");
    }

    #[test]
    fn a_missing_placeholder_is_an_error() {
        let error = render("{{nope}}", None, None, "body").expect_err("missing");
        assert_eq!(error.code(), "MISSING_PLACEHOLDER");
        assert!(error.message().contains("nope"));
    }

    #[test]
    fn unclosed_and_empty_placeholders_are_template_errors() {
        assert_eq!(
            render("{{name", None, None, "body").expect_err("unclosed"),
            RenderError::Unclosed
        );
        assert_eq!(
            render("{{ }}", None, None, "body").expect_err("empty"),
            RenderError::Empty
        );
    }

    #[test]
    fn non_scalar_values_are_rejected_even_when_unused() {
        let values = json!({ "used": "x", "bad": { "nested": 1 } })
            .as_object()
            .expect("object")
            .clone();
        assert_eq!(
            validate_values(&values).expect_err("object value"),
            RenderError::NonScalar("bad".to_owned())
        );
        assert_eq!(
            render("{{bad}}", Some(&values), None, "").expect_err("object value"),
            RenderError::NonScalar("bad".to_owned())
        );
    }

    #[test]
    fn names_are_trimmed_inside_braces() {
        let rendered = render("{{ body }}", None, None, "x").expect("render");
        assert_eq!(rendered, "x");
    }

    #[test]
    fn rendering_is_single_pass_and_does_not_escape() {
        let values = json!({ "a": "{{body}}" })
            .as_object()
            .expect("object")
            .clone();
        let rendered = render("{{a}}", Some(&values), None, "body").expect("render");
        assert_eq!(rendered, "{{body}}");
    }
}
