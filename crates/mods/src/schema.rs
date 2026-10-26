//! JSON Schema 2020-12 handling at the mod boundary. Remote and file
//! references are never fetched: a schema must be self-contained.

use jsonschema::Validator;
use serde_json::Value;

/// Compile `schema` as JSON Schema 2020-12, regardless of its `$schema`.
pub(crate) fn compile(schema: &Value) -> Result<Validator, String> {
    jsonschema::options()
        .with_draft(jsonschema::Draft::Draft202012)
        .build(schema)
        .map_err(|error| error.to_string())
}

/// Every reason `instance` fails `validator`, joined; `None` when it passes.
pub(crate) fn violations(validator: &Validator, instance: &Value) -> Option<String> {
    let errors: Vec<String> = validator
        .iter_errors(instance)
        .map(|error| error.to_string())
        .collect();
    (!errors.is_empty()).then(|| errors.join("; "))
}

/// Whether the schema's root is `"type": "object"`.
pub(crate) fn is_object(schema: &Value) -> bool {
    schema.get("type").and_then(Value::as_str) == Some("object")
}
