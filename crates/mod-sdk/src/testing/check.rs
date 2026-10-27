use serde_json::Value;

use crate::manifest::ModManifest;
use crate::registry::validate_and_compile;
use crate::schema;

/// Every problem that keeps `manifest` from conforming:
/// - anything the registry would reject it for (first such reason only);
/// - an operation name that is not a valid MCP tool name (1-128 characters
///   of `A-Za-z0-9_.-`, MCP specification 2025-11-25 and later);
/// - an empty description;
/// - an output schema that is not an object schema (MCP structured content
///   must be an object);
/// - an input schema without `examples`, or with an example it rejects.
pub fn check_manifest(manifest: &ModManifest) -> Result<(), Vec<String>> {
    let mut problems = Vec::new();
    if let Err(diagnostic) = validate_and_compile(manifest, false) {
        problems.push(diagnostic.message().to_owned());
    }
    for operation in &manifest.operations {
        let name = &operation.name;
        if !is_mcp_tool_name(name) {
            problems.push(format!(
                "operation `{name}` is not a valid MCP tool name (1-128 of A-Za-z0-9_.-)"
            ));
        }
        if operation.description.trim().is_empty() {
            problems.push(format!("operation `{name}` has no description"));
        }
        if !schema::is_object(&operation.output_schema) {
            problems.push(format!(
                "operation `{name}` output_schema root must be `\"type\": \"object\"`"
            ));
        }
        let examples = operation
            .input_schema
            .get("examples")
            .and_then(Value::as_array)
            .filter(|examples| !examples.is_empty());
        let Some(examples) = examples else {
            problems.push(format!("operation `{name}` input_schema has no `examples`"));
            continue;
        };
        if let Ok(validator) = schema::compile(&operation.input_schema) {
            for (index, example) in examples.iter().enumerate() {
                if let Some(why) = schema::violations(&validator, example) {
                    problems.push(format!(
                        "operation `{name}` example {index} fails its input_schema: {why}"
                    ));
                }
            }
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems)
    }
}

/// Check `instance` against `schema` (JSON Schema 2020-12).
pub fn validate(schema: &Value, instance: &Value) -> Result<(), String> {
    let validator = schema::compile(schema)?;
    schema::violations(&validator, instance).map_or(Ok(()), Err)
}

fn is_mcp_tool_name(name: &str) -> bool {
    (1..=128).contains(&name.len())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}
