//! Tool input/output schema validation.
//!
//! Validates tool inputs against JSON Schema before handler execution
//! and enforces output size limits.

#![allow(clippy::collapsible_if)]
#![allow(clippy::unnecessary_map_or)]

use serde_json::Value;

use crate::error::{ToolError, ToolErrorCode};
use crate::ids::ToolName;

/// Maximum allowed tool output size in bytes (1 MB).
pub const MAX_TOOL_OUTPUT_SIZE: usize = 1024 * 1024;

/// Validate tool input against the provided JSON Schema.
///
/// Returns Ok(()) if validation passes, or Err with an InvalidInput error
/// containing details about what failed validation.
pub fn validate_tool_input(
    schema: &Value,
    input: &Value,
    tool_name: &ToolName,
) -> Result<(), ToolError> {
    // Basic JSON Schema validation
    if let Err(e) = validate_value_against_schema(schema, input) {
        return Err(ToolError::new(
            ToolErrorCode::InvalidInput,
            format!("input validation failed: {e}"),
        )
        .with_tool(tool_name.clone()));
    }

    Ok(())
}

/// Validate tool output size against the configured limit.
///
/// Returns Ok(()) if the output is within limits, or Err with an
/// ExecutionFailed error if it exceeds the limit.
pub fn validate_tool_output_size(output: &Value, tool_name: &ToolName) -> Result<(), ToolError> {
    let size = serde_json::to_string(output).map(|s| s.len()).unwrap_or(0);

    if size > MAX_TOOL_OUTPUT_SIZE {
        return Err(ToolError::new(
            ToolErrorCode::ExecutionFailed,
            format!(
                "tool output too large: {} bytes exceeds limit of {} bytes",
                size, MAX_TOOL_OUTPUT_SIZE
            ),
        )
        .with_tool(tool_name.clone()));
    }

    Ok(())
}

/// Simple JSON Schema validation implementation.
fn validate_value_against_schema(schema: &Value, value: &Value) -> Result<(), String> {
    // Handle empty schema (allows anything)
    if schema.as_object().is_none_or(|o| o.is_empty()) {
        return Ok(());
    }

    // Validate type if specified
    if let Some(expected_type) = schema.get("type") {
        let actual_type = match value {
            Value::Null => "null",
            Value::Bool(_) => "boolean",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        };

        let expected_str = expected_type.as_str().unwrap_or("");
        if expected_str != actual_type {
            return Err(format!(
                "expected type '{}', got '{}'",
                expected_str, actual_type
            ));
        }
    }

    // Validate enum if specified
    if let Some(enum_values) = schema.get("enum") {
        if let Some(enum_array) = enum_values.as_array() {
            if !enum_array.contains(value) {
                return Err(format!("value '{}' not in allowed enum values", value));
            }
        }
    }

    // Validate object properties if specified
    if let (Some(properties), Some(obj)) = (schema.get("properties"), value.as_object()) {
        if let Some(props_obj) = properties.as_object() {
            // Check required fields
            if let Some(required) = schema.get("required") {
                if let Some(required_array) = required.as_array() {
                    for req_field in required_array {
                        if let Some(field_name) = req_field.as_str() {
                            if !obj.contains_key(field_name) {
                                return Err(format!("missing required field '{}'", field_name));
                            }
                        }
                    }
                }
            }

            // Validate each property against its schema
            for (key, prop_schema) in props_obj {
                if let Some(prop_value) = obj.get(key) {
                    validate_value_against_schema(prop_schema, prop_value)?;
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool_name() -> ToolName {
        ToolName::new("test.tool")
    }

    #[test]
    fn valid_input_passes_validation() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "number" }
            },
            "required": ["name"]
        });
        let input = json!({
            "name": "John",
            "age": 30
        });

        assert!(validate_tool_input(&schema, &input, &tool_name()).is_ok());
    }

    #[test]
    fn missing_required_field_fails_validation() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "number" }
            },
            "required": ["name"]
        });
        let input = json!({
            "age": 30
        });

        let result = validate_tool_input(&schema, &input, &tool_name());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code(), ToolErrorCode::InvalidInput);
    }

    #[test]
    fn wrong_type_fails_validation() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            }
        });
        let input = json!({
            "name": 123
        });

        let result = validate_tool_input(&schema, &input, &tool_name());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code(), ToolErrorCode::InvalidInput);
    }

    #[test]
    fn wrong_enum_value_fails_validation() {
        let schema = json!({
            "type": "object",
            "properties": {
                "mode": { "type": "string", "enum": ["agent", "build"] }
            }
        });
        let input = json!({
            "mode": "invalid"
        });

        let result = validate_tool_input(&schema, &input, &tool_name());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code(), ToolErrorCode::InvalidInput);
    }

    #[test]
    fn valid_enum_value_passes_validation() {
        let schema = json!({
            "type": "object",
            "properties": {
                "mode": { "type": "string", "enum": ["agent", "build"] }
            }
        });
        let input = json!({
            "mode": "agent"
        });

        assert!(validate_tool_input(&schema, &input, &tool_name()).is_ok());
    }

    #[test]
    fn output_within_size_limit() {
        let output = json!({"result": "success"});
        assert!(validate_tool_output_size(&output, &tool_name()).is_ok());
    }

    #[test]
    fn output_exceeds_size_limit() {
        // Create a large output that exceeds the 1MB limit
        let large_string = "x".repeat(MAX_TOOL_OUTPUT_SIZE + 1);
        let output = json!({"data": large_string});

        let result = validate_tool_output_size(&output, &tool_name());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code(), ToolErrorCode::ExecutionFailed);
    }
}
