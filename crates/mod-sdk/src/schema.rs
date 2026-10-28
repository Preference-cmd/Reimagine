//! JSON Schema 2020-12 handling at the mod boundary. Remote and file
//! references are never fetched: a schema must be self-contained.

use jsonschema::Validator;
use serde_json::Value;

/// Compile `schema` as JSON Schema 2020-12, regardless of its `$schema`.
pub fn compile(schema: &Value) -> Result<Validator, String> {
    jsonschema::options()
        .with_draft(jsonschema::Draft::Draft202012)
        .build(schema)
        .map_err(|error| error.to_string())
}

/// Every reason `instance` fails `validator`, joined; `None` when it passes.
pub fn violations(validator: &Validator, instance: &Value) -> Option<String> {
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

/// Whether `schema` has the asset marker (`"format": "asset"`).
pub(crate) fn is_asset_marker(schema: &Value) -> bool {
    schema.get("format").and_then(Value::as_str) == Some("asset")
}

/// Whether `schema` is a valid asset field (`"type": "string"`, `"format": "asset"`).
pub(crate) fn is_asset_schema(schema: &Value) -> bool {
    schema.get("type").and_then(Value::as_str) == Some("string")
        && schema.get("format").and_then(Value::as_str) == Some("asset")
}

/// Recursively checks that any asset marker in `schema` (`"format": "asset"`)
/// appears ONLY directly as a property in `properties` or as `items`.
/// Rejects markers anywhere else (e.g. under `oneOf`, `anyOf`, `allOf`, `not`, root).
pub fn check_asset_marker_locations(schema: &Value) -> Result<(), String> {
    fn walk(val: &Value, in_allowed: bool) -> Result<(), String> {
        if is_asset_marker(val) && !in_allowed {
            return Err(
                "asset markers are allowed only under `properties` and `items`".to_string(),
            );
        }
        match val {
            Value::Object(map) => {
                for (k, v) in map {
                    if k == "properties" {
                        if let Value::Object(props) = v {
                            for prop_val in props.values() {
                                walk(prop_val, true)?;
                            }
                        } else {
                            walk(v, false)?;
                        }
                    } else if k == "items" {
                        walk(v, true)?;
                    } else {
                        walk(v, false)?;
                    }
                }
            }
            Value::Array(arr) => {
                for item in arr {
                    walk(item, false)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    walk(schema, false)
}

/// Represents an asset field discovered within a payload instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetField {
    /// JSON Pointer (RFC 6901), e.g. `"/doc"` or `"/assets/0"`.
    pub pointer: String,
    /// Pattern or concrete media type declared in `contentMediaType`, if any.
    pub media_type: Option<String>,
}

/// Walk schema and value together and return the JSON pointers and media type patterns
/// of all populated asset fields.
pub fn find_asset_fields(schema: &Value, value: &Value) -> Vec<AssetField> {
    let mut fields = Vec::new();
    walk_schema_value(schema, value, "", &mut fields);
    fields
}

/// Whether `schema` declares any asset field (`"format": "asset"`).
pub fn has_asset_field(schema: &Value) -> bool {
    fn walk(val: &Value) -> bool {
        if is_asset_marker(val) {
            return true;
        }
        match val {
            Value::Object(map) => map.values().any(walk),
            Value::Array(arr) => arr.iter().any(walk),
            _ => false,
        }
    }
    walk(schema)
}

fn escape_pointer_segment(seg: &str) -> String {
    seg.replace('~', "~0").replace('/', "~1")
}

fn walk_schema_value(schema: &Value, value: &Value, current_ptr: &str, out: &mut Vec<AssetField>) {
    if is_asset_schema(schema) {
        if value.is_string() {
            let media_type = schema
                .get("contentMediaType")
                .and_then(Value::as_str)
                .map(|s| s.to_owned());
            out.push(AssetField {
                pointer: current_ptr.to_owned(),
                media_type,
            });
        }
        return;
    }

    match (schema, value) {
        (Value::Object(s_map), Value::Object(v_map)) => {
            if let Some(Value::Object(props)) = s_map.get("properties") {
                for (prop_name, prop_schema) in props {
                    if let Some(prop_val) = v_map.get(prop_name) {
                        let seg = escape_pointer_segment(prop_name);
                        let next_ptr = format!("{current_ptr}/{seg}");
                        walk_schema_value(prop_schema, prop_val, &next_ptr, out);
                    }
                }
            }
        }
        (Value::Object(s_map), Value::Array(v_arr)) => {
            if let Some(items_schema) = s_map.get("items") {
                for (idx, item_val) in v_arr.iter().enumerate() {
                    let next_ptr = format!("{current_ptr}/{idx}");
                    walk_schema_value(items_schema, item_val, &next_ptr, out);
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn asset_marker_locations_valid() {
        let valid_schema = json!({
            "type": "object",
            "properties": {
                "doc": {
                    "type": "string",
                    "format": "asset",
                    "contentMediaType": "text/*"
                },
                "assets": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "format": "asset"
                    }
                }
            }
        });
        assert!(check_asset_marker_locations(&valid_schema).is_ok());
    }

    #[test]
    fn asset_marker_locations_rejects_under_oneof() {
        let invalid_schema = json!({
            "type": "object",
            "properties": {
                "doc": {
                    "oneOf": [
                        { "type": "string", "format": "asset" },
                        { "type": "null" }
                    ]
                }
            }
        });
        assert!(check_asset_marker_locations(&invalid_schema).is_err());
    }

    #[test]
    fn asset_marker_locations_rejects_at_root() {
        let invalid_schema = json!({
            "type": "string",
            "format": "asset"
        });
        assert!(check_asset_marker_locations(&invalid_schema).is_err());
    }

    #[test]
    fn find_asset_fields_extracts_pointers() {
        let schema = json!({
            "type": "object",
            "properties": {
                "doc": {
                    "type": "string",
                    "format": "asset",
                    "contentMediaType": "text/markdown"
                },
                "assets": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "format": "asset"
                    }
                },
                "nested": {
                    "type": "object",
                    "properties": {
                        "file": {
                            "type": "string",
                            "format": "asset",
                            "contentMediaType": "image/png"
                        }
                    }
                },
                "other": { "type": "string" }
            }
        });

        let value = json!({
            "doc": "asset-doc-1",
            "assets": ["asset-a", "asset-b"],
            "nested": {
                "file": "asset-nested-1"
            },
            "other": "hello"
        });

        let fields = find_asset_fields(&schema, &value);
        assert_eq!(
            fields,
            vec![
                AssetField {
                    pointer: "/assets/0".to_string(),
                    media_type: None,
                },
                AssetField {
                    pointer: "/assets/1".to_string(),
                    media_type: None,
                },
                AssetField {
                    pointer: "/doc".to_string(),
                    media_type: Some("text/markdown".to_string()),
                },
                AssetField {
                    pointer: "/nested/file".to_string(),
                    media_type: Some("image/png".to_string()),
                },
            ]
        );
    }

    #[test]
    fn has_asset_field_detects_markers() {
        let with_asset = json!({
            "type": "object",
            "properties": {
                "file": { "type": "string", "format": "asset" }
            }
        });
        assert!(has_asset_field(&with_asset));

        let without_asset = json!({
            "type": "object",
            "properties": {
                "id": { "type": "string" },
                "version": { "type": "integer" }
            }
        });
        assert!(!has_asset_field(&without_asset));
    }
}
