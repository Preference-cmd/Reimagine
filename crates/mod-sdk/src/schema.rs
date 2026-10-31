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

/// Whether `schema` has the inline-text marker (`"format": "inline-text"`).
pub(crate) fn is_inline_text_marker(schema: &Value) -> bool {
    schema.get("format").and_then(Value::as_str) == Some("inline-text")
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
    check_marker_locations(
        schema,
        is_asset_marker,
        "asset markers are allowed only under `properties` and `items`",
    )
}

/// The inline-text counterpart of [`check_asset_marker_locations`]: a
/// `"format": "inline-text"` marker must appear only directly as a property
/// in `properties` or as `items`, the same places the walk that finds inline
/// fields follows.
pub fn check_inline_text_marker_locations(schema: &Value) -> Result<(), String> {
    check_marker_locations(
        schema,
        is_inline_text_marker,
        "inline-text markers are allowed only under `properties` and `items`",
    )
}

fn check_marker_locations(
    schema: &Value,
    is_marker: fn(&Value) -> bool,
    message: &str,
) -> Result<(), String> {
    let mut rejection = None;
    walk_marker_positions(
        schema,
        None,
        "",
        true,
        false,
        &mut |node, _value, _pointer, allowed| {
            if rejection.is_none() && !allowed && is_marker(node) {
                rejection = Some(message.to_owned());
            }
        },
    );
    match rejection {
        Some(message) => Err(message),
        None => Ok(()),
    }
}

/// Walk every schema node from the root and report whether it sits in a place
/// a marker may annotate.
///
/// `on_walk` is true for the root and for every node the marker finders follow;
/// `allowed` is true only for followed nodes other than the root, i.e. values of
/// a `properties` object or of `items`, transitively. Any other keyword
/// (`oneOf`, `allOf`, `not`, `additionalProperties`, …) is scanned with both
/// flags false: a marker there is rejected by the checks and is never returned
/// by the finders, so validation and discovery cannot disagree.
///
/// `value` is the payload paired with `schema` when finding fields; validation
/// passes `None` to walk the schema structurally.
fn walk_marker_positions(
    schema: &Value,
    value: Option<&Value>,
    pointer: &str,
    on_walk: bool,
    allowed: bool,
    visit: &mut impl FnMut(&Value, Option<&Value>, &str, bool),
) {
    visit(schema, value, pointer, allowed);

    let Value::Object(map) = schema else {
        if let Value::Array(items) = schema {
            for (index, item) in items.iter().enumerate() {
                let next = format!("{pointer}/{index}");
                walk_marker_positions(item, None, &next, false, false, visit);
            }
        }
        return;
    };

    if !on_walk {
        for (key, child) in map {
            let next = format!("{pointer}/{}", escape_pointer_segment(key));
            walk_marker_positions(child, None, &next, false, false, visit);
        }
        return;
    }

    if let Some(properties) = map.get("properties") {
        if let Value::Object(properties) = properties {
            for (name, property) in properties {
                let next = format!("{pointer}/{}", escape_pointer_segment(name));
                let property_value = value
                    .and_then(Value::as_object)
                    .and_then(|object| object.get(name));
                walk_marker_positions(property, property_value, &next, true, true, visit);
            }
        } else {
            let next = format!("{pointer}/properties");
            walk_marker_positions(properties, None, &next, false, false, visit);
        }
    }
    if let Some(items) = map.get("items") {
        match value.and_then(Value::as_array) {
            Some(array) if !array.is_empty() => {
                for (index, item_value) in array.iter().enumerate() {
                    let next = format!("{pointer}/{index}");
                    walk_marker_positions(items, Some(item_value), &next, true, true, visit);
                }
            }
            _ => {
                let next = format!("{pointer}/items");
                walk_marker_positions(items, None, &next, true, true, visit);
            }
        }
    }
    for (key, child) in map {
        if key == "properties" || key == "items" {
            continue;
        }
        let next = format!("{pointer}/{}", escape_pointer_segment(key));
        walk_marker_positions(child, None, &next, false, false, visit);
    }
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
    walk_marker_positions(
        schema,
        Some(value),
        "",
        true,
        false,
        &mut |node, value, pointer, allowed| {
            if !allowed || !is_asset_schema(node) {
                return;
            }
            if value.is_some_and(Value::is_string) {
                let media_type = node
                    .get("contentMediaType")
                    .and_then(Value::as_str)
                    .map(|s| s.to_owned());
                fields.push(AssetField {
                    pointer: pointer.to_owned(),
                    media_type,
                });
            }
        },
    );
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

/// A populated inline-text field discovered within a payload instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InlineTextField {
    /// JSON Pointer (RFC 6901), e.g. `"/text"` or `"/texts/0"`.
    pub pointer: String,
    /// Length of the value in UTF-8 bytes.
    pub bytes: usize,
}

/// Walk schema and value together and return the JSON pointers and UTF-8 byte
/// lengths of all populated inline-text fields.
///
/// A field is inline text when its schema carries `"format": "inline-text"`
/// and the value is a string. The walk follows `properties` and `items`, the
/// same places the asset marker is allowed, so an inline document inside an
/// array of strings or a nested object is found.
pub fn find_inline_text_fields(schema: &Value, value: &Value) -> Vec<InlineTextField> {
    let mut fields = Vec::new();
    walk_marker_positions(
        schema,
        Some(value),
        "",
        true,
        false,
        &mut |node, value, pointer, allowed| {
            if !allowed || !is_inline_text_marker(node) {
                return;
            }
            if let Some(text) = value.and_then(Value::as_str) {
                fields.push(InlineTextField {
                    pointer: pointer.to_owned(),
                    bytes: text.len(),
                });
            }
        },
    );
    fields
}

fn escape_pointer_segment(seg: &str) -> String {
    seg.replace('~', "~0").replace('/', "~1")
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
    fn inline_text_marker_locations_valid() {
        let valid_schema = json!({
            "type": "object",
            "properties": {
                "text": { "type": "string", "format": "inline-text" },
                "texts": {
                    "type": "array",
                    "items": { "type": "string", "format": "inline-text" }
                },
                "nested": {
                    "type": "object",
                    "properties": {
                        "text": { "type": "string", "format": "inline-text" }
                    }
                }
            }
        });
        assert!(check_inline_text_marker_locations(&valid_schema).is_ok());
    }

    #[test]
    fn inline_text_marker_locations_reject_other_positions() {
        for invalid_schema in [
            json!({ "type": "string", "format": "inline-text" }),
            json!({
                "type": "object",
                "properties": {
                    "text": {
                        "oneOf": [
                            { "type": "string", "format": "inline-text" },
                            { "type": "null" }
                        ]
                    }
                }
            }),
            json!({
                "type": "object",
                "additionalProperties": { "type": "string", "format": "inline-text" }
            }),
        ] {
            assert!(check_inline_text_marker_locations(&invalid_schema).is_err());
        }
    }

    #[test]
    fn markers_under_oneof_are_rejected_even_inside_nested_properties() {
        let inline_under_oneof = json!({
            "type": "object",
            "properties": {
                "wrapper": {
                    "oneOf": [{
                        "type": "object",
                        "properties": {
                            "text": { "type": "string", "format": "inline-text" }
                        }
                    }]
                }
            }
        });
        assert!(check_inline_text_marker_locations(&inline_under_oneof).is_err());
        assert!(
            find_inline_text_fields(&inline_under_oneof, &json!({ "wrapper": { "text": "x" } }))
                .is_empty(),
            "a rejected marker must not be discovered"
        );

        let asset_under_allof = json!({
            "type": "object",
            "properties": {
                "wrapper": {
                    "allOf": [{
                        "type": "object",
                        "properties": {
                            "doc": { "type": "string", "format": "asset" }
                        }
                    }]
                }
            }
        });
        assert!(check_asset_marker_locations(&asset_under_allof).is_err());
        assert!(
            find_asset_fields(
                &asset_under_allof,
                &json!({ "wrapper": { "doc": "asset-1" } })
            )
            .is_empty(),
            "a rejected asset marker must not be discovered"
        );
    }

    #[test]
    fn every_accepted_inline_text_marker_is_found() {
        let marker = json!({ "type": "string", "format": "inline-text" });
        let schema = json!({
            "type": "object",
            "properties": {
                "text": marker,
                "nested": {
                    "type": "object",
                    "properties": { "text": marker }
                },
                "texts": {
                    "type": "array",
                    "items": marker
                },
                "grid": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": marker
                    }
                }
            }
        });
        assert!(check_inline_text_marker_locations(&schema).is_ok());

        let value = json!({
            "text": "a",
            "nested": { "text": "bb" },
            "texts": ["c", "dd"],
            "grid": [["e"], ["ff"]]
        });
        assert_eq!(
            find_inline_text_fields(&schema, &value),
            vec![
                InlineTextField {
                    pointer: "/grid/0/0".to_owned(),
                    bytes: 1,
                },
                InlineTextField {
                    pointer: "/grid/1/0".to_owned(),
                    bytes: 2,
                },
                InlineTextField {
                    pointer: "/nested/text".to_owned(),
                    bytes: 2,
                },
                InlineTextField {
                    pointer: "/text".to_owned(),
                    bytes: 1,
                },
                InlineTextField {
                    pointer: "/texts/0".to_owned(),
                    bytes: 1,
                },
                InlineTextField {
                    pointer: "/texts/1".to_owned(),
                    bytes: 2,
                },
            ]
        );
    }

    #[test]
    fn every_accepted_asset_marker_is_found() {
        let marker = json!({ "type": "string", "format": "asset" });
        let schema = json!({
            "type": "object",
            "properties": {
                "doc": marker,
                "nested": {
                    "type": "object",
                    "properties": { "file": marker }
                },
                "assets": {
                    "type": "array",
                    "items": marker
                },
                "grid": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": marker
                    }
                }
            }
        });
        assert!(check_asset_marker_locations(&schema).is_ok());

        let value = json!({
            "doc": "asset-doc",
            "nested": { "file": "asset-file" },
            "assets": ["asset-a", "asset-b"],
            "grid": [["asset-c"]]
        });
        assert_eq!(
            find_asset_fields(&schema, &value)
                .into_iter()
                .map(|field| field.pointer)
                .collect::<Vec<_>>(),
            [
                "/assets/0",
                "/assets/1",
                "/doc",
                "/grid/0/0",
                "/nested/file"
            ]
        );
    }

    #[test]
    fn inline_text_and_asset_markers_are_checked_separately() {
        let asset_under_oneof = json!({
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
        assert!(check_inline_text_marker_locations(&asset_under_oneof).is_ok());
        assert!(check_asset_marker_locations(&asset_under_oneof).is_err());
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

    #[test]
    fn find_inline_text_fields_reports_direct_fields_in_byte_length() {
        let schema = json!({
            "type": "object",
            "properties": {
                "text": { "type": "string", "format": "inline-text" }
            }
        });
        let fields = find_inline_text_fields(&schema, &json!({ "text": "héllo" }));
        assert_eq!(
            fields,
            vec![InlineTextField {
                pointer: "/text".to_owned(),
                bytes: 6,
            }]
        );
    }

    #[test]
    fn find_inline_text_fields_walks_nested_objects_and_array_items() {
        let schema = json!({
            "type": "object",
            "properties": {
                "doc": {
                    "type": "object",
                    "properties": {
                        "text": { "type": "string", "format": "inline-text" }
                    }
                },
                "texts": {
                    "type": "array",
                    "items": { "type": "string", "format": "inline-text" }
                }
            }
        });
        let value = json!({
            "doc": { "text": "ab" },
            "texts": ["c", "dé"]
        });
        assert_eq!(
            find_inline_text_fields(&schema, &value),
            vec![
                InlineTextField {
                    pointer: "/doc/text".to_owned(),
                    bytes: 2,
                },
                InlineTextField {
                    pointer: "/texts/0".to_owned(),
                    bytes: 1,
                },
                InlineTextField {
                    pointer: "/texts/1".to_owned(),
                    bytes: 3,
                },
            ]
        );
    }

    #[test]
    fn find_inline_text_fields_ignores_non_string_values() {
        let schema = json!({
            "type": "object",
            "properties": {
                "text": { "type": "string", "format": "inline-text" },
                "items": {
                    "type": "array",
                    "items": { "type": "string", "format": "inline-text" }
                }
            }
        });
        let value = json!({
            "text": 7,
            "items": ["ok", 1, null, { "text": "nested" }]
        });
        assert_eq!(
            find_inline_text_fields(&schema, &value),
            vec![InlineTextField {
                pointer: "/items/0".to_owned(),
                bytes: 2,
            }]
        );
    }

    #[test]
    fn find_inline_text_fields_has_no_false_positives() {
        let schema = json!({
            "type": "object",
            "properties": {
                "doc": { "type": "string", "format": "asset" },
                "plain": { "type": "string" },
                "nested": {
                    "type": "object",
                    "properties": {
                        "plain": { "type": "string" }
                    }
                }
            }
        });
        let value = json!({
            "doc": "asset-1",
            "plain": "just text",
            "nested": { "plain": "more text" }
        });
        assert!(find_inline_text_fields(&schema, &value).is_empty());
    }

    #[test]
    fn find_inline_text_fields_escapes_pointer_segments() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a/b": { "type": "string", "format": "inline-text" },
                "a~b": { "type": "string", "format": "inline-text" }
            }
        });
        let value = json!({ "a/b": "x", "a~b": "y" });
        let pointers: Vec<String> = find_inline_text_fields(&schema, &value)
            .into_iter()
            .map(|field| field.pointer)
            .collect();
        assert_eq!(pointers, ["/a~1b", "/a~0b"]);
    }
}
