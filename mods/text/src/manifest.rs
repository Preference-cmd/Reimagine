//! The mod manifest: the six operation contracts and their schemas.

use mod_sdk::serde_json::{Map, Value, json};
use mod_sdk::{ModManifest, OperationAnnotations, OperationSpec};

use crate::ID;

pub fn manifest() -> ModManifest {
    ModManifest {
        id: ID.to_owned(),
        name: "Text".to_owned(),
        version: "0.1.0".to_owned(),
        api: "0.1".to_owned(),
        operations: vec![
            operation(
                "text.normalize",
                "Normalize a document's whitespace, full-width punctuation, line endings and Unicode form.",
                document_input(
                    json!({
                        "whitespace": {
                            "type": "boolean",
                            "default": true,
                            "description": "Collapse runs of spaces and tabs and trim line ends."
                        },
                        "punctuation": {
                            "type": "string",
                            "enum": ["none", "halfwidth"],
                            "default": "none",
                            "description": "Map common full-width ASCII punctuation to half-width."
                        },
                        "newline": {
                            "type": "boolean",
                            "default": true,
                            "description": "Unify CRLF and CR line endings to LF."
                        },
                        "unicode": {
                            "type": "string",
                            "enum": ["NFC", "none"],
                            "default": "NFC",
                            "description": "Unicode normalization form."
                        }
                    }),
                    &[],
                    json!([{ "text": "Hello,  world!\r\n" }]),
                ),
                document_output(json!({}), &[]),
                annotations(false, true),
            ),
            operation(
                "text.replace",
                "Replace literal strings, whole words or regular expressions and report the count per pair.",
                document_input(
                    json!({
                        "pairs": {
                            "type": "array",
                            "description": "Replacement pairs, applied in order.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "find": { "type": "string", "minLength": 1 },
                                    "replace": {
                                        "type": "string",
                                        "description": "In `regex` mode `$1` refers to a capture group; in the other modes the text is literal."
                                    },
                                    "mode": { "type": "string", "enum": ["literal", "word", "regex"] },
                                    "case_sensitive": { "type": "boolean", "default": true }
                                },
                                "required": ["find", "replace", "mode"],
                                "additionalProperties": false
                            }
                        }
                    }),
                    &["pairs"],
                    json!([{
                        "text": "AI is useful",
                        "pairs": [{ "find": "AI", "replace": "A.I.", "mode": "literal" }]
                    }]),
                ),
                document_output(
                    json!({
                        "counts": {
                            "type": "array",
                            "description": "Replacements made per pair, in input order.",
                            "items": { "type": "integer", "minimum": 0 }
                        }
                    }),
                    &["counts"],
                ),
                annotations(false, false),
            ),
            operation(
                "text.inspect",
                "Read a document's title, ATX headings, flat front matter and text statistics as JSON.",
                document_input(json!({}), &[], json!([{ "text": "# Title\n\nBody." }])),
                inspect_output(),
                annotations(true, false),
            ),
            operation(
                "text.template",
                "Render a `{{name}}` template from values, the document's front matter and its body.",
                document_input(
                    json!({
                        "template": {
                            "type": "string",
                            "description": "Placeholders are `{{name}}`, resolved from `values`, then front matter, then `body`."
                        },
                        "values": {
                            "type": "object",
                            "default": {},
                            "additionalProperties": { "type": ["string", "number", "boolean"] }
                        }
                    }),
                    &["template"],
                    json!([{
                        "text": "---\ntitle: T\n---\nBody",
                        "template": "{{title}}: {{body}}"
                    }]),
                ),
                document_output(json!({}), &[]),
                annotations(false, false),
            ),
            operation(
                "text.split",
                "Split a document at a heading level or a literal separator; every piece carries a title.",
                document_input(
                    json!({
                        "by": {
                            "type": "object",
                            "properties": {
                                "kind": { "type": "string", "enum": ["heading", "separator"] },
                                "level": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 6,
                                    "description": "Heading level to split on; required when `kind` is `heading`."
                                },
                                "separator": {
                                    "type": "string",
                                    "minLength": 1,
                                    "description": "Literal separator; required when `kind` is `separator`."
                                }
                            },
                            "required": ["kind"],
                            "additionalProperties": false
                        }
                    }),
                    &["by"],
                    json!([{
                        "text": "## One\n\nfirst\n\n## Two\n\nsecond",
                        "by": { "kind": "heading", "level": 2 }
                    }]),
                ),
                split_output(),
                annotations(false, false),
            ),
            operation(
                "text.join",
                "Join documents in order with a separator, or render a template for each and concatenate.",
                join_input(),
                document_output(json!({}), &[]),
                annotations(false, false),
            ),
        ],
        events: Vec::new(),
    }
}

fn operation(
    name: &str,
    description: &str,
    input_schema: Value,
    output_schema: Value,
    annotations: OperationAnnotations,
) -> OperationSpec {
    OperationSpec {
        name: name.to_owned(),
        description: description.to_owned(),
        input_schema,
        output_schema,
        annotations,
    }
}

/// A deterministic, project-local operation.
fn annotations(read_only: bool, idempotent: bool) -> OperationAnnotations {
    OperationAnnotations {
        read_only,
        destructive: false,
        idempotent,
        deterministic: true,
        open_world: false,
    }
}

/// An input schema whose document is `doc` (asset) or `text` (inline), plus
/// the operation's extra properties.
fn document_input(extra: Value, required: &[&str], examples: Value) -> Value {
    let mut properties = document_properties("text/*");
    if let Value::Object(extra) = extra {
        for (key, value) in extra {
            properties.insert(key, value);
        }
    }
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
        "examples": examples,
    })
}

/// An output schema mirroring the input form: `doc` for an asset input,
/// `text` for an inline input, plus the operation's extra properties.
fn document_output(extra: Value, required: &[&str]) -> Value {
    let mut properties = document_properties("text/markdown");
    if let Value::Object(extra) = extra {
        for (key, value) in extra {
            properties.insert(key, value);
        }
    }
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}

fn document_properties(media_type: &str) -> Map<String, Value> {
    let mut properties = Map::new();
    properties.insert(
        "doc".to_owned(),
        json!({
            "type": "string",
            "format": "asset",
            "contentMediaType": media_type,
            "description": "An asset path. Give exactly one of `doc` and `text`."
        }),
    );
    properties.insert(
        "text".to_owned(),
        json!({
            "type": "string",
            "format": "inline-text",
            "description": "Inline text (at most 256 KiB). Give exactly one of `doc` and `text`."
        }),
    );
    properties
}

fn inspect_output() -> Value {
    json!({
        "type": "object",
        "properties": {
            "title": { "type": "string" },
            "headings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "level": { "type": "integer", "minimum": 1, "maximum": 6 },
                        "text": { "type": "string" }
                    },
                    "required": ["level", "text"],
                    "additionalProperties": false
                }
            },
            "front_matter": {
                "type": ["object", "null"],
                "additionalProperties": { "type": "string" }
            },
            "stats": {
                "type": "object",
                "properties": {
                    "chars": { "type": "integer", "minimum": 0 },
                    "lines": { "type": "integer", "minimum": 0 },
                    "paragraphs": { "type": "integer", "minimum": 0 }
                },
                "required": ["chars", "lines", "paragraphs"],
                "additionalProperties": false
            }
        },
        "required": ["title", "headings", "front_matter", "stats"],
        "additionalProperties": false,
    })
}

fn split_output() -> Value {
    json!({
        "type": "object",
        "properties": {
            "docs": {
                "type": "array",
                "description": "Pieces of an asset input, written as `.md` files in the output directory.",
                "items": {
                    "type": "object",
                    "properties": {
                        "doc": { "type": "string", "format": "asset", "contentMediaType": "text/markdown" },
                        "title": { "type": "string" },
                        "chars": { "type": "integer", "minimum": 1 }
                    },
                    "required": ["doc", "title", "chars"],
                    "additionalProperties": false
                }
            },
            "pieces": {
                "type": "array",
                "description": "Pieces of an inline input.",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": { "type": "string", "format": "inline-text" },
                        "title": { "type": "string" },
                        "chars": { "type": "integer", "minimum": 1 }
                    },
                    "required": ["text", "title", "chars"],
                    "additionalProperties": false
                }
            }
        },
        "additionalProperties": false,
    })
}

fn join_input() -> Value {
    json!({
        "type": "object",
        "properties": {
            "docs": {
                "type": "array",
                "description": "Documents to join, in order. Give exactly one of `docs` and `texts`.",
                "items": {
                    "type": "object",
                    "properties": {
                        "doc": { "type": "string", "format": "asset", "contentMediaType": "text/*" },
                        "title": { "type": "string" }
                    },
                    "required": ["doc"],
                    "additionalProperties": false
                }
            },
            "texts": {
                "type": "array",
                "description": "Inline documents to join, in order. Give exactly one of `docs` and `texts`.",
                "items": { "type": "string", "format": "inline-text" }
            },
            "separator": {
                "type": "string",
                "description": "Literal separator between pieces. Give exactly one of `separator` and `template`."
            },
            "template": {
                "type": "string",
                "description": "Rendered once per piece with `{{name}}` placeholders, then concatenated. Give exactly one of `separator` and `template`."
            }
        },
        "additionalProperties": false,
        "examples": [{
            "texts": ["one", "two"],
            "separator": "\n\n---\n\n"
        }],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_document_input_declares_doc_and_inline_text() {
        let manifest = manifest();
        assert_eq!(manifest.id, "text");
        assert_eq!(manifest.version, "0.1.0");
        assert_eq!(manifest.api, "0.1");
        assert!(manifest.events.is_empty());
        for operation in &manifest.operations {
            if operation.name == "text.join" {
                continue;
            }
            let properties = operation.input_schema["properties"]
                .as_object()
                .expect("properties");
            assert_eq!(properties["doc"]["format"], "asset");
            assert_eq!(properties["doc"]["contentMediaType"], "text/*");
            assert_eq!(properties["text"]["format"], "inline-text");
        }
    }

    #[test]
    fn annotations_match_the_contract() {
        let manifest = manifest();
        let annotations = |name: &str| {
            manifest
                .operations
                .iter()
                .find(|operation| operation.name == name)
                .expect("declared operation")
                .annotations
        };
        for operation in &manifest.operations {
            assert!(operation.annotations.deterministic, "{}", operation.name);
            assert!(!operation.annotations.open_world, "{}", operation.name);
            assert!(!operation.annotations.destructive, "{}", operation.name);
        }
        assert!(annotations("text.normalize").idempotent);
        assert!(annotations("text.inspect").read_only);
        assert!(!annotations("text.replace").idempotent);
        assert!(!annotations("text.inspect").idempotent);
    }
}
