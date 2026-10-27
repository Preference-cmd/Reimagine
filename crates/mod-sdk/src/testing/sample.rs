use std::path::Path;
use std::sync::OnceLock;

use async_trait::async_trait;
use reimagine_core::diagnostic::Diagnostic;
use serde_json::{Value, json};

use crate::context::ModContext;
use crate::diagnostic::{caller_error, cancelled, internal_error};
use crate::interface::{Invocation, Mod};
use crate::manifest::{ModEvent, ModManifest, OperationAnnotations, OperationSpec};

const ID: &str = "sample";

/// A reference mod, id `sample`, that exercises every interface behavior:
///
/// - `sample.echo {text}` returns `{text}` and publishes `sample.echoed`;
/// - `sample.fail {kind}` fails as the caller's (`"caller"`) or its own
///   (`"internal"`) error;
/// - `sample.wait {}` publishes `sample.progress.started`, then waits for
///   cancellation and returns `MOD/CALLER_CANCELLED`;
/// - `sample.transform {input_file, escape, symlink}` reads an input asset file
///   and writes a transformed asset file to `output_dir`;
/// - `stop` publishes `sample.progress.stopped`;
/// - [`SampleMod::failing_start`] fails `start` with `MOD/INTERNAL_START_FAILED`.
///
/// Every observable effect goes through the [`Mod`] trait or the event bus,
/// so the same checks hold wherever the mod runs.
pub struct SampleMod {
    manifest: ModManifest,
    fail_start: bool,
    ctx: OnceLock<ModContext>,
}

impl SampleMod {
    pub fn new() -> Self {
        Self {
            manifest: manifest(),
            fail_start: false,
            ctx: OnceLock::new(),
        }
    }

    /// A sample mod whose `start` fails.
    pub fn failing_start() -> Self {
        Self {
            fail_start: true,
            ..Self::new()
        }
    }

    fn publish(&self, kind: &str, payload: Value) -> Result<(), Diagnostic> {
        let ctx = self
            .ctx
            .get()
            .ok_or_else(|| internal_error(ID, "NOT_STARTED", "sample mod was not started"))?;
        ctx.publish(kind, payload)
    }
}

impl Default for SampleMod {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Mod for SampleMod {
    fn manifest(&self) -> &ModManifest {
        &self.manifest
    }

    async fn start(&self, ctx: ModContext) -> Result<(), Diagnostic> {
        if self.fail_start {
            return Err(internal_error(
                ID,
                "START_FAILED",
                "sample mod refused to start",
            ));
        }
        self.ctx
            .set(ctx)
            .map_err(|_| internal_error(ID, "STARTED_TWICE", "sample mod started twice"))
    }

    async fn invoke(&self, invocation: Invocation) -> Result<Value, Diagnostic> {
        match invocation.operation.as_str() {
            "sample.echo" => {
                let text = invocation.input["text"].clone();
                self.publish("sample.echoed", json!({ "text": text }))?;
                Ok(json!({ "text": text }))
            }
            "sample.fail" => match invocation.input["kind"].as_str() {
                Some("caller") => Err(caller_error(ID, "SAMPLE", "the caller asked to fail")),
                _ => Err(internal_error(ID, "SAMPLE", "the sample mod failed")),
            },
            "sample.wait" => {
                self.publish("sample.progress.started", json!({ "stage": "started" }))?;
                invocation.cancel.cancelled().await;
                Err(cancelled(ID))
            }
            "sample.transform" => {
                let input_file_str = invocation.input["input_file"]
                    .as_str()
                    .ok_or_else(|| caller_error(ID, "INVALID_INPUT", "missing input_file"))?;
                if invocation
                    .input
                    .get("escape")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    return Ok(json!({ "output_file": "../escaped.txt" }));
                }
                if invocation
                    .input
                    .get("escape_second")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    let content = std::fs::read_to_string(input_file_str)
                        .map_err(|e| internal_error(ID, "IO", e.to_string()))?;
                    let transformed = format!("{content}\ntransformed");
                    let out_file = invocation.output_dir.join("transformed.txt");
                    std::fs::write(&out_file, transformed)
                        .map_err(|e| internal_error(ID, "IO", e.to_string()))?;
                    return Ok(json!({
                        "output_file": "transformed.txt",
                        "second_file": "../escaped.txt"
                    }));
                }
                if invocation
                    .input
                    .get("symlink")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    let symlink_path = invocation.output_dir.join("symlink.txt");
                    #[cfg(unix)]
                    std::os::unix::fs::symlink(Path::new(input_file_str), &symlink_path)
                        .map_err(|e| internal_error(ID, "IO", e.to_string()))?;
                    #[cfg(windows)]
                    std::os::windows::fs::symlink_file(Path::new(input_file_str), &symlink_path)
                        .map_err(|e| internal_error(ID, "IO", e.to_string()))?;
                    return Ok(json!({ "output_file": "symlink.txt" }));
                }
                let content = std::fs::read_to_string(input_file_str)
                    .map_err(|e| internal_error(ID, "IO", e.to_string()))?;
                let transformed = format!("{content}\ntransformed");
                let out_file = invocation.output_dir.join("transformed.txt");
                std::fs::write(&out_file, transformed)
                    .map_err(|e| internal_error(ID, "IO", e.to_string()))?;
                Ok(json!({ "output_file": "transformed.txt" }))
            }
            other => Err(caller_error(
                ID,
                "UNKNOWN_OPERATION",
                format!("sample mod has no operation `{other}`"),
            )),
        }
    }

    async fn stop(&self) -> Result<(), Diagnostic> {
        self.publish("sample.progress.stopped", json!({ "stage": "stopped" }))
    }
}

fn manifest() -> ModManifest {
    let object = |properties: Value, examples: Value| {
        let required: Vec<&String> = properties
            .as_object()
            .into_iter()
            .flatten()
            .map(|(k, _)| k)
            .collect();
        json!({
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false,
            "examples": examples,
        })
    };
    let text = json!({ "text": { "type": "string" } });
    let operation = |name: &str,
                     description: &str,
                     input: Value,
                     output: Value,
                     read_only: bool,
                     deterministic: bool,
                     open_world: bool| OperationSpec {
        name: name.to_owned(),
        description: description.to_owned(),
        input_schema: input,
        output_schema: output,
        annotations: OperationAnnotations {
            read_only,
            destructive: false,
            idempotent: read_only,
            deterministic,
            open_world,
        },
    };
    let stage = json!({
        "type": "object",
        "properties": { "stage": { "enum": ["started", "stopped"] } },
        "required": ["stage"],
    });
    ModManifest {
        id: ID.to_owned(),
        name: "Sample".to_owned(),
        version: "0.1.0".to_owned(),
        api: "0.1".to_owned(),
        operations: vec![
            operation(
                "sample.echo",
                "Return the text and publish sample.echoed.",
                object(text.clone(), json!([{ "text": "hello" }])),
                object(text.clone(), json!([{ "text": "hello" }])),
                true,
                true,
                false,
            ),
            operation(
                "sample.fail",
                "Fail as the caller's or the mod's own error.",
                object(
                    json!({ "kind": { "enum": ["caller", "internal"] } }),
                    json!([{ "kind": "caller" }, { "kind": "internal" }]),
                ),
                json!({ "type": "object" }),
                true,
                true,
                false,
            ),
            operation(
                "sample.wait",
                "Publish sample.progress.started and wait until cancelled.",
                object(json!({}), json!([{}])),
                json!({ "type": "object" }),
                false,
                false,
                false,
            ),
            operation(
                "sample.transform",
                "Read input_file and write transformed.txt to output_dir.",
                json!({
                    "type": "object",
                    "properties": {
                        "input_file": {
                            "type": "string",
                            "format": "asset",
                            "contentMediaType": "text/plain"
                        },
                        "escape": { "type": "boolean" },
                        "escape_second": { "type": "boolean" },
                        "symlink": { "type": "boolean" }
                    },
                    "required": ["input_file"],
                    "additionalProperties": false,
                    "examples": [{ "input_file": "sample-asset-1" }],
                }),
                json!({
                    "type": "object",
                    "properties": {
                        "output_file": {
                            "type": "string",
                            "format": "asset",
                            "contentMediaType": "text/plain"
                        },
                        "second_file": {
                            "type": "string",
                            "format": "asset",
                            "contentMediaType": "text/plain"
                        }
                    },
                    "required": ["output_file"],
                    "additionalProperties": false,
                    "examples": [{ "output_file": "transformed.txt" }],
                }),
                false,
                true,
                false,
            ),
        ],
        events: vec![
            ModEvent {
                kind: "sample.echoed".to_owned(),
                schema: object(text, json!([{ "text": "hello" }])),
            },
            ModEvent {
                kind: "sample.progress.*".to_owned(),
                schema: stage,
            },
        ],
    }
}
