use std::sync::OnceLock;

use async_trait::async_trait;
use reimagine_core::diagnostic::Diagnostic;
use reimagine_core::task::CancelToken;
use serde_json::{Value, json};

use crate::context::ModContext;
use crate::diagnostic::{caller_error, cancelled, internal_error};
use crate::interface::Mod;
use crate::manifest::{ModEvent, ModManifest, ModOperation, OperationAnnotations};

const ID: &str = "sample";

/// A reference mod, id `sample`, that exercises every interface behavior:
///
/// - `sample.echo {text}` returns `{text}` and publishes `sample.echoed`;
/// - `sample.fail {kind}` fails as the caller's (`"caller"`) or its own
///   (`"internal"`) error;
/// - `sample.wait {}` publishes `sample.progress.started`, then waits for
///   cancellation and returns `MOD/CALLER_CANCELLED`;
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

    async fn invoke(
        &self,
        operation: &str,
        payload: Value,
        cancel: CancelToken,
    ) -> Result<Value, Diagnostic> {
        match operation {
            "sample.echo" => {
                let text = payload["text"].clone();
                self.publish("sample.echoed", json!({ "text": text }))?;
                Ok(json!({ "text": text }))
            }
            "sample.fail" => match payload["kind"].as_str() {
                Some("caller") => Err(caller_error(ID, "SAMPLE", "the caller asked to fail")),
                _ => Err(internal_error(ID, "SAMPLE", "the sample mod failed")),
            },
            "sample.wait" => {
                self.publish("sample.progress.started", json!({ "stage": "started" }))?;
                cancel.cancelled().await;
                Err(cancelled(ID))
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
    let operation =
        |name: &str, description: &str, input: Value, output: Value, read_only| ModOperation {
            name: name.to_owned(),
            description: description.to_owned(),
            input_schema: input,
            output_schema: output,
            annotations: OperationAnnotations {
                read_only,
                destructive: false,
                idempotent: read_only,
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
            ),
            operation(
                "sample.wait",
                "Publish sample.progress.started and wait until cancelled.",
                object(json!({}), json!([{}])),
                json!({ "type": "object" }),
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
