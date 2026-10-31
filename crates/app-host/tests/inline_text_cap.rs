//! Host-side enforcement of the 256 KiB inline-text cap. The probe mod returns
//! inline text sized by `output_bytes` and always offers one output asset, so a
//! rejected output proves assets are not ingested.

use std::fs;
use std::sync::Arc;

use async_trait::async_trait;
use mod_sdk::{
    Invocation, Mod, ModContext, ModManifest, ModRegistry, OperationAnnotations, OperationSpec,
};
use reimagine_app_host::{
    InvokeRequest, MAX_INLINE_TEXT_BYTES, WorkspaceHost, WorkspaceHostBuilder, WorkspaceScope,
};
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandProvenance, OperationContext,
};
use reimagine_core::diagnostic::Diagnostic;
use reimagine_core::event::EventBus;
use reimagine_core::model::ProjectId;
use reimagine_core::project::ProjectMetadata;
use reimagine_core::step::{StepOutcome, StepRecord};
use reimagine_core::task::CancelToken;
use serde_json::{Value, json};

const OP_NAME: &str = "inline_probe.emit";

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-app-host-inline-{prefix}-{nonce}"))
}

fn test_context() -> OperationContext {
    OperationContext::new(
        CommandActor::new(CommandActorKind::Human).with_id("test-user"),
        CommandProvenance::Direct,
    )
}

struct InlineTextMod {
    manifest: ModManifest,
}

impl InlineTextMod {
    fn new() -> Self {
        Self {
            manifest: ModManifest {
                id: "inline_probe".to_owned(),
                name: "Inline probe".to_owned(),
                version: "0.1.0".to_owned(),
                api: "0.1".to_owned(),
                operations: vec![OperationSpec {
                    name: OP_NAME.to_owned(),
                    description: "Return inline text of output_bytes and one asset".to_owned(),
                    input_schema: json!({
                        "type": "object",
                        "properties": {
                            "text": { "type": "string", "format": "inline-text" },
                            "texts": {
                                "type": "array",
                                "items": { "type": "string", "format": "inline-text" }
                            },
                            "output_bytes": { "type": "integer", "minimum": 0 }
                        },
                        "additionalProperties": false
                    }),
                    output_schema: json!({
                        "type": "object",
                        "properties": {
                            "text": { "type": "string", "format": "inline-text" },
                            "output_file": {
                                "type": "string",
                                "format": "asset",
                                "contentMediaType": "text/plain"
                            }
                        },
                        "required": ["text", "output_file"],
                        "additionalProperties": false
                    }),
                    annotations: OperationAnnotations {
                        read_only: false,
                        destructive: false,
                        idempotent: false,
                        deterministic: true,
                        open_world: false,
                    },
                }],
                events: vec![],
            },
        }
    }
}

#[async_trait]
impl Mod for InlineTextMod {
    fn manifest(&self) -> &ModManifest {
        &self.manifest
    }

    async fn start(&self, _ctx: ModContext) -> Result<(), Diagnostic> {
        Ok(())
    }

    async fn invoke(&self, invocation: Invocation) -> Result<Value, Diagnostic> {
        let output_bytes = invocation.input["output_bytes"].as_u64().unwrap_or(0) as usize;
        std::fs::write(invocation.output_dir.join("emitted.txt"), "emitted")
            .expect("write emitted asset");
        Ok(json!({
            "text": "x".repeat(output_bytes),
            "output_file": "emitted.txt"
        }))
    }

    async fn stop(&self) -> Result<(), Diagnostic> {
        Ok(())
    }
}

async fn build_host(base: &std::path::Path) -> (WorkspaceHost, ProjectId, std::path::PathBuf) {
    let event_bus = EventBus::new();
    let mod_registry =
        Arc::new(ModRegistry::start(vec![Arc::new(InlineTextMod::new())], &event_bus).await);
    let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test-inline-cap"), base)
        .event_bus(event_bus)
        .mod_registry(mod_registry)
        .build()
        .await
        .expect("build host");

    let project_id = ProjectId::new("proj-inline");
    host.project_service()
        .create_project(
            project_id.clone(),
            ProjectMetadata::new(
                "P",
                "desc",
                "2026-01-01T00:00:00Z".into(),
                "2026-01-01T00:00:00Z".into(),
            ),
        )
        .await
        .expect("create project");

    let project_dir = host.config().paths().project_dir(project_id.as_str());
    (host, project_id, project_dir)
}

fn invoke_request(input: Value) -> InvokeRequest {
    InvokeRequest {
        name: OP_NAME.to_owned(),
        input,
        context: test_context(),
        cancel: CancelToken::new(),
        run: None,
    }
}

fn line_count(path: &std::path::Path) -> usize {
    match fs::read_to_string(path) {
        Ok(content) => content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .count(),
        Err(_) => 0,
    }
}

fn step_line_count(project_dir: &std::path::Path) -> usize {
    line_count(&project_dir.join("steps.jsonl"))
}

fn asset_line_count(project_dir: &std::path::Path) -> usize {
    line_count(&project_dir.join("assets.jsonl"))
}

#[tokio::test]
async fn inline_text_input_at_cap_is_accepted_and_over_cap_writes_no_step_record() {
    let base = temp_dir("input-cap");
    let (host, project_id, project_dir) = build_host(&base).await;

    let accepted = host
        .operation_service()
        .invoke(
            &project_id,
            invoke_request(json!({
                "text": "a".repeat(MAX_INLINE_TEXT_BYTES),
                "output_bytes": 0
            })),
        )
        .await
        .expect("inline text at the cap is accepted");
    assert_eq!(accepted.output["text"], "");
    let recorded_steps = step_line_count(&project_dir);
    assert_eq!(
        recorded_steps, 2,
        "an accepted call records start and finish"
    );

    let err = host
        .operation_service()
        .invoke(
            &project_id,
            invoke_request(json!({
                "text": "a".repeat(MAX_INLINE_TEXT_BYTES + 1),
                "output_bytes": 0
            })),
        )
        .await
        .expect_err("inline text over the cap must be rejected");

    assert_eq!(
        err.code().as_str(),
        "INLINE_PROBE/CALLER_INLINE_TEXT_TOO_LARGE"
    );
    assert!(
        err.message().contains(&MAX_INLINE_TEXT_BYTES.to_string()),
        "diagnostic must name the limit: {}",
        err.message()
    );
    assert_eq!(
        step_line_count(&project_dir),
        recorded_steps,
        "an over-cap call must not write a step record"
    );

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn inline_text_input_budget_is_summed_across_fields_and_array_items() {
    let base = temp_dir("input-sum");
    let (host, project_id, project_dir) = build_host(&base).await;
    let half = MAX_INLINE_TEXT_BYTES / 2;

    host.operation_service()
        .invoke(
            &project_id,
            invoke_request(json!({
                "text": "a".repeat(half),
                "texts": ["b".repeat(MAX_INLINE_TEXT_BYTES - half)],
                "output_bytes": 0
            })),
        )
        .await
        .expect("fields summing to the cap are accepted");
    let recorded_steps = step_line_count(&project_dir);

    let err = host
        .operation_service()
        .invoke(
            &project_id,
            invoke_request(json!({
                "text": "a".repeat(half),
                "texts": ["b".repeat(MAX_INLINE_TEXT_BYTES - half + 1)],
                "output_bytes": 0
            })),
        )
        .await
        .expect_err("fields summing over the cap must be rejected");
    assert_eq!(
        err.code().as_str(),
        "INLINE_PROBE/CALLER_INLINE_TEXT_TOO_LARGE"
    );
    assert_eq!(
        step_line_count(&project_dir),
        recorded_steps,
        "an over-cap sum must not write a step record"
    );

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn preview_rejects_inline_text_over_the_cap() {
    let base = temp_dir("preview-cap");
    let (host, project_id, project_dir) = build_host(&base).await;

    let preview = host
        .operation_service()
        .preview(
            &project_id,
            OP_NAME,
            json!({ "text": "a".repeat(MAX_INLINE_TEXT_BYTES) }),
            "at-cap",
            CancelToken::new(),
        )
        .await
        .expect("inline text at the cap previews");
    assert!(
        preview["output_file"]
            .as_str()
            .expect("preview uri")
            .starts_with("reimagine-asset://preview/at-cap/")
    );

    let err = host
        .operation_service()
        .preview(
            &project_id,
            OP_NAME,
            json!({ "text": "a".repeat(MAX_INLINE_TEXT_BYTES + 1) }),
            "over-cap",
            CancelToken::new(),
        )
        .await
        .expect_err("inline text over the cap must not preview");
    assert_eq!(
        err.code().as_str(),
        "INLINE_PROBE/CALLER_INLINE_TEXT_TOO_LARGE"
    );
    assert_eq!(
        step_line_count(&project_dir),
        0,
        "previews never write step records"
    );

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn inline_output_over_the_cap_is_invalid_output_and_ingests_no_asset() {
    let base = temp_dir("output-cap");
    let (host, project_id, project_dir) = build_host(&base).await;

    let err = host
        .operation_service()
        .invoke(
            &project_id,
            invoke_request(json!({
                "text": "ok",
                "output_bytes": MAX_INLINE_TEXT_BYTES + 1
            })),
        )
        .await
        .expect_err("inline output over the cap must be rejected");
    assert_eq!(err.code().as_str(), "INLINE_PROBE/INTERNAL_INVALID_OUTPUT");
    assert!(
        err.message().contains("inline text output"),
        "diagnostic must describe the inline output: {}",
        err.message()
    );
    assert_eq!(
        asset_line_count(&project_dir),
        0,
        "an over-cap output must not ingest its asset"
    );

    let steps = fs::read_to_string(project_dir.join("steps.jsonl")).expect("steps.jsonl");
    let last: StepRecord = serde_json::from_str(steps.lines().last().expect("step record"))
        .expect("parse step record");
    assert!(
        matches!(last.outcome, StepOutcome::Failed { .. }),
        "the started step must finish as failed"
    );

    let accepted = host
        .operation_service()
        .invoke(
            &project_id,
            invoke_request(json!({
                "text": "ok",
                "output_bytes": MAX_INLINE_TEXT_BYTES
            })),
        )
        .await
        .expect("inline output at the cap is accepted");
    assert_eq!(
        accepted.output["text"].as_str().expect("text").len(),
        MAX_INLINE_TEXT_BYTES
    );
    assert_eq!(
        asset_line_count(&project_dir),
        1,
        "an accepted output ingests its asset"
    );

    let _ = fs::remove_dir_all(&base);
}
