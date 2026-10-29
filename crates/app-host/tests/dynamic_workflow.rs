//! End-to-end coverage for dynamic workflow documents and batch runs.
//!
//! The operations these workflows call are registered by the test itself, so
//! the suite does not depend on a sample mod or a worker process.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use mod_sdk::{OperationAnnotations, OperationSpec};
use reimagine_app_host::{
    AssetStore, InvokeRequest, KernelCall, KernelOp, StepLog, WorkspaceHost, WorkspaceHostBuilder,
    WorkspaceScope,
};
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandProvenance, OperationContext,
};
use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::dynamic_workflow::{RunDocument, RunState, SlotState};
use reimagine_core::event::Event;
use reimagine_core::model::{AssetId, CollectionId, DiagnosticId, ProjectId, RunId};
use reimagine_core::project::ProjectMetadata;
use reimagine_core::task::CancelToken;
use serde_json::{Value, json};
use tokio::sync::Notify;

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-dynamic-workflow-{prefix}-{nonce}"))
}

fn test_context() -> OperationContext {
    OperationContext::new(
        CommandActor::new(CommandActorKind::Human).with_id("test-user"),
        CommandProvenance::Direct,
    )
}

fn test_error(code: &str, message: impl Into<String>) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new(format!("test-op:{code}")),
        DiagnosticCode::new(format!("TEST/CALLER_{code}")),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("test_ops"),
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new("test")),
    )
}

fn annotations(
    read_only: bool,
    idempotent: bool,
    deterministic: bool,
    open_world: bool,
) -> OperationAnnotations {
    OperationAnnotations {
        read_only,
        destructive: false,
        idempotent,
        deterministic,
        open_world,
    }
}

fn asset_input_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "doc": { "type": "string", "format": "asset" }
        },
        "required": ["doc"],
        "additionalProperties": false
    })
}

fn asset_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "doc": { "type": "string", "format": "asset" }
        },
        "required": ["doc"],
        "additionalProperties": false
    })
}

#[derive(Default)]
struct Counters {
    copy: AtomicUsize,
    inspect: AtomicUsize,
    wait: AtomicUsize,
    pause: AtomicUsize,
    gate: AtomicUsize,
    open_world: AtomicUsize,
    render: AtomicUsize,
}

/// Copies the input asset; rejects content containing `bad`.
struct CopyOp {
    spec: OperationSpec,
    counters: Arc<Counters>,
}

impl CopyOp {
    fn new(counters: Arc<Counters>) -> Self {
        Self {
            spec: OperationSpec {
                name: "test.copy".to_owned(),
                description: "Copy an asset".to_owned(),
                input_schema: asset_input_schema(),
                output_schema: asset_output_schema(),
                annotations: annotations(false, true, true, false),
            },
            counters,
        }
    }
}

#[async_trait]
impl KernelOp for CopyOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        self.counters.copy.fetch_add(1, Ordering::SeqCst);
        let input = call
            .invocation
            .input
            .get("doc")
            .and_then(Value::as_str)
            .ok_or_else(|| test_error("INVALID_INPUT", "missing doc"))?;
        let bytes = tokio::fs::read(input)
            .await
            .map_err(|error| test_error("IO", error.to_string()))?;
        if String::from_utf8_lossy(&bytes).contains("bad") {
            return Err(test_error("BAD_INPUT", "asset content is rejected"));
        }
        tokio::fs::write(call.invocation.output_dir.join("out.txt"), bytes)
            .await
            .map_err(|error| test_error("IO", error.to_string()))?;
        Ok(json!({ "doc": "out.txt" }))
    }
}

/// Read-only op: produces no asset, used to check that run steps are logged.
struct InspectOp {
    spec: OperationSpec,
    counters: Arc<Counters>,
}

impl InspectOp {
    fn new(counters: Arc<Counters>) -> Self {
        Self {
            spec: OperationSpec {
                name: "test.inspect".to_owned(),
                description: "Inspect an asset without producing one".to_owned(),
                input_schema: asset_input_schema(),
                output_schema: json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
                annotations: annotations(true, true, true, false),
            },
            counters,
        }
    }
}

#[async_trait]
impl KernelOp for InspectOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    async fn invoke(&self, _call: KernelCall) -> Result<Value, Diagnostic> {
        self.counters.inspect.fetch_add(1, Ordering::SeqCst);
        Ok(json!({}))
    }
}

/// Waits until the run is cancelled.
struct WaitOp {
    spec: OperationSpec,
    counters: Arc<Counters>,
}

impl WaitOp {
    fn new(counters: Arc<Counters>) -> Self {
        Self {
            spec: OperationSpec {
                name: "test.wait".to_owned(),
                description: "Wait for cancellation".to_owned(),
                input_schema: asset_input_schema(),
                output_schema: asset_output_schema(),
                annotations: annotations(false, false, false, false),
            },
            counters,
        }
    }
}

#[async_trait]
impl KernelOp for WaitOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        self.counters.wait.fetch_add(1, Ordering::SeqCst);
        call.invocation.cancel.cancelled().await;
        Err(test_error("CANCELLED", "wait was cancelled"))
    }
}

/// Waits for the test to release it, then copies; used to hold a task inside
/// a shutdown window.
struct GateOp {
    spec: OperationSpec,
    counters: Arc<Counters>,
    started: Arc<Notify>,
    release: Arc<Notify>,
}

impl GateOp {
    fn new(counters: Arc<Counters>, started: Arc<Notify>, release: Arc<Notify>) -> Self {
        Self {
            spec: OperationSpec {
                name: "test.gate".to_owned(),
                description: "Wait for the test gate, then copy".to_owned(),
                input_schema: asset_input_schema(),
                output_schema: asset_output_schema(),
                annotations: annotations(false, true, true, false),
            },
            counters,
            started,
            release,
        }
    }
}

#[async_trait]
impl KernelOp for GateOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        self.counters.gate.fetch_add(1, Ordering::SeqCst);
        self.started.notify_one();
        self.release.notified().await;
        let input = call
            .invocation
            .input
            .get("doc")
            .and_then(Value::as_str)
            .ok_or_else(|| test_error("INVALID_INPUT", "missing doc"))?;
        let bytes = tokio::fs::read(input)
            .await
            .map_err(|error| test_error("IO", error.to_string()))?;
        tokio::fs::write(call.invocation.output_dir.join("out.txt"), bytes)
            .await
            .map_err(|error| test_error("IO", error.to_string()))?;
        Ok(json!({ "doc": "out.txt" }))
    }
}

/// Pauses on its first invocation, then copies.
struct PauseOp {
    spec: OperationSpec,
    counters: Arc<Counters>,
    started: Arc<Notify>,
}

impl PauseOp {
    fn new(counters: Arc<Counters>, started: Arc<Notify>) -> Self {
        Self {
            spec: OperationSpec {
                name: "test.pause".to_owned(),
                description: "Pause once, then copy".to_owned(),
                input_schema: asset_input_schema(),
                output_schema: asset_output_schema(),
                annotations: annotations(false, false, false, false),
            },
            counters,
            started,
        }
    }
}

#[async_trait]
impl KernelOp for PauseOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        let call_number = self.counters.pause.fetch_add(1, Ordering::SeqCst) + 1;
        if call_number == 1 {
            self.started.notify_one();
            std::future::pending::<()>().await;
        }
        let input = call
            .invocation
            .input
            .get("doc")
            .and_then(Value::as_str)
            .ok_or_else(|| test_error("INVALID_INPUT", "missing doc"))?;
        let bytes = tokio::fs::read(input)
            .await
            .map_err(|error| test_error("IO", error.to_string()))?;
        tokio::fs::write(call.invocation.output_dir.join("out.txt"), bytes)
            .await
            .map_err(|error| test_error("IO", error.to_string()))?;
        Ok(json!({ "doc": "out.txt" }))
    }
}

/// Pauses on its first invocation, then renders its text; used by group runs.
struct PauseRenderOp {
    spec: OperationSpec,
    counters: Arc<Counters>,
    started: Arc<Notify>,
}

impl PauseRenderOp {
    fn new(counters: Arc<Counters>, started: Arc<Notify>) -> Self {
        Self {
            spec: OperationSpec {
                name: "test.pause_render".to_owned(),
                description: "Pause once, then render".to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": { "text": { "type": "string" } },
                    "required": ["text"],
                    "additionalProperties": false
                }),
                output_schema: asset_output_schema(),
                annotations: annotations(false, false, false, false),
            },
            counters,
            started,
        }
    }
}

#[async_trait]
impl KernelOp for PauseRenderOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        let call_number = self.counters.pause.fetch_add(1, Ordering::SeqCst) + 1;
        if call_number == 1 {
            self.started.notify_one();
            std::future::pending::<()>().await;
        }
        let text = call
            .invocation
            .input
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| test_error("INVALID_INPUT", "missing text"))?;
        tokio::fs::write(call.invocation.output_dir.join("out.txt"), text)
            .await
            .map_err(|error| test_error("IO", error.to_string()))?;
        Ok(json!({ "doc": "out.txt" }))
    }
}

/// Pauses on its first invocation and is declared `open_world` + non-idempotent.
struct OpenWorldOp {
    spec: OperationSpec,
    counters: Arc<Counters>,
    started: Arc<Notify>,
}

impl OpenWorldOp {
    fn new(counters: Arc<Counters>, started: Arc<Notify>) -> Self {
        Self {
            spec: OperationSpec {
                name: "test.open_world".to_owned(),
                description: "Open-world step that pauses once".to_owned(),
                input_schema: asset_input_schema(),
                output_schema: asset_output_schema(),
                annotations: annotations(false, false, false, true),
            },
            counters,
            started,
        }
    }
}

#[async_trait]
impl KernelOp for OpenWorldOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        let call_number = self.counters.open_world.fetch_add(1, Ordering::SeqCst) + 1;
        if call_number == 1 {
            self.started.notify_one();
            std::future::pending::<()>().await;
        }
        let input = call
            .invocation
            .input
            .get("doc")
            .and_then(Value::as_str)
            .ok_or_else(|| test_error("INVALID_INPUT", "missing doc"))?;
        let bytes = tokio::fs::read(input)
            .await
            .map_err(|error| test_error("IO", error.to_string()))?;
        tokio::fs::write(call.invocation.output_dir.join("out.txt"), bytes)
            .await
            .map_err(|error| test_error("IO", error.to_string()))?;
        Ok(json!({ "doc": "out.txt" }))
    }
}

/// Renders a parameter group into an asset.
struct RenderOp {
    spec: OperationSpec,
    counters: Arc<Counters>,
}

impl RenderOp {
    fn new(counters: Arc<Counters>) -> Self {
        Self {
            spec: OperationSpec {
                name: "test.render".to_owned(),
                description: "Render text into an asset".to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": { "text": { "type": "string" } },
                    "required": ["text"],
                    "additionalProperties": false
                }),
                output_schema: asset_output_schema(),
                annotations: annotations(false, true, true, false),
            },
            counters,
        }
    }
}

#[async_trait]
impl KernelOp for RenderOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        self.counters.render.fetch_add(1, Ordering::SeqCst);
        let text = call
            .invocation
            .input
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| test_error("INVALID_INPUT", "missing text"))?;
        tokio::fs::write(call.invocation.output_dir.join("out.txt"), text)
            .await
            .map_err(|error| test_error("IO", error.to_string()))?;
        Ok(json!({ "doc": "out.txt" }))
    }
}

struct TestOps {
    counters: Arc<Counters>,
    pause_started: Arc<Notify>,
    gate_started: Arc<Notify>,
    gate_release: Arc<Notify>,
    open_world_started: Arc<Notify>,
}

impl TestOps {
    fn new() -> Self {
        Self {
            counters: Arc::new(Counters::default()),
            pause_started: Arc::new(Notify::new()),
            gate_started: Arc::new(Notify::new()),
            gate_release: Arc::new(Notify::new()),
            open_world_started: Arc::new(Notify::new()),
        }
    }

    fn register(&self, host: &WorkspaceHost) {
        let service = host.operation_service();
        service.register_kernel_op(Arc::new(CopyOp::new(Arc::clone(&self.counters))));
        service.register_kernel_op(Arc::new(InspectOp::new(Arc::clone(&self.counters))));
        service.register_kernel_op(Arc::new(WaitOp::new(Arc::clone(&self.counters))));
        service.register_kernel_op(Arc::new(GateOp::new(
            Arc::clone(&self.counters),
            Arc::clone(&self.gate_started),
            Arc::clone(&self.gate_release),
        )));
        service.register_kernel_op(Arc::new(PauseOp::new(
            Arc::clone(&self.counters),
            Arc::clone(&self.pause_started),
        )));
        service.register_kernel_op(Arc::new(PauseRenderOp::new(
            Arc::clone(&self.counters),
            Arc::clone(&self.pause_started),
        )));
        service.register_kernel_op(Arc::new(OpenWorldOp::new(
            Arc::clone(&self.counters),
            Arc::clone(&self.open_world_started),
        )));
        service.register_kernel_op(Arc::new(RenderOp::new(Arc::clone(&self.counters))));
    }
}

async fn build_host(base: &Path) -> WorkspaceHost {
    WorkspaceHostBuilder::new(WorkspaceScope::new("test-dynamic-workflow"), base)
        .build()
        .await
        .expect("build host")
}

/// Simulate a restart: wait for the aborted tasks to exit, then build a host
/// over the same directory and register the test operations again.
async fn reopen_host(base: &Path, ops: &TestOps) -> WorkspaceHost {
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    let host = build_host(base).await;
    ops.register(&host);
    host
}

async fn create_project(host: &WorkspaceHost, id: &str) -> ProjectId {
    let project_id = ProjectId::new(id);
    host.project_service()
        .create_project(
            project_id.clone(),
            ProjectMetadata::new(
                "Dynamic workflow test",
                "test project",
                "2026-01-01T00:00:00Z".into(),
                "2026-01-01T00:00:00Z".into(),
            ),
        )
        .await
        .expect("create project");
    project_id
}

async fn invoke(host: &WorkspaceHost, project: &ProjectId, name: &str, input: Value) -> Value {
    host.operation_service()
        .invoke(
            project,
            InvokeRequest {
                name: name.to_owned(),
                input,
                context: test_context(),
                cancel: CancelToken::new(),
                run: None,
            },
        )
        .await
        .unwrap_or_else(|error| panic!("{name} failed: {}", error.message()))
        .output
}

async fn invoke_as(
    host: &WorkspaceHost,
    project: &ProjectId,
    name: &str,
    input: Value,
    context: OperationContext,
) -> Value {
    host.operation_service()
        .invoke(
            project,
            InvokeRequest {
                name: name.to_owned(),
                input,
                context,
                cancel: CancelToken::new(),
                run: None,
            },
        )
        .await
        .unwrap_or_else(|error| panic!("{name} failed: {}", error.message()))
        .output
}

async fn try_invoke(
    host: &WorkspaceHost,
    project: &ProjectId,
    name: &str,
    input: Value,
) -> Result<Value, Diagnostic> {
    host.operation_service()
        .invoke(
            project,
            InvokeRequest {
                name: name.to_owned(),
                input,
                context: test_context(),
                cancel: CancelToken::new(),
                run: None,
            },
        )
        .await
        .map(|result| result.output)
}

async fn import_assets(
    host: &WorkspaceHost,
    project: &ProjectId,
    base: &Path,
    files: &[(&str, &str)],
) -> Vec<String> {
    let mut paths = Vec::new();
    for (index, (name, content)) in files.iter().enumerate() {
        let path = base.join(format!("import-{index}-{name}"));
        tokio::fs::write(&path, content)
            .await
            .expect("write import");
        paths.push(path.to_string_lossy().to_string());
    }
    let output = invoke(host, project, "asset.import", json!({ "paths": paths })).await;
    output["assets"]
        .as_array()
        .expect("assets")
        .iter()
        .map(|value| value.as_str().expect("asset id").to_owned())
        .collect()
}

async fn create_collection(
    host: &WorkspaceHost,
    project: &ProjectId,
    assets: &[String],
) -> CollectionId {
    let output = invoke(
        host,
        project,
        "collection.create",
        json!({ "name": "Batch", "assets": assets }),
    )
    .await;
    CollectionId::new(output["collection_id"].as_str().expect("collection id"))
}

async fn create_workflow(host: &WorkspaceHost, project: &ProjectId, workflow: Value) -> String {
    let output = invoke(
        host,
        project,
        "dynamic_workflow.create",
        json!({ "name": "Test workflow", "workflow": workflow }),
    )
    .await;
    output["dynamic_workflow_id"]
        .as_str()
        .expect("dynamic workflow id")
        .to_owned()
}

async fn start_run(
    host: &WorkspaceHost,
    project: &ProjectId,
    workflow_id: &str,
    target: Value,
    params: Option<Value>,
) -> RunId {
    start_run_as(host, project, workflow_id, target, params, test_context()).await
}

async fn start_run_as(
    host: &WorkspaceHost,
    project: &ProjectId,
    workflow_id: &str,
    target: Value,
    params: Option<Value>,
    context: OperationContext,
) -> RunId {
    let mut input = json!({ "dynamic_workflow_id": workflow_id, "target": target });
    if let Some(params) = params {
        input["params"] = params;
    }
    let output = invoke_as(host, project, "dynamic_workflow.run", input, context).await;
    RunId::new(output["run_id"].as_str().expect("run id"))
}

async fn get_run(host: &WorkspaceHost, project: &ProjectId, run_id: &RunId) -> RunDocument {
    let output = invoke(
        host,
        project,
        "dynamic_workflow.get_run",
        json!({ "run_id": run_id.as_str() }),
    )
    .await;
    serde_json::from_value(output).expect("run document")
}

async fn wait_for_run(
    host: &WorkspaceHost,
    project: &ProjectId,
    run_id: &RunId,
    done: impl Fn(&RunDocument) -> bool,
) -> RunDocument {
    for _ in 0..500 {
        let run = get_run(host, project, run_id).await;
        if done(&run) {
            return run;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("run `{run_id}` did not settle");
}

fn copy_workflow(extra_steps: Vec<Value>, output: Value) -> Value {
    let mut steps = vec![json!({
        "id": "check",
        "op": "test.inspect",
        "input": { "doc": { "$from": "inputs.doc" } }
    })];
    steps.extend(extra_steps);
    json!({
        "inputs": { "doc": { "type": "string", "format": "asset" } },
        "params": { "type": "object" },
        "steps": steps,
        "output": output
    })
}

fn group_workflow(op: &str) -> Value {
    json!({
        "inputs": {},
        "params": {
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"]
        },
        "steps": [{
            "id": "render",
            "op": op,
            "input": { "text": { "$from": "params.text" } }
        }],
        "output": { "$from": "steps.render.doc" }
    })
}

async fn slot_contents(
    host: &WorkspaceHost,
    project: &ProjectId,
    collection: &CollectionId,
) -> Vec<String> {
    let output = invoke(
        host,
        project,
        "collection.get",
        json!({ "collection_id": collection.as_str() }),
    )
    .await;
    let store = AssetStore::new(host.config().paths().project_dir(project.as_str()));
    let mut contents = Vec::new();
    for slot in output["slots"].as_array().expect("slots") {
        let asset = AssetId::new(slot["asset_id"].as_str().expect("asset id"));
        let record = store
            .get_asset(&asset)
            .await
            .expect("asset lookup")
            .expect("asset exists");
        let bytes = tokio::fs::read(store.content_path(&record.hash))
            .await
            .expect("asset bytes");
        contents.push(String::from_utf8(bytes).expect("utf8 asset"));
    }
    contents
}

async fn collection_file_count(host: &WorkspaceHost, project: &ProjectId) -> usize {
    let dir = host
        .config()
        .paths()
        .project_dir(project.as_str())
        .join("collections");
    let mut entries = match tokio::fs::read_dir(dir).await {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return 0,
        Err(error) => panic!("read collections: {error}"),
    };
    let mut count = 0;
    while let Some(entry) = entries.next_entry().await.expect("collection entry") {
        if entry.path().extension().and_then(|ext| ext.to_str()) == Some("json") {
            count += 1;
        }
    }
    count
}

#[tokio::test]
async fn multi_slot_run_emits_slot_events_and_logs_read_only_steps() {
    let base = temp_dir("multi-slot");
    let ops = TestOps::new();
    let host = build_host(&base).await;
    let project = create_project(&host, "proj").await;
    let assets = import_assets(
        &host,
        &project,
        &base,
        &[("a.txt", "alpha"), ("b.txt", "beta"), ("c.txt", "gamma")],
    )
    .await;
    let collection = create_collection(&host, &project, &assets).await;
    let workflow_id = create_workflow(
        &host,
        &project,
        copy_workflow(
            vec![json!({
                "id": "copy",
                "op": "test.copy",
                "input": { "doc": { "$from": "inputs.doc" } }
            })],
            json!({ "$from": "steps.copy.doc" }),
        ),
    )
    .await;
    let stored = invoke(
        &host,
        &project,
        "dynamic_workflow.get",
        json!({ "dynamic_workflow_id": workflow_id }),
    )
    .await;
    assert_eq!(stored["id"], workflow_id.as_str());
    assert_eq!(stored["name"], "Test workflow");
    assert_eq!(stored["version"], 1);
    assert_eq!(stored["workflow"]["steps"][0]["id"], "check");
    ops.register(&host);

    let mut events = host.event_bus().subscribe();
    let run_id = start_run(
        &host,
        &project,
        &workflow_id,
        json!({ "collection_id": collection.as_str() }),
        None,
    )
    .await;

    let run = wait_for_run(&host, &project, &run_id, |run| {
        !matches!(run.state, RunState::Queued | RunState::Running)
    })
    .await;
    assert_eq!(run.state, RunState::Completed);
    assert_eq!(run.items.len(), 3);
    for item in &run.items {
        assert_eq!(item.state, SlotState::Succeeded);
        assert!(item.slot_id.is_some());
    }

    let mut slot_events = 0;
    let mut run_events = 0;
    while let Ok(event) = events.try_recv() {
        if let Event::Document {
            kind,
            document_id,
            version,
            ..
        } = event
            && document_id == run_id.as_str()
        {
            match kind.as_str() {
                "dynamic_workflow.slot_changed" => {
                    assert!(version >= 2);
                    slot_events += 1;
                }
                "dynamic_workflow.run_changed" => run_events += 1,
                _ => {}
            }
        }
    }
    assert!(
        slot_events >= run.items.len(),
        "expected at least one slot event per item, got {slot_events}"
    );
    assert!(run_events >= 2, "running and completed are announced");

    // The collection now points at run outputs.
    let output = invoke(
        &host,
        &project,
        "collection.get",
        json!({ "collection_id": collection.as_str() }),
    )
    .await;
    let slots = output["slots"].as_array().expect("slots");
    assert_eq!(slots.len(), 3);
    for slot in slots {
        assert!(!assets.contains(&slot["asset_id"].as_str().unwrap().to_owned()));
    }

    // Read-only steps still get a merged record with the run context.
    let step_log = StepLog::new(host.config().paths().project_dir(project.as_str()));
    let records = step_log.records().await.expect("step records");
    let run_records: Vec<_> = records
        .iter()
        .filter(|record| record.run.as_ref().is_some_and(|run| run.run_id == run_id))
        .collect();
    assert_eq!(
        run_records
            .iter()
            .filter(|record| record.op == "test.copy")
            .count(),
        3
    );
    assert_eq!(
        run_records
            .iter()
            .filter(|record| record.op == "test.inspect")
            .count(),
        3
    );
    for record in run_records {
        let run = record.run.as_ref().expect("run context");
        assert!(matches!(run.step_id.as_str(), "check" | "copy"));
        assert!(matches!(
            record.outcome,
            reimagine_core::step::StepOutcome::Succeeded { .. }
        ));
    }

    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn one_failed_slot_does_not_stop_the_others() {
    let base = temp_dir("one-failure");
    let ops = TestOps::new();
    let host = build_host(&base).await;
    let project = create_project(&host, "proj").await;
    let assets = import_assets(
        &host,
        &project,
        &base,
        &[("good.txt", "good content"), ("bad.txt", "bad content")],
    )
    .await;
    let collection = create_collection(&host, &project, &assets).await;
    let workflow_id = create_workflow(
        &host,
        &project,
        copy_workflow(
            vec![json!({
                "id": "copy",
                "op": "test.copy",
                "input": { "doc": { "$from": "inputs.doc" } }
            })],
            json!({ "$from": "steps.copy.doc" }),
        ),
    )
    .await;
    ops.register(&host);

    let run_id = start_run(
        &host,
        &project,
        &workflow_id,
        json!({ "collection_id": collection.as_str() }),
        None,
    )
    .await;
    let run = wait_for_run(&host, &project, &run_id, |run| {
        !matches!(run.state, RunState::Queued | RunState::Running)
    })
    .await;

    assert_eq!(run.state, RunState::Failed);
    let states: Vec<SlotState> = run.items.iter().map(|item| item.state).collect();
    assert_eq!(
        states
            .iter()
            .filter(|state| **state == SlotState::Succeeded)
            .count(),
        1
    );
    let failed = run
        .items
        .iter()
        .find(|item| item.state == SlotState::Failed)
        .expect("one failed item");
    assert_eq!(
        failed
            .diagnostic
            .as_ref()
            .expect("diagnostic")
            .code()
            .as_str(),
        "TEST/CALLER_BAD_INPUT"
    );

    let output = invoke(
        &host,
        &project,
        "collection.get",
        json!({ "collection_id": collection.as_str() }),
    )
    .await;
    let slots = output["slots"].as_array().expect("slots");
    let unchanged = slots
        .iter()
        .filter(|slot| assets.contains(&slot["asset_id"].as_str().unwrap().to_owned()))
        .count();
    assert_eq!(unchanged, 1, "the failed slot keeps its input asset");

    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn cancel_stops_the_run_and_skips_later_steps() {
    let base = temp_dir("cancel");
    let ops = TestOps::new();
    let host = build_host(&base).await;
    let project = create_project(&host, "proj").await;
    let assets = import_assets(&host, &project, &base, &[("a.txt", "alpha")]).await;
    let collection = create_collection(&host, &project, &assets).await;
    let workflow = json!({
        "inputs": { "doc": { "type": "string", "format": "asset" } },
        "params": { "type": "object" },
        "steps": [
            {
                "id": "wait",
                "op": "test.wait",
                "input": { "doc": { "$from": "inputs.doc" } }
            },
            {
                "id": "copy",
                "op": "test.copy",
                "input": { "doc": { "$from": "steps.wait.doc" } }
            }
        ],
        "output": { "$from": "steps.copy.doc" }
    });
    let workflow_id = create_workflow(&host, &project, workflow).await;
    ops.register(&host);

    let run_id = start_run(
        &host,
        &project,
        &workflow_id,
        json!({ "collection_id": collection.as_str() }),
        None,
    )
    .await;

    for _ in 0..500 {
        if ops.counters.wait.load(Ordering::SeqCst) == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(ops.counters.wait.load(Ordering::SeqCst), 1);

    let output = invoke(
        &host,
        &project,
        "dynamic_workflow.cancel",
        json!({ "run_id": run_id.as_str() }),
    )
    .await;
    assert_eq!(output["state"], "cancelled");

    let run = get_run(&host, &project, &run_id).await;
    assert_eq!(run.state, RunState::Cancelled);
    assert_eq!(run.items[0].state, SlotState::Cancelled);
    assert_eq!(
        ops.counters.copy.load(Ordering::SeqCst),
        0,
        "the step after the cancelled one must not run"
    );

    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn shutdown_during_run_registration_leaves_no_step_running() {
    let base = temp_dir("shutdown-race");
    let ops = TestOps::new();
    let host = build_host(&base).await;
    let project = create_project(&host, "proj").await;
    let assets = import_assets(&host, &project, &base, &[("a.txt", "alpha")]).await;
    let collection = create_collection(&host, &project, &assets).await;
    let workflow = json!({
        "inputs": { "doc": { "type": "string", "format": "asset" } },
        "params": { "type": "object" },
        "steps": [
            {
                "id": "gate",
                "op": "test.gate",
                "input": { "doc": { "$from": "inputs.doc" } }
            },
            {
                "id": "copy",
                "op": "test.copy",
                "input": { "doc": { "$from": "steps.gate.doc" } }
            }
        ],
        "output": { "$from": "steps.copy.doc" }
    });
    let workflow_id = create_workflow(&host, &project, workflow).await;
    ops.register(&host);

    // Shutdown races the registration of a new run: the claim may land
    // before or after the drain. Either way nothing may execute afterwards.
    let (run_result, ()) = tokio::join!(
        try_invoke(
            &host,
            &project,
            "dynamic_workflow.run",
            json!({
                "dynamic_workflow_id": workflow_id,
                "target": { "collection_id": collection.as_str() },
            }),
        ),
        async {
            tokio::task::yield_now().await;
            host.shutdown().await;
        }
    );
    if let Err(error) = run_result {
        assert_eq!(
            error.code().as_str(),
            "DYNAMIC_WORKFLOW/CALLER_SHUTTING_DOWN",
            "a run that arrives after shutdown is rejected, not started"
        );
    }

    // Release the gate if a task managed to reach it; an aborted or rejected
    // run must not continue into the copy step.
    ops.gate_release.notify_one();
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(
        ops.counters.copy.load(Ordering::SeqCst),
        0,
        "no step may run after shutdown"
    );

    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn interrupted_run_resumes_after_the_last_successful_step() {
    let base = temp_dir("resume");
    let ops = TestOps::new();
    let host = build_host(&base).await;
    let project = create_project(&host, "proj").await;
    let assets = import_assets(&host, &project, &base, &[("a.txt", "alpha")]).await;
    let collection = create_collection(&host, &project, &assets).await;
    let workflow = json!({
        "inputs": { "doc": { "type": "string", "format": "asset" } },
        "params": { "type": "object" },
        "steps": [
            {
                "id": "copy",
                "op": "test.copy",
                "input": { "doc": { "$from": "inputs.doc" } }
            },
            {
                "id": "pause",
                "op": "test.pause",
                "input": { "doc": { "$from": "steps.copy.doc" } }
            }
        ],
        "output": { "$from": "steps.pause.doc" }
    });
    let workflow_id = create_workflow(&host, &project, workflow).await;
    ops.register(&host);

    let run_id = start_run(
        &host,
        &project,
        &workflow_id,
        json!({ "collection_id": collection.as_str() }),
        None,
    )
    .await;
    ops.pause_started.notified().await;
    assert_eq!(ops.counters.copy.load(Ordering::SeqCst), 1);

    host.shutdown().await;
    drop(host);
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }

    // Simulated restart: a new host over the same directory.
    let host = build_host(&base).await;
    ops.register(&host);
    let run = get_run(&host, &project, &run_id).await;
    assert_eq!(run.state, RunState::Interrupted);

    let output = invoke(
        &host,
        &project,
        "dynamic_workflow.resume",
        json!({ "run_id": run_id.as_str() }),
    )
    .await;
    assert_eq!(output["state"], "completed");

    let run = get_run(&host, &project, &run_id).await;
    assert_eq!(run.state, RunState::Completed);
    assert_eq!(run.items[0].state, SlotState::Succeeded);
    assert_eq!(
        ops.counters.copy.load(Ordering::SeqCst),
        1,
        "the successful step is reused, not re-run"
    );
    assert_eq!(
        ops.counters.pause.load(Ordering::SeqCst),
        2,
        "the interrupted non-open-world step is re-run"
    );

    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn a_failing_run_task_marks_the_run_failed() {
    let base = temp_dir("task-error");
    let ops = TestOps::new();
    let host = build_host(&base).await;
    let project = create_project(&host, "proj").await;
    let assets = import_assets(&host, &project, &base, &[("a.txt", "alpha")]).await;
    let collection = create_collection(&host, &project, &assets).await;
    let workflow = json!({
        "inputs": { "doc": { "type": "string", "format": "asset" } },
        "params": { "type": "object" },
        "steps": [
            {
                "id": "copy",
                "op": "test.copy",
                "input": { "doc": { "$from": "inputs.doc" } }
            },
            {
                "id": "pause",
                "op": "test.pause",
                "input": { "doc": { "$from": "steps.copy.doc" } }
            }
        ],
        "output": { "$from": "steps.pause.doc" }
    });
    let workflow_id = create_workflow(&host, &project, workflow).await;
    ops.register(&host);

    let run_id = start_run(
        &host,
        &project,
        &workflow_id,
        json!({ "collection_id": collection.as_str() }),
        None,
    )
    .await;
    ops.pause_started.notified().await;

    host.shutdown().await;
    drop(host);
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }

    // A new session whose target collection is gone: preparing the resumed
    // run fails before any item executes.
    let host = build_host(&base).await;
    ops.register(&host);
    let collection_path = host
        .config()
        .paths()
        .project_dir(project.as_str())
        .join("collections")
        .join(format!("{}.json", collection.as_str()));
    tokio::fs::remove_file(&collection_path)
        .await
        .expect("remove collection");

    let output = invoke(
        &host,
        &project,
        "dynamic_workflow.resume",
        json!({ "run_id": run_id.as_str() }),
    )
    .await;
    assert_eq!(output["state"], "failed");

    let run = get_run(&host, &project, &run_id).await;
    assert_eq!(run.state, RunState::Failed);
    assert!(run.finished_at.is_some(), "a failed run is finished");
    for item in &run.items {
        assert_eq!(
            item.state,
            SlotState::Failed,
            "items the run never settled are failed with it"
        );
        assert_eq!(
            item.diagnostic
                .as_ref()
                .expect("item diagnostic")
                .code()
                .as_str(),
            "DYNAMIC_WORKFLOW/RUN_FAILED"
        );
    }

    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn interrupted_open_world_step_needs_confirmation_and_is_not_replayed() {
    let base = temp_dir("needs-confirmation");
    let ops = TestOps::new();
    let host = build_host(&base).await;
    let project = create_project(&host, "proj").await;
    let assets = import_assets(&host, &project, &base, &[("a.txt", "alpha")]).await;
    let collection = create_collection(&host, &project, &assets).await;
    let workflow = json!({
        "inputs": { "doc": { "type": "string", "format": "asset" } },
        "params": { "type": "object" },
        "steps": [
            {
                "id": "copy",
                "op": "test.copy",
                "input": { "doc": { "$from": "inputs.doc" } }
            },
            {
                "id": "publish",
                "op": "test.open_world",
                "input": { "doc": { "$from": "steps.copy.doc" } }
            }
        ],
        "output": { "$from": "steps.publish.doc" }
    });
    let workflow_id = create_workflow(&host, &project, workflow).await;
    ops.register(&host);

    let run_id = start_run(
        &host,
        &project,
        &workflow_id,
        json!({ "collection_id": collection.as_str() }),
        None,
    )
    .await;
    ops.open_world_started.notified().await;
    assert_eq!(ops.counters.open_world.load(Ordering::SeqCst), 1);

    host.shutdown().await;
    drop(host);
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }

    let host = build_host(&base).await;
    ops.register(&host);
    assert_eq!(
        get_run(&host, &project, &run_id).await.state,
        RunState::Interrupted
    );

    let output = invoke(
        &host,
        &project,
        "dynamic_workflow.resume",
        json!({ "run_id": run_id.as_str() }),
    )
    .await;
    assert_eq!(output["state"], "needs_confirmation");

    let run = get_run(&host, &project, &run_id).await;
    assert_eq!(run.state, RunState::NeedsConfirmation);
    assert_eq!(run.items[0].state, SlotState::NeedsConfirmation);
    assert_eq!(
        ops.counters.open_world.load(Ordering::SeqCst),
        1,
        "an interrupted open-world non-idempotent step is not replayed"
    );
    assert_eq!(ops.counters.copy.load(Ordering::SeqCst), 1);

    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn group_target_builds_a_collection_in_group_order() {
    let base = temp_dir("groups");
    let ops = TestOps::new();
    let host = build_host(&base).await;
    let project = create_project(&host, "proj").await;
    let workflow = json!({
        "inputs": {},
        "params": {
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"]
        },
        "steps": [{
            "id": "render",
            "op": "test.render",
            "input": { "text": { "$from": "params.text" } }
        }],
        "output": { "$from": "steps.render.doc" }
    });
    let workflow_id = create_workflow(&host, &project, workflow).await;
    ops.register(&host);

    let run_id = start_run(
        &host,
        &project,
        &workflow_id,
        json!({
            "groups": [
                { "text": "first" },
                { "text": "second" },
                { "text": "third" }
            ],
            "name": "Rendered batch"
        }),
        None,
    )
    .await;
    let run = wait_for_run(&host, &project, &run_id, |run| {
        !matches!(run.state, RunState::Queued | RunState::Running)
    })
    .await;
    assert_eq!(run.state, RunState::Completed);
    assert_eq!(run.items.len(), 3);
    for (index, item) in run.items.iter().enumerate() {
        assert_eq!(item.item_id, format!("group-{}", index + 1));
        assert_eq!(item.state, SlotState::Succeeded);
    }
    let collection_id = run.collection_id.clone().expect("created collection");

    let output = invoke(
        &host,
        &project,
        "collection.get",
        json!({ "collection_id": collection_id.as_str() }),
    )
    .await;
    let slots = output["slots"].as_array().expect("slots");
    assert_eq!(slots.len(), 3);

    let store = AssetStore::new(host.config().paths().project_dir(project.as_str()));
    let mut contents = Vec::new();
    for slot in slots {
        let asset = AssetId::new(slot["asset_id"].as_str().expect("asset id"));
        let record = store
            .get_asset(&asset)
            .await
            .expect("asset lookup")
            .expect("asset exists");
        let bytes = tokio::fs::read(store.content_path(&record.hash))
            .await
            .expect("asset bytes");
        contents.push(String::from_utf8(bytes).expect("utf8 asset"));
    }
    assert_eq!(contents, vec!["first", "second", "third"]);

    let output = invoke(
        &host,
        &project,
        "dynamic_workflow.list_runs",
        json!({ "collection_id": collection_id.as_str() }),
    )
    .await;
    let runs = output["runs"].as_array().expect("runs");
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0]["id"], run_id.as_str());

    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn output_must_be_produced_by_this_runs_steps() {
    let base = temp_dir("output-unproduced");
    let ops = TestOps::new();
    let host = build_host(&base).await;
    let project = create_project(&host, "proj").await;
    let assets = import_assets(&host, &project, &base, &[("a.txt", "alpha")]).await;
    let collection = create_collection(&host, &project, &assets).await;
    // The output binds to the input asset, which exists but no step made it.
    let workflow_id = create_workflow(
        &host,
        &project,
        copy_workflow(vec![], json!({ "$from": "inputs.doc" })),
    )
    .await;
    ops.register(&host);

    let run_id = start_run(
        &host,
        &project,
        &workflow_id,
        json!({ "collection_id": collection.as_str() }),
        None,
    )
    .await;
    let run = wait_for_run(&host, &project, &run_id, |run| {
        !matches!(run.state, RunState::Queued | RunState::Running)
    })
    .await;

    assert_eq!(run.state, RunState::Failed);
    let item = &run.items[0];
    assert_eq!(item.state, SlotState::Failed);
    assert_eq!(
        item.diagnostic
            .as_ref()
            .expect("diagnostic")
            .code()
            .as_str(),
        "DYNAMIC_WORKFLOW/OUTPUT_NOT_PRODUCED"
    );
    assert_eq!(
        ops.counters.copy.load(Ordering::SeqCst),
        0,
        "no copy step ran"
    );

    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn resume_reads_the_input_asset_pinned_at_run_creation() {
    let base = temp_dir("resume-pinned-input");
    let ops = TestOps::new();
    let host = build_host(&base).await;
    let project = create_project(&host, "proj").await;
    let assets = import_assets(
        &host,
        &project,
        &base,
        &[("a.txt", "alpha"), ("b.txt", "beta")],
    )
    .await;
    let collection = create_collection(&host, &project, &assets[..1]).await;
    let workflow = json!({
        "inputs": { "doc": { "type": "string", "format": "asset" } },
        "params": { "type": "object" },
        "steps": [
            {
                "id": "pause",
                "op": "test.pause",
                "input": { "doc": { "$from": "inputs.doc" } }
            },
            {
                "id": "copy",
                "op": "test.copy",
                "input": { "doc": { "$from": "inputs.doc" } }
            }
        ],
        "output": { "$from": "steps.copy.doc" }
    });
    let workflow_id = create_workflow(&host, &project, workflow).await;
    ops.register(&host);

    let run_id = start_run(
        &host,
        &project,
        &workflow_id,
        json!({ "collection_id": collection.as_str() }),
        None,
    )
    .await;
    ops.pause_started.notified().await;

    host.shutdown().await;
    drop(host);
    let host = reopen_host(&base, &ops).await;

    // The slot is repointed while the run is interrupted.
    let output = invoke(
        &host,
        &project,
        "collection.get",
        json!({ "collection_id": collection.as_str() }),
    )
    .await;
    let slot_id = output["slots"][0]["id"]
        .as_str()
        .expect("slot id")
        .to_owned();
    invoke(
        &host,
        &project,
        "collection.set_slot",
        json!({
            "collection_id": collection.as_str(),
            "slot_id": slot_id,
            "asset": assets[1],
        }),
    )
    .await;

    let output = invoke(
        &host,
        &project,
        "dynamic_workflow.resume",
        json!({ "run_id": run_id.as_str() }),
    )
    .await;
    assert_eq!(output["state"], "completed");

    assert_eq!(
        slot_contents(&host, &project, &collection).await,
        vec!["alpha".to_owned()],
        "the resumed steps read the asset pinned at run creation"
    );

    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn resume_uses_the_workflow_snapshot() {
    let base = temp_dir("resume-snapshot");
    let ops = TestOps::new();
    let host = build_host(&base).await;
    let project = create_project(&host, "proj").await;
    let assets = import_assets(&host, &project, &base, &[("a.txt", "alpha")]).await;
    let collection = create_collection(&host, &project, &assets).await;
    let workflow = json!({
        "inputs": { "doc": { "type": "string", "format": "asset" } },
        "params": { "type": "object" },
        "steps": [
            {
                "id": "pause",
                "op": "test.pause",
                "input": { "doc": { "$from": "inputs.doc" } }
            },
            {
                "id": "copy",
                "op": "test.copy",
                "input": { "doc": { "$from": "inputs.doc" } }
            }
        ],
        "output": { "$from": "steps.copy.doc" }
    });
    let workflow_id = create_workflow(&host, &project, workflow).await;
    ops.register(&host);

    let run_id = start_run(
        &host,
        &project,
        &workflow_id,
        json!({ "collection_id": collection.as_str() }),
        None,
    )
    .await;
    ops.pause_started.notified().await;

    host.shutdown().await;
    drop(host);
    let host = reopen_host(&base, &ops).await;

    // Rewrite the saved workflow; the run must resume from its own snapshot.
    let workflow_path = host
        .config()
        .paths()
        .project_dir(project.as_str())
        .join("dynamic_workflows")
        .join(format!("{workflow_id}.json"));
    let mut document: Value = serde_json::from_slice(
        &tokio::fs::read(&workflow_path)
            .await
            .expect("read dynamic workflow"),
    )
    .expect("workflow json");
    document["workflow"] = json!({
        "inputs": { "doc": { "type": "string", "format": "asset" } },
        "params": { "type": "object" },
        "steps": [{
            "id": "other",
            "op": "test.missing",
            "input": { "doc": { "$from": "inputs.doc" } }
        }],
        "output": { "$from": "steps.other.doc" }
    });
    tokio::fs::write(
        &workflow_path,
        serde_json::to_vec_pretty(&document).expect("serialize workflow"),
    )
    .await
    .expect("write dynamic workflow");

    let output = invoke(
        &host,
        &project,
        "dynamic_workflow.resume",
        json!({ "run_id": run_id.as_str() }),
    )
    .await;
    assert_eq!(output["state"], "completed");
    assert_eq!(
        ops.counters.pause.load(Ordering::SeqCst),
        2,
        "the snapshot's paused step, not the edited one, was resumed"
    );
    assert_eq!(ops.counters.copy.load(Ordering::SeqCst), 1);

    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn resume_rejects_a_run_that_is_not_interrupted() {
    let base = temp_dir("resume-not-interrupted");
    let ops = TestOps::new();
    let host = build_host(&base).await;
    let project = create_project(&host, "proj").await;
    let assets = import_assets(&host, &project, &base, &[("a.txt", "alpha")]).await;
    let collection = create_collection(&host, &project, &assets).await;
    let workflow_id = create_workflow(
        &host,
        &project,
        copy_workflow(
            vec![json!({
                "id": "copy",
                "op": "test.copy",
                "input": { "doc": { "$from": "inputs.doc" } }
            })],
            json!({ "$from": "steps.copy.doc" }),
        ),
    )
    .await;
    ops.register(&host);

    let run_id = start_run(
        &host,
        &project,
        &workflow_id,
        json!({ "collection_id": collection.as_str() }),
        None,
    )
    .await;
    wait_for_run(&host, &project, &run_id, |run| {
        !matches!(run.state, RunState::Queued | RunState::Running)
    })
    .await;
    assert_eq!(
        ops.counters.copy.load(Ordering::SeqCst),
        1,
        "the run completed before the resume attempt"
    );

    let error = try_invoke(
        &host,
        &project,
        "dynamic_workflow.resume",
        json!({ "run_id": run_id.as_str() }),
    )
    .await
    .expect_err("a completed run cannot resume");
    assert_eq!(
        error.code().as_str(),
        "DYNAMIC_WORKFLOW/CALLER_NOT_INTERRUPTED"
    );
    assert_eq!(ops.counters.copy.load(Ordering::SeqCst), 1);

    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn concurrent_resumes_start_one_task() {
    let base = temp_dir("resume-concurrent");
    let ops = TestOps::new();
    let host = build_host(&base).await;
    let project = create_project(&host, "proj").await;
    let assets = import_assets(&host, &project, &base, &[("a.txt", "alpha")]).await;
    let collection = create_collection(&host, &project, &assets).await;
    let workflow = json!({
        "inputs": { "doc": { "type": "string", "format": "asset" } },
        "params": { "type": "object" },
        "steps": [
            {
                "id": "copy",
                "op": "test.copy",
                "input": { "doc": { "$from": "inputs.doc" } }
            },
            {
                "id": "pause",
                "op": "test.pause",
                "input": { "doc": { "$from": "steps.copy.doc" } }
            }
        ],
        "output": { "$from": "steps.pause.doc" }
    });
    let workflow_id = create_workflow(&host, &project, workflow).await;
    ops.register(&host);

    let run_id = start_run(
        &host,
        &project,
        &workflow_id,
        json!({ "collection_id": collection.as_str() }),
        None,
    )
    .await;
    ops.pause_started.notified().await;

    host.shutdown().await;
    drop(host);
    let host = reopen_host(&base, &ops).await;

    assert_eq!(
        get_run(&host, &project, &run_id).await.state,
        RunState::Interrupted
    );

    let service = host.dynamic_workflow_service();
    let (first, second) = tokio::join!(
        service.resume(&project, &run_id),
        service.resume(&project, &run_id)
    );
    let results = [first, second];
    assert_eq!(
        results.iter().filter(|result| result.is_ok()).count(),
        1,
        "exactly one resume may claim the run"
    );
    let error = results
        .iter()
        .find_map(|result| result.as_ref().err())
        .expect("the other resume is rejected");
    assert!(
        matches!(
            error.code().as_str(),
            "DYNAMIC_WORKFLOW/CALLER_NOT_INTERRUPTED"
                | "DYNAMIC_WORKFLOW/CALLER_RUN_ALREADY_ACTIVE"
        ),
        "unexpected rejection: {}",
        error.code().as_str()
    );

    let run = get_run(&host, &project, &run_id).await;
    assert_eq!(run.state, RunState::Completed);
    assert_eq!(
        ops.counters.pause.load(Ordering::SeqCst),
        2,
        "the resumed step ran once, not once per resume"
    );
    assert_eq!(
        ops.counters.copy.load(Ordering::SeqCst),
        1,
        "the successful step was reused"
    );

    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn group_run_resumes_and_creates_its_collection_once() {
    let base = temp_dir("group-resume");
    let ops = TestOps::new();
    let host = build_host(&base).await;
    let project = create_project(&host, "proj").await;
    let workflow_id = create_workflow(&host, &project, group_workflow("test.pause_render")).await;
    ops.register(&host);

    let run_id = start_run(
        &host,
        &project,
        &workflow_id,
        json!({ "groups": [{ "text": "only" }], "name": "Rendered batch" }),
        None,
    )
    .await;
    ops.pause_started.notified().await;

    host.shutdown().await;
    drop(host);
    let host = reopen_host(&base, &ops).await;

    let output = invoke(
        &host,
        &project,
        "dynamic_workflow.resume",
        json!({ "run_id": run_id.as_str() }),
    )
    .await;
    assert_eq!(output["state"], "completed");

    let run = get_run(&host, &project, &run_id).await;
    assert_eq!(run.state, RunState::Completed);
    let collection = run.collection_id.clone().expect("created collection");
    assert_eq!(collection_file_count(&host, &project).await, 1);
    assert_eq!(
        slot_contents(&host, &project, &collection).await,
        vec!["only".to_owned()]
    );
    assert_eq!(ops.counters.pause.load(Ordering::SeqCst), 2);

    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn resuming_a_group_run_that_has_a_collection_does_not_create_another() {
    let base = temp_dir("group-existing-collection");
    let ops = TestOps::new();
    let host = build_host(&base).await;
    let project = create_project(&host, "proj").await;
    let workflow_id = create_workflow(&host, &project, group_workflow("test.render")).await;
    ops.register(&host);

    let run_id = start_run(
        &host,
        &project,
        &workflow_id,
        json!({
            "groups": [{ "text": "first" }, { "text": "second" }],
            "name": "Rendered batch"
        }),
        None,
    )
    .await;
    let run = wait_for_run(&host, &project, &run_id, |run| {
        !matches!(run.state, RunState::Queued | RunState::Running)
    })
    .await;
    assert_eq!(run.state, RunState::Completed);
    let original_collection = run.collection_id.clone().expect("created collection");
    assert_eq!(collection_file_count(&host, &project).await, 1);

    // A crash between recording the collection and settling the run leaves a
    // still-running document that already names its collection.
    let run_path = host
        .config()
        .paths()
        .project_dir(project.as_str())
        .join("dynamic_workflow_runs")
        .join(format!("{}.json", run_id.as_str()));
    let mut document: Value =
        serde_json::from_slice(&tokio::fs::read(&run_path).await.expect("read run document"))
            .expect("run json");
    document["state"] = json!("interrupted");
    document["version"] = json!(document["version"].as_u64().expect("version") + 1);
    tokio::fs::write(
        &run_path,
        serde_json::to_vec_pretty(&document).expect("serialize run"),
    )
    .await
    .expect("write run document");

    let output = invoke(
        &host,
        &project,
        "dynamic_workflow.resume",
        json!({ "run_id": run_id.as_str() }),
    )
    .await;
    assert_eq!(output["state"], "completed");

    let run = get_run(&host, &project, &run_id).await;
    assert_eq!(run.collection_id.as_ref(), Some(&original_collection));
    assert_eq!(
        collection_file_count(&host, &project).await,
        1,
        "the existing collection is reused, not duplicated"
    );
    assert_eq!(
        slot_contents(&host, &project, &original_collection).await,
        vec!["first".to_owned(), "second".to_owned()]
    );

    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn run_set_slot_uses_the_run_context_and_can_be_undone() {
    let base = temp_dir("set-slot-undo");
    let ops = TestOps::new();
    let host = build_host(&base).await;
    let project = create_project(&host, "proj").await;
    let assets = import_assets(&host, &project, &base, &[("a.txt", "alpha")]).await;
    let collection = create_collection(&host, &project, &assets).await;
    let workflow_id = create_workflow(
        &host,
        &project,
        copy_workflow(
            vec![json!({
                "id": "copy",
                "op": "test.copy",
                "input": { "doc": { "$from": "inputs.doc" } }
            })],
            json!({ "$from": "steps.copy.doc" }),
        ),
    )
    .await;
    ops.register(&host);

    let run_context = OperationContext::new(
        CommandActor::new(CommandActorKind::Human).with_id("run-actor"),
        CommandProvenance::Direct,
    );
    let run_id = start_run_as(
        &host,
        &project,
        &workflow_id,
        json!({ "collection_id": collection.as_str() }),
        None,
        run_context,
    )
    .await;
    let run = wait_for_run(&host, &project, &run_id, |run| {
        !matches!(run.state, RunState::Queued | RunState::Running)
    })
    .await;
    assert_eq!(run.state, RunState::Completed);

    // Every step of the run, including the SetSlot path's siblings, carries
    // the initiating actor.
    let step_log = StepLog::new(host.config().paths().project_dir(project.as_str()));
    let records = step_log.records().await.expect("step records");
    let run_records: Vec<_> = records
        .iter()
        .filter(|record| record.run.as_ref().is_some_and(|run| run.run_id == run_id))
        .collect();
    assert!(!run_records.is_empty());
    for record in run_records {
        assert_eq!(record.context.actor().id(), Some("run-actor"));
    }

    let snapshot = invoke(
        &host,
        &project,
        "collection.get",
        json!({ "collection_id": collection.as_str() }),
    )
    .await;
    let output_asset = snapshot["slots"][0]["asset_id"]
        .as_str()
        .expect("asset id")
        .to_owned();
    assert_ne!(output_asset, assets[0], "the slot points at the run output");

    let undo = invoke(
        &host,
        &project,
        "collection.undo",
        json!({ "collection_id": collection.as_str() }),
    )
    .await;
    assert_eq!(undo["applied"], true);
    let snapshot = invoke(
        &host,
        &project,
        "collection.get",
        json!({ "collection_id": collection.as_str() }),
    )
    .await;
    assert_eq!(
        snapshot["slots"][0]["asset_id"].as_str(),
        Some(assets[0].as_str()),
        "the run's SetSlot can be undone"
    );

    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn dynamic_workflow_create_is_not_deterministic() {
    let base = temp_dir("create-annotations");
    let host = build_host(&base).await;
    let project = create_project(&host, "proj").await;
    let infos = host.operation_service().list(&project).await;
    let create = infos
        .iter()
        .find(|info| info.spec.name == "dynamic_workflow.create")
        .expect("dynamic_workflow.create is registered");
    assert!(
        !create.spec.annotations.deterministic,
        "every create call allocates a new id"
    );
    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn unknown_target_keys_are_caller_diagnostics() {
    let base = temp_dir("unknown-target-key");
    let ops = TestOps::new();
    let host = build_host(&base).await;
    let project = create_project(&host, "proj").await;
    let assets = import_assets(&host, &project, &base, &[("a.txt", "alpha")]).await;
    let collection = create_collection(&host, &project, &assets).await;
    let workflow_id = create_workflow(
        &host,
        &project,
        copy_workflow(
            vec![json!({
                "id": "copy",
                "op": "test.copy",
                "input": { "doc": { "$from": "inputs.doc" } }
            })],
            json!({ "$from": "steps.copy.doc" }),
        ),
    )
    .await;
    ops.register(&host);

    // `slot` is a misspelling of `slots`; it must not degrade to "all slots".
    let error = try_invoke(
        &host,
        &project,
        "dynamic_workflow.run",
        json!({
            "dynamic_workflow_id": workflow_id,
            "target": { "collection_id": collection.as_str(), "slot": ["slot-1"] }
        }),
    )
    .await
    .expect_err("a misspelled target key is rejected");
    assert_eq!(
        error.code().as_str(),
        "DYNAMIC_WORKFLOW/CALLER_INVALID_TARGET"
    );
    assert!(
        error.message().contains("slot"),
        "the diagnostic names the unknown key: {}",
        error.message()
    );

    let output = invoke(
        &host,
        &project,
        "dynamic_workflow.list_runs",
        json!({ "collection_id": collection.as_str() }),
    )
    .await;
    assert_eq!(
        output["runs"].as_array().expect("runs").len(),
        0,
        "no run was created"
    );

    let _ = std::fs::remove_dir_all(&base);
}
