# Runtime Module Architecture

> Status: working draft
> Crate: `crates/runtime`

## Role

`runtime` is the host-independent workflow execution layer. It executes `core::ExecutionPlan` values using node capabilities and Candle-backed inference, manages run sessions, schedules stages, handles cancellation, routes artifacts, and emits `core::RunEvent` values through a host-provided sink.

It must not depend on Tauri or Axum.

## Responsibilities

- Execute workflows through `ExecutionRunner`.
- Own `RunSession` and run state.
- Schedule DAG stages and parallel nodes.
- Handle cancellation.
- Coordinate node execution and backend calls.
- Route preview and saved artifacts under `base_path`.
- Emit run events through `RunEventSink`.

## Non-Responsibilities

- Workflow command application.
- Workflow history.
- Tauri IPC.
- HTTP/WebSocket routing.
- Agent reasoning.
- ComfyUI import.
- UI state.

## Dependencies

```text
runtime -> core
runtime -> nodes
runtime -> candle-integration
runtime must not -> tauri
runtime must not -> axum
```

## Suggested Module Layout

```text
src/
  lib.rs
  runtime.rs
  runner.rs
  run_session.rs
  scheduler.rs
  cancellation.rs
  artifacts.rs
  events.rs
  node_context.rs
```

Use modern Rust module layout. Do not introduce `mod.rs`, and prefer ordinary `mod foo;` declarations over `#[path = "..."]` attributes.

## RunEventSink

`core` owns `RunEvent`. `runtime` owns `RunEventSink`.

```text
RunEventSink
  emit(RunEvent)
  emit_many(Vec<RunEvent>)
```

Host implementations:

```text
TauriRunEventSink
  app.emit("run_event", event)

BroadcastRunEventSink
  tokio broadcast channel for future Axum SSE/WebSocket

VecRunEventSink
  test sink
```

Sink failure does not automatically fail the run. Runtime records/logs sink failure and continues when possible. Actual execution failures still fail the run.

## Cancellation

Cancellation belongs to `runtime` and uses a scheduler-aware cancellation token.

```text
RunSession
  run_id
  workflow_id
  workflow_version
  state
  cancellation_token
  started_at
  finished_at
```

Scheduler rules:

- check cancellation before scheduling a stage;
- check cancellation before scheduling each node;
- pass the token into node execution context;
- stop downstream scheduling after cancellation;
- emit `RunCancelled` and `NodeCancelled` as appropriate.

```text
NodeExecutionContext
  run_id
  node_id
  inputs
  params
  artifacts
  cancellation_token
  correlation_id
```

`NodeCancelled` means cancellation stopped or prevented execution. `NodeSkipped` means upstream failure or readiness conditions prevented execution.

Cancellation is not a diagnostic error by default. Cleanup failures may produce diagnostics.

## Failure Strategy

V1 uses fail-fast downstream scheduling.

When a node in a parallel stage fails:

```text
Stage: A, B, C
A fails
```

Runtime behavior:

- emit `NodeFailed` for the failed node;
- transition the run into a failing state;
- stop scheduling downstream stages;
- request cancellation for already-running sibling nodes when possible;
- wait for running siblings to finish or observe cancellation;
- discard late sibling outputs if the run is already failing;
- emit `NodeSkipped` for downstream nodes that cannot run because an upstream dependency failed;
- emit `RunFailed` with diagnostics.

This avoids continuing expensive downstream work after the run is already unrecoverable, while still leaving room for in-flight tasks to shut down cleanly.

## Runtime Store

Runtime state is layered:

```text
Runtime
  RunStore
    RunSession
      RunValueStore
```

`Runtime` is long-lived and held by the host as a handle:

```text
Runtime
  config
  run_store
  node_catalog
  model_descriptor_resolver
  backend_model_store
```

`RunStore` tracks active runs and summaries:

```text
RunStore
  active: RunId -> RunHandle
  summaries: RunId -> RunSummary
```

V1 can use `RwLock<HashMap<...>>`; it does not need a concurrent map unless profiling shows contention.

`RunHandle` is host-visible metadata/control, not a value-store handle:

```text
RunHandle
  run_id
  state
  cancellation
```

`RunSession` is internal to the runner task:

```text
RunSession
  run_id
  workflow_id
  workflow_version
  plan
  state
  values: RunValueStore
  artifacts: ArtifactStore
  cancellation
  correlation_id
```

`RunValueStore` stores intermediate node outputs only for the lifetime of a run:

```text
RunValueStore
  values: OutputKey -> RuntimeValue

OutputKey
  node_id
  slot_id
```

`RuntimeValue` may contain backend-native handles for cheap in-run sharing:

```text
Param
Model
Clip
Vae
Latent
Conditioning
Image
Artifact
Null
```

These values are not owned by app state and are not exposed to UI. Host state keeps run handles, summaries, diagnostics, and artifact references only.

`core::model::NodeValue` remains the public semantic value model. It is not the runtime store's primary representation.

Node executors should not receive the whole `RunValueStore`. Runtime resolves graph dependencies and passes only the node's inputs through `NodeExecutionContext`.

Run cleanup:

```text
completed/failed/cancelled
  -> persist artifacts as needed
  -> store RunSummary
  -> drop RunSession
  -> drop RunValueStore and intermediate tensors
```

V1 does not persist run event logs or intermediate values. UI can recover with `RunSummary` and workflow snapshots.

## Model Resolution and Loading

Runtime does not scan model directories and does not own the manifest. It receives a model descriptor resolver from `model-manager`.

Execution flow:

```text
Workflow ModelRef
  -> model-manager readiness resolver reports availability diagnostics
  -> runtime resolves full ModelDescriptor through model-manager
  -> backend model store get_or_load(descriptor, role, device)
  -> RuntimeValue::Model / Clip / Vae
```

The loaded backend payload stays in the backend model store. Runtime passes typed handles between nodes.
