# Runtime Module Architecture

> Status: working draft
> Crate: `crates/runtime`

## Role

`runtime` is the host-independent workflow execution layer. It executes `core::ExecutionPlan` values using injected node capabilities, manages run sessions, schedules stages, handles cancellation, routes artifacts through injected artifact capabilities, and emits `core::RunEvent` values through a host-provided sink.

It must not depend on Tauri or Axum.

## Responsibilities

- Execute prepared `core::ExecutionPlan` values through `RuntimeService` / `ExecutionRunner`.
- Own `RunSession` and run state.
- Schedule DAG stages and parallel nodes.
- Handle cancellation.
- Coordinate node execution through a `NodeExecutorRegistry`.
- Route preview and saved artifacts through injected artifact capabilities.
- Emit run events through `RunEventSink`.

## Non-Responsibilities

- Workflow command application.
- Workflow history.
- Workflow readiness planning.
- Workspace config loading.
- Model manifest scanning or model reference policy.
- Tauri IPC.
- HTTP/WebSocket routing.
- Agent reasoning.
- ComfyUI import.
- UI state.

## Dependencies

```text
runtime -> core
runtime must not -> tauri
runtime must not -> axum
runtime must not -> model-manager
runtime must not -> candle-integration
```

Concrete node executors and backend capabilities are assembled by `app-host`. Runtime defines the execution traits and consumes trait objects/registries.

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
  executor.rs
  store.rs
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

`RuntimeService` is long-lived and held by `app-host` as a handle:

```text
RuntimeService
  run_store
  node_executor_registry
```

Workspace config, model manager, backend model stores, and concrete node executor construction belong to `app-host`.

`RunStore` tracks active runs and summaries:

```text
RunStore
  active: RunId -> RunHandle
  snapshots: RunId -> RunSnapshot
  summaries: RunId -> RunSummary
```

V1 can use `RwLock<HashMap<...>>`; it does not need a concurrent map unless profiling shows contention.

`RunHandle` is host-visible metadata/control, not a value-store handle:

```text
RunHandle
  run_id
  cancellation
```

`RunHandle` is a control handle, not the canonical state source. Hosts query `RunStore` for `RunSnapshot` and `RunSummary`.

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

Runtime does not scan model directories, own the manifest, or depend on `model-manager`. Model resolution and backend loading are provided through node executors or runtime capabilities assembled by `app-host`.

Execution flow:

```text
Workflow ModelRef
  -> model-manager readiness resolver reports availability diagnostics
  -> app-host builds execution plan and node executor capabilities
  -> checkpoint/model-loading node executor resolves/loads through injected capability
  -> RuntimeValue::Model / Clip / Vae
```

The loaded backend payload stays in the backend model store owned by the injected backend capability. Runtime passes typed handles between nodes.
