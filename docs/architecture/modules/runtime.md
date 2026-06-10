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
  resources.rs
  store.rs
```

Use modern Rust module layout. Do not introduce `mod.rs`, and prefer ordinary `mod foo;` declarations over `#[path = "..."]` attributes.

## Public Run Boundary

`RuntimeService::run` starts a complete workflow run for one prepared `core::ExecutionPlan`.

```text
RuntimeService::run(plan, run_inputs, options, sink) -> RunHandle
```

It does not mean "execute one internal execution unit". The public runtime boundary remains plan-based because callers such as `app-host`, Tauri, Axum, and Agent tools reason about workflow runs, targets, diagnostics, and run state. They should not need to know how the runtime scheduler subdivides the plan.

Runtime may compile an `ExecutionPlan` into internal scheduling structures:

```text
ExecutionPlan
  -> PreparedRun
    -> ScheduledGraph
      -> ExecutionUnit
```

`PreparedRun`, `ScheduledGraph`, and `ExecutionUnit` are runtime-internal concepts. They may model stage nodes, executor calls, artifact writes, or future fused operations, but they are not the API consumed by hosts.

V1 `run` should spawn a background runner task and return a `RunHandle` immediately. Hosts observe progress through `RunStore` snapshots/summaries and `RunEventSink`, rather than blocking on the whole run.

```text
RuntimeService::run(...)
  -> create run_id and cancellation token
  -> insert active RunHandle and initial RunSnapshot
  -> spawn runner task with RunSession
  -> return RunHandle
```

The future may add lower-level APIs for testing or scheduler introspection, but host-facing execution stays run/plan-oriented.

## Host and Agent Boundary

`runtime` exposes a host-neutral service contract for `app-host`. It must not know about Tauri, Axum, or Agent tools.

```text
RuntimeService
  run(plan, run_inputs, options, sink) -> RunHandle
  cancel(run_id) -> OperationReport
  snapshot(run_id) -> Option<RunSnapshot>
  summary(run_id) -> Option<RunSummary>
```

`app-host` decides which runtime operations are reachable from UI, future Axum routes, or Agent tools.

```text
UI/Tauri -> app-host -> runtime
Future Axum -> app-host -> runtime
Agent -> app-host -> runtime only when policy explicitly allows

Agent must not -> runtime
runtime must not -> agent
```

V1 Agent policy does not expose runtime run/cancel tools. Agent may inspect workflow/model/diagnostic facades through `app-host`, and runtime status may be projected into diagnostics or summaries by `app-host`, but Agent does not receive direct `RuntimeService` access.

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

V1 should use `Arc<RwLock<RunStoreInner>>`; it does not need a concurrent map unless profiling shows contention. The runner task owns the mutable `RunSession` and publishes snapshots/summaries back into the store. Hosts never get direct mutable access to a session.

`RunHandle` is host-visible metadata/control, not a value-store handle:

```text
RunHandle
  run_id
  cancellation
```

`RunHandle` is a control handle, not the canonical state source. Hosts query `RunStore` for `RunSnapshot` and `RunSummary`.

`RunSnapshot` is the live host-neutral observation shape:

```text
RunSnapshot
  run_id
  workflow_id
  workflow_version
  state
  node_states
  diagnostics
  artifacts
  started_at
  updated_at
```

`RunSummary` is the completed/terminal observation shape:

```text
RunSummary
  run_id
  workflow_id
  workflow_version
  state
  diagnostics
  artifacts
  started_at
  finished_at
```

Snapshots and summaries must not expose `RunValueStore`, backend tensor payloads, loaded model payloads, or mutable session handles.

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

`RunValueStore` should store `Arc<RuntimeValue>` handles. Large tensor buffers and loaded model payloads must not be copied into `RunValueStore`; they remain in backend-owned stores and are referenced by lightweight handles such as `BackendTensorHandle` or `RuntimeModelHandle`.

These values are not owned by app state and are not exposed to UI. Host state keeps run handles, summaries, diagnostics, and artifact references only.

`core::model::NodeValue` remains the public semantic value model. It is not the runtime store's primary representation.

Node executors should not receive the whole `RunValueStore`. Runtime resolves graph dependencies and passes only the node's inputs through `NodeExecutionContext`.

`NodeExecutorRegistry` is keyed by `core::model::NodeTypeId`. The node catalog defines node metadata; the executor registry defines how a node type runs. Do not merge those responsibilities.

V1 `NodeExecutor` may use `async-trait` for a readable async trait-object boundary. If profiling later shows this boundary matters, it can be replaced with boxed futures without changing the runtime's public run/plan API.

V1 scheduling executes stages in order and may run nodes within a stage concurrently. Deterministic stage order and deterministic event/snapshot semantics are still required even when same-stage work is concurrent.

## Backend Resource Lifecycle

Runtime owns run dependency lifetime. Backend capabilities own real model/tensor/device memory lifetime.

```text
Runtime owns:
  RunValueStore
  OutputKey -> Arc<RuntimeValue>
  dependency order
  stage/last-use knowledge
  run cancellation and cleanup timing

Backend owns:
  loaded model payloads
  tensor payloads
  device allocations
  model cache policy
  tensor cache policy
  device transfer/offload policy
  memory budget and eviction
```

Runtime should not directly unload a specific model, move a tensor to a device, compact a memory pool, or decide GPU/CPU placement. It should call a thin resource capability that communicates lifecycle intent:

```text
RunResourceBackend
  begin_run(run_id)
  release_runtime_value(run_id, value)
  cleanup_run(run_id)
  memory_snapshot()
```

The backend decides the mechanism:

- `begin_run` may create run-scoped allocation state or pin already loaded models.
- `release_runtime_value` may release a tensor immediately, decrement a backend refcount, or keep it in a pool.
- `cleanup_run` releases run-scoped tensor payloads and run pins. Cached models may remain loaded according to backend policy.
- `memory_snapshot` returns backend-specific memory/cache observations for diagnostics and UI, without giving runtime ownership of backend internals.

V1 may keep all intermediate tensor handles until run cleanup. The runtime type shape should still allow future last-use analysis to call `release_runtime_value` earlier after the last downstream consumer completes.

Model loading remains executor/backend capability work:

```text
checkpoint_loader executor
  -> backend model capability get_or_load(...)
  -> RuntimeValue::Model(handle)
```

Runtime stores and routes the returned handle, but it does not resolve manifests or own the loaded payload.

Device placement and transfer are backend policy decisions surfaced through handles, run events, diagnostics, or memory snapshots. Runtime may pass cancellation and context to executors, but backend capabilities decide whether CPU/GPU transfer, offload, or eviction is allowed.

Run cleanup:

```text
completed/failed/cancelled
  -> persist artifacts as needed
  -> cleanup backend run resources
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
