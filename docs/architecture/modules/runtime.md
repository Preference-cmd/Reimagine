# Runtime Module Architecture

> Status: working draft
> Crate: `crates/runtime`

## Role

`runtime` is the host-independent workflow execution layer. It executes `core::ExecutionPlan` values using injected node capabilities, manages run sessions, schedules stages, handles cancellation, routes artifacts through injected artifact capabilities, and emits `core::RunEvent` values through a host-provided sink.

It must not depend on Tauri or Axum.

## Responsibilities

- Execute prepared `core::ExecutionPlan` values through `RuntimeService` / `ExecutionRunner`.
- Own `RunSession` and run state.
- Schedule DAG stages. The target architecture supports same-stage
  parallelism, but the current runner remains sequential.
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
runtime must not -> inference
runtime must not -> inference-core
runtime must not -> inference-backends/*
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

The migration target is that `store.rs`, `node_context.rs`, `executor.rs`, and
`resources.rs` refer to `core::ExecutionValue`. They must not define the
canonical value enum themselves. A temporary `RuntimeValue` alias may exist
only as a migration shim while downstream crates are updated.

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

When a node in a stage fails, and especially once same-stage parallelism is
introduced:

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

Current implementation note: `scheduler.rs` owns the small
`StageExecutionPolicy` used by the sequential runner to decide whether a
workflow node invocation should execute or be skipped after the first failure.
The policy is intentionally backend-neutral and does not touch value stores,
artifact stores, or node executor internals. Future same-stage concurrency can
deepen this module without changing the public `RuntimeService::run` seam.

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

`runtime` does not know whether a `NodeExecutor` is hand-written, inference
backed, Candle backed, or remote. It schedules nodes and calls the
`NodeExecutor` trait object. Inference-backed executors are adapters outside
runtime:

```text
runtime scheduler
  -> NodeExecutor::execute(NodeExecutionContext)
  -> inference-backed executor
  -> typed inference-core capability call
  -> selected backend typed capability method
  -> core::ExecutionValue outputs
```

This prevents `inference` from becoming a second runtime. `runtime` owns run
state, scheduling, cancellation, value storage, snapshots, summaries, and run
events. `inference` owns only built-in node orchestration over abstract
handles. The typed backend capability boundary lives in `inference-core`, not
in `runtime`.

Runtime's execution unit is a workflow node invocation:

```text
ExecutionNode
  node_id
  type_id
  input bindings
  params
```

The scheduler invokes one `NodeExecutor` for each execution node. It does not
know whether that node uses SDXL, Flux, a remote backend, or a future fused
graph. Model identity enters runtime only as opaque `ExecutionValue` handles
flowing between node invocations.

Asynchronous and parallel scheduling are orthogonal to this boundary. The
scheduler may run independent node invocations concurrently, await async
executors, cancel in-flight siblings, or release values after last use. Those
policies change when node invocations are called, not what the execution unit
is. Runtime must not turn `text_encode`, `diffusion_sample`, SDXL CLIP, or a
backend kernel graph into its scheduling unit.

Runtime does not select inference backends per node. Backend selection,
resource compatibility, and explicit transfer/bridge policy belong to the
inference runtime/router assembled by app-host. Runtime executes the registry
it was given, passes opaque runtime handles into `NodeExecutionContext`, and
does not perform implicit cross-backend tensor conversion.

`RunResourceBackend` remains a runtime lifecycle hook, not the inference
execution API. A concrete backend may implement both `RunResourceBackend` and
the inference backend trait, but runtime only sees `RunResourceBackend`.

## Review Notes

As of 2026-06-15, the public runtime seam is still correct. The current runner
implementation should be treated as a sequential scheduler with a small
scheduler-owned fail-fast policy, not as the final DAG-parallel implementation
described above.

Follow-up candidates:

- deepen the internal scheduler module further so stage concurrency, fail-fast
  sibling cancellation, cancellation checks, and snapshot cadence live behind
  one implementation;
- keep `RuntimeService::run` plan-oriented even if the internal scheduler later
  compiles plans into lower-level execution units.

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

`RunSnapshot.artifacts` and `RunSummary.artifacts` carry host-neutral
`RunArtifactRef` values:

```text
RunArtifactRef
  id
  node_id
  reference
```

`reference` is a `core::model::ArtifactRef`, such as a workspace-relative
output path. It is not a backend tensor payload, image buffer, or absolute
filesystem capability.

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
  values: OutputKey -> ExecutionValue

OutputKey
  node_id
  slot_id
```

`core::ExecutionValue` may contain backend-affine handles for cheap in-run
sharing:

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

`RunValueStore` should store `Arc<core::ExecutionValue>` handles. Large tensor
buffers and loaded model payloads must not be copied into `RunValueStore`; they
remain in backend-owned stores and are referenced by lightweight handles such
as `BackendTensorHandle` or `RuntimeModelHandle`.

`ExecutionConditioning` is one of the `core::ExecutionValue` variants. Its
current `metadata` lives with the conditioning value and should be treated as
public execution context rather than a separate runtime subsystem. Existing code
may temporarily expose the old `RuntimeConditioning` name as a compatibility
alias during migration.

These values are not owned by app state and are not exposed to UI. Host state keeps run handles, summaries, diagnostics, and artifact references only.

`core::model::NodeValue` remains the public semantic value model. It is not the runtime store's primary representation.

Node executors should not receive the whole `RunValueStore`. Runtime resolves graph dependencies and passes only the node's inputs through `NodeExecutionContext`.

`NodeExecutorRegistry` is keyed by `core::model::NodeTypeId`. The node catalog defines node metadata; the executor registry defines how a node type runs. Do not merge those responsibilities.

V1 `NodeExecutor` may use `async-trait` for a readable async trait-object boundary. If profiling later shows this boundary matters, it can be replaced with boxed futures without changing the runtime's public run/plan API.

Current V1 scheduling executes stages in order and runs nodes within a stage
sequentially. The architecture should still preserve deterministic stage order
and deterministic event/snapshot semantics when a follow-up scheduler deepening
issue introduces same-stage concurrency.

## Backend Resource Lifecycle

Runtime owns run dependency lifetime. Backend capabilities own real model/tensor/device memory lifetime.

```text
Runtime owns:
  RunValueStore
  OutputKey -> Arc<ExecutionValue>
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

Runtime scheduling and backend resource scheduling are separate concerns. A
workflow node invocation remains the runtime execution unit, while model
pinning, tensor residency, CPU/GPU placement, offload, and eviction remain
backend-owned policies.

For an SDXL workflow that generates multiple images from the same prompt, the
desired behavior is:

```text
checkpoint_loader
  -> backend loads or reuses one bundle
  -> returns Model / Clip / Vae handles

clip_text_encode positive / negative
  -> uses Clip once
  -> stores reusable Conditioning handles
  -> CLIP payload may become unpinned after the final text_encode consumer

empty_latent_image x N
  -> creates independent Latent handles

ksampler x N
  -> consumes the same Model and Conditioning handles
  -> consumes one Latent per sample
  -> diffusion model should stay pinned for the generation group
  -> concurrency is controlled by scheduler/backend resource policy

vae_decode x N
  -> may start as soon as each sampled latent is ready
  -> may run on a different device or backend, such as CPU VAE decode
  -> cross-backend/device transfer must go through explicit bridge policy

save_image / preview_image x N
  -> emits an artifact event as each image completes
  -> UI does not need to wait for the whole run to finish before showing output
```

This requires the scheduler to expose lifecycle intent and progressive
completion without taking ownership of backend memory policy:

```text
Scheduler / RunSession
  knows last downstream consumer for an ExecutionValue
  can release a value after its last consumer completes
  can emit artifact observations when a target/save/preview node completes
  can keep deterministic run state while independent nodes execute concurrently

RunResourceBackend / inference backend
  decides whether release means unpin, decrement refcount, move to CPU, evict,
  pool, or no-op
  decides whether a model such as the diffusion model remains pinned across a
  generation group
  decides whether VAE decode can run on CPU while diffusion sampling continues
```

The scheduler may discover that `Clip` is no longer needed while
`Conditioning` remains live, or that the diffusion model is still needed by
later KSampler nodes. It reports those lifecycle facts through resource
capabilities; it does not unload CLIP, UNet, VAE, tensors, or device buffers
directly.

Model loading remains inference/backend adapter work:

```text
checkpoint_loader executor
  -> inference-core typed capability call
  -> backend load_bundle(...)
  -> core::ExecutionValue::Model(handle)
```

Runtime stores and routes the returned handle, but it does not resolve
manifests, inspect the resolved model descriptor, or own the loaded payload.
Only the inference backend knows what concrete model object, graph, tokenizer,
weights, or device allocations sit behind that handle.

Device placement and transfer are inference runtime/backend policy decisions
surfaced through handles, run events, diagnostics, or memory snapshots. Runtime
may pass cancellation and context to executors, but the inference
runtime/router and backend capabilities decide whether CPU/GPU transfer,
offload, or eviction is allowed.

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
  -> app-host builds execution plan and node executor adapters
  -> checkpoint node executor calls inference-core typed capability
  -> inference backend resolves concrete loaded model object
  -> core::ExecutionValue::Model / Clip / Vae
```

The loaded backend payload stays in the backend model store owned by the
inference backend. Runtime passes typed handles between nodes and never learns
which concrete model architecture or checkpoint was loaded.
