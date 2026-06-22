# Runtime Module Architecture

> Status: working draft
> Crate: `crates/runtime`

## Role

`runtime` is the host-independent workflow execution layer. It executes
`core::ExecutionPlan` values using the `inference` executor facade, manages run
sessions, schedules stages, handles cancellation, routes artifacts through
injected artifact capabilities, and emits `core::RunEvent` values through a
host-provided sink.

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
runtime -> inference
runtime must not -> tauri
runtime must not -> axum
runtime must not -> model-manager
runtime must not -> inference-backends/*
```

Concrete node executors and backend capabilities are assembled by `app-host`.
Runtime should consume executor contracts, execution values, execution outputs,
and backend handle types through the `inference` facade. The former physical
`crates/inference-core` crate has been folded into `crates/inference`.

Runtime must not depend on concrete backend crates.

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
  store.rs
```

Use modern Rust module layout. Do not introduce `mod.rs`, and prefer ordinary `mod foo;` declarations over `#[path = "..."]` attributes.

Runtime imports executor contracts, node context, execution outputs, and
retention policies from the `inference` facade. Runtime must not define the
canonical value enum or own the node executor contract. A temporary
`RuntimeValue` alias may exist only as a migration shim while downstream crates
are updated.

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

`runtime` does not know whether a `NodeExecutor` is hand-written,
inference-backed, Candle-backed, or remote. It schedules nodes and calls the
`inference::NodeExecutor` trait object. Built-in inference executors are
provided by `inference` and assembled by `app-host`:

```text
runtime scheduler
  -> inference::NodeExecutor::execute(NodeExecutionContext)
  -> inference-backed executor
  -> typed inference capability call
  -> selected backend typed capability method
  -> inference::ExecutionOutput values
```

This prevents `inference` from becoming a second runtime. `runtime` owns run
state, scheduling, cancellation, value storage, snapshots, summaries, and run
events. `inference` owns only built-in node orchestration over abstract
handles. The typed backend capability boundary lives in `inference`, not
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
graph. Model identity enters runtime only as opaque `inference::ExecutionValue`
handles flowing between node invocations.

Asynchronous and parallel scheduling are orthogonal to this boundary. The
scheduler may run independent node invocations concurrently, await async
executors, cancel in-flight siblings, or release values after last use. Those
policies change when node invocations are called, not what the execution unit
is. Runtime must not turn `text_encode`, `diffusion_sample`, SDXL CLIP, or a
backend kernel graph into its scheduling unit.

Runtime does not select inference backends per node. Backend selection,
resource compatibility, and explicit transfer/bridge policy belong to the
inference runtime/router assembled by app-host. Runtime executes the registry
it was given, passes opaque execution handles into `NodeExecutionContext`, and
does not perform implicit cross-backend tensor conversion.

Runtime may own high-level policy inputs such as run priority, cancellation,
budget hints, target selection, and scheduling pressure. Those inputs may be
projected by app-host into inference router configuration, but runtime still
does not call concrete backends. Backend selection is applied by the
`InferenceRuntime` router:

```text
runtime
  -> executes workflow node
  -> dyn inference::NodeExecutor
  -> InferenceRuntime typed capability call
  -> router applies BackendSelectionPolicy / BackendBridgePolicy
  -> selected backend
```

Model loading follows the same rule. Runtime executes the checkpoint-loader
node; the inference executor constructs `LoadBundleRequest`; the router chooses
the backend when no backend-bound handles exist; returned `Model`, `Clip`, and
`Vae` handles carry the selected `BackendInstance`. Later nodes are constrained by
those handle affinities unless explicit bridge policy permits transfer.

Runtime lifecycle is driven by `Arc<ExecutionValue>` ownership and
producer-declared retention policies. Runtime should not introduce a separate
backend release-intent protocol for ordinary value lifecycle. Backend caches may
remain live by holding their own handles or internal owners.

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
  values: OutputKey -> RuntimeValueRecord

OutputKey
  node_id
  slot_id

RuntimeValueRecord
  value: Arc<ExecutionValue>
  retention: ExecutionValueRetention
```

`inference::ExecutionValue` may contain backend-affine handles for cheap in-run
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

`RunValueStore` should store `Arc<ExecutionValue>` handles with an
`ExecutionValueRetention` declared by the producer. Large tensor buffers and
loaded model payloads must not be copied into `RunValueStore`; they remain in
backend-owned stores and are referenced by lightweight handles such as
`BackendTensorHandle` or `RuntimeModelHandle`.

`ExecutionConditioning` is one of the `inference::ExecutionValue` variants. Its
current `metadata` lives with the conditioning value and should be treated as
internal execution context rather than a separate runtime subsystem. Existing
code may temporarily expose the old `RuntimeConditioning` name as a
compatibility alias during migration.

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

Runtime owns run dependency lifetime. Backend capabilities own real
model/tensor/device memory lifetime.

```text
Runtime owns:
  RunValueStore
  OutputKey -> Arc<ExecutionValue> + ExecutionValueRetention
  dependency order
  producer-declared retention enforcement
  run cancellation and cleanup timing

Backend implements concrete mechanisms for:
  loaded model payloads
  tensor payloads
  device allocations
  model cache ownership
  tensor cache ownership
  device transfer/offload execution
  memory observation and local eviction mechanics
```

Runtime should not directly unload a specific model, move a tensor to a device,
compact a memory pool, or decide GPU/CPU placement. Ordinary value lifecycle is
managed by inserting and dropping `Arc<ExecutionValue>` records according to
retention:

```text
SingleUse
  dropped after its unique consumer completes
  fan-out means edge-sourced consumers in the active execution plan
  fan-out greater than one fails the run when the value is produced
  fan-out zero is kept until cleanup in V1

RunScoped
  kept until terminal run cleanup

WorkspaceScoped
  may be passed during the run
  dropping the run reference must not imply workspace cache eviction
```

If a backend or workspace cache needs a value to outlive the run, it must hold
its own internal resource owner. Runtime dropping its run-scoped reference is
not an imperative unload command. Memory/cache observations should be surfaced
through explicit host-neutral snapshot/diagnostic shapes rather than backend
internals.

Runtime scheduling and backend resource scheduling are separate concerns. A
workflow node invocation remains the runtime execution unit, while model
pinning, tensor residency, CPU/GPU placement, offload, and eviction remain
backend-owned mechanisms. Runtime or a future resource coordinator owns global
resource policy because it has the active-run, execution-plan, and multi-backend
view that individual backends do not have. That coordinator should communicate
through backend mechanism traits defined in `inference`, not through
concrete backend types and not by interpreting backend-private payloads.

The mechanism contract is plugin-aligned through backend instances:

```text
app-host
  -> static PluginExtension { extends: HostSurface::InferenceBackend }
  -> constructs BackendInstanceDescriptor { plugin, extension, backend, instance }
  -> registers typed InferenceBackend adapter
  -> registers resource mechanism adapter for the same BackendInstance

runtime
  -> calls coarse lifecycle/observation trait object supplied by app-host
  -> never loads, unloads, moves, pins, or frees a concrete payload
```

There is no separate `HostSurface::ResourceBackend` in V1. Resource mechanisms
are part of an inference backend instance's host wiring. This keeps plugin
identity, backend selection, and resource observation aligned around the same
`BackendInstance` unit.

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
  -> concurrency is controlled by scheduler policy and backend mechanisms

vae_decode x N
  -> may start as soon as each sampled latent is ready
  -> may run on a different device or backend, such as CPU VAE decode
  -> cross-backend/device transfer must go through explicit bridge policy

save_image / preview_image x N
  -> emits an artifact event as each image completes
  -> UI does not need to wait for the whole run to finish before showing output
```

This requires the scheduler to honor retention and progressive completion
without taking ownership of backend memory mechanisms:

```text
Scheduler / RunSession
  stores producer-declared retention for each ExecutionValue
  can drop SingleUse values after their unique consumer completes
  can keep RunScoped values until terminal cleanup
  can emit artifact observations when a target/save/preview node completes
  can keep deterministic run state while independent nodes execute concurrently

inference backend / workspace cache
  keeps any workspace-scoped resources it owns
  implements model pinning, pooling, eviction, and device placement mechanics
  reports whether VAE decode can run on CPU while diffusion sampling continues
```

The scheduler may drop a run-owned `Clip` handle while `Conditioning` remains
live, or keep a `Model` handle across later KSampler nodes. Dropping runtime's
`Arc` does not directly unload CLIP, UNet, VAE, tensors, or device buffers; the
backend/cache lifetime follows its own ownership.

Model loading remains inference/backend adapter work:

```text
checkpoint_loader executor
  -> typed inference capability call
  -> backend load_bundle(...)
  -> inference::ExecutionValue::Model(handle)
```

Runtime stores and routes the returned handle, but it does not resolve
manifests, inspect the resolved model descriptor, or own the loaded payload.
Only the inference backend knows what concrete model object, graph, tokenizer,
weights, or device allocations sit behind that handle.

Device placement and transfer are coordinated through inference router policy
and backend mechanism contracts, then surfaced through handles, run events,
diagnostics, or memory snapshots. Runtime may pass cancellation and context to
executors, but it must not directly command concrete CPU/GPU transfer, offload,
or eviction.

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

## Remaining Runtime Work

The old `runtime/05` planning slice is split conceptually:

```text
runtime/05a progressive artifact output
  save/preview artifacts become observable as each node completes

runtime/05b scheduler concurrency foundation
  same-stage independent node invocations may run concurrently while preserving
  deterministic event/snapshot semantics and fail-fast cancellation

runtime/05c resource observation integration
  runtime/app-host can collect backend-instance resource snapshots for
  diagnostics and future policy without direct backend memory commands
```

Resource coordination policy comes after these slices. It should be based on
backend-neutral observations, run priorities, active plans, and configured
budgets. It should not start by adding per-value release or pin/offload
commands to the ordinary runtime lifecycle.

## Model Resolution and Loading

Runtime does not scan model directories, own the manifest, or depend on `model-manager`. Model resolution and backend loading are provided through node executors or runtime capabilities assembled by `app-host`.

Execution flow:

```text
Workflow ModelRef
  -> model-manager readiness resolver reports availability diagnostics
  -> app-host builds execution plan and node executor adapters
  -> checkpoint node executor calls typed inference capability
  -> inference backend resolves concrete loaded model object
  -> inference::ExecutionValue::Model / Clip / Vae
```

The loaded backend payload stays in the backend model store owned by the
inference backend. Runtime passes typed handles between nodes and never learns
which concrete model architecture or checkpoint was loaded.
