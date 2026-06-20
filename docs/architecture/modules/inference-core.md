# Inference Core Module Architecture

> Status: proposed
> Crate: `crates/inference-core`

## Role

`inference-core` is the low-level inference execution contract crate. It
defines the canonical execution value envelope, backend-affine handles, typed
capability protocol, inference runtime router, and concrete inference backend
traits.

It must not own workflow scheduling, node orchestration, concrete backend
payloads, host DTOs, workflow JSON, or UI/Agent observation shapes.

## Responsibilities

- Define `InferenceRuntime`, the executor-facing router trait.
- Define `InferenceBackend`, the concrete backend adapter trait.
- Define `InferenceBackendRegistry` keyed by `BackendKind`.
- Own `ExecutionValue`, `ExecutionValueKind`, `ExecutionConditioning`, and
  backend-affine tensor/model/image handles.
- Define typed capability request/response DTOs.
- Define backend capability reports.
- Define model resolver handoff DTOs and traits.
- Define bridge/transfer policy traits.
- Define inference errors and diagnostic projection helpers.

## Non-Responsibilities

- Runtime scheduling, cancellation, run store, snapshots, or summaries.
- Built-in node executor implementation.
- Workflow graph validation.
- Host-facing execution DTOs, run snapshots, run summaries, or workflow JSON.
- Concrete Candle, ONNX, remote, or Comfy implementation.
- Model manifest scanning or persistence.
- Tauri IPC, Axum routes, UI state, or Agent policy.

## Dependency Direction

```text
inference-core -> core

inference-core must not -> runtime
inference-core must not -> inference
inference-core must not -> inference-backends/*
inference-core must not -> model-manager
inference-core must not -> app-host
```

Consumers:

```text
inference -> inference-core
inference-backends/candle -> inference-core
app-host -> inference-core
```

Runtime-facing code should consume these contracts through the `inference`
facade (`runtime -> inference -> inference-core`). A temporary direct
`runtime -> inference-core` dependency is acceptable only during the
inference-runtime-boundary migration, before the node executor contract is
moved out of `runtime`.

This keeps backend data flow and crate dependency direction separate.
`inference-core` may depend on lightweight `core::model` ids, params, artifact
refs, and run metadata, but `core` must not depend back on inference execution
types. Concrete backend crates depend on `inference-core`; `inference-core`
does not depend on any concrete backend crate.

## Internal Execution Values

The shared execution value envelope is owned by `inference-core` and
re-exported by `inference` as the runtime-facing facade:

```text
inference_core::ExecutionValue
inference_core::BackendKind
inference_core::BackendPayloadKey
inference_core::BackendTensorHandle
inference_core::BackendTensorMetadata
inference_core::RuntimeModelHandle
inference_core::RuntimeClipHandle
inference_core::RuntimeVaeHandle
inference_core::RuntimeLatent
inference_core::ExecutionConditioning
inference_core::ConditioningMetadata
inference_core::RuntimeImage
```

The canonical architecture name is `ExecutionValue`. A temporary
`RuntimeValue` alias is acceptable only as a migration aid while moving old
runtime-facing imports.

Reason:

- `runtime` stores and passes these values through the `inference` facade.
- `inference` must inspect them when constructing requests.
- backend adapters must construct them as opaque handles to backend-owned
  payloads.
- host observations must not expose them as external DTOs.

Concrete backend types such as `CandleTensor`, `LoadedSdxlBundle`, tokenizer
state, scheduler graphs, and backend-local store keys must not cross this
execution value boundary.

`ExecutionValue` is not workflow file content, not a run event payload, not an
Axum/Tauri DTO, and not an Agent tool result. Host-safe observations remain
snapshots, summaries, diagnostics, node/run states, artifact references, and
memory/cache summaries projected through explicit observation shapes.

`ExecutionConditioning` belongs to this execution-value set. Requests may carry
it, routers may inspect its handles and metadata for compatibility, and
backends may use its payload keys to resolve backend-owned conditioning
tensors. Its metadata stays inside `ExecutionConditioning` for V1. Existing
code may temporarily expose the old `RuntimeConditioning` name as a
compatibility alias during migration.

## Inference Runtime / Router

Executors depend on an executor-facing `InferenceRuntime`, not on a single
concrete backend.

```text
InferenceRuntime
  load_bundle(...)
  text_encode(...)
  create_empty_latent(...)
  diffusion_sample(...)
  latent_decode(...)
  image_save(...)
  image_preview(...)
```

The router owns:

- backend selection from handle affinity and capability support;
- validation that request handles are compatible;
- explicit bridge/transfer policy;
- structured inference diagnostics;
- dispatch to the selected `InferenceBackend`.

V1 must implement a registry-backed router even when only one backend is
registered. Do not add a separate single-backend router special case. A
single-backend workspace is just a registry with one backend.

Suggested concrete shape:

```text
DefaultInferenceRuntime
  registry: Arc<InferenceBackendRegistry>
  bridge_policy: Arc<dyn BackendBridgePolicy>

impl InferenceRuntime for DefaultInferenceRuntime
  validate request handles
  choose backend from explicit request target or handle affinity
  ask bridge policy before any cross-backend transfer
  dispatch to Arc<dyn InferenceBackend>
```

`InferenceRuntime` and `InferenceBackend` are intentionally not equivalent
interfaces, even when they expose the same capability method names.

```text
InferenceRuntime
  executor-facing router interface
  validates public handles and request invariants
  selects a backend from explicit target / handle affinity / capability support
  applies bridge policy before crossing backend affinity
  turns routing failures into inference diagnostics
  calls one selected InferenceBackend

InferenceBackend
  concrete backend adapter interface
  assumes the request has been routed to this backend
  resolves backend-private payload keys
  runs model graph / tensor / device implementation
  returns ExecutionValue handles or typed backend responses
```

Do not collapse these traits because a V1 workspace has only one backend. A
single-backend workspace is still routed through `InferenceRuntime`; the router
is where multi-backend readiness, bridge diagnostics, capability reports, and
future backend selection policy live. The backend trait is the adapter seam for
one implementation such as Candle, ONNX, remote inference, or a test fake.

## Backend Registry

```text
InferenceBackendRegistry
  register(kind: BackendKind, backend: Arc<dyn InferenceBackend>)
  get(kind: &BackendKind) -> Option<Arc<dyn InferenceBackend>>
  capabilities() -> merged capability report
```

`BackendKind` is a stable inference execution label used by config, runtime
handles, diagnostics, and registry lookup. It should not replace the backend
trait with a giant closed enum.

## Backend Capability Trait

`InferenceBackend` is the concrete backend adapter trait behind the router:

```text
InferenceBackend
  backend_kind()
  capabilities()
  load_bundle(...)
  text_encode(...)
  create_empty_latent(...)
  diffusion_sample(...)
  latent_decode(...)
  image_save(...)
  image_preview(...)
```

V1 should use the same `async_trait` style already used by `runtime` and
`agent` for async trait objects:

```rust
#[async_trait]
pub trait InferenceBackend: Send + Sync + 'static {
    fn backend_kind(&self) -> &BackendKind;
    fn capabilities(&self) -> InferenceBackendCapabilities;

    async fn diffusion_sample(
        &self,
        request: DiffusionSampleRequest,
    ) -> Result<DiffusionSampleResponse, InferenceError>;
}
```

## Capability Identity

`operation_id` is not part of the target execution interface.

The primary execution interface is the typed capability method itself:

```text
InferenceRuntime::diffusion_sample(DiffusionSampleRequest)
InferenceBackend::diffusion_sample(DiffusionSampleRequest)
```

Do not route V1 execution through:

```text
execute(InferenceRequest { operation_id, inputs, params })
match operation_id
```

The remaining stable identity concept should be a capability label/kind used
for diagnostics, capability reports, tracing, and bridge policy context:

```text
InferenceCapability
  LoadBundle
  TextEncode
  CreateEmptyLatent
  DiffusionSample
  LatentDecode
  ImageSave
  ImagePreview
```

`InferenceCapability` may render to strings such as `diffusion.sample` for logs
or external diagnostics, but it must not be the runtime/backend dispatch key.
Generic `InferenceRequest`, `InferenceResponse`, and `InferenceOperationId`
may exist only as migration shims while old code is moved to typed DTOs. New
executor and backend code should not add fields that require a caller to set an
operation id correctly before a typed method can run.

## Typed Requests And Responses

Typed requests own cheap, shareable handles rather than borrowing from
`runtime::NodeExecutionContext`. This keeps backend calls simple across
`.await` while preserving zero-copy behavior for tensors and loaded models,
because large data remains in backend-owned stores referenced by execution
handles.

Example DTOs:

```text
LoadBundleRequest
  resolved_model: ResolvedInferenceModel
  run_id
  node_id
  correlation_id

TextEncodeRequest
  clip: RuntimeClipHandle
  text: String
  run_id
  node_id

DiffusionSampleRequest
  model: RuntimeModelHandle
  positive: ExecutionConditioning
  negative: ExecutionConditioning
  latent: RuntimeLatent
  seed / steps / cfg / sampler / scheduler / denoise
  run_id
  node_id
```

Typed requests do not carry `operation_id`; the method call already identifies
the capability. They may carry explicit target/backend preferences only when
that is a routing decision rather than an operation identifier.

Typed responses return execution handles or artifact intents. They should not
carry output `SlotId`; slot mapping belongs to the inference executor that knows
which workflow node it is running.

## Bridge Policy

Cross-backend transfer is allowed only through explicit bridge capability.

```text
BackendBridgePolicy
  plan_transfer(source, target_backend, context) -> BridgePlan

BackendBridge
  can_transfer(source, target_backend) -> BridgeSupport
  transfer(source, target_backend, context) -> ExecutionValue
```

The router may use bridges to normalize a request before calling a backend. It
must not silently reinterpret a `BackendPayloadKey` from one backend as another
backend's payload key.

V1 defines the bridge interfaces and ships `RejectAllBridgePolicy` as the
default. The router must fail explicitly when a request would require
cross-backend transfer. It must not silently copy, reinterpret, or coerce
backend payload keys.

Minimum bridge diagnostics:

```text
BackendBridgeRequired
  source_backend
  target_backend
  value_kind
  capability

BackendBridgeUnsupported
  source_backend
  target_backend
  value_kind
  capability
  reason
```

These diagnostics belong to `inference-core` error/diagnostic projection, not
runtime. Runtime may surface them through node failure events, but it does not
decide bridge policy.

## Resource And Device Policy Handoff

`inference-core` routes typed capability calls, but it is not a memory manager.
It should carry enough public context for routing and diagnostics while leaving
real resource policy to the selected backend.

```text
InferenceRuntime
  validates backend affinity on handles
  chooses the backend for a typed capability call
  asks bridge policy before cross-backend/device transfer
  forwards request context and cancellation/provenance metadata

InferenceBackend
  owns loaded model cache and tensor payload store
  pins/unpins backend-private resources according to backend policy
  decides CPU/GPU placement, offload, transfer, and eviction
  returns public ExecutionValue handles or typed responses
```

For multi-image generation, the router may repeatedly route `diffusion_sample`
to the same backend because the `Model` and `Conditioning` handles have the
same backend affinity. It does not decide that the diffusion model stays loaded
between samples; that is a backend cache/pinning decision informed by runtime
lifecycle intent and backend policy.

If `latent_decode` is routed to a different backend or device than
`diffusion_sample`, `InferenceRuntime` must require an explicit bridge plan.
It must not silently reinterpret a latent payload key produced by one backend
as a payload key for another backend.

## Model Resolver Handoff

`checkpoint_loader` receives a workflow `ModelRef`. `inference-core` defines the
resolver trait and resolved DTO shape consumed by `LoadBundleRequest`.

```text
ModelRef
  -> app-host/model-manager adapter
  -> ResolvedInferenceModel
  -> LoadBundleRequest
```

`ResolvedInferenceModel` should reuse stable `core::model` semantics such as
`ModelId`, `ModelSeries`, `ModelVariant`, and `ModelRole`, while leaving
manifest-only details inside `model-manager`.

## Suggested Module Layout

```text
src/
  lib.rs
  error.rs
  diagnostic.rs
  capability.rs
  backend.rs
  runtime.rs
  registry.rs
  bridge.rs
  resolver.rs
  request.rs
  response.rs
  request/
    model.rs
    text.rs
    latent.rs
    diffusion.rs
    image.rs
  response/
    model.rs
    text.rs
    latent.rs
    diffusion.rs
    image.rs
```

Use modern Rust module layout. Do not introduce `mod.rs`, and keep `lib.rs` as a
facade with explicit re-exports.

`runtime.rs` should contain the executor-facing router trait and the default
router implementation. `backend.rs` should contain only the concrete backend
adapter trait. `registry.rs` should own registration and lookup. `bridge.rs`
should own bridge policy, bridge plans, and reject-all behavior. Request and
response modules should stay typed by capability, not by workflow node type.
`capability.rs` should own capability report types and the optional diagnostic
label/kind, not a stringly execution dispatcher.
