# Inference Core Module Architecture

> Status: proposed
> Crate: `crates/inference-core`

## Role

`inference-core` is the backend contract crate. It defines the typed
capability protocol between inference node executors, the inference runtime
router, and concrete inference backends.

It must not own workflow scheduling, node orchestration, concrete backend
payloads, or public execution values. Public execution values and backend-affine
handles belong to `core`.

## Responsibilities

- Define `InferenceRuntime`, the executor-facing router trait.
- Define `InferenceBackend`, the concrete backend adapter trait.
- Define `InferenceBackendRegistry` keyed by `core::BackendKind`.
- Define typed capability request/response DTOs.
- Define backend capability reports.
- Define model resolver handoff DTOs and traits.
- Define bridge/transfer policy traits.
- Define inference errors and diagnostic projection helpers.

## Non-Responsibilities

- Runtime scheduling, cancellation, run store, snapshots, or summaries.
- Built-in node executor implementation.
- Workflow graph validation.
- Public execution value ownership.
- Concrete Candle, ONNX, remote, or Comfy implementation.
- Model manifest scanning or persistence.
- Tauri IPC, Axum routes, UI state, or Agent policy.

## Dependency Direction

```text
inference-core -> core

runtime must not -> inference-core
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

This keeps backend data flow and crate dependency direction separate. A backend
may construct `core::ExecutionValue` handles and return them through
`inference-core` responses, but `inference-core` does not depend on that backend
crate and `runtime` does not depend on `inference-core`.

## Public Values Stay In Core

The shared execution value envelope is owned by `core`, not `inference-core`:

```text
core::ExecutionValue
core::BackendKind
core::BackendPayloadKey
core::BackendTensorHandle
core::RuntimeModelHandle
core::RuntimeClipHandle
core::RuntimeVaeHandle
core::RuntimeLatent
core::ExecutionConditioning
core::RuntimeImage
```

The canonical architecture name is `ExecutionValue`. A temporary
`RuntimeValue` alias is acceptable only as a migration aid while moving the
current implementation out of `crates/runtime`.

Reason:

- `runtime` must store and pass these values without depending on inference.
- `inference` must inspect them when constructing requests.
- backend adapters must construct them as opaque handles to backend-owned
  payloads.
- host observations may expose selected artifact/value metadata.

Concrete backend types such as `CandleTensor`, `LoadedSdxlBundle`, tokenizer
state, scheduler graphs, and backend-local store keys must not cross this
public value boundary.

`ExecutionConditioning` belongs to this public execution-value set. It is not
an `inference-core` DTO. `inference-core` requests may carry it, routers may
inspect its public handles and metadata for compatibility, and backends may use
its payload keys to resolve backend-owned conditioning tensors. Its metadata
stays inside `ExecutionConditioning` for V1. Existing code may temporarily
expose the old `RuntimeConditioning` name as a compatibility alias during
migration.

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

## Backend Registry

```text
InferenceBackendRegistry
  register(kind: BackendKind, backend: Arc<dyn InferenceBackend>)
  get(kind: &BackendKind) -> Option<Arc<dyn InferenceBackend>>
  capabilities() -> merged capability report
```

`BackendKind` is a stable core-owned label used by config, runtime handles,
diagnostics, and registry lookup. It should not replace the backend trait with a
giant closed enum.

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

## Typed Requests And Responses

Typed requests own cheap, shareable handles rather than borrowing from
`runtime::NodeExecutionContext`. This keeps backend calls simple across
`.await` while preserving zero-copy behavior for tensors and loaded models,
because large data remains in backend-owned stores referenced by core handles.

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

Typed responses return public core handles or artifact intents. They should not
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
  operation

BackendBridgeUnsupported
  source_backend
  target_backend
  value_kind
  operation
  reason
```

These diagnostics belong to `inference-core` error/diagnostic projection, not
runtime. Runtime may surface them through node failure events, but it does not
decide bridge policy.

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
