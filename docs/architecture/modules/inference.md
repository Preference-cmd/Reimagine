# Inference Module Architecture

> Status: working draft
> Crate: `crates/inference`

## Role

`inference` is the backend-neutral node orchestration layer for built-in
generation nodes. It defines the runtime-facing executor contract/facade and
implements built-in executors that map a workflow node invocation to typed
backend capability calls.

It does not define the backend contract itself. Shared backend traits, typed
capability request/response DTOs, backend registry, router, bridge policy, and
inference errors belong to [`inference-core`](inference-core.md).

## Responsibilities

- Provide built-in inference-backed node executors.
- Define runtime-facing node executor contracts, node execution context, output
  contracts, and executor registration helpers.
- Convert node execution context inputs and params into typed `inference-core`
  requests.
- Call the injected `inference-core::InferenceRuntime` router.
- Validate typed responses against the node's output slots.
- Record save/preview artifacts through injected artifact capabilities.
- Provide executor registration helpers for app-host bootstrap.
- Re-export `inference-core` execution values and handles as the facade runtime
  consumes.

## Non-Responsibilities

- Runtime scheduling, cancellation, run state, snapshots, or value-store
  ownership.
- Backend trait definitions, backend registry, bridge policy, or capability
  DTO ownership.
- Concrete Candle, ONNX, remote, or Comfy implementations.
- Model manifest scanning or persistence.
- Tauri IPC, Axum routes, or UI state.
- Agent policy.

## Dependency Direction

```text
inference -> core
inference -> inference-core

inference must not -> inference-backends/*
inference must not -> runtime
inference must not -> model-manager
inference must not -> app-host
inference must not -> tauri
inference must not -> axum
```

`runtime` depends on `inference` as its executor/value facade. `app-host`
composes the pieces by constructing an `inference-core` router and asking
`inference` to register node executors into the executor registry consumed by
runtime.

## Node Executor Contract

The executor contract — `NodeExecutor` trait, `NodeExecutorError`,
`NodeExecutionContext` / `NodeInputs` / `NodeParams`,
`NodeExecutorRegistry` (with `BoxedNodeExecutor`,
`NodeExecutorRegistryError`), plus the `ArtifactPublisher` /
`NodeCancellation` abstractions and the `ArtifactEventKind` enum — is
owned by `inference` (this crate). The runtime consumes the contract
and provides the concrete impls:

- `runtime::CancellationToken` implements `inference::NodeCancellation`.
- `runtime::RuntimeNodeArtifactCapability` implements
  `inference::ArtifactPublisher` and holds the runtime-owned
  `ArtifactStore` and `RunEventSink`.

The runner task constructs each `NodeExecutionContext` by wrapping a
fresh `CancellationToken` and `RuntimeNodeArtifactCapability` in
`Arc<dyn NodeCancellation>` and `Arc<dyn ArtifactPublisher>` and
hands the context to `dyn NodeExecutor::execute`. Executors never see
the runtime's concrete artifact or cancellation types.

`runtime::NodeExecutor`, `runtime::NodeExecutionContext`,
`runtime::NodeExecutorRegistry`, `runtime::ArtifactEventKind`, and
the other moved types are re-exported from `reimagine_inference` for
backward compatibility with call sites that pre-date the inversion.

## Boundary

`runtime` owns the execution loop. `inference` owns node orchestration.
`inference-core` owns the backend contract. Concrete backends own payloads,
model graphs, tensors, and device policy.

```text
runtime scheduler
  -> inference::NodeExecutor::execute(NodeExecutionContext)
  -> inference executor
  -> inference-core typed request
  -> inference-core InferenceRuntime/router
  -> selected inference-core InferenceBackend method
  -> backend-private model graph / payload store
  -> inference::ExecutionValue outputs
```

The runtime execution unit remains a workflow node invocation. Backend
capability calls are not runtime scheduling units.

`inference` calls the executor-facing `InferenceRuntime` router, not
`InferenceBackend` directly. The two traits are deliberately not equivalent:
the runtime/router trait validates handles, selects a backend, applies bridge
policy, and emits routing diagnostics; the backend trait is the concrete
adapter seam for one backend implementation. Inference executors should depend
on the router trait even if a workspace currently registers only Candle.

## Execution Value Usage

`inference` consumes and returns execution values owned by `inference-core` and
re-exported by `inference`.

```text
inference::ExecutionValue
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

`inference` may inspect the public handle shape, such as `BackendKind`,
`BackendPayloadKey`, tensor shape, dtype, model role, and device label. It must
not inspect backend-local tensors, loaded model objects, tokenizer state, or
kernel graphs.

`ExecutionValue` is internal execution data. It must not be exposed through
workflow JSON, run snapshots, run summaries, run events, Axum/Tauri DTOs, or
Agent tool results.

## Executor Shape

Each executor should be explicit and small:

- read required inputs and params from `NodeExecutionContext`;
- build a typed `inference-core` request;
- call the corresponding `InferenceRuntime` capability method;
- validate response value kinds and required outputs;
- map typed responses into `ExecutionOutput` / `NodeExecutionOutputs`;
- record artifacts through injected artifact capabilities when needed.

The initial executor set remains:

```text
builtin.string
builtin.checkpoint_loader
builtin.clip_text_encode
builtin.empty_latent_image
builtin.ksampler
builtin.vae_decode
builtin.save_image
builtin.preview_image
```

Executor registration belongs in a narrow API such as:

```text
register_builtin_inference_executors(
  registry,
  Arc<dyn inference_core::InferenceRuntime>,
  Arc<dyn inference_core::ModelResolver>,
)
```

The registration helper must not require `app-host`, `model-manager`, or a
concrete backend crate.

## KSampler Example

```text
core::NodeCatalog
  get("builtin.ksampler") -> NodeDef

inference::NodeExecutorRegistry
  get("builtin.ksampler") -> Arc<dyn NodeExecutor>

inference::KSamplerExecutor
  reads: Model + positive Conditioning + negative Conditioning + Latent
  reads: seed / steps / cfg / sampler / scheduler / denoise
  builds: DiffusionSampleRequest
  calls: InferenceRuntime::diffusion_sample(...)
  returns: ExecutionValue::Latent

inference-core::InferenceRuntime
  validates handle compatibility
  applies explicit bridge policy if available
  routes to selected backend

inference-backend
  resolves backend-private payload keys
  runs model graph / kernel adapter
  stores new payload
  returns internal ExecutionValue handle
```

Rust-shaped sketch:

```rust
pub struct KSamplerExecutor {
    inference: Arc<dyn inference_core::InferenceRuntime>,
}

#[async_trait]
impl inference::NodeExecutor for KSamplerExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<NodeExecutionOutputs, NodeExecutorError> {
        let request = DiffusionSampleRequest::from_context(&context)?;
        let response = self.inference.diffusion_sample(request).await?;
        Ok(latent_output(response))
    }
}
```

`KSamplerExecutor` may encode node-level semantics. It may not encode
SDXL-specific tensor shapes, Candle device policy, loaded bundle internals, or
future model-family branches.

## Artifact Boundary

Save/preview executors are runtime-facing adapters:

```text
image.save executor
  -> InferenceRuntime::image_save(...)
  -> validate image/artifact response
  -> NodeArtifactCapability::record(slot_id, ArtifactRef, ArtifactEventKind)
  -> ExecutionValue::Artifact(...)
```

Backend requests must not carry runtime's artifact store. Backends may encode
or write image data according to their configured output policy, but artifact
recording remains an executor/runtime concern.

## Model Resolution Handoff

Model manifest semantics stay outside `inference`.

```text
workflow ModelRef
  -> app-host ModelService / model-manager resolver
  -> inference-core ModelResolver adapter
  -> LoadBundleRequest
  -> InferenceRuntime::load_bundle(...)
  -> ExecutionValue::Model / Clip / Vae handles
```

`inference` depends on a resolver trait from `inference-core`, not directly on
`model-manager`.

## Suggested Module Layout

```text
src/
  lib.rs
  value.rs
  executor.rs
  node_context.rs
  artifacts.rs
  executors.rs
  executors/
    common.rs
    validation.rs
    string.rs
    model.rs
    text.rs
    latent.rs
    diffusion.rs
    image.rs
  registry.rs
  testing.rs
```

`lib.rs` should stay as a facade of private modules plus explicit public
re-exports. Do not collect executor logic into one large file.

Executor code architecture:

```text
value.rs
  re-export inference-core execution values and handles

executor.rs
  NodeExecutor
  NodeExecutorError
  ExecutionOutput
  ExecutionValueRetention
  NodeExecutionOutputs
  NodeExecutorRegistry

node_context.rs
  NodeExecutionContext
  NodeInputs
  NodeParams
  cancellation facade
  artifact capability facade

executors.rs
  register_builtin_inference_executors(...)
  public executor re-exports

executors/common.rs
  typed input extraction helpers
  param conversion helpers
  InferenceRuntime error mapping
  NodeExecutionOutputs builders

executors/validation.rs
  output slot and value-kind validation

executors/model.rs
  CheckpointLoaderExecutor

executors/text.rs
  ClipTextEncodeExecutor

executors/latent.rs
  EmptyLatentImageExecutor
  VaeDecodeExecutor

executors/diffusion.rs
  KSamplerExecutor

executors/image.rs
  SaveImageExecutor
  PreviewImageExecutor
```

The helper layer should remove boilerplate, but concrete executor structs
remain explicit. Do not replace them with a mandatory table-driven
`operation_id -> backend call` interpreter.

## V1 Strategy

The next correction is to make existing executors depend on
`inference-core::InferenceRuntime` rather than a single
`Arc<dyn InferenceBackend>` or stringly `operation_id` dispatch.

`operation_id` is not part of the target executor path. A concrete executor
should call a typed method such as:

```text
InferenceRuntime::diffusion_sample(DiffusionSampleRequest)
```

It should not build a generic envelope whose correctness depends on:

```text
operation_id = "diffusion.sample"
```

If labels are needed for diagnostics, tracing, capability reports, or bridge
policy, they should be derived from an `InferenceCapability`/capability label
owned by `inference-core`, not used as the primary dispatch key.

Real CLIP/UNet/VAE work should land behind the same typed backend capability
protocol. Do not add model-family-specific executor infrastructure unless a
genuinely new capability kind is required.
