# Inference Module Architecture

> Status: working draft
> Crate: `crates/inference`

## Role

`inference` is the backend-neutral node orchestration layer for built-in
generation nodes. It implements `runtime::NodeExecutor` adapters that map a
workflow node invocation to typed backend capability calls.

It does not define the backend contract itself. Shared backend traits, typed
capability request/response DTOs, backend registry, router, bridge policy, and
inference errors belong to [`inference-core`](inference-core.md).

## Responsibilities

- Provide built-in inference-backed node executors.
- Convert `runtime::NodeExecutionContext` inputs and params into typed
  `inference-core` requests.
- Call the injected `inference-core::InferenceRuntime` router.
- Validate typed responses against the node's output slots.
- Record save/preview artifacts through runtime artifact capabilities.
- Provide executor registration helpers for app-host bootstrap.

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
inference -> runtime
inference -> inference-core

inference must not -> inference-backends/*
inference must not -> model-manager
inference must not -> app-host
inference must not -> tauri
inference must not -> axum
```

`runtime` must not depend on `inference`. `app-host` composes the pieces by
constructing an `inference-core` router and asking `inference` to register
node executors into a runtime `NodeExecutorRegistry`.

## Boundary

`runtime` owns the execution loop. `inference` owns node orchestration.
`inference-core` owns the backend contract. Concrete backends own payloads,
model graphs, tensors, and device policy.

```text
runtime scheduler
  -> NodeExecutor::execute(NodeExecutionContext)
  -> inference executor
  -> inference-core typed request
  -> inference-core InferenceRuntime/router
  -> selected inference-core InferenceBackend method
  -> backend-private model graph / payload store
  -> core::ExecutionValue outputs
```

The runtime execution unit remains a workflow node invocation. Backend
capability calls are not runtime scheduling units.

## Runtime Value Usage

`inference` consumes and returns public execution values owned by `core`.

```text
core::ExecutionValue
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

## Executor Shape

Each executor should be explicit and small:

- read required inputs and params from `NodeExecutionContext`;
- build a typed `inference-core` request;
- call the corresponding `InferenceRuntime` capability method;
- validate response value kinds and required outputs;
- map typed responses into `NodeExecutionOutputs`;
- record artifacts through runtime artifact capabilities when needed.

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

runtime::NodeExecutorRegistry
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
  returns public ExecutionValue handle
```

Rust-shaped sketch:

```rust
pub struct KSamplerExecutor {
    inference: Arc<dyn inference_core::InferenceRuntime>,
}

#[async_trait]
impl runtime::NodeExecutor for KSamplerExecutor {
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

Real CLIP/UNet/VAE work should land behind the same typed backend capability
protocol. Do not add model-family-specific executor infrastructure unless a
genuinely new capability kind is required.
