# Inference Module Architecture

> Status: working draft
> Crate: `crates/inference`

## Role

`inference` is the unified backend-neutral inference abstraction/facade. It
owns built-in node orchestration, the runtime-facing executor contract,
execution values, typed backend capability DTOs, the executor-facing router,
backend adapter contracts, bridge policy, and inference errors.

The earlier `inference-core` contract layer is folded into this module for
architecture, issue planning, and code ownership. The physical
`crates/inference-core` crate has been removed, and new design work should be
tracked under the `inference` module.

## Responsibilities

- Provide built-in inference-backed node executors.
- Define runtime-facing node executor contracts, node execution context, output
  contracts, and executor registration helpers.
- Own and re-export execution values and backend-affine handles consumed by
  runtime.
- Define typed backend capability request/response DTOs.
- Define `InferenceRuntime`, the executor-facing router trait.
- Define `InferenceBackend`, the concrete backend adapter trait.
- Define backend registry, capability reports, bridge policy, model resolver
  handoff DTOs, inference errors, and diagnostic projection helpers.
- Convert node execution context inputs and params into typed inference
  requests.
- Call the injected `InferenceRuntime` router.
- Validate typed responses against the node's output slots.
- Record save/preview artifacts through injected artifact capabilities.
- Provide executor registration helpers for app-host bootstrap.

## Non-Responsibilities

- Runtime scheduling, cancellation, run state, snapshots, or value-store
  ownership.
- Concrete Candle, ONNX, remote, or Comfy implementations.
- Model manifest scanning or persistence.
- Plugin loading or plugin package lifecycle.
- Tauri IPC, Axum routes, or UI state.
- Agent policy.

## Dependency Direction

```text
inference -> core

inference must not -> inference-backends/*
inference must not -> runtime
inference must not -> model-manager
inference must not -> app-host
inference must not -> tauri
inference must not -> axum
```

`runtime` depends on `inference` as its executor/value facade. `app-host`
composes the pieces by constructing the inference router/backend registry and
asking `inference` to register node executors into the executor registry
consumed by runtime.

Implementation note: backend contracts, router contracts, execution values,
typed capability DTOs, resource contracts, and node executor contracts are all
imported from `reimagine_inference`.

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

`reimagine-runtime` temporarily re-exports the moved executor types from
`reimagine-inference` for backward compatibility with call sites that pre-date
the inversion. New code should import the executor contract from
`reimagine-inference`.

## Boundary

`runtime` owns the execution loop. `inference` owns node orchestration and the
backend contract. Concrete backends own payloads, model graphs, tensors, and
device mechanisms.

```text
runtime scheduler
  -> inference::NodeExecutor::execute(NodeExecutionContext)
  -> inference executor
  -> typed inference request
  -> inference::InferenceRuntime/router
  -> selected inference::InferenceBackend method
  -> backend-private model graph / payload store
  -> inference::ExecutionValue outputs
```

The runtime execution unit remains a workflow node invocation. Backend
capability calls are not runtime scheduling units.

`inference` calls the executor-facing `InferenceRuntime` router, not
`InferenceBackend` directly. The two traits are deliberately not equivalent:
the runtime/router trait validates handles, selects a backend instance, applies
bridge policy, and emits routing diagnostics; the backend trait is the concrete
adapter seam for one backend implementation. Inference executors should depend
on the router trait even if a workspace currently registers only Candle.

The router must be configurable and must support safe fallback, but inference
executors do not own that policy. Executors construct typed requests from node
inputs and params, then call `InferenceRuntime`. App-host/runtime policy inputs
configure the router; the router applies those inputs to capability support,
handle affinity, and bridge policy.

```text
inference executor
  -> typed request
  -> InferenceRuntime router
    -> BackendSelectionPolicy
    -> BackendBridgePolicy
    -> selected backend
```

Fallback is valid only before a request has backend-bound handles or before a
failed attempt produces visible execution handles. Once a `Model`, `Clip`,
`Vae`, `Latent`, `Conditioning`, or `Image` handle exists, the handle's backend
affinity constrains later routing. Cross-backend execution after that point
requires an explicit bridge/transfer policy rather than silent fallback.

Backend identity should not be modeled as a closed `BackendKind` enum. Built-in
and future external backends are plugin extensions over the
`HostSurface::InferenceBackend` surface. Inference routing should work with an
open `Backend` label for the backend implementation and concrete
`BackendInstance` descriptors provided by app-host, with optional plugin
provenance (`Plugin` / `Extension`) for diagnostics and registry
introspection.

## Backend Resource Mechanisms

`inference` also owns the backend-neutral resource mechanism contracts used by
runtime and app-host. These contracts are attached to configured backend
instances; they are not a second plugin surface and they are not a concrete
memory manager.

Plugin alignment:

```text
PluginPackage
  -> PluginExtension { extends: HostSurface::InferenceBackend }
  -> app-host constructs one or more BackendInstance values
  -> app-host registers:
       InferenceBackend adapter for typed capabilities
       BackendResourceMechanism adapter for lifecycle/observation
```

The plugin metadata tells the host which package and extension contributed the
backend. The `BackendInstance` is the runtime selection and observation unit.
A single plugin extension can later produce multiple instances such as
`"candle:cpu"` and `"candle:metal"`, each with its own resource observations.

V1 should keep the mechanism surface coarse:

```text
BackendRunLifecycle
  begin_run(run_id)
  cleanup_run(run_id)

BackendResourceObservation
  resource_snapshot() -> BackendResourceSnapshot

BackendResourceMechanism
  BackendRunLifecycle + BackendResourceObservation
```

The existing `RunResourceBackend` name should be treated as historical. The
replacement name should communicate that this is a backend-instance mechanism,
not a runtime-owned backend manager.

Resource snapshots are host-neutral observations. They may include backend
instance identity, open backend label, optional plugin provenance, device
profile, cache counts, approximate bytes, and diagnostics. They must not expose
backend-private tensors, loaded model structs, tokenizer state, graph objects,
or file handles.

V1 must not reintroduce ordinary per-value release, pin, unpin, offload, evict,
or prepare-value commands. Runtime value lifetime is governed by
`Arc<ExecutionValue>` ownership and producer-declared
`ExecutionValueRetention`; backend caches and concrete payload stores retain
their own internal owners. Future budget, transfer, preparation, or pinning
interfaces should be separate mechanism traits added only when runtime has a
concrete policy that needs them.

## Execution Value Usage

`inference` consumes and returns execution values as its public runtime-facing
facade. These types are physically defined in `crates/inference`.

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

`inference` may inspect the public handle shape, such as backend affinity,
backend payload key, tensor shape, dtype, model role, and device label. It
must not inspect backend-local tensors, loaded model objects, tokenizer state,
or kernel graphs.

`ExecutionValue` is internal execution data. It must not be exposed through
workflow JSON, run snapshots, run summaries, run events, Axum/Tauri DTOs, or
Agent tool results.

## Executor Shape

Each executor should be explicit and small:

- read required inputs and params from `NodeExecutionContext`;
- build a typed inference request;
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
  Arc<dyn inference::InferenceRuntime>,
  Arc<dyn inference::ModelResolver>,
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

inference::InferenceRuntime
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
    inference: Arc<dyn inference::InferenceRuntime>,
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
  -> inference ModelResolver adapter
  -> LoadBundleRequest
  -> InferenceRuntime::load_bundle(...)
  -> ExecutionValue::Model / Clip / Vae handles
```

`inference` depends on a resolver trait, not directly on `model-manager`.

Model backend choice is applied by the router, not by runtime and not by the
executor itself:

```text
checkpoint_loader executor
  -> ModelResolver resolves ModelRef to ResolvedInferenceModel
  -> LoadBundleRequest
  -> InferenceRuntime::load_bundle
  -> router selects backend from config/policy when no handle affinity exists
  -> returned Model / Clip / Vae handles carry selected backend affinity
```

Later executors route through those returned handles. For example,
`clip_text_encode` is pinned by the `Clip` handle's backend affinity, and
`ksampler` is constrained by the `Model`, `Conditioning`, and `Latent` handle
affinities unless an explicit bridge policy permits transfer.

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
  define or facade execution values and handles

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

The corrected executor path depends on `InferenceRuntime` rather than a single
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
owned by `inference`, not used as the primary dispatch key.

Real CLIP/UNet/VAE work should land behind the same typed backend capability
protocol. Do not add model-family-specific executor infrastructure unless a
genuinely new capability kind is required.
