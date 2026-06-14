# Inference Module Architecture

> Status: working draft
> Crate: `crates/inference`

## Role

`inference` is the backend-neutral image generation inference layer. It defines
an operation-based backend protocol, executor factory shape, backend-neutral
errors, and runtime value conventions needed to run built-in generation nodes
without making `runtime`, `app-host`, Tauri, or Axum depend on a concrete
inference backend.

Concrete inference adapters live under grouped backend crates:

```text
crates/inference/
  Cargo.toml              # reimagine-inference

crates/inference-backends/
  candle/
    Cargo.toml            # reimagine-inference-candle
  fake/                   # optional test/dev backend
  remote/                 # future
  onnx/                   # future
```

This keeps the top-level `crates/` directory readable while preserving separate
Cargo crates and clean optional dependencies for each backend.

## Responsibilities

- Define backend-neutral inference backend and operation protocol.
- Define executor factory / registration helpers for V1 built-in nodes.
- Define backend-neutral inference errors and diagnostic projection.
- Define model resolution capability interfaces consumed by executors.
- Preserve runtime value handle conventions for model, CLIP, VAE, latent,
  conditioning, image, and artifact values.
- Provide fake/stub backend seams for tests when useful.

## Non-Responsibilities

- Runtime scheduling.
- Workflow graph validation.
- Model manifest scanning or persistence.
- Concrete Candle, ONNX, remote, or Comfy implementations.
- Tauri IPC, Axum routes, or UI state.
- Agent policy.

## Dependency Direction

```text
app-host -> inference
app-host -> inference-backends/candle     # V1 configured default backend assembly

inference -> runtime
inference -> core

inference-backends/candle -> inference
inference-backends/candle -> runtime
inference-backends/candle -> core

runtime must not -> inference
runtime must not -> inference-backends/*
axum-host must not -> inference-backends/*
src-tauri should not directly -> inference-backends/*
```

`app-host` is the composition root. It chooses the configured backend, builds
the backend adapter object, registers inference-backed executors into
`RuntimeService`, and hands host adapters an `Arc<WorkspaceHost>`.

The configured backend applies at workspace/run/execution-unit granularity in
V1. `inference` does not define node-level backend overrides, implicit
cross-backend transfer, or backend-specific scheduler decisions. Those require
an explicit future bridge/conversion design.

## Runtime And Inference Boundary

`runtime` owns the execution loop. `inference` must not become a second
runtime.

```text
runtime
  owns:
    ExecutionPlan
    DAG stage scheduling
    RunSession / RunValueStore
    cancellation
    RunSnapshot / RunSummary
    RunEventSink
    NodeExecutor trait
    RuntimeValue envelope
    RunResourceBackend lifecycle hook

inference
  owns:
    operation protocol
    built-in node -> operation mapping
    backend-neutral inference errors
    model resolver capability shape
    fake backend for tests
    inference-backed NodeExecutor registration

backend adapter
  owns:
    supported operation matrix
    tensor/model cache
    device/dtype/offload policy
    concrete operation implementation
```

`NodeExecutor` remains a runtime trait. `inference` produces `NodeExecutor`
implementations that map node execution contexts into inference operations.

```text
NodeExecutionContext
  -> inference-backed NodeExecutor
  -> InferenceRequest
  -> InferenceBackend::execute
  -> InferenceResponse
  -> Vec<(SlotId, Arc<RuntimeValue>)>
```

## Suggested Module Layout

```text
src/
  lib.rs
  error.rs
  operation.rs
  request.rs
  response.rs
  capability.rs
  backend.rs
  resolver.rs
  registry.rs
  executors.rs
  executors/
    validation.rs
    string.rs
    model.rs
    text.rs
    latent.rs
    diffusion.rs
    image.rs
  testing.rs
```

`lib.rs` should stay as a facade of private modules plus explicit public
re-exports. Keep the operation protocol, backend trait, resolver trait, and
executor adapters in separate files; do not collect the crate's core model into
one large `lib.rs` or `backend.rs`.

`testing.rs` may expose fake/stub helpers behind `#[cfg(any(test, feature =
"testing"))]` if downstream tests need them. The default public API should not
require test-only fake types.

## Operation Protocol

The central abstraction is an operation-based backend adapter:

```text
InferenceBackend
  backend_kind()
  capabilities()
  execute(request)
  memory_snapshot()
```

V1 should use the same `async_trait` style already used by `runtime` and
`agent` for async trait objects:

```text
#[async_trait]
trait InferenceBackend: Send + Sync + 'static {
  async fn execute(&self, request: InferenceRequest)
    -> Result<InferenceResponse, InferenceError>;
}
```

The request owns cheap, shareable handles rather than borrowing from
`NodeExecutionContext`. In practice that means `SlotId -> Arc<RuntimeValue>`
maps, typed params, and resolved model DTOs are cloned into the request. This
keeps the backend call lifetime simple across `.await` while preserving
zero-copy behavior for tensors and loaded models, because large data remains
inside backend-owned stores referenced by runtime handles.

`InferenceRequest` carries:

- `operation_id`;
- `models: Vec<ResolvedInferenceModel>` when the operation needs model
  context;
- input `RuntimeValue` values keyed by `SlotId`;
- typed node params;
- run/correlation context.

Use a vector even for single-model operations. A request with one checkpoint
bundle simply carries one resolved model, while future operations can carry a
base model plus LoRA, ControlNet, refiner, or multiple text encoders without
changing the protocol.

`InferenceRequest` must not carry `NodeArtifactCapability`. Artifact recording
belongs to the inference-backed executor adapter because runtime owns the
artifact store. A backend may return an image value, an artifact intent, or a
backend payload handle, but the executor records the artifact through
`NodeArtifactCapability::record`.

`InferenceResponse` carries slot-aware named `RuntimeValue` outputs or an
`InferenceError`:

```text
InferenceResponse
  outputs: Vec<InferenceOutput>

InferenceOutput
  slot_id: SlotId
  value: Arc<RuntimeValue>
```

`SlotId` is the core model identifier for a node input or output slot. It is
not a backend field name and it is not an array index. Inference executors use
the workflow/node declaration to translate operation inputs and outputs into
these slot ids. This mirrors runtime's `NodeExecutionOutputs` shape and avoids
relying on array order for multi-output nodes such as
`builtin.checkpoint_loader`.

Response validation belongs to the inference-backed executor adapter:

- every returned `slot_id` must be declared by the node's `output_slots`;
- required output slots must be present before the executor returns;
- returned values must match the expected runtime value kind for the slot;
- duplicate output slot ids are errors;
- extra backend outputs are errors unless the node type explicitly declares an
  extension/dynamic output policy.

This keeps runtime generic. Runtime stores whatever
`Vec<(SlotId, Arc<RuntimeValue>)>` the executor returns; it does not understand
inference operation ids or backend-native output names.

Put shared validation code in `executors/validation.rs` rather than duplicating
it across node executors. Individual executor modules should declare their
expected output slots and value kinds, call the shared validator, and then
return `NodeExecutionOutputs`.

`InferenceError` should convert into runtime's `NodeExecutorError` through an
explicit method such as `InferenceError::into_executor_error()`. Avoid a broad
`From<InferenceError> for NodeExecutorError` implementation in V1 so call sites
make the inference-to-runtime boundary visible.

The operation protocol is model-family-neutral. SDXL is a V1 consumer of the
protocol, not the shape of the protocol.

Initial operation ids:

```text
model.load_bundle
text.encode
latent.create_empty
diffusion.sample
latent.decode
image.save
image.preview
```

Backends publish a capability report that says which operations they support
for which model families/variants:

```text
InferenceBackendCapabilities
  backend_kind
  operations: Vec<InferenceOperationSupport>

InferenceOperationSupport
  operation_id
  model_series: Option<ModelSeries>
  variant: Option<ModelVariant>
  roles: Vec<ModelRole>
```

Future Flux, video, ONNX, or remote backends extend the support matrix. They do
not require new infrastructure traits per model family unless a genuinely new
operation kind is needed.

## Runtime Integration

`runtime` already exposes:

```text
NodeExecutorRegistry
NodeExecutor::execute(NodeExecutionContext)
RuntimeValue
NodeArtifactCapability
```

`inference` should produce `NodeExecutor` implementations or factories that use
`InferenceBackend::execute` and return `RuntimeValue` handles. No concrete
backend tensor type may cross this boundary.

The initial executor set remains:

```text
builtin.string
builtin.checkpoint_loader
builtin.clip_text_encode
builtin.empty_latent_image
builtin.ksampler
builtin.vae_decode
builtin.save_image
```

### Executor Adapter Organization

Inference-backed executors are node adapters, not backend implementations.
Each executor should be small and focused:

- read required inputs/params from `NodeExecutionContext`;
- produce one `InferenceRequest` with a stable `InferenceOperationId`;
- call `InferenceBackend::execute`;
- validate response slot names and value kinds;
- record artifacts through `NodeArtifactCapability` for save/preview
  operations;
- return `NodeExecutionOutputs`.

Keep backend-specific behavior out of these executor files. For example,
`executors/diffusion.rs` may map `builtin.ksampler` to
`diffusion.sample`, but it must not branch on Candle model internals.

Executor registration belongs in a narrow API such as:

```text
register_builtin_inference_executors(registry, backend, resolver)
```

The exact function name can vary, but the registration entrypoint should live
in `registry.rs` or a similarly focused module. It should not require
`app-host`, `model-manager`, or a concrete backend crate.

Save/preview executors should treat artifact recording as a runtime-facing
adapter concern:

```text
image.save executor
  -> InferenceBackend::execute(image.save)
  -> validate image/artifact response
  -> NodeArtifactCapability::record(slot_id, ArtifactRef, ArtifactEventKind)
  -> RuntimeValue::Artifact(...)
```

This keeps concrete backends independent from runtime's artifact store while
still letting remote or local backends produce saveable image data.

## RunResourceBackend Relationship

`RunResourceBackend` and `InferenceBackend` are separate roles.

```text
RunResourceBackend
  called by runtime
  begin_run / release_runtime_value / cleanup_run / memory_snapshot

InferenceBackend
  called by inference-backed node executors
  execute(operation_request)
```

A concrete backend adapter, such as Candle, may implement both roles. Runtime
still only knows the `RunResourceBackend` trait, while inference-backed
executors know the `InferenceBackend` trait.

## Model Resolution Handoff

Model manifest semantics stay outside inference. `checkpoint_loader` receives a
workflow `ModelRef`; a host-supplied resolver capability maps that reference to
a resolved descriptor/path before loading.

```text
workflow ModelRef
  -> app-host ModelService / model-manager resolver
  -> inference model resolver capability
  -> InferenceRequest(model.load_bundle)
  -> backend adapter
  -> RuntimeValue::Model / Clip / Vae handles
```

This keeps `model-manager` independent from backend crates and keeps runtime
free of model manifest knowledge.

The model resolver trait should return inference-layer resolved model metadata,
not `model-manager::ModelDescriptor` directly. `app-host` can adapt
`ModelDescriptor` into that shape.

`ResolvedInferenceModel` is type-independent from `model-manager`, but it
should reuse stable cross-module semantics from `core::model`. In practice,
that means fields such as `ModelId`, `ModelSeries`, `ModelVariant`, and
`ModelRole` come from `core`, while manifest-only details such as scan source,
root declarations, user classification rules, fingerprint status, or
model-manager diagnostics stay outside inference.

Suggested shape:

```text
ResolvedInferenceModel
  model_id
  model_series
  variant
  role
  source_path
  format
  metadata
```

This prevents `inference` from depending on `model-manager` while still
preserving the path and identity needed by `model.load_bundle`.

## Backend Crate Placement

Backend adapters should be grouped under `crates/inference-backends/` instead
of being fully flat in `crates/`.

Reasons:

- the top-level `crates/` directory stays focused on architectural layers;
- each backend remains a separate Cargo crate with independent dependencies;
- optional backend selection remains clean for future packaging;
- backend implementations are visually subordinate to the inference layer.

## V1 Strategy

1. Introduce `crates/inference` with the operation-based backend protocol and
   generic executor registration shape.
2. Directly migrate the legacy `crates/candle-integration` placeholder into
   `crates/inference-backends/candle` as `reimagine-inference-candle`.
3. Wire app-host to select the configured backend enum value, with Candle as
   the V1 default backend.
4. Prove the SDXL example workflow runs through Axum using the same app-host
   and runtime path.
5. Replace stubbed backend kernels with real Candle CLIP/UNet/VAE behavior
   behind the same operation protocol.
