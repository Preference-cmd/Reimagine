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
app-host -> inference-backends/candle     # V1 default backend assembly

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

## Operation Protocol

The central abstraction is an operation-based backend adapter:

```text
InferenceBackend
  backend_kind()
  capabilities()
  execute(request)
  memory_snapshot()
```

`InferenceRequest` carries:

- `operation_id`;
- model context or resolved model input when required;
- input `RuntimeValue` values;
- typed node params;
- run/correlation context;
- artifact capability when the operation can publish files.

`InferenceResponse` carries named `RuntimeValue` outputs or an
`InferenceError`. The operation protocol is model-family-neutral. SDXL is a V1
consumer of the protocol, not the shape of the protocol.

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
CandleBackend supports:
  model.load_bundle    stable_diffusion/sdxl
  text.encode          stable_diffusion/sdxl
  latent.create_empty  *
  diffusion.sample     stable_diffusion/sdxl
  latent.decode        stable_diffusion/sdxl
  image.save           *
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
2. Move or replace `crates/candle-integration` with
   `crates/inference-backends/candle` as `reimagine-inference-candle`.
3. Wire app-host to use the Candle backend as the V1 default backend.
4. Prove the SDXL example workflow runs through Axum using the same app-host
   and runtime path.
5. Replace stubbed backend kernels with real Candle CLIP/UNet/VAE behavior
   behind the same operation protocol.
