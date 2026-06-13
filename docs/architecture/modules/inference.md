# Inference Module Architecture

> Status: working draft
> Crate: `crates/inference`

## Role

`inference` is the backend-neutral image generation inference layer. It defines
the capability traits, executor factory shape, backend-neutral errors, and
runtime value conventions needed to run built-in generation nodes without
making `runtime`, `app-host`, Tauri, or Axum depend on a concrete inference
backend.

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

- Define backend-neutral inference backend and SDXL capability traits.
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
the backend capability object, registers the backend-neutral executors into
`RuntimeService`, and hands host adapters an `Arc<WorkspaceHost>`.

## Backend Adapter Shape

The central abstraction is a backend adapter that can provide executor
capabilities without taking over the runtime loop:

```text
InferenceBackend
  backend_kind()
  register_executors(registry, services)
  memory_snapshot()
```

The exact Rust API may split this into narrower traits, but the ownership
remains the same: inference owns the adapter boundary, runtime owns execution,
and concrete backend crates own tensors/model caches.

For SDXL, the backend capability surface should be closer to:

```text
SdxlModelLoader
SdxlTextEncoder
SdxlLatentFactory
SdxlSampler
SdxlVaeDecoder
SdxlImageWriter
```

The V1 built-in executors depend on these traits, not on Candle. Candle is only
one implementation of the traits.

## Runtime Integration

`runtime` already exposes:

```text
NodeExecutorRegistry
NodeExecutor::execute(NodeExecutionContext)
RuntimeValue
NodeArtifactCapability
```

`inference` should produce `NodeExecutor` implementations or factories that use
backend capabilities and return `RuntimeValue` handles. No concrete backend
tensor type may cross this boundary.

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

## Model Resolution Handoff

Model manifest semantics stay outside inference. `checkpoint_loader` receives a
workflow `ModelRef`; a host-supplied resolver capability maps that reference to
a resolved descriptor/path before loading.

```text
workflow ModelRef
  -> app-host ModelService / model-manager resolver
  -> inference model resolver capability
  -> backend adapter model loader
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

1. Introduce `crates/inference` with the backend-neutral adapter boundary and
   SDXL executor registration shape.
2. Move or replace `crates/candle-integration` with
   `crates/inference-backends/candle` as `reimagine-inference-candle`.
3. Wire app-host to use the Candle backend as the V1 default backend.
4. Prove the SDXL example workflow runs through Axum using the same app-host
   and runtime path.
5. Replace stubbed backend kernels with real Candle CLIP/UNet/VAE behavior
   behind the same inference traits.

