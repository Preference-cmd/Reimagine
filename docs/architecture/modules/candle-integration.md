# Candle Integration Module Architecture

> Status: working draft
> Crate: `crates/candle-integration`

## Role

`candle-integration` is the Candle-specific inference backend. It implements backend behavior behind the `core` inference contracts.

## V1 Target

V1 must support SDXL base-only text-to-image inference. SDXL refiner support is deferred.

## Responsibilities

- Session management.
- Device and dtype configuration.
- Model loading and cache.
- CLIP, UNet, VAE implementations.
- Tensor conversion between `core` data and Candle tensors.
- Built-in SDXL node executor implementations for the runtime executor registry.
- Artifact encoding for image outputs produced by `builtin.save_image`.

## Non-Responsibilities

- Workflow graph semantics.
- Tauri IPC.
- Runtime scheduling.
- Agent tools.
- ComfyUI import.

## Dependency Direction

```text
app-host -> candle-integration
candle-integration -> runtime
candle-integration -> core
candle-integration must not -> app-host
candle-integration must not -> tauri
candle-integration must not -> axum
candle-integration must not -> model-manager
```

`app-host` resolves model descriptors through `model-manager`, then injects
the resolved backend capability and node executor registry into runtime.
`candle-integration` consumes resolved paths/metadata; it does not scan
directories or read manifests.

## Runtime Integration

The runtime already owns the host-neutral execution boundary:

```text
RuntimeService
  -> NodeExecutorRegistry
  -> NodeExecutor::execute(NodeExecutionContext)
  -> Vec<(SlotId, Arc<RuntimeValue>)>
```

`candle-integration` provides concrete `NodeExecutor` implementations for the
V1 built-ins. It must register them into a `NodeExecutorRegistry` assembled by
`app-host`; runtime remains backend-agnostic.

Initial executor set:

```text
builtin.string
builtin.checkpoint_loader
builtin.clip_text_encode
builtin.empty_latent_image
builtin.ksampler
builtin.vae_decode
builtin.save_image
```

`builtin.preview_image` may be added later, but it is not required to prove the
base text-to-image save path.

## Backend Session Shape

V1 should introduce a Candle backend service with explicit configuration and
interior synchronization so executors can share it safely:

```text
CandleBackend
  config: CandleBackendConfig
  session/cache: Arc<...>
```

The backend owns:

- Candle device and dtype policy;
- loaded checkpoint/model component cache;
- backend tensor payload store keyed by `BackendPayloadKey`;
- conversion between backend tensors and `RuntimeValue` handles;
- image encoding/write helpers used by `save_image`.

The runtime only sees lightweight `RuntimeValue` handles:

```text
RuntimeValue::Model(RuntimeModelHandle)
RuntimeValue::Clip(RuntimeClipHandle)
RuntimeValue::Vae(RuntimeVaeHandle)
RuntimeValue::Conditioning(RuntimeConditioning)
RuntimeValue::Latent(RuntimeLatent)
RuntimeValue::Image(RuntimeImage)
RuntimeValue::Artifact(ArtifactRef)
```

No `candle_core::Tensor` should appear in `runtime`, `app-host`, `axum-host`,
or workflow JSON.

## Model Resolution Handoff

`checkpoint_loader` receives a workflow `ModelRef` as a static param. The
executor needs a host-supplied resolver capability that maps that `ModelRef`
to a resolved model descriptor/path before loading.

The dependency direction is:

```text
workflow ModelRef
  -> app-host ModelService::resolve_descriptor
  -> resolved descriptor/path
  -> candle-integration checkpoint loader executor
  -> RuntimeValue::Model / Clip / Vae handles
```

The resolver capability should be defined at the app-host/candle boundary, not
inside runtime. Runtime passes params to the executor but does not resolve
models itself.

## M1 Strategy

M1 should prioritize an executable vertical slice over complete SDXL quality:

1. Register concrete executors through app-host into runtime.
2. Prove the existing SDXL workflow executes through Axum HTTP using the real
   registry path.
3. Initially allow backend stubs for heavy kernels where needed, but keep the
   runtime value shapes and artifact path identical to the eventual real SDXL
   path.
4. Replace stubs with real Candle CLIP/UNet/VAE implementation behind the same
   executor/backend API.

The first M1 issue should not try to perfect sampling quality, device offload,
or streaming progress. It should make the SDXL example produce a deterministic
artifact or a precise backend-not-implemented diagnostic through the real
executor registration path.
