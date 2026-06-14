# Candle Backend Adapter Architecture

> Status: working draft
> Crate: `crates/inference-backends/candle` (`reimagine-inference-candle`)
> Legacy placeholder: `crates/candle-integration`

## Role

The Candle backend adapter is the local Candle implementation of the
backend-neutral inference layer. It implements the operation protocol from
`crates/inference` and owns Candle-specific model loading, tensor storage,
device policy, and image encoding.

Concrete inference backend crates are grouped under
`crates/inference-backends/*`. `crates/candle-integration` is a legacy
placeholder from the pre-`inference` architecture and should be migrated,
renamed, or removed as the Candle adapter becomes `reimagine-inference-candle`.
The preferred V1 path is direct migration into the grouped backend crate, not a
long-lived compatibility layer.

## V1 Target

V1 must support SDXL base-only text-to-image inference. SDXL refiner support is deferred.

## Responsibilities

- Session management.
- Device and dtype configuration.
- Model loading and cache.
- CLIP, UNet, VAE implementations.
- Tensor conversion between `core` data and Candle tensors.
- Operation implementations consumed by inference-layer executors.
- Artifact encoding for image outputs produced by `builtin.save_image`.

## Non-Responsibilities

- Workflow graph semantics.
- Tauri IPC.
- Runtime scheduling.
- Agent tools.
- ComfyUI import.

## Dependency Direction

```text
app-host -> inference
app-host -> inference-backends/candle
inference-backends/candle -> inference
inference-backends/candle -> runtime
inference-backends/candle -> core
inference-backends/candle must not -> app-host
inference-backends/candle must not -> tauri
inference-backends/candle must not -> axum
inference-backends/candle must not -> model-manager
```

`app-host` resolves model descriptors through `model-manager`, then injects the
chosen inference backend and node executor registry into runtime. The Candle
adapter consumes resolved paths/metadata; it does not scan directories or read
manifests.

Backend selection is owned by app-host/config. Candle can be the default V1
backend, but it should be selected through an enum/config value rather than
being hard-coded into runtime, inference executors, Axum, or Tauri.

## Runtime Integration

The runtime already owns the host-neutral execution boundary:

```text
RuntimeService
  -> NodeExecutorRegistry
  -> NodeExecutor::execute(NodeExecutionContext)
  -> Vec<(SlotId, Arc<RuntimeValue>)>
```

`inference` provides backend-neutral `NodeExecutor` implementations or
factories for the V1 built-ins. The Candle adapter implements the operations
consumed by those executors. Runtime remains backend-agnostic.

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

`builtin.preview_image` follows the same backend boundary as `save_image`; it
is useful for UI/runtime parity but is not required to prove the base
text-to-image save path.

## Candle Backend Shape

V1 should introduce a Candle backend service with explicit configuration and
interior synchronization so executors can share it safely:

```text
CandleBackend
  config: CandleBackendConfig
  store: Arc<CandleStore>
  model_cache: Arc<CandleModelCache>
```

The backend owns:

- Candle device and dtype policy;
- loaded checkpoint/model component cache;
- backend tensor payload store keyed by `BackendPayloadKey`;
- conversion between backend tensors and `RuntimeValue` handles;
- image encoding/write helpers used by `save_image`.

`CandleBackend` owns both the backend payload store and the model cache. Runtime
and app-host must not hold Candle tensors or loaded Candle model objects
directly.

Payload lifetimes are split by intent:

```text
CandleModelCache
  cross-run model payloads
  loaded checkpoint / UNet / CLIP / VAE objects
  backend cache policy controls eviction

CandleStore
  run-scoped payloads
  latent / conditioning / decoded image tensors
  indexed by RunId and BackendPayloadKey
```

`RunResourceBackend` communicates lifecycle intent to the backend. It does not
force a release strategy:

- `begin_run` can create run-local indexes or pins;
- `release_runtime_value` can release immediately, decrement backend refs, or
  keep a payload in a pool;
- `cleanup_run` releases run-scoped payloads and run pins;
- cached model payloads may remain loaded according to backend policy;
- `memory_snapshot` reports backend cache/device observations without exposing
  backend internals.

It may implement both:

```text
InferenceBackend
  execute(operation_request)

RunResourceBackend
  begin_run / release_runtime_value / cleanup_run / memory_snapshot
```

These roles stay separate even when implemented by the same concrete object.

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

`model.load_bundle` should avoid duplicating loaded SDXL components. Internally,
the backend may store one loaded bundle and expose separate typed handles for
the workflow outputs:

```text
LoadedSdxlBundle
  diffusion model / UNet
  CLIP
  VAE
  metadata

model.load_bundle outputs
  RuntimeValue::Model(handle to bundle model role)
  RuntimeValue::Clip(handle to bundle clip role)
  RuntimeValue::Vae(handle to bundle vae role)
```

The workflow/runtime surface still sees three typed values. Only the Candle
backend knows whether those handles point into one shared bundle or separate
backend payloads.

## Model Resolution Handoff

`checkpoint_loader` receives a workflow `ModelRef` as a static param. The
executor needs a host-supplied resolver capability that maps that `ModelRef`
to a resolved model descriptor/path before loading.

The dependency direction is:

```text
workflow ModelRef
  -> app-host ModelService::resolve_descriptor
  -> inference model resolver capability
  -> InferenceRequest(model.load_bundle)
  -> CandleBackend
  -> RuntimeValue::Model / Clip / Vae handles
```

The resolver capability should be defined at the app-host/inference boundary,
not inside runtime. Runtime passes params to the executor but does not resolve
models itself.

## M1 Strategy

M1 should prioritize an executable vertical slice over complete SDXL quality:

1. Introduce the `inference` crate boundary and backend-neutral executor
   registration shape.
2. Directly migrate the current `candle-integration` placeholder into
   `crates/inference-backends/candle` as `reimagine-inference-candle`, replacing
   the legacy crate unless a very small temporary shim is required.
3. Register concrete executors through app-host into runtime.
4. Prove the existing SDXL workflow executes through Axum HTTP using the real
   registry path.
5. Initially allow backend stubs for heavy kernels where needed, but keep the
   runtime value shapes and artifact path identical to the eventual real SDXL
   path.
6. Replace stubs with real Candle CLIP/UNet/VAE implementation behind the same
   inference/backend API.

The first M1 issue should not try to perfect sampling quality, device offload,
or streaming progress. It should make the SDXL example produce a deterministic
artifact or a precise backend-not-implemented diagnostic through the real
executor registration path. It is acceptable for `model.load_bundle` and
`latent.create_empty` to succeed and for the run to fail at the first heavy
unimplemented operation, such as `text.encode`, as long as the diagnostic names
the unsupported backend operation precisely.
