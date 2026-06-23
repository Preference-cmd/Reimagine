# Candle Backend Adapter Architecture

> Status: working draft
> Crate: `crates/inference-backends/candle` (`reimagine-inference-candle`)

## Role

The Candle backend adapter is the local Candle implementation of the
backend-neutral inference contract. It implements typed backend capabilities
owned by `inference` and owns Candle-specific model loading, tensor storage,
device mechanism, and image encoding.

Concrete inference backend crates are grouped under
`crates/inference-backends/*`. The previous standalone Candle integration path
has been replaced by `crates/inference-backends/candle`; do not introduce a
compatibility crate or a second backend path.

## V1 Target

V1 must support SDXL base-only text-to-image inference. SDXL refiner support is
deferred. SDXL is the first backend implementation behind typed backend
capabilities; it is not a public capability family.

## Responsibilities

- Session management.
- Device and dtype configuration.
- Model loading and cache.
- CLIP, UNet, VAE implementations.
- Tensor conversion between inference execution handles and Candle tensors.
- Typed backend capability implementations consumed by inference-layer executors.
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
inference-backends/candle -> core
inference-backends/candle must not -> runtime
inference-backends/candle must not -> app-host
inference-backends/candle must not -> tauri
inference-backends/candle must not -> axum
inference-backends/candle must not -> model-manager
```

`app-host` resolves model descriptors through `model-manager`, registers the
Candle adapter in the inference backend registry, constructs the inference
runtime/router, then injects inference-backed node executors
into runtime through the inference facade. The Candle adapter consumes resolved
paths/metadata; it does not scan directories or read manifests.

Backend construction is owned by app-host/config. Candle can be the default V1
backend, but it should be registered behind `BackendKind` and reached through
the inference runtime/router rather than being hard-coded into runtime,
inference executors, Axum, or Tauri.

## Runtime Integration

The runtime already owns the host-neutral execution boundary:

```text
RuntimeService
  -> inference::NodeExecutorRegistry
  -> inference::NodeExecutor::execute(NodeExecutionContext)
  -> Vec<ExecutionOutput>
```

`inference` provides backend-neutral `NodeExecutor` implementations or
factories for the V1 built-ins. Those executors call the inference
runtime/router. The Candle adapter implements typed backend capability methods
selected by the router. Runtime remains backend-agnostic.

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

The Candle backend service has explicit configuration and interior
synchronization so executors can share it safely:

```text
CandleBackend
  config: CandleBackendConfig
  store: Arc<CandleStore>
  model_cache: Arc<CandleModelCache>
```

The backend owns:

- Candle device and dtype mechanism;
- loaded checkpoint/model component cache;
- backend tensor payload store keyed by `BackendPayloadKey`;
- conversion between backend tensors and `ExecutionValue` handles;
- image encoding/write helpers used by `save_image`.

`CandleBackend` owns both the backend payload store and the model cache. Runtime
and app-host must not hold Candle tensors or loaded Candle model objects
directly.

Payload lifetimes are split by intent:

```text
CandleModelCache
  cross-run model payloads
  loaded checkpoint / UNet / CLIP / VAE objects
  backend cache owner implements local eviction mechanics

CandleStore
  run-scoped payloads
  latent / conditioning / decoded image tensors
  indexed by RunId and BackendPayloadKey
```

Runtime communicates ordinary execution value lifetime by dropping its
`Arc<ExecutionValue>` references according to producer-declared retention. It
does not send a separate release-intent command for normal value lifecycle.
Candle decides whether cached model payloads, tensor payloads, or device
allocations remain live by holding its own owners or cache entries.

Candle implements:

```text
inference::InferenceBackend
  load_bundle / text_encode / diffusion_sample / ...
```

Memory/cache observations should be exposed through explicit host-neutral
summary or diagnostic shapes, without making runtime own Candle internals.

Runtime only sees lightweight `ExecutionValue` handles through the inference
facade:

```text
ExecutionValue::Model(RuntimeModelHandle)
ExecutionValue::Clip(RuntimeClipHandle)
ExecutionValue::Vae(RuntimeVaeHandle)
ExecutionValue::Conditioning(ExecutionConditioning)
ExecutionValue::Latent(RuntimeLatent)
ExecutionValue::Image(RuntimeImage)
ExecutionValue::Artifact(ArtifactRef)
```

No `candle_core::Tensor` should appear in `runtime`, `app-host`, `axum-host`,
or workflow JSON.

`load_bundle` should avoid duplicating loaded SDXL components. Internally,
the backend may store one loaded bundle and expose separate typed handles for
the workflow outputs:

```text
LoadedSdxlBundle
  diffusion model / UNet
  CLIP
  VAE
  metadata

load_bundle outputs
  ExecutionValue::Model(handle to bundle model role)
  ExecutionValue::Clip(handle to bundle clip role)
  ExecutionValue::Vae(handle to bundle vae role)
```

The workflow/runtime/inference-executor surface still sees three typed values.
Only the Candle backend knows whether those handles point into one shared
bundle, separate backend payloads, placeholder V1 objects, or real loaded
Candle modules. Higher modules must not infer model architecture from the
handle shape.

## Code Organization

The Candle backend should keep the same modern module style as the rest of the
workspace. Do not introduce `mod.rs` files or `#[path = "..."]` attributes.
Keep `lib.rs` as a facade of private modules plus explicit re-exports.

Suggested layout for the real Candle backend path:

```text
src/
  backend.rs
  config.rs
  device.rs
  error.rs
  lib.rs
  operation.rs
  operation/
    diffusion.rs
    image.rs
    latent.rs
    model.rs
    text.rs
  resource.rs
  store.rs
  models.rs
  models/
    stable_diffusion.rs
    stable_diffusion/
      sdxl.rs
      sdxl/
        bundle.rs
        text.rs
        diffusion.rs
        vae.rs
        tokenizer.rs
```

Backend capability modules translate typed backend-neutral request DTOs into
backend-local model/store calls and translate results back into typed response
DTOs. They should not become large kernel implementation files.

Standard backend capabilities must stay standard:

- `operation/text.rs` owns the `text_encode` request/response contract.
- `operation/diffusion.rs` owns the `diffusion_sample` request/response
  contract.
- `operation/latent.rs` owns `create_empty_latent` and `latent_decode`
  request/response contracts.
- `operation/image.rs` owns `image_save` and `image_preview` request/response
  contracts.

These modules may inspect backend-local model metadata and dispatch to a model
series/variant implementation, but they must not rename the capability into an
SDXL-specific concept or encode SDXL-only assumptions into the public operation
protocol. SDXL is the first V1 implementation behind those capabilities, not
the capability shape itself.

Node orchestration remains above the backend in the inference layer. The Candle
backend receives typed capability requests over abstract execution handles; it
does not define what `builtin.ksampler` or `builtin.clip_text_encode` means as a
workflow node. Its job is to interpret handles such as `Model`, `Clip`, `Vae`,
`Latent`, and `Conditioning` against backend-owned loaded bundles and payloads.

The same rule applies inside the Candle backend implementation. It is acceptable
for V1 to route a typed capability to an SDXL implementation because only SDXL
is currently loaded. It is not acceptable for the backend design to imply that
every future model family should receive a parallel copy of the same operation
infrastructure.

Preferred shape:

```text
operation/text.rs
  validates `text_encode`
  resolves loaded text encoder role(s)
  dispatches to a model graph / kernel adapter selected by loaded bundle metadata

operation/diffusion.rs
  validates `diffusion_sample`
  resolves diffusion model + scheduler configuration
  dispatches to a sampler/kernel adapter selected by loaded bundle metadata

operation/latent.rs
  validates `latent_decode`
  resolves VAE/image decoder role
  dispatches to a decoder adapter selected by loaded bundle metadata
```

The backend-private trait chain for sampling can be:

```text
inference::InferenceBackend::diffusion_sample(DiffusionSampleRequest)
  -> CandleBackend::diffusion_sample(...)
  -> LoadedModelGraph::diffusion_sampler(...)
  -> DiffusionSampler::sample(...)
  -> LoadedSdxlBundle / future model implementation
```

These backend-private traits may use Candle tensors, tokenizer state, scheduler
configuration, device policy, and loaded bundle metadata. They must not use
workflow `NodeTypeId` values or catalog metadata.

In this shape, `models/stable_diffusion/sdxl/*` is the first concrete backend
implementation selected by the loaded bundle. It is not the shape of the
capability protocol, and it should not be copied as a template for every future
model family.

Model-specific code may exist below `models/<series>/<variant>/...`, but it
should trend toward reusable backend-local traits or graph adapters where the
algorithm is common and only loaded weights/config differ. If the implementation
for a new model would duplicate most of SDXL's text/sampling/decode operation
logic, that is a signal to deepen the backend model graph abstraction before
adding the model.

Model-family code belongs below `models/<series>/<variant>/...`. For V1 this
means stable-diffusion SDXL helpers for loading, tokenization, text encoding,
sampling, and VAE decode under `models/stable_diffusion/sdxl/*`.

`store.rs` owns backend payload maps and safe access helpers. As real tensors
land, `CandlePayload` may become:

```text
CandlePayload
  Latent(...)
  Conditioning(...)
  Image(...)
```

Callers should access payloads through typed store methods such as
`get_latent`, `insert_conditioning`, or `take_image_for_save` rather than
matching on the payload enum throughout operation modules. This keeps lock
scope, error messages, and payload ownership policy centralized.

The variant bundle module owns the concrete loaded bundle type. The
inference-facing result of `load_bundle` remains three lightweight
`ExecutionValue` handles; the loaded Candle objects stay behind the cache entry.

## Resource And Device Lifecycle

Candle owns the concrete model and tensor lifecycle behind internal
`ExecutionValue` handles. Runtime owns only its `Arc<ExecutionValue>` records
and retention policy; Candle decides concrete cache and device behavior.

```text
Runtime value retention
  SingleUse
  RunScoped
  WorkspaceScoped

CandleBackend mechanisms
  pin or unpin loaded model bundles
  keep diffusion model hot across repeated KSampler calls
  release or pool run-scoped latent / conditioning / image tensors
  decide CPU / GPU / Metal / CUDA placement
  decide whether VAE decode runs on CPU while diffusion sampling continues
  report memory/cache observations without exposing Candle internals
```

For the common SDXL multi-image path, Candle should be able to keep the
diffusion model loaded across repeated `diffusion_sample` calls, reuse
conditioning tensors produced by one `text_encode`, and run `latent_decode`
for each completed latent without waiting for every sample in the run. The
runtime scheduler can emit value release and artifact completion events as
nodes finish, but Candle decides whether a cache owner keeps a payload pinned,
evicts it, pools it, or moves it between devices.

Candle's concrete mechanisms remain backend-local. Global resource policy, if
needed, belongs above individual backends because it needs the active-run,
execution-plan, and multi-backend view. Higher modules must not special-case
CLIP, UNet, VAE, SDXL tensors, or Candle device handles. They observe public
handles, diagnostics, artifact references, and memory snapshots only.

## Artifact Boundary

Image save/preview has two responsibilities that must stay distinct:

- Candle backend code may encode a backend-owned image payload and write bytes
  to a safe workspace-relative destination, or return a backend image/artifact
  intent that lets the executor write through a host capability.
- The inference executor records the artifact with runtime
  `NodeArtifactCapability` so run snapshots and run events stay host-neutral.

Typed backend requests must not carry `NodeArtifactCapability`. Runtime owns
the artifact store and event semantics. The backend owns only image
tensor/payload conversion and encoding details. A saved file path must stay
under the workspace output directory selected by app-host/config.

## Model Resolution Handoff

`checkpoint_loader` receives a workflow `ModelRef` as a static param. The
executor needs a host-supplied resolver capability that maps that `ModelRef`
to a resolved model descriptor/path before loading.

The dependency direction is:

```text
workflow ModelRef
  -> app-host ModelService::resolve_descriptor
  -> inference model resolver capability
  -> LoadBundleRequest
  -> inference::InferenceBackend::load_bundle(...)
  -> ExecutionValue::Model / Clip / Vae handles
```

The resolver capability should be defined at the app-host/inference boundary,
not inside runtime. Runtime passes params to the executor but does not resolve
models itself.

## Current Implementation Notes

The first executable SDXL path now runs through the real app-host/runtime/
inference/backend registration path and can produce a PNG artifact under the
workspace output directory. The current backend math is still a V1 placeholder
for the heavy model kernels; real weight-driven CLIP/UNet/VAE execution should
land behind the same typed backend capability protocol.

See [Real SDXL Roadmap](../real-sdxl-roadmap.md) for the tracked route from
the current placeholder path to real SDXL base inference.

Follow-up implementation should prioritize backend-internal model graph and
kernel adapter boundaries before adding another model family. If supporting a
new model would require copying SDXL text/sampling/decode operation modules,
deepen the backend abstraction first.
