# Real SDXL Roadmap

> Status: tracked roadmap; not yet an implementation spec

The Axum E2E workflow can currently execute the SDXL-shaped graph and write a
PNG artifact, but the Candle backend still uses placeholder math for the heavy
model path. This document defines the route from that placeholder path to real
SDXL base inference while preserving the runtime / inference / backend
boundary.

## Current Placeholder Boundary

The current path proves the host/runtime shape:

```text
workflow ModelRef
  -> model-manager manifest resolution
  -> app-host inference runtime composition
  -> runtime node execution
  -> inference executor
  -> typed backend capability
  -> Candle backend placeholder implementation
  -> output PNG artifact
```

The current path does not yet prove real model math:

- model loading validates a readable `.safetensors` source, but does not load
  real SDXL CLIP / UNet / VAE weights;
- tokenizer code is a simplified shape-correct stand-in, not real CLIP BPE;
- text encode returns shape-correct placeholder conditioning;
- diffusion sampling is deterministic placeholder math;
- VAE decode is deterministic placeholder upscaling / remapping.

Real SDXL work must replace backend-private Candle internals behind the same
typed capabilities. It must not introduce SDXL-specific runtime execution
units, workflow schema fields, public inference operations, or node catalog
special cases.

## Fixed Architecture Constraints

The runtime execution unit remains a workflow node invocation:

```text
runtime scheduler
  -> inference::NodeExecutor::execute(NodeExecutionContext)
  -> inference executor
  -> typed inference request
  -> inference::InferenceRuntime router
  -> selected inference::InferenceBackend method
  -> backend-private Candle model graph / payload store
  -> inference::ExecutionValue outputs
```

The public capability names stay model-neutral:

```text
model.load_bundle
text.encode
latent.create_empty
diffusion.sample
latent.decode
image.save
image.preview
```

SDXL is the first concrete implementation behind these capabilities, not the
shape of the capability layer.

The Candle backend may own SDXL-specific implementation below:

```text
crates/inference-backends/candle/src/models/stable_diffusion/sdxl/
  bundle.rs
  tokenizer.rs
  text.rs
  diffusion.rs
  vae.rs
```

Higher layers may observe only backend-neutral handles, diagnostics, snapshots,
and artifact references. `candle_core::Tensor`, loaded model structs,
tokenizer state, scheduler graphs, and file handles must not leak into
runtime, app-host, Axum, Tauri, workflow JSON, or Agent tool outputs.

## First Supported Model Shape

V1 real SDXL should start with **single-file SDXL base safetensors** referenced
from the model manifest:

```text
<base_path>/models/checkpoints/sdxl_base_1.0.safetensors
```

Reasons:

- it matches the current E2E guide and manifest examples;
- it keeps model-manager resolution stable: `ModelRef` still maps to one
  `ModelDescriptor`;
- it avoids adding diffusers directory manifest semantics before real kernels
  are proven.

Diffusers directory layouts are explicitly deferred. Supporting them later
should be a model-manager + Candle loader extension, not a workflow JSON
change.

## Dependency Direction And Candidate Crates

The current workspace only depends on `candle-core` for the Candle backend.
Real SDXL will likely need additional backend-local dependencies. Candidate
crates include:

```text
candle-nn
candle-transformers
tokenizers
safetensors
hf-hub
image
```

Allowed direction:

```text
inference-backends/candle -> candle-* / tokenizers / safetensors / image
```

Forbidden direction:

```text
runtime -> candle-*
app-host -> candle tensor/model types
axum-host -> candle-*
src-tauri -> candle-*
model-manager -> candle-*
workflow JSON -> candle concepts
```

`hf-hub` is not required for V1 local execution. If added later, it should be
behind an explicit download/cache feature and must not make local manifest
resolution depend on network access.

The Candle project has official Stable Diffusion examples and safetensors
loading entry points. Those examples may guide backend-private implementation,
but Reimagine must still preserve its own typed capability boundary and
workspace/model-manager semantics.

## Device Target

The first real path should be **CPU-correct first**, with Metal treated as an
optimization target after correctness. The workspace config already carries a
`candle_device` label, so real SDXL should keep device selection behind Candle
backend configuration:

```json
{
  "schema_version": "1",
  "backend": "candle",
  "candle_device": "cpu"
}
```

Metal support can be wired through the same backend config once correctness and
test coverage exist. The runtime and inference executors must not branch on
CPU vs Metal.

## Slice Order

### 1. Real bundle metadata and weight loading

Goal: replace placeholder bundle construction with a backend-local loaded SDXL
bundle that reads the configured single-file safetensors checkpoint.

The slice should:

- validate the source file and format;
- load or index required tensors without exposing them outside Candle;
- populate `LoadedSdxlBundle` / model cache with role-specific handles for
  diffusion model, CLIP, and VAE;
- preserve `load_bundle` outputs as three typed execution handles:
  `Model`, `Clip`, `Vae`;
- fail clearly when required SDXL tensors are missing or unsupported.

It should not yet require full text encode, sampling, or VAE decode to be
production quality.

### 2. Real tokenizer and text encoder inputs

Goal: replace the simplified tokenizer with real CLIP BPE-compatible
tokenization for SDXL prompts.

The slice should:

- load tokenizer vocabulary/merge data from backend-local resources or model
  sidecar data;
- produce token tensors and attention masks compatible with both SDXL text
  encoders;
- keep tokenizer state inside the loaded bundle or backend-local graph;
- preserve the `text.encode` request/response shape.

### 3. Real CLIP text encode

Goal: implement `text.encode` using the loaded SDXL text encoder weights.

The slice should:

- run CLIP-L and CLIP-G paths as required by SDXL base;
- produce conditioning and pooled embeddings with the existing public
  `ExecutionConditioning` / runtime handle shape;
- keep text encoder modules and tensors backend-private;
- include deterministic tests for tensor shapes and non-placeholder behavior.

### 4. Scheduler and UNet sampling

Goal: implement `diffusion.sample` using loaded UNet weights and scheduler
configuration while keeping `builtin.ksampler` model-neutral.

The slice should:

- support the current workflow params: seed, steps, cfg, sampler, scheduler,
  and denoise;
- map those params to backend-local sampler/scheduler objects;
- produce latent tensors through backend-local payload storage;
- maintain backend affinity constraints through existing execution handles;
- avoid adding SDXL-specific executor code above the Candle backend.

### 5. Real VAE decode

Goal: implement `latent.decode` using the loaded SDXL VAE decoder weights.

The slice should:

- consume `Latent` and `Vae` handles through the existing typed capability;
- run real VAE decode into an image payload;
- preserve `image.save` / `image.preview` artifact recording semantics;
- keep artifact references host-neutral and output files under
  `<base_path>/output`.

### 6. Performance and resource tuning

Goal: improve repeated generations and device behavior without changing the
public workflow/inference/runtime boundary.

Focus areas:

- keep diffusion model cache hot across repeated KSampler calls;
- reuse text conditioning when prompt text does not change;
- allow VAE decode to run on CPU or another configured backend/device when
  safe;
- add backend-instance memory/cache observations for loaded models and
  run-scoped payloads;
- avoid introducing runtime commands that directly unload or move concrete
  Candle tensors.

## Testing Strategy

Real weights must not be committed.

Use three tiers of tests:

```text
Unit tests
  no large weights
  validate parsing, tensor-name mapping, tokenizer behavior, shape contracts,
  error messages, and path handling

Tiny/synthetic fixture tests
  checked in only if legally and practically small
  prove loader/schema paths without pretending to be SDXL quality

Manual/local E2E tests
  use developer-provided SDXL weights under <base_path>/models
  run through Axum E2E guide
  verify non-placeholder PNG artifact and event/snapshot behavior
```

CI should continue to run without real model weights. Tests that require real
weights must be opt-in, ignored by default, or gated behind an explicit local
environment/config flag.

## Acceptance For "Real SDXL Base"

The placeholder label can be removed only when:

- a real SDXL base single-file safetensors checkpoint loads from the manifest;
- real tokenizer + CLIP text encode produce conditioning;
- real UNet/scheduler sampling produces latent output;
- real VAE decode produces image output;
- the canonical Axum E2E workflow can generate a PNG from a developer-provided
  SDXL checkpoint;
- no runtime, app-host, Axum, Tauri, workflow JSON, or Agent tool result
  exposes Candle tensors or SDXL-specific internals.

## Deferred

- SDXL refiner.
- Diffusers directory model layout.
- Remote model download or `hf-hub` cache management.
- Quantization and low-memory variants.
- Multi-backend component placement policy.
- Persistent run/artifact history.
- UI model download and progress flows.
