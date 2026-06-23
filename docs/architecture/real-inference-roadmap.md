# Real Inference Roadmap

> Status: tracked roadmap; not yet an implementation spec

The Axum E2E workflow can currently execute the SDXL-shaped graph and write a
PNG artifact, but the Candle backend still uses placeholder math for the heavy
model path. This document defines the route from that placeholder path to real
local inference while preserving the runtime / inference / backend boundary.

SDXL base is the first concrete example used to validate the stack. It is not
the architecture target. New infrastructure must be justified by model-neutral
inference concepts such as model loading, text encoding, diffusion sampling,
latent decoding, artifact writing, backend routing, and resource observation.
Do not design runtime, inference, node catalog, app-host, or host DTOs as
SDXL-specific systems.

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

Real inference work must replace backend-private Candle placeholder internals
behind the same typed capabilities. The first replacement target happens to be
SDXL base, but it must not introduce SDXL-specific runtime execution units,
workflow schema fields, public inference operations, host DTOs, or node catalog
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
shape of the capability layer. If supporting another diffusion model would
require copying large portions of public executor/runtime infrastructure, the
abstraction is wrong and should be revisited before adding that model.

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

## Workspace Capability Discovery

Real inference needs a workspace-level initialization phase that reports
hardware and backend capability information to top-level hosts before a user
starts a run. This initialization is coordinated by `app-host` during
`WorkspaceHost` bootstrap. Concrete backend crates participate by reporting
host-neutral profiles; they do not own the workspace service graph.

```text
Axum / Tauri
  -> app-host WorkspaceHost bootstrap
  -> load workspace config
  -> discover registered backend providers
  -> collect backend/device/memory profiles
  -> validate persisted user backend/device selection
  -> construct backend instances and routing policy
  -> construct inference runtime and runtime service
```

The discovery result must be returned to top-level host surfaces so the UI or
HTTP clients can show available compute targets and let users save a preferred
configuration:

```text
WorkspaceComputeProfile
  generated_at
  backend profiles
  diagnostics

BackendProfile
  backend implementation label
  backend instance candidates
  plugin / extension provenance
  capability support
  diagnostics

DeviceProfile
  id                 # "cpu:0", "metal:0", "cuda:0", "remote:..."
  name
  accelerator        # cpu | metal | cuda | remote | unknown
  location           # local | remote | unknown
  available
  memory summary     # optional / approximate
  supported dtypes
  diagnostics
```

These profiles are DTOs and observations, not backend handles. They must not
contain Candle `Device`, CUDA/Metal handles, loaded tensors, model graph
objects, or OS-specific resource owners.

Persisted config stores the user's selection and fallback preferences, not the
full discovery snapshot:

```json
{
  "schema_version": "1",
  "backend": "candle",
  "device": "metal:0",
  "fallback_devices": ["cpu:0"],
  "precision": "fp16"
}
```

On startup, app-host re-runs discovery and validates the saved selection. If it
is unavailable, bootstrap should produce diagnostics and either use a configured
fallback or a conservative default. Runtime and inference executors must not
scan hardware, branch on CPU/GPU, or persist discovered system details.

This shape is intentionally compatible with future remote runtime support. A
remote provider can later report the same `BackendProfile` / `DeviceProfile`
shape; app-host can import those as backend instance candidates without
rewriting runtime scheduling or workflow JSON.

## First Example Model Sources

The first real inference example should support both checkpoint-bundle and
split-component model sources through the model manifest:

```text
<base_path>/models/checkpoints/sdxl_base_1.0.safetensors

or

<base_path>/models/diffusion/...
<base_path>/models/clip/...
<base_path>/models/vae/...
<base_path>/models/tokenizers/...
```

Supported V1 source shapes:

- checkpoint bundle: one model descriptor resolves the model roles needed by
  the loaded graph, such as diffusion model, CLIP/text encoder, VAE, and
  tokenizer resources;
- split components: separate descriptors can resolve diffusion model, CLIP/text
  encoder, VAE, and tokenizer resources independently, while the backend builds
  one compatible loaded component graph for the run.

Workflow JSON still stores stable `ModelRef` values, not local file paths or
Candle-specific model layout fields. Model-manager owns descriptor persistence
and path resolution. The Candle backend owns interpretation of the resolved
sources and compatibility checks for the loaded component graph.

Diffusers directory layouts remain a possible loader extension. Supporting
them should be a model-manager + backend loader change, not a workflow JSON
change and not a reason to fork runtime/inference abstractions.

## Dependency Direction And Candidate Crates

The current workspace only depends on `candle-core` for the Candle backend.
The first real inference example will likely need additional backend-local
dependencies. Candidate crates include:

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

The Candle project has official Stable Diffusion examples that support multiple
Stable Diffusion variants, local file inputs, and GPU feature flags. Those
examples may guide backend-private device construction, safetensors loading,
tokenizer/model wiring, and scheduler defaults. Reimagine must still preserve
its own typed capability boundary, workspace config, and model-manager
semantics. Candle example CLI ownership, automatic download behavior, and
direct output-file flow must not leak upward into app-host, runtime, or
workflow JSON.

## Device Target

The first real path should support CPU and GPU-capable Candle devices. Exact
availability depends on the local build features and platform, but the
architecture must not hard-code CPU-only behavior. Device selection stays
behind app-host bootstrap, backend discovery, and inference routing
configuration.

Runtime and inference executors must not branch on CPU, Metal, CUDA, or future
remote devices. They call typed capabilities through `InferenceRuntime`; the
selected backend instance owns concrete device construction and execution.

## Slice Order

### 0. Workspace hardware profile and backend discovery

Goal: add a top-level discovery surface so app-host can report backend,
device, memory, dtype, and capability profiles to Axum/Tauri/UI and validate
user backend/device configuration during workspace bootstrap.

The slice should:

- define host-neutral compute/backend/device profile DTOs in the shared
  app-host/inference boundary;
- let the Candle backend report CPU and GPU-capable device candidates without
  exposing Candle `Device` handles;
- expose the latest workspace compute profile through app-host and host
  adapters;
- validate saved backend/device selection against the discovery result and
  produce diagnostics for unavailable selections;
- keep runtime out of hardware probing and config persistence.

It should not yet implement real SDXL kernels.

### 1. Model source loading and component graph

Goal: replace placeholder bundle construction with a backend-local loaded model
component graph that can be built from either a checkpoint bundle or split
component sources. SDXL base is the first graph implementation.

The slice should:

- validate source files, formats, and component compatibility;
- load or index required tensors/resources without exposing them outside
  Candle;
- resolve tokenizer resources as part of backend model loading rather than as
  runtime or workflow concepts;
- populate the backend model cache with role-specific handles for diffusion
  model, CLIP/text encoder, VAE, tokenizer state, and metadata;
- preserve `load_bundle` outputs as three typed execution handles:
  `Model`, `Clip`, `Vae`;
- fail clearly when required tensors/resources are missing or unsupported.

It should not yet require full text encode, sampling, or VAE decode to be
production quality.

### 2. Real tokenizer and text encoder inputs

Goal: replace the simplified tokenizer path with real tokenizer execution for
the loaded model graph. For the first example this means CLIP BPE-compatible
tokenization for SDXL prompts.

The slice should:

- use tokenizer resources resolved by backend model loading;
- produce token tensors and attention masks compatible with both SDXL text
  encoders;
- keep tokenizer state inside the loaded bundle or backend-local graph;
- preserve the `text.encode` request/response shape.

### 3. Real text encode for the example model

Goal: implement `text.encode` using the loaded SDXL text encoder weights.

The slice should:

- run CLIP-L and CLIP-G paths as required by SDXL base;
- produce conditioning and pooled embeddings with the existing public
  `ExecutionConditioning` / runtime handle shape;
- keep text encoder modules and tensors backend-private;
- include deterministic tests for tensor shapes and non-placeholder behavior.

### 4. Scheduler and diffusion sampling

Goal: implement `diffusion.sample` using loaded diffusion weights and scheduler
configuration while keeping `builtin.ksampler` model-neutral. For the first
example this means SDXL UNet sampling, but the public capability remains
`diffusion.sample`.

The slice should:

- support the current workflow params: seed, steps, cfg, sampler, scheduler,
  and denoise;
- map those params to backend-local sampler/scheduler objects;
- produce latent tensors through backend-local payload storage;
- maintain backend affinity constraints through existing execution handles;
- avoid adding SDXL-specific executor code above the Candle backend.

### 5. Real latent decode for the example model

Goal: implement `latent.decode` using the loaded example model's decoder
weights. For the first example this means SDXL VAE decode, but the public
capability remains `latent.decode`.

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
  select CPU or GPU backend/device config from discovery
  run through Axum E2E guide
  verify non-placeholder PNG artifact and event/snapshot behavior
```

CI should continue to run without real model weights. Tests that require real
weights must be opt-in, ignored by default, or gated behind an explicit local
environment/config flag.

## Acceptance For "Real Local Inference, SDXL Example"

The placeholder label can be removed only when:

- app-host can report available backend/device profiles and validate the
  selected CPU or GPU backend instance;
- real SDXL base checkpoint-bundle and split-component sources can load through
  the manifest/model resolver path;
- real tokenizer + CLIP text encode produce conditioning;
- real UNet/scheduler sampling produces latent output;
- real VAE decode produces image output;
- the canonical Axum E2E workflow can generate a PNG from a developer-provided
  SDXL checkpoint;
- no runtime, app-host, Axum, Tauri, workflow JSON, or Agent tool result
  exposes Candle tensors or SDXL-specific internals.

## Deferred

- SDXL refiner.
- Remote model download or `hf-hub` cache management.
- Quantization and low-memory variants.
- Multi-backend component placement policy.
- Persistent run/artifact history.
- UI model download and progress flows.
