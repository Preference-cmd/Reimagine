# Runtime / Inference / Backend Review

> Date: 2026-06-15
> Status: recorded
> Scope: `crates/runtime`, `crates/inference`, `crates/inference-backends/candle`

## Summary

The current three-module shape is directionally sound:

- `runtime` owns workflow run state, scheduling, cancellation, value storage,
  snapshots, summaries, run events, and artifact recording.
- `inference` owns built-in node orchestration and the typed backend capability
  router/registry that lets runtime node execution call backend capabilities
  without binding executors to one backend.
- `inference-backends/candle` owns Candle-specific model loading, tensor stores,
  device policy, operation implementations, and artifact encoding.

The main risk is not the existence of the seams. The risk is that some modules
are becoming shallow or drifting from first principles as the SDXL vertical slice
lands.

## Highest-priority design risk

### Standard operations must not become model-series implementations

Recent issues added text encode, diffusion sample, and latent decode milestones
with SDXL as the first working path. That was useful for closing the vertical
slice, but it creates a design risk: model-specific modules can start looking
like each model needs its own text encoder, sampler, and decoder infrastructure.

First-principles rule:

- Runtime's execution unit is a workflow node invocation. It is not an SDXL
  text encoder invocation, an SDXL sampler invocation, or any other
  model-specific execution unit.
- `text_encode` is a typed standard backend capability.
- `diffusion_sample` is a typed standard backend capability.
- `latent_decode` is a typed standard backend capability.
- `inference` knows how to orchestrate built-in nodes over abstract handles and
  typed backend capabilities, but it does not know what concrete model object
  was loaded.
- Only the concrete `inference-backend` knows what model architecture, weights,
  tokenizer, scheduler graph, and device allocations sit behind a runtime
  handle.
- Model families, variants, and checkpoints should select loaded weights,
  model config, tensor shapes, schedulers, tokenizers, conditioning schemas, and
  kernel graphs behind those operations.
- They should not force duplicated operation modules or duplicated executor
  infrastructure per model family.

The correct seam is:

```text
inference executor
  -> typed backend capability method
  -> inference runtime/router
  -> backend adapter
  -> loaded model bundle / model graph / kernel adapter
```

The incorrect drift is:

```text
SDXLTextEncode
SDXLKSampler
SDXLVaeDecode
FluxTextEncode
FluxKSampler
FluxVaeDecode
...
```

Backend implementation may still have model-specific internals, but those
internals should be selected through loaded model metadata and capabilities, not
through a growing set of model-specific public operations.

## Review candidates

### 1. Deepen the runtime scheduler

Current friction:

- `docs/architecture/modules/runtime.md` describes DAG stage parallelism.
- Current `Runner::run_to_completion` executes nodes in each stage
  sequentially.
- Scheduling, node context construction, snapshot publishing, failure policy,
  and cancellation policy are concentrated in one large implementation.

Candidate:

- Keep `RuntimeService::run(plan, inputs, options) -> RunHandle` as the public
  interface.
- Move stage execution policy into an internal scheduler module that can own
  stage concurrency, fail-fast sibling cancellation, snapshot cadence, and
  cancellation checks.

Why:

- Locality: scheduling bugs concentrate in one module.
- Leverage: runtime tests exercise scheduling through the same public run
  interface.
- The public runtime interface remains plan-oriented and host-neutral.

### 2. Collapse operation mapping boilerplate

Current friction:

- Most inference executors repeat the same pattern:
  read inputs/params, build a typed backend request, call backend, validate
  response.
- The repeated code makes executor modules shallow: each interface is nearly as
  complex as the implementation.

Candidate:

- Introduce shared executor helpers for input preparation, output construction,
  backend error mapping, and response validation.
- Keep concrete node executors readable and explicit; do not introduce a
  stringly `operation_id` dispatch layer as the primary backend call mechanism.

Why:

- Locality: mapping and validation rules concentrate.
- Leverage: new built-in inference nodes become cheaper to add.
- Tests cover one deeper module instead of many shallow adapters.

### 3. Carry artifact references through runtime observations

Current friction:

- `ArtifactRecord` stores `ArtifactRef`.
- `RunArtifactRef` currently exposes only artifact id and node id.
- Hosts can observe that an artifact exists, but not the actual
  workspace-relative destination.

Candidate:

- Add `ArtifactRef` to runtime snapshot/summary artifact observations and host
  DTOs.
- Keep `NodeArtifactCapability` as the only artifact recording path.

Why:

- Locality: artifact semantics stay in runtime's artifact module.
- Leverage: Tauri, Axum, and UI can render or open the produced artifact without
  backend-specific knowledge.

### 4. Split Candle payload storage from image artifact writing

Current friction:

- `CandleStore` is useful and should remain, but payload lifetime, model cache,
  memory accounting, and image artifact persistence now change for different
  reasons.
- `operation/image.rs` contains request translation, path safety, PNG encoding,
  and artifact response construction.

Candidate:

- Keep `CandleBackend` as the external adapter.
- Split internal modules into payload store, model cache, and image artifact
  writer/encoder.

Why:

- Locality: file/path/encoding bugs do not live beside tensor lifetime policy.
- Leverage: future formats or remote artifact strategies can change one module.

### 5. Repair context and documentation drift

Current friction:

- `CONTEXT.md` is local ignored project context and can drift from the tracked
  architecture docs.
- Architecture docs can accidentally describe intended behavior, such as
  parallel runtime stage execution, without clearly marking the current code
  state.

Candidate:

- Keep `CONTEXT.md` and module docs synchronized with the current crate layout
  and distinguish current implementation from intended follow-up.
- Keep `.scratch/issue-status.md` aligned with architecture docs when issues
  become implementation-ready.

Why:

- Locality: architectural decisions are found in one place.
- Leverage: humans and agents navigate the current codebase with fewer stale
  assumptions.
