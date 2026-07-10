# Burn package workspace bootstrap and smoke workflow

Status: done

Depends on: burn/15i

## Parent

[Burn Axum real-image E2E](./05-burn-axum-real-image-e2e.md)

## Architecture source

- [Axum E2E Workflow Guide](../../../docs/architecture/e2e-workflow-axum.md)
- [Burn Backend Adapter Architecture](../../../docs/architecture/modules/burn-integration.md)

## What to build

Add the repeatable workspace bootstrap and repository example needed by the
Burn Axum E2E. The slice consumes an existing package report already inside a
dedicated workspace `models/` directory, imports it through `ModelService`,
selects `burn:wgpu:default`, and proves that the real host can open the Burn
smoke workflow with a truthful compute profile.

Add `examples/workflows/sdxl-base-burn-smoke-workflow.json` as a separate
example. Preserve the standard checkpoint, positive/negative CLIP, empty
latent, sampler, VAE decode, preview, and save graph. Fix the smoke envelope to
batch 1, 256x256, one Euler/normal step, and `denoise=1.0`. The initial model id
is `burn-real-sdxl-smoke-burn`.

## Acceptance criteria

- [x] The opt-in setup reads `REIMAGINE_BURN_AXUM_WORKSPACE` and
      `REIMAGINE_BURN_AXUM_PACKAGE_REPORT`.
- [x] The canonical package report path must remain under the workspace
      `models/` directory; an invalid or escaping path fails clearly.
- [x] Import uses `ModelService::import_burn_converted_package()` and validates
      the returned descriptor id instead of writing a test-only manifest.
- [x] Workspace config explicitly selects `burn:wgpu:default` and real
      `WorkspaceHost` bootstrap resolves that instance without Candle fallback
      execution hooks.
- [x] `GET /compute-profile` reports the Burn WGPU instance as available with
      every capability required by the smoke workflow.
- [x] The Burn smoke workflow is read from disk, validates successfully, and
      opens through the Axum workflow route.
- [x] The setup does not copy, convert, delete, or mutate package data.

## Non-goals

- Running inference to completion; that is e2e-workflow/05b.
- Editing the canonical 1024x1024, 30-step SDXL example.
- Model acquisition or conversion.

## Blocked by

- burn/15i (done)

## Outcome

Landed on main:

- `examples/workflows/sdxl-base-burn-smoke-workflow.json` — 256x256,
  one-Euler/normal-step Burn SDXL smoke workflow with model id
  `burn-real-sdxl-smoke-burn`.
- `crates/axum-host/tests/burn_real_e2e.rs` — opt-in ignored test gated
  on `REIMAGINE_BURN_AXUM_WORKSPACE` and
  `REIMAGINE_BURN_AXUM_PACKAGE_REPORT`. Imports via
  `ModelService::import_burn_converted_package`, writes
  `inference_backend.json` selecting `burn:wgpu:default`, asserts
  `/compute-profile` carries the truthful Burn capability set
  (LoadBundle, TextEncode, CreateEmptyLatent, DiffusionSample,
  LatentDecode, ImageSave, ImagePreview) and NOT ImageImport, and
  verifies `/workflows/open` accepts `sdxl-base-burn-smoke-workflow.json`.
- `crates/model-manager/src/manifest/burn_package.rs` — match
  `expected_component_role` casing to the actual converter output
  (lowercased).
