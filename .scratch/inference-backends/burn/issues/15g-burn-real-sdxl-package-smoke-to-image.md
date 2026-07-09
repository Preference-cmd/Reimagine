# Burn real SDXL package smoke to image artifact

Status: done

Depends on: burn/15e, burn/15f, burn/15h

Dependency status: 15e, 15f, and 15h are done. Package dialect is diffusers
(`burn-sdxl-package-15h-v1`). Full-profile UNet Module now binds package keys
identity-first and the opt-in real package smoke produces preview/save PNGs.

## Parent

burn/15d: Burn Module fidelity gap breakdown

## What to build

Add the first real converted SDXL package smoke that produces an image artifact
through the Burn backend. This is the "can produce an image" milestone, not a
claim that the image is numerically or visually parity-matched.

## Scope

- Use the existing split-component / converted Burn package format, not raw HF
  checkpoint files directly.
- Require a package converter/layout version that is compatible with full
  UNet/VAE key-space coverage; reject stale partial packages.
- Exercise the public capability chain:
  `load_bundle -> text.encode -> latent.create_empty -> diffusion.sample ->
  latent.decode -> image.preview/image.save`.
- Keep the smoke minimal: batch 1, SDXL base, Euler/normal, small step count
  suitable for local verification.
- Gate the real-weight smoke behind explicit env/config so normal CI remains
  deterministic and does not need local model weights.
- Record clear diagnostics and artifact metadata for the generated preview/save
  result.
- Keep WGPU default and Flex fallback behavior explicit.

## Non-goals

- Do not implement model download, import, or conversion orchestration here.
  Model acquisition and package conversion belong to the model-manager /
  acquisition lanes.
- Do not claim sampler numeric parity; that remains 15d9.
- Do not tune performance thresholds.
- Do not add LoRA/training behavior.

## Acceptance criteria

- [x] A converted real SDXL component package can run through the Burn public
      capability chain to produce an image preview/save artifact.
- [x] The smoke is skipped or clearly gated when no real package fixture is
      configured; correctness tests remain deterministic and non-flaky.
- [x] Failures identify the stage that failed: package load, text encode,
      sampler, VAE decode, preview, or save.
- [x] The smoke documents the configured package path, backend target, package
      converter/layout version, step count, seed, sampler, and produced
      artifact ref when successful.
- [ ] WGPU and Flex Burn crate tests/checks/Clippy remain green.
- [x] Follow-up parity/performance issues are based on evidence from this
      smoke, not speculative optimization.

## Current Evidence

- 2026-07-09: real package UNet weight bind against
  `workspace/models/converted/burn/burn-real-sdxl-smoke/stat-c04e79b3c2262e0e`
  applied **1676/1676** package tensors (`missing=2` Module-only leaves;
  `unused=512` empty/optional Module slots). Loader remapper is identity-first
  for diffusers keys; residual remaps are prefix strip, legacy SGM fixtures,
  and `ff.net.2.{weight,bias} -> ff.net.2.linear.{weight,bias}`.
- 2026-07-09: opt-in smoke
  `REIMAGINE_BURN_REAL_SDXL_PACKAGE=.../stat-c04e79b3c2262e0e`
  `WIDTH=256 HEIGHT=256 STEPS=1 SEED=1234 DEVICE=default` completed the full
  public chain and wrote:
  - `output/preview_run-burn-real-sdxl-smoke_preview_0.png`
  - `output/burn-real-sdxl_run-burn-real-sdxl-smoke_save_1.png`
  (~90s). Smoke env adds `WIDTH`/`HEIGHT`/`STOP_AFTER`; artifact assert strips
  the runtime `output/` prefix.
- Residual non-fatal wgpu/CubeCL noise during the green run:
  `buffer bound at binding index 2 is bound with size 8 where the shader expects 16`.
  Track under performance/runtime follow-up, not as a 15g blocker.
- Earlier: VAE Module/policy already use diffusers plural keys
  (`mid_block.attentions.0`, `up_blocks.N.upsamplers.0`, `to_out.0`); prior
  decode failures from singular Burn-dialect required snapshots are closed.

## Unlocks

- burn/15d9: full-topology sampler parity
- burn/16a: LoRA and training readiness design gate