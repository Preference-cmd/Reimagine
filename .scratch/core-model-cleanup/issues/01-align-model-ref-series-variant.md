# Align model refs with model series and variant

Status: ready-for-human

## What to build

Update the committed core model naming to match the current architecture decision:

- replace `ModelFamily` with `ModelSeries` and `ModelVariant`;
- replace `ModelRole::Denoiser` with `ModelRole::DiffusionModel`;
- update `ModelRef` so it stores `model_series`, `variant`, and `role`;
- update runtime value tests and public facade tests accordingly.

This is a naming and data-shape cleanup only. Do not implement model manager, manifest persistence, scanning, runtime loading, or SDXL nodes in this slice.

## Acceptance criteria

- [ ] `core::model` publicly exports `ModelSeries` and `ModelVariant`.
- [ ] `ModelRef` exposes `model_series()` and `variant()` accessors.
- [ ] No committed Rust code refers to `ModelFamily`.
- [ ] No committed Rust code refers to `ModelRole::Denoiser`.
- [ ] Runtime model handle tests use `ModelRole::DiffusionModel`.
- [ ] `cargo test -p reimagine-core` passes.
- [ ] `cargo test -p reimagine-runtime` passes.
- [ ] `cargo check --workspace` passes.

## Blocked by

None - can start immediately.
