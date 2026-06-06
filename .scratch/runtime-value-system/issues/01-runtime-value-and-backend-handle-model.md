# Runtime value and backend handle model

Status: ready-for-human

## What to build

Introduce the first runtime-layer value model for SDXL execution without making `runtime` depend on Candle concrete types. The goal is to let a run pass model, clip, vae, latent, conditioning, image, artifact, and param values between nodes while keeping backend-native payloads behind lightweight handles.

This slice should define the data shapes only. It should not implement the scheduler, node executor protocol, Candle loading, or SDXL nodes.

The implementation should respect the existing architecture decisions:

- `core::model::NodeValue` remains the public semantic value model.
- `runtime::RuntimeValue` is the per-run execution value model.
- `RunValueStore` should eventually store `RuntimeValue`, not raw `NodeValue`.
- Runtime values may reference backend-native payloads through handles, but must not expose `candle_core::Tensor` or other Candle concrete types.
- Use modern Rust module layout. Do not add `mod.rs` files.

## Acceptance criteria

- [ ] A runtime-layer value module exists in `crates/runtime` or the appropriate runtime crate location agreed by the current workspace structure.
- [ ] `RuntimeValue` includes V1 variants for param, model, clip, vae, latent, conditioning, image, artifact, and null values.
- [ ] Runtime model handles distinguish model, clip, and vae roles without exposing backend payload types.
- [ ] Backend tensor handles carry enough metadata for scheduling and diagnostics, including backend identity, payload key, dtype, shape, and device label.
- [ ] SDXL-oriented runtime wrappers exist for latent, conditioning, and image values.
- [ ] `RuntimeConditioning` can represent SDXL text embeddings plus optional pooled embeddings and conditioning metadata such as width, height, crop, and target size.
- [ ] Public constructors/accessors make the types usable without exposing internal fields unnecessarily.
- [ ] Tests verify the public runtime value API can express the minimal SDXL base workflow intermediate values:
  - checkpoint loader outputs model, clip, and vae handles;
  - prompt encode outputs positive and negative conditioning values;
  - empty latent and sampler outputs latent values;
  - VAE decode outputs an image value;
  - save image can return an artifact value.
- [ ] Tests verify no Candle type is required to construct or inspect runtime values.
- [ ] No new `mod.rs` files are added.
- [ ] `cargo test -p reimagine-runtime` passes if a runtime crate exists; otherwise the focused crate test command for the newly introduced runtime value module passes.
- [ ] `cargo check --workspace` passes.

## Blocked by

- [02-refine-core-model-ids-and-tensor-data.md](../../core-model-facade/issues/02-refine-core-model-ids-and-tensor-data.md)
