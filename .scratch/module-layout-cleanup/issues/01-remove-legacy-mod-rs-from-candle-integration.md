# Remove legacy mod.rs from candle integration

Status: ready-for-human

## What to build

Migrate the existing Candle integration model module away from the legacy `mod.rs` layout. The project standard is to use modern Rust module files and keep related code split by domain concept rather than collecting module declarations in `mod.rs`.

This slice should only change module organization. Do not change model loading behavior, runtime value design, Candle execution semantics, or public architecture decisions.

The expected shape is a facade file such as `crates/candle-integration/src/models.rs` that privately wires existing submodules like `models/clip.rs` and re-exports only the intended public surface.

## Acceptance criteria

- [ ] `crates/candle-integration/src/models/mod.rs` is removed.
- [ ] `crates/candle-integration` still exposes the same public model module surface needed by existing code.
- [ ] Existing model-family files remain split by concept, for example `models/clip.rs`.
- [ ] No new `mod.rs` files are added anywhere in the workspace.
- [ ] `find crates -name mod.rs -print` produces no output.
- [ ] `cargo check --workspace` passes.

## Blocked by

None - can start immediately.
