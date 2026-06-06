# Refine core model IDs and TensorData

Status: ready-for-human

## What to build

Refine the shared `core::model` surface after the initial facade landed. The goal is to make shared IDs type-safe and ergonomic across modules, and make `TensorData` backend-neutral, cheap to clone, and convenient for Candle conversion from `crates/candle-integration`.

Keep implementation inside `crates/core` except for tests needed to verify the public interface. Do not add `mod.rs` files.

## Acceptance criteria

- [ ] Shared ID newtypes include `WorkflowId`, `NodeId`, `EdgeId`, `RunId`, `ArtifactId`, `DiagnosticId`, `HistoryEntryId`, `CommandBatchId`, `ProposalId`, and `ModelId`.
- [ ] Shared ID newtypes expose `new(...)`, `as_str()`, `Display`, `From<String>`, and `From<&str>`.
- [ ] If available without adding unnecessary dependencies, shared ID newtypes derive or implement serde `Serialize` / `Deserialize` as string values.
- [ ] `TensorData` stores data as private `Arc<[f32]>` plus private shape data.
- [ ] `TensorData` exposes `from_vec(...)`, `as_slice()`, `to_vec()`, `shape()`, and `numel()`.
- [ ] Existing inference contracts continue to use `core::model::TensorData`.
- [ ] Tests verify ID ergonomics through the public `reimagine_core::model::{...}` facade.
- [ ] Tests verify `TensorData` cheap-clone semantics through public behavior, not private fields.
- [ ] No new `mod.rs` files are added.
- [ ] `cargo test -p reimagine-core` and `cargo check --workspace` pass.

## Blocked by

- [01-core-model-facade.md](./01-core-model-facade.md)
