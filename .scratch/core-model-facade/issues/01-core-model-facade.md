# Add core model facade and shared domain data types

Status: ready-for-agent

## What to build

Create the shared `core::model` facade in `crates/core` and move the common domain data model into focused modules. This gives downstream crates one stable import path for IDs, node definitions, socket/param kinds, artifact references, and shared runtime values.

The facade should use private submodules plus public re-exports, so callers import from `reimagine_core::model::{...}` rather than from internal paths.

Use the modern Rust module layout. Do not introduce `mod.rs`. Keep files focused by concept instead of piling the shared data model into one large file.

## Acceptance criteria

- [ ] `reimagine_core::model::{NodeValue, NodeDef, SocketKind, ParamKind, ArtifactRef, WorkflowId, NodeId, EdgeId, RunId}` or their agreed equivalents are available through the facade.
- [ ] `model.rs` declares private submodules and re-exports the public model surface.
- [ ] Shared model files are split by concept, such as IDs, values, node definitions, sockets, params, artifacts, and time wrappers.
- [ ] `TensorData` belongs to the shared value/model layer, with existing inference contracts updated to use it from there.
- [ ] `core::model` does not contain workflow sessions, history, Agent tools, provider config, ComfyUI schema, Candle tensor types, or Tauri IPC DTOs.
- [ ] No new `mod.rs` files are added.
- [ ] `cargo check --workspace` passes.

## Blocked by

None - can start immediately.
