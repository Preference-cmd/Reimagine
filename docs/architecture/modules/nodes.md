# Nodes Module Architecture

> Status: working draft
> Crate: `crates/nodes`

## Role

`nodes` provides the V1 built-in node catalog, static registry, execution capability metadata, and external aliases. It uses `core` node-definition schemas and must not define a competing schema.

## Responsibilities

- Built-in `NodeDef` catalog.
- Static `NodeRegistry`.
- Execution capabilities.
- ComfyUI aliases.
- Node-local definition validation.

## Non-Responsibilities

- Canonical workflow storage.
- Core graph validation.
- ComfyUI parsing.
- Tauri IPC.
- Candle internals.
- Agent reasoning.

## Suggested Module Layout

```text
src/
  lib.rs
  def.rs
  registry.rs
  builtins.rs
  builtins/
    inputs.rs
    model.rs
    conditioning.rs
    latent.rs
    sampling.rs
    image.rs
  aliases.rs
  aliases/
    comfy.rs
```
