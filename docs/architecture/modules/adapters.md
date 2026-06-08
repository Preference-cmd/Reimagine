# Adapters Module Architecture

> Status: working draft
> Crate: `crates/adapters`

## Role

`adapters` converts external workflow formats into canonical Reimagine workflows. V1 supports ComfyUI import-only.

## Responsibilities

- Parse ComfyUI JSON.
- Map external node types and parameters through `nodes` aliases.
- Produce canonical workflow data or command batches.
- Emit loss/unsupported diagnostics.

## Non-Responsibilities

- V1 ComfyUI export.
- Core validation rules.
- Runtime execution.
- UI rendering.
