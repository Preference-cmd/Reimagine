# UI Module Architecture

> Status: working draft
> Package: `ui`

## Role

The UI provides the human editing experience and projects Rust canonical state into React Flow. It does not own canonical workflow semantics.

## Responsibilities

- React Flow draft graph.
- Canonical workflow projection.
- Node library and properties panels.
- Agent panel and proposal review.
- Diagnostic and run overlays.
- Rust-owned history/undo/redo controls.

## Non-Responsibilities

- Authoritative validation.
- Agent tool execution.
- Runtime execution.
- Model scanning.
- ComfyUI conversion.
