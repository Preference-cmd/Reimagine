# Tauri Host Architecture

> Status: working draft
> Host: `src-tauri`

## Role

`src-tauri` is the V1 desktop host adapter. It is a thin connector between the frontend and reusable Rust crates.

## Responsibilities

- Tauri IPC commands.
- Desktop settings and window integration.
- Tauri event bridge.
- Host stores/handles for workflow, run, and Agent sessions.
- Workdir selection UI integration.

## Non-Responsibilities

- Workflow mutation logic.
- Runtime scheduling.
- Agent reasoning.
- Candle inference.
- ComfyUI mapping.
