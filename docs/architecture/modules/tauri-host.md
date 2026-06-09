# Tauri Host Architecture

> Status: working draft
> Host: `src-tauri`

## Role

`src-tauri` is the V1 desktop host adapter. It is a thin connector between the frontend and `app-host`.

## Responsibilities

- Tauri IPC commands.
- Desktop settings and window integration.
- Tauri event bridge.
- Tauri state injection for `AppHost` / `WorkspaceHost`.
- Workdir selection UI integration.

## Non-Responsibilities

- Workflow mutation logic.
- Workflow session registry.
- Workflow readiness orchestration.
- Runtime scheduling.
- Agent reasoning.
- Candle inference.
- ComfyUI mapping.

Tauri commands should call `app-host` facade methods rather than directly composing `core`, `runtime`, `model-manager`, or `agent`.
