# reimagine

A Tauri + Candle + React desktop app for node-based image generation workflows.

If `AGENTS.local.md` exists, read it after this file.

## Workspace

- `src-tauri/` - Tauri 2 binary crate. Keep it as a thin shell and IPC layer.
- `crates/` - Rust domain crates. Domain logic belongs here, not in `src-tauri/`.
- `ui/` - React 19 + Vite 7 frontend, managed with Bun.
- `assets/` - static placeholders and non-secret assets.

## Commands

- Type-check: `cargo check --workspace`
- Test: `cargo test --workspace`
- Dev: `cd src-tauri && cargo tauri dev`
- Frontend build: `cd ui && bun install && bun run build`
- Release bundle: `cd src-tauri && cargo tauri build`

## Conventions

- Domain crates must not depend on `tauri`.
- Dependency versions are centralized in root `Cargo.toml` under `[workspace.dependencies]`.
- AI/ML inference backend code belongs in `crates/inference-backends/`.
- Do not commit generated build outputs, local runtime data, model weights, secrets, or machine-local planning files.
- Prefer the existing crate and module boundaries over adding cross-cutting logic to host crates.
