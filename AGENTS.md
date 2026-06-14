# reimagine

A Tauri + Candle + React desktop app for node-based image generation workflows — a ComfyUI alternative.

## Workspace

- `src-tauri/` — Tauri 2 binary crate (shell + IPC handlers).
- `crates/core/` — `reimagine-core`: workflow DAG + executor. Pure Rust, no Tauri.
- `ui/` — React 19 + Vite 7 frontend (bun-managed).
- `assets/` — static resources (models, sample images).
- `docs/agents/` — per-repo agent config (tracked).
- `docs/architecture/` — architecture source of truth (tracked).
- `docs/design/` — design source material (tracked).

## Commands

- Type-check: `cargo check --workspace`
- Test: `cargo test --workspace`
- Dev (Vite HMR + Tauri window): `cd src-tauri && cargo tauri dev`
- Frontend build only: `cd ui && bun install && bun run build`
- Release bundle: `cd src-tauri && cargo tauri build`

## Conventions

- Tauri commands in `src-tauri/src/lib.rs` are thin wrappers; domain logic lives in `crates/*`.
- Domain crates must not depend on `tauri` — keep them unit-testable without a Tauri runtime.
- Dependency versions are centralized in root `Cargo.toml` under `[workspace.dependencies]`.
- AI/ML inference code belongs in `crates/inference-backends/candle/`, never in `src-tauri/`.

## Agent skills

### Issue tracker

Local markdown — issues and PRDs live as files under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — one local `CONTEXT.md` plus tracked `docs/architecture/`. See `docs/agents/domain.md`.
