# Reimagine

A Tauri + Burn + React desktop app for node-based image generation workflows
with an integrated AI agent daemon.

- `AGENTS.local.md` Additional local specs. Must loaded after this file if exists.

## Workspace

- `src-tauri/` - Tauri 2 binary crate. Keep it as a thin shell and IPC layer.
- `crates/core/` - `reimagine-core`: workflow DAG, command/history, readiness, events. Pure Rust, no Tauri.
- `crates/config/` - `reimagine-config`: workspace config infrastructure.
- `crates/model-manager/` - `reimagine-model-manager`: manifest, scan, series config, resolver.
- `crates/nodes/` - `reimagine-nodes`: V1 builtin SDXL node catalog.
- `crates/runtime/` - `reimagine-runtime`: RuntimeService, scheduler, run/value store.
- `crates/inference/` - `reimagine-inference`: backend contract, router, execution values, built-in executors.
- `crates/inference-backends/` - Concrete backend adapters (Burn primary, Candle deprecated).
- `crates/app-host/` - `reimagine-app-host`: WorkspaceHost, service facade, Agent tools.
- `crates/agent/` - `reimagine-agent`: tool registry, policy, provider boundary, Agent loop, ContextManager.
- `crates/agent-daemon/` - `reimagine-agent-daemon`: standalone daemon binary. JSON-RPC over stdio, self-contained execution engine with full WorkspaceHost.
- `crates/axum-host/` - `reimagine-axum-host`: HTTP E2E test harness.
- `ui/` - React 19 + Vite 7 frontend, managed with Bun.
- `assets/` - static placeholders and non-secret assets.

## Agent Architecture

The agent runs as an independent daemon process (`agent-daemon`) that owns a
complete `WorkspaceHost` — including tool registry, provider catalog, and all
workspace tools. Communication with the Tauri app uses JSON-RPC 2.0 over
stdio (newline-delimited JSON).

```
┌──────────────────┐  JSON-RPC  ┌──────────────────────────────────────┐
│  Tauri App        │ ◄───────► │  Agent Daemon                         │
│  ├─ UI (React)   │  stdio    │  ├─ WorkspaceHost (full init)         │
│  ├─ Workflow State│           │  │   ├─ AgentService                  │
│  └─ Thin Bridge  │           │  │   │   ├─ AgentLoop                  │
│     (just IPC)   │           │  │   │   ├─ AgentToolRegistry          │
└──────────────────┘           │  │   │   │   └─ workspace tools        │
                                │  │   │   └─ AgentProviderCatalog       │
                                │  │   └─ WorkflowService               │
                                │  ├─ ContextManager (per session)       │
                                │  ├─ Session Persistence               │
                                │  └─ JSON-RPC Transport                │
                                └──────────────────────────────────────┘
```

## Commands

- Type-check: `cargo check --workspace`
- Test: `cargo test --workspace`
- Dev: `cd src-tauri && cargo tauri dev`
- Frontend build: `cd ui && bun install && bun run build`
- Release bundle: `cd src-tauri && cargo tauri build`
- Daemon build: `cargo build -p reimagine-agent-daemon`
- Daemon test: `cargo test -p reimagine-agent-daemon`

## Conventions

- Domain crates must not depend on `tauri`.
- `agent-daemon` depends on `app-host` (not `tauri`) — it is host-neutral.
- Dependency versions are centralized in root `Cargo.toml` under `[workspace.dependencies]`.
- AI/ML inference backend code belongs in `crates/inference-backends/`.
- Do not commit generated build outputs, local runtime data, model weights, secrets, or machine-local planning files.
- Prefer the existing crate and module boundaries over adding cross-cutting logic to host crates.
- Agent tool implementations belong in `crates/app-host/src/tools.rs`, not in `agent-daemon`.

