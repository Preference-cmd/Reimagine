# reimagine

Reimagine is an agentic workflow studio for AIGC.

It combines node-based workflow editing, AI agent collaboration, local model
management, and high-performance inference into a desktop-first creative
environment.

> Reimagine is in early development. The public repository currently focuses on
> the workflow model, runtime scheduler, host layer, agent plumbing, local model
> management, and inference backend scaffolding. The UI and real inference path
> are still under active construction. Real model weights and local workspace
> data are not included.

## Features

- Node-based workflow editing and execution.
- AI agent collaboration for workflow construction and iteration.
- Local model management with manifest-based model references.
- Host-independent runtime shared by desktop and HTTP interfaces.
- High-performance local inference foundation with Candle as the first backend.
- Structured validation, diagnostics, run events, and artifact routing.

Reimagine is still early. The current Candle SDXL path is a vertical-slice
example for validating workflow execution and artifact output while real model
execution is being completed.

## Stack

- **Domain/runtime**: Rust workspace
- **Desktop host**: Tauri 2
- **HTTP host**: Axum
- **Inference backend**: Candle first, behind backend adapter boundaries
- **UI**: React 19 + Vite 7
- **Frontend tooling**: Bun

## Requirements

- Rust 1.96 or newer
- Bun
- Platform prerequisites for Tauri 2

## Quick start

```bash
cd ui && bun install && cd ..
cd src-tauri && cargo tauri dev
```

## Development commands

```bash
cargo check --workspace
cargo test --workspace
cd ui && bun install && bun run build
```

Run the Axum host for API-oriented workflow testing:

```bash
cargo run -p reimagine-axum-host -- --addr 127.0.0.1:7878
```

Pass `--base-path <path>` to use a specific workspace directory.

## Layout

- `src-tauri/` — Tauri shell and IPC layer.
- `crates/` — Rust domain crates.
- `crates/core/` — workflow data model, commands, validation, planning, and events.
- `crates/runtime/` — workflow execution runtime.
- `crates/inference/` — inference-facing abstractions, executors, routing, and execution values.
- `crates/inference-backends/` — concrete inference backend adapters.
- `crates/app-host/` — shared host assembly used by app frontends.
- `crates/axum-host/` — HTTP host for local API testing.
- `ui/` — React frontend.
- `assets/` — static placeholders and non-secret assets.

## Local data

Generated build outputs, model weights, runtime workspaces, local configuration,
and private planning notes are intentionally ignored by git. Put local runtime
files under a workspace base path rather than committing them to the repository.

See [AGENTS.md](AGENTS.md) for the full workspace map and agent conventions.

## License

Reimagine is licensed under GPL-3.0-or-later. See [LICENSE](LICENSE).
