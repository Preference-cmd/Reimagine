<div align="center">

# Reimagine

**An agentic workflow studio for AIGC — node-based workflow editing**
**with a first-class agent loop, in a single Rust + React workspace.**

[![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue?style=flat-square)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.96%2B-orange?style=flat-square&logo=rust)](https://www.rust-lang.org)
[![Bun](https://img.shields.io/badge/Bun-black?style=flat-square&logo=bun)](https://bun.sh)
[![Tauri 2](https://img.shields.io/badge/Tauri-2-purple?style=flat-square&logo=tauri)](https://tauri.app)

</div>

AIGC workflows are usually hand-wired: compose nodes, wire slots, run,
inspect, tweak — then repeat. The bottleneck is the editing loop itself.

Reimagine closes that loop. A first-class agent reads, authors, and executes
workflows on the same runtime behind the desktop and HTTP hosts, exposing a
typed tool surface over the graph, validation pipeline, and scheduler.
Describe what you want — the agent composes, validates, runs, observes, and
iterates.



## Architecture

### Desktop app + headless server

The workflow engine runs in two modes: a **desktop app** with a visual
node editor, and a **headless server** for automation and scripting.
Workflows built in the GUI run on the server without changes.

### Cross-platform inference

Two interchangeable inference backends ship with the app. One is optimized
for Apple Silicon; the other covers Windows, Linux, and macOS with GPU
acceleration and a CPU fallback. Pick the best fit for your hardware.

### Local-first models

Models stay on your machine — downloaded from HuggingFace on demand,
verified for integrity, and managed through a local workspace store.
No accounts, no cloud, no telemetry.



## Stack

| Layer             | Technology                                          |
| ----------------- | --------------------------------------------------- |
| Domain / runtime  | Rust 2024 workspace                                 |
| Desktop host      | Tauri 2                                             |
| HTTP host         | Axum 0.8                                            |
| Inference         | Candle 0.10 · Burn 0.21 (`wgpu` default, `flex` CPU) |
| UI                | React 19 + Vite 7 (Bun)                             |

## Requirements

- Rust MSRV **1.96**
- Bun
- Platform prerequisites for **Tauri 2** (see the Tauri docs for your OS)
- A working **WGPU** adapter for the default Burn backend (Metal on Apple
  Silicon, Vulkan on Linux/Windows), or use the `flex` CPU backend

## Quick start

```bash
# Desktop host (Vite HMR + Tauri window)
cd ui && bun install && cd ..
cd src-tauri && cargo tauri dev

# HTTP host (script-driven / remote workflow execution)
cargo run -p reimagine-axum-host -- --base-path ./workspace

# Frontend build only
cd ui && bun run build
```

The Axum host defaults to `127.0.0.1:7878`; pass `--addr` to override.
Sample workflows live under `examples/workflows/`.

## Workspace layout

```text
src-tauri/        Tauri 2 desktop shell
crates/           Domain modules (workflow model, runtime, inference, agent, …)
ui/               React 19 + Vite 7 frontend
assets/           Static placeholders
examples/         Sample workflow JSON
```

`src-tauri/` and `crates/axum-host/` are thin host adapters that share the
`reimagine-app-host` facade; all domain logic lives under `crates/`.

## Conventions

- `src-tauri/` and `crates/axum-host/` are thin host adapters. Domain
  logic belongs in `crates/`.
- Domain crates must not depend on `tauri`.
- Dependency versions are centralized in the root `Cargo.toml` under
  `[workspace.dependencies]`.
- AI/ML inference backend code belongs in `crates/inference-backends/`.

## Local data

Generated build outputs, model weights, runtime workspaces, local
configuration, and private planning notes are intentionally ignored by
git. Put local runtime files under a workspace base path (for example
`--base-path ./workspace`) rather than committing them.

See [AGENTS.md](AGENTS.md) for the full workspace map and agent conventions.

## Status

Reimagine is currently in active development. There will be breaking
changes to internal APIs, node contracts, and the workflow document schema
while the agent–runtime–backend integration stabilizes. The runtime
semantics for built-in nodes are intended to be stable; expect schema
revisions for new capabilities.

## License

Reimagine is licensed under **GPL-3.0-or-later**. See [LICENSE](LICENSE).
Opening a pull request is assumed to signal agreement with these
licensing terms.