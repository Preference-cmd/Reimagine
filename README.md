<div align="center">

# Reimagine

**The AI agent that understands your AIGC workflows**
**Node-based editing + native agent, in a single Rust + React workspace.**

[![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue?style=flat-square)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.96%2B-orange?style=flat-square&logo=rust)](https://www.rust-lang.org)
[![Bun](https://img.shields.io/badge/Bun-black?style=flat-square&logo=bun)](https://bun.sh)
[![Tauri 2](https://img.shields.io/badge/Tauri-2-purple?style=flat-square&logo=tauri)](https://tauri.app)

</div>

AIGC workflows are usually hand-wired: compose nodes, wire slots, run,
inspect, tweak — then repeat. The bottleneck is the editing loop itself.
External agents (Comfy MCP) can help, but they don't understand node graph
semantics and can't operate as true collaborators.

Reimagine puts an AI agent at the center. The agent reads, authors, and
executes workflows on the same runtime behind the desktop and HTTP hosts,
exposing a typed tool surface over the graph, validation pipeline, and
scheduler. It is not a chatbot bolted onto a node editor — it is a peer
operator that shares the same domain model and operation paths as the user.
Describe what you want; the agent composes, validates, runs, observes, and
iterates alongside you.



## Architecture

### Agent-first design

Reimagine is built around a native AI agent that deeply understands AIGC
workflows. The agent is not an external tool talking through MCP — it is a
peer operator that shares the same domain model and operation paths as the
user. Every modification, whether from a human click or an agent tool call,
flows through the same `WorkflowCommand` path with full undo/redo support.

Starting from image generation, the agent's domain understanding extends
across video, animation, and audio — composing cross-domain pipelines from
a single natural-language intent.

### Desktop app + headless server

The workflow engine runs in two modes: a **desktop app** with a visual
node editor, and a **headless server** for automation and scripting.
Workflows built in the GUI run on the server without changes.

### Cross-platform inference

Burn is the primary inference backend: it runs on Metal, Vulkan or the CPU
`flex` backend, so one workflow runs on Apple Silicon, Windows, and Linux.
Candle, the earlier backend, is deprecated and kept only as a V1 fallback.

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

All 24 Cargo workspace members are listed here; `Cargo.toml` `[workspace]` is the
authority.

```text
src-tauri/                       reimagine - Tauri 2 desktop shell, thin IPC layer
crates/core/                     reimagine-core - project, board, workflow DSL, commands, history, events
crates/config/                   reimagine-config - workspace config, path layout
crates/model-manager/            reimagine-model-manager - manifest, scan, series config, resolver
crates/model-acquisition/        reimagine-model-acquisition - HF catalog, component resolution, staged download
crates/nodes/                    reimagine-nodes - V1 builtin SDXL node catalog
crates/runtime/                  reimagine-runtime - RuntimeService, scheduler, run/value store
crates/inference/                reimagine-inference - backend contract, router, execution values, executors
crates/inference-backends/       candle/ burn/ burn-worker/ - concrete backends (Burn primary, Candle deprecated)
crates/plugin/                   reimagine-plugin - plugin package and extension metadata contract
crates/app-host/                 reimagine-app-host - WorkspaceHost, Project/Board/Workflow/Agent services, agent tools
crates/axum-host/                reimagine-axum-host - HTTP E2E test harness
crates/agent-daemon/             reimagine-agent-daemon - frozen experimental sidecar
crates/agent-stack/              agent-harness/ agent-legacy-stack/ agent-extension/ agent-macros/
crates/backend-worker-protocol/  reimagine-backend-worker-protocol - worker wire protocol
crates/backend-worker-host/      reimagine-backend-worker-host - out-of-process worker management
crates/backend-worker-transport-quic/   reimagine-backend-worker-transport-quic - QUIC transport
crates/backend-worker-transport-grpc/   reimagine-backend-worker-transport-grpc - gRPC transport
crates/backend-worker-transfer/  reimagine-backend-worker-transfer - transfer planning and execution
ui/                              React 19 + Vite 7 frontend, Bun-managed
assets/                          static placeholders and non-secret assets
examples/                        sample workflow JSON
```

`src-tauri/` and `crates/axum-host/` are thin host adapters that share the
`reimagine-app-host` facade; all domain logic lives under `crates/`.

The agent-stack crates are frozen or staging code. The open-source publish set
lives in the sibling repository `../agent-stack-oss` (`Causa`: `causa`,
`causa-kernel`, `causa-protocol`, `causa-runtime`, `causa-provider`,
`causa-extension`) and is consumed through path dependencies while the
extraction is in transition.

## Conventions

Repository conventions - enforced layering, module layout, commit titles,
generated catalogs, dependency policy - are in [AGENTS.md](AGENTS.md)
`Conventions` and `Layering`. Do not keep a second copy here.

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