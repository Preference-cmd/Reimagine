<div align="center">

# Reimagine

**An agent-native project canvas for AIGC**
**Materials, notes, workflows and runs on one canvas, in a single Rust + React workspace.**

[![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue?style=flat-square)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.96%2B-orange?style=flat-square&logo=rust)](https://www.rust-lang.org)
[![Bun](https://img.shields.io/badge/Bun-black?style=flat-square&logo=bun)](https://bun.sh)
[![Tauri 2](https://img.shields.io/badge/Tauri-2-purple?style=flat-square&logo=tauri)](https://tauri.app)

</div>

Creative AIGC projects sprawl: reference material in one place, node graphs in
another, outputs in a folder, decisions in a chat log. Node editors such as
ComfyUI cover one part of the job - wiring and running a graph - and external
agents that drive them over MCP see tool calls, not the project around them.

Reimagine keeps the whole project in one place. Each project is an infinite
canvas holding source materials, notes, workflow frames, runs and generated
artifacts. Workflows are a first-class DSL on that canvas, not the whole
product. An AI agent is meant to be the main way to work: you describe what you
want, and it gathers and organizes material, builds and previews workflows,
runs them and iterates with you - starting with image generation and extending
to video, animation and audio.



## Architecture

### One operation pipeline for humans and the agent

Humans and the agent operate the same pipeline in `reimagine-app-host`. Canvas
edits go through `BoardCommand` and workflow edits through `WorkflowCommand`;
both can be previewed, undone and redone, and every change records who made it
(`CommandActor`). The agent is a peer operator that shares the user's domain
model and operation paths - not a chatbot bolted onto an editor, and not an
external tool that only sees an API.

Today the pipeline is driven by humans only: the legacy agent integration has
been removed, and a new one built on Causa is pending. Editing, saving and
undo/redo work fully without it.

### Project, Board and Workflow

A **Project** is the top-level container. Its **Board** is the persisted canvas
document; board items reference workflows, assets and runs, or hold notes. A
**Workflow** is a node graph stored as a single JSON document; on the canvas it
appears as a workflow frame. Runs and generated files stay outside the workflow
document, under the workspace base path.

### Desktop app

The engine runs as a **desktop app** with the canvas editor (`src-tauri`).

### Cross-platform inference

Burn is the inference backend: it runs on Metal, Vulkan or the CPU
`flex` backend, so one workflow runs on Apple Silicon, Windows, and Linux.

### Local-first models

Models stay on your machine — downloaded from HuggingFace on demand,
verified for integrity, and managed through a local workspace store.
No accounts, no cloud, no telemetry.



## Stack

| Layer             | Technology                                          |
| ----------------- | --------------------------------------------------- |
| Domain / runtime  | Rust 2024 workspace                                 |
| Desktop host      | Tauri 2                                             |
| Inference         | Burn 0.21 (`wgpu` default, `flex` CPU)              |
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

# Frontend build only
cd ui && bun run build
```

Sample workflows live under `examples/workflows/`.

## Workspace layout

All 16 Cargo workspace members are listed here; `Cargo.toml` `[workspace]` is the
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
crates/inference-backends/       burn/ burn-worker/ - concrete backend
crates/mod-sdk/                  mod-sdk - mod manifest, interface, JSON-RPC protocol, SDK, registry, process host, conformance kit
mods/text/                       text-mod - first-party text mod: six deterministic document operations over the mod protocol
crates/aigc/                     aigc - optional AIGC module: inference composition, worker management, model services
crates/app-host/                 reimagine-app-host - WorkspaceHost, Project/Board/Workflow/Run services, proposals
crates/backend-worker-protocol/  reimagine-backend-worker-protocol - worker wire protocol, gRPC contract, local and TLS transports
crates/backend-worker-host/      reimagine-backend-worker-host - out-of-process worker management
ui/                              React 19 + Vite 7 frontend, Bun-managed
assets/                          static placeholders and non-secret assets
examples/                        sample workflow JSON
```

`src-tauri/` is the thin desktop host adapter driving the
`reimagine-app-host` facade; all domain logic lives under `crates/`.

No workspace crate depends on an agent stack. The open-source agent kernel
lives in the sibling repository `../agent-stack-oss` (`Causa`: `causa`,
`causa-kernel`, `causa-protocol`, `causa-runtime`, `causa-provider`,
`causa-extension`). A future agent integration will depend on it from an
operator adapter crate above `reimagine-app-host`, alongside the Tauri desktop
host; nothing depends on it today.

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

Reimagine is in active development; internal APIs, node contracts and the
workflow document schema will still change. Where it stands today:

- The Project, Board and Workflow model, its command pipeline and undo/redo are
  implemented in Rust. The desktop UI is a board, library and activity shell
  for notes and materials, without a workflow-graph editor.
- Image generation (SDXL base on Burn) is the only domain implemented; video,
  animation and audio are planned.
- The agent is not wired in; see the architecture section above.

## License

Reimagine is licensed under **GPL-3.0-or-later**. See [LICENSE](LICENSE).
Opening a pull request is assumed to signal agreement with these
licensing terms.