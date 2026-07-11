<div align="center">

# Reimagine

**An agentic workflow studio for AIGC — node-based workflow editing**
**with a first-class agent loop, in a single Rust + React workspace.**

`GPL-3.0-or-later` · `Rust 1.96+` · `Bun`

</div>

---

AIGC workflows are typically hand-wired. The user authors a graph node by
node, wires up slots by hand, runs it, inspects the output, and tweaks a few
parameters — then repeats. The bottleneck is the editing loop itself:
composing new nodes, fixing validation errors, adjusting slot connections,
and translating *intent* ("make it more cinematic") into parameter changes.

Reimagine closes that loop. A first-class agent can read, author, and execute
workflows on the same runtime that powers the desktop and HTTP hosts. The
agent exposes a typed tool surface over the graph, the validation pipeline,
and the runtime scheduler — so a user can describe what they want, and the
agent composes a workflow, validates it, runs it, observes the diagnostics,
and iterates.

---


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

## One workspace, two hosts

A Tauri 2 desktop shell and an Axum HTTP host share the same
host-neutral facade (`reimagine-app-host`). The desktop host owns the
window, IPC channels, and runtime event hubs; the Axum host owns routing,
serialization, and a CLI for headless operation. Both call into the same
services; neither reimplements them.

The Axum host doubles as the canonical end-to-end test harness for
workflow execution.

## Dual inference backends

|                  | Candle                        | Burn (`wgpu` default)        |
| ---------------- | ----------------------------- | ---------------------------- |
| Apple Silicon    | ✅ Metal + Accelerate         | ✅ Metal via WGPU             |
| Linux / Windows  | ✅ CPU                        | ✅ Vulkan via WGPU            |
| CPU fallback     | —                             | ✅ `flex` backend            |

Both backends sit behind the same `reimagine-inference` facade: a backend
trait, typed request/response DTOs, capability report, model resolver,
router, and registry. Swapping backends is a workspace-config change; no
node or workflow change required.

## Local-first model stack

Manifest-based model references with fingerprinting, classification,
scanning, and resolution — paired with a HuggingFace download pipeline
(`hf-hub`). Models live in a workspace-scoped store the user fully
controls.

| Stage         | Crate                       | Responsibility                                       |
| ------------- | --------------------------- | ---------------------------------------------------- |
| Registration  | `reimagine-model-manager`   | Manifests, fingerprints, classification, resolution   |
| Acquisition   | `reimagine-model-acquisition` | HuggingFace download pipeline (`hf-hub`)           |

---

## Stack

| Layer             | Technology                                          |
| ----------------- | --------------------------------------------------- |
| Domain / runtime  | Rust 2024 workspace                    |
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


Domain logic lives under `crates/`; `src-tauri/` and `crates/axum-host/` are
thin host adapters that share the `reimagine-app-host` facade.

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