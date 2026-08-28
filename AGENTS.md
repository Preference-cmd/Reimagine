# Reimagine

**Agent for AIGC** — a Tauri + Burn + React desktop app with a native AI agent
that deeply understands the project canvas and its workflow DSL. The agent is a
peer operator, not an external tool; it shares the same domain model and
operation paths as the user, across image generation, video, animation, and
audio.

- `AGENTS.local.md` Additional local specs. Must loaded after this file if exists.

## Workspace

- `src-tauri/` - Tauri 2 binary crate. Thin shell and IPC layer for the desktop app.
- `crates/core/` - `reimagine-core`: project, board, workflow DSL, command/history, readiness, events. Pure Rust, no Tauri.
- `crates/config/` - `reimagine-config`: workspace config and path layout infrastructure.
- `crates/model-manager/` - `reimagine-model-manager`: manifest, scan, series config, resolver.
- `crates/nodes/` - `reimagine-nodes`: V1 builtin SDXL node catalog.
- `crates/runtime/` - `reimagine-runtime`: RuntimeService, scheduler, run/value store.
- `crates/inference/` - `reimagine-inference`: backend contract, router, execution values, built-in executors.
- `crates/inference-backends/` - Concrete backend adapters (Burn primary, Candle deprecated).
- `crates/app-host/` - `reimagine-app-host`: WorkspaceHost, Project/Board/Workflow/Agent services, Agent tools, provider config + adapter wiring.
- `crates/agent-stack/` - Agent libraries:
  - `ai-protocol/` - `reimagine-ai-protocol`: wire-protocol layer (`Protocol` discriminator, DTO translation, `CompletionBackend` seam). Transport-free.
  - `agent-harness/` - `reimagine-agent-harness`: host-neutral agent harness domain (tool registry, policy, provider boundary, Agent loop, ContextManager, `LlmModelCatalog`).
  - `agent-provider/` - `reimagine-agent-provider`: concrete provider adapters (reqwest OpenAI-compatible / Anthropic / Responses).
  - `agent-macros/` - `reimagine-agent-macros`: `#[agent_tool]` attribute macro. Candidate for removal or real adoption; see refine roadmap DP-4.
- `crates/agent-daemon/` - `reimagine-agent-daemon`: **frozen experimental sidecar**. Not used by the V1 production path.
- `crates/axum-host/` - `reimagine-axum-host`: HTTP E2E test harness.
- `ui/` - React 19 + Vite 7 frontend: infinite project canvas + WorkflowFrame editing, managed with Bun.
- `assets/` - static placeholders and non-secret assets.

## Agent Architecture

V1 runs the agent **embedded in `app-host`**, in the same process and the same
`WorkspaceHost` as the user's project state. There is exactly one
WorkflowService, one AgentService, one provider catalog, and one inference
runtime in the production path.

The agent operates as a **peer operator** on the project canvas. Human edits and
agent tool calls converge on the same command engines:

- Board edits go through `BoardCommand`.
- Workflow edits go through `WorkflowCommand`.
- Both support preview, version guards, undo/redo, and audit.

`agent-daemon` / `AgentBridge` are frozen. A sidecar may return later only if
there is a real headless, remote, or second-client requirement; that future
sidecar must not duplicate domain state.

```
┌─────────────────────────────────────────────────────────────────┐
│ Tauri App / app-host process                                     │
│                                                                  │
│  UI (React infinite canvas)                                      │
│   ├─ Board view                                                  │
│   ├─ WorkflowFrame editing                                       │
│   └─ AgentThread chat                                            │
│              │ Tauri IPC                                         │
│              ▼                                                   │
│  WorkspaceHost                                                   │
│   ├─ ProjectService                                              │
│   ├─ BoardService                                                │
│   ├─ WorkflowService          ← human + agent                    │
│   ├─ AgentService             ← AgentThread + AgentLoop          │
│   │    ├─ AgentToolRegistry                                      │
│   │    ├─ AgentProviderCatalog                                   │
│   │    └─ ContextManager per thread                              │
│   └─ Inference/RuntimeService                                    │
└─────────────────────────────────────────────────────────────────┘
```

## Commands

- Type-check: `cargo check --workspace`
- Test: `cargo test --workspace`
- Dev: `cd src-tauri && cargo tauri dev`
- Frontend build: `cd ui && bun install && bun run build`
- Release bundle: `cd src-tauri && cargo tauri build`
- Frozen experimental daemon build: `cargo build -p reimagine-agent-daemon`
- Frozen experimental daemon test: `cargo test -p reimagine-agent-daemon`

## Conventions

- Domain crates must not depend on `tauri`.
- `agent-daemon` is frozen; do not add production wiring that spawns it from Tauri.
- Agent layering mirrors the Pi agent toolkit: `agent-harness` (loop, tools, policy, model catalog) ← `ai-protocol` (Protocol discriminator, DTO translation, `CompletionBackend` seam — transport-free) ← `agent-provider` (concrete reqwest adapters, must not depend on `app-host`) ← `app-host` (provider config documents, adapter wiring, `AgentProviderCatalog`).
- Dependency versions are centralized in root `Cargo.toml` under `[workspace.dependencies]`.
- AI/ML inference backend code belongs in `crates/inference-backends/`.
- Do not commit generated build outputs, local runtime data, model weights, secrets, or machine-local planning files.
- Prefer the existing crate and module boundaries over adding cross-cutting logic to host crates.
- Agent tool implementations belong in `crates/app-host/src/tools.rs`, not in `agent-daemon`.
- Project is the domain aggregate root. Board, Workflow, Asset, Run, AgentThread, and ProjectMemory all belong to a Project.
- Canvas is UI view state, not a persisted domain document. Board is the persisted project canvas document.

