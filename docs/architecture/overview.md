# Reimagine Architecture Overview

> Status: working draft
> Last updated: 2026-06-13

## Purpose

Reimagine is a Tauri + Candle + React desktop application for node-based image generation workflows. V1 targets an SDXL base-only text-to-image workflow, first-class AI Agent collaboration, ComfyUI import, local model management, and a host-independent execution runtime.

This document is the architecture entry point. Module-level details live under `docs/architecture/modules/`.

## Architecture Principles

### Host adapters are thin

`src-tauri` is the V1 desktop host adapter. It owns Tauri IPC, desktop settings, window integration, and event bridging. It does not own workflow mutation, validation, Agent policy, execution scheduling, model inference, or ComfyUI mapping logic.

`axum-host` is a peer host adapter for remote/headless operation and backend E2E testing. It reuses the same `app-host` facade as Tauri instead of copying desktop logic into server routes.

### UI draft, Rust canonical truth

The UI owns draft editor state and view projection:

- React Flow nodes and edges used for interaction;
- selection, viewport, panels, and immediate affordances;
- projection of Rust-owned workflow state, history, diagnostics, and run events.

Rust owns canonical workflow state:

- workflow schema;
- validation;
- persistence;
- command application;
- history and undo/redo;
- execution readiness;
- run event schema.

### Human and Agent are peer operators

Human gestures and Agent tool calls both become structured workflow commands. They pass through the same `WorkflowSession`, validation, history, provenance, versioning, and diagnostics path.

Agent modes:

- `agent`: automatically applies allowed low-risk edits.
- `build`: produces a full-proposal diff over the existing workflow; V1 accepts or rejects the proposal as a whole.

### WorkflowCommand is the mutation language

`WorkflowCommand` means "change the canonical workflow." It is not a generic button/action/event language.

Workflow mutations use `WorkflowCommand`:

- add/remove node;
- connect/disconnect;
- update parameter;
- move/apply layout;
- annotate;
- update metadata.

Non-mutations are separate host/service actions:

- run/cancel workflow;
- save/open workflow;
- import ComfyUI workflow;
- start/send Agent session;
- list/rescan models.

Those actions may produce or apply workflow commands, but they are not themselves `WorkflowCommand` variants.

### Diagnostics are shared and structured

`core` defines a stable diagnostic model for command validation, workflow validation, execution readiness, adapter import, model discovery, Agent policy, and runtime failures.

Diagnostics are user/Agent-facing and structured. Logs/traces are developer/runtime-facing. They are linked with correlation IDs but remain separate systems.

### Modern Rust module layout

New Rust modules use the standard modern file layout and avoid the old `mod.rs` directory pattern. Prefer ordinary `mod foo;` declarations that resolve to `foo.rs` or `foo/bar.rs`; do not use `#[path = "..."]` unless there is a concrete interop reason. Large files should be split by domain concept before they become review-hostile.

## Crate and Host Map

```text
crates/core
  Pure domain kernel:
  - canonical workflow schema
  - shared NodeDef / SocketDef / ParamDef schema
  - WorkflowSession, WorkflowCommand, history, diagnostics
  - execution plan and RunEvent schema
  - backend-agnostic inference contracts

crates/nodes
  Built-in node package:
  - V1 built-in NodeDef catalog
  - static NodeRegistry
  - execution capabilities
  - ComfyUI aliases

crates/adapters
  External workflow adapters:
  - V1 ComfyUI import
  - loss diagnostics
  - future export shape

crates/agent
  Agent runtime:
  - AgentSession
  - agent/build policy
  - tool registry
  - WorkflowProposal and diff generation
  - Reimagine-owned AgentProvider trait

crates/agent-provider
  Agent provider adapters:
  - Rig-backed V1 provider adapter
  - OpenAI-compatible provider implementation
  - Anthropic provider implementation
  - provider config mapping to AgentProvider instances

crates/agent-macros
  Agent tool macro:
  - #[agent_tool] wrapper generation
  - schema metadata derivation
  - no policy bypass

crates/app-host
  Application service layer:
  - WorkspaceHost / AppHost facade
  - workflow session registry
  - model/runtime/config/agent composition
  - core readiness orchestration
  - concrete Agent tools

crates/config
  Workspace-scoped configuration infrastructure:
  - AppPaths and base_path layout
  - ConfigStore and typed ConfigHandle<T>
  - ConfigDocument trait
  - atomic JSON config persistence
  - config IO diagnostics

crates/model-manager
  Local model management:
  - model manifest schema and persistence
  - model roots and source status
  - model scanning and update policy
  - model series classification and id policy
  - model descriptor/readiness resolution

crates/runtime
  Host-independent execution runtime:
  - ExecutionRunner
  - RunSession
  - scheduler
  - cancellation
  - artifact routing
  - RunEventSink boundary

crates/candle-integration
  Candle backend:
  - Session
  - model loader/cache
  - SDXL base-only inference implementation

crates/axum-host
  HTTP host adapter:
  - V1 REST API for health, workflow open/run, run snapshot, and run events
  - app-host state injection
  - backend E2E workflow test harness

src-tauri
  V1 desktop host adapter:
  - IPC
  - settings/window integration
  - Tauri event bridge
  - app-host state injection

ui
  Editing experience:
  - React Flow draft graph
  - canonical projection
  - node library/properties
  - Agent panel/proposal review
  - run and diagnostic overlays
```

## V1 Scope

V1 includes:

- SDXL base-only text-to-image execution;
- UI/workflow/runtime end-to-end path;
- Rust-owned workflow history and undo/redo;
- V1 model management with local manifest, directory scanning, and resolver;
- single-file workflow JSON;
- `base_path` with `input/`, `output/`, `models/`, `workflows/`, and `config/`;
- workspace-scoped config infrastructure shared by module services;
- ComfyUI import-only;
- Agent as a V1 feature;
- Rig-backed provider layer for OpenAI-compatible endpoints and Anthropic;
- host-independent runtime crate.

V1 defers:

- SDXL refiner;
- ComfyUI export;
- third-party plugin loading;
- Python/WASM custom node runtime;
- model download;
- partial proposal acceptance;
- Agent auto-run;

## Workflow Format and Base Path

Workflow files are single JSON documents. They store graph structure, parameters, layout, annotations, provenance, and model references.

They do not store generated images, previews, run metadata, or run history. Local app-managed files live under `base_path`:

```text
base_path/
  input/
  output/
  models/
  workflows/
  config/
```

By default, the app creates an app-owned `base_path`. Users can select another directory in settings.

Model references in workflow JSON point to manifest `ModelId` values. Workflow JSON does not store model file paths.

## V1 Built-In Nodes

Input and utility:

- `core.text`
- `core.string`
- `core.integer`
- `core.float`
- `core.seed`
- `core.note`

Model:

- `core.checkpoint_loader`

Conditioning:

- `core.clip_text_encode`

Latent:

- `core.empty_latent_image`

Sampling:

- `core.ksampler`

VAE and image:

- `core.vae_decode`
- `core.preview_image`
- `core.save_image`

## Module Docs

- [Core](./modules/core.md)
- [App host](./modules/app-host.md)
- [Config](./modules/config.md)
- [Model manager](./modules/model-manager.md)
- [Runtime](./modules/runtime.md)
- [Nodes](./modules/nodes.md)
- [Agent](./modules/agent.md)
- [Agent provider](./modules/agent-provider.md)
- [Adapters](./modules/adapters.md)
- [Candle integration](./modules/candle-integration.md)
- [Tauri host](./modules/tauri-host.md)
- [UI](./modules/ui.md)
- [Axum host](./modules/axum-host.md)
