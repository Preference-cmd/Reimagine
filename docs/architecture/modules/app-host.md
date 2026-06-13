# App Host Module Architecture

> Status: working draft
> Crate: `crates/app-host`

## Role

`app-host` is the application service and composition layer. It assembles the reusable domain crates into one host-neutral surface for Tauri, future Axum, and Agent workflows.

It is not a concrete host adapter. It must not contain Tauri command attributes, Axum routes, React/UI state, or backend inference kernels.

## Responsibilities

- Own the application/workspace service facade.
- Hold workspace-level services and registries.
- Coordinate config, workflow sessions, model manager, runtime, node executors, and Agent service.
- Build model readiness snapshots and pass them to core readiness.
- Call `core` to validate workflows, apply workflow commands, and build execution plans.
- Call `runtime` to run prepared execution plans.
- Provide concrete Agent tools over workflow/model/diagnostic operations.
- Offer a shared API surface for Tauri, future Axum, and Agent tool execution.

## Non-Responsibilities

- Workflow command semantics.
- Runtime scheduling semantics.
- Model scanning and manifest rules.
- Candle tensor kernels or model loading internals.
- Tauri IPC binding.
- Axum routing or WebSocket/SSE transport.
- Agent provider SDK details.
- UI projection state.

## Dependency Direction

```text
app-host -> core
app-host -> config
app-host -> model-manager
app-host -> runtime
app-host -> candle-integration
app-host -> agent
app-host -> agent-provider
app-host -> agent-macros

src-tauri -> app-host
future axum-host -> app-host
```

Reusable domain crates must not depend on `app-host`.

`agent` should not depend on `app-host`. Instead, `app-host` depends on `agent` and injects concrete tools or tool capabilities into Agent sessions.

## State Shape

```text
AppHost
  workspace: Arc<WorkspaceHost>
```

```text
WorkspaceHost
  services: Arc<WorkspaceServices>
  agent_service

WorkspaceServices
  workspace_scope
  config
  workflow_service
  model_service
  runtime_service
  node_catalog
```

`WorkspaceHost` is the shared application state center. Tauri and future Axum hold it through their own host state mechanisms. Agent sessions are bound to one workspace, not to global `AppHost` state.

`WorkspaceServices` is the app-host service container captured by concrete Agent tools. Tools capture `Arc<WorkspaceServices>`, not `Arc<WorkspaceHost>`, so they cannot recurse back through `AgentService` or mutate the registry while handling a tool call.

V1 can keep `AppHost` single-workspace:

```text
AppHost
  workspace: Arc<WorkspaceHost>
```

The type shape should still make the workspace boundary explicit so future multi-workspace support can route Agent sessions and host requests to the correct `WorkspaceHost`.

`app-host` owns the unified bootstrap entry that assembles `WorkspaceHost`. Tauri and Axum adapters should receive an already-built workspace handle rather than duplicating service composition.

Workspace construction follows a fixed capability-surface flow:

```text
WorkspaceHost::new
  -> build Arc<WorkspaceServices>
  -> build AgentToolRegistry
  -> register_app_tools(&mut registry, Arc<WorkspaceServices>)
  -> freeze registry as Arc<AgentToolRegistry>
  -> create AgentService with workspace_scope and registry
  -> return WorkspaceHost
```

The V1 registry is frozen after workspace construction. Built-in tools are not dynamically added or removed while Agent sessions are running.

Typical bootstrap flow:

```text
app-host bootstrap
  -> resolve AppConfig / AppPaths
  -> build WorkflowService / ModelService / RuntimeService / NodeRegistry
  -> construct WorkspaceHost
  -> hand Arc<WorkspaceHost> to src-tauri or axum-host
```

## Service Facades

`WorkflowService` owns app-level workflow session management:

```text
WorkflowService
  sessions: WorkflowId -> WorkflowSession
  proposals: WorkflowProposalStore
```

It coordinates:

- opening workflow JSON;
- saving workflow JSON;
- applying `WorkflowCommand` batches;
- previewing command batches;
- creating and approving workflow proposals;
- building readiness plans for a selected target.

`core` still owns the rules for a single workflow/session. `WorkflowService` owns the app-level registry.

`WorkflowService` is also the persistence boundary for workflow JSON in V1. It may expose save/open helpers for host adapters, but graph mutation still goes through `WorkflowSession` command preview/apply APIs from `core`.

`ModelService` wraps `model-manager` operations:

- load/save manifest;
- scan model roots;
- list models;
- resolve `ModelRef`;
- produce readiness diagnostics/snapshots for core readiness.

`ModelService` must not change model-manager validation, scan, identity, or resolver rules. It owns the host-facing cache/snapshot of the latest manifest and delegates domain decisions to `model-manager`.

Core readiness expects a synchronous `ExternalReadinessProvider`, while model-manager resolution is async because it can touch the filesystem. `app-host` bridges this by building a snapshot before calling core:

```text
ModelService::build_readiness_snapshot(workflow)
  -> collect ModelRef subjects from the workflow/session snapshot
  -> asynchronously resolve each ModelRef through model-manager
  -> store diagnostics keyed by ExternalReadinessSubject
  -> return SnapshotExternalReadinessProvider

core::readiness::build_execution_plan(..., Some(&snapshot_provider))
  -> synchronously reads diagnostics from the snapshot
```

No async work should happen inside the core readiness callback.

`RuntimeService` is provided by `crates/runtime` and is used through a host-neutral API:

```text
run(plan, run_inputs, run_options, sink) -> RunHandle
cancel(run_id)
snapshot(run_id)
summary(run_id)
```

## Run Workflow Flow

```text
AppHost::run_workflow(workflow_id, target_selection, run_inputs)
  -> WorkflowService returns workflow/session snapshot
  -> ModelService builds an external readiness snapshot
  -> core::readiness::build_execution_plan(...)
  -> if readiness report has errors, return diagnostics
  -> RuntimeService::run(plan, run_inputs, options, sink)
  -> return run id / initial snapshot
```

This flow belongs in `app-host`, not in `runtime` and not in `src-tauri`.

`run_workflow` is a host action in V1. Agent tools do not expose runtime run/cancel, so Agent-created workflow edits must be accepted/applied first and then a human/host action may call `run_workflow`.

## Agent Tool Boundary

Agent tools may use an attribute macro to keep tool metadata close to implementation:

```rust
#[agent_tool(
    name = "workflow.preview_commands",
    description = "Preview workflow command changes",
    modes = ["agent", "build"],
    permission = "workflow.read"
)]
async fn preview_commands(
    ctx: ToolContext,
    input: PreviewCommandsInput,
) -> ToolResult<PreviewCommandsOutput> {
    // calls app-host workflow facade
}
```

Tool abstractions live in `crates/agent`:

```text
AgentTool
ToolRegistry
ToolContext
ToolPolicy
ToolSpec
```

`ToolContext` is an execution context supplied by `app-host` at call time. The type lives in `agent`, but it remains generic and host-neutral. App-specific access should be captured by concrete app-host tool functions/closures instead of being stored inside `ToolContext`.

```text
ToolContext
  workspace_scope
  agent_session_id
  mode
  correlation_id
  actor
  permissions

app-host concrete tool
  captures Arc<WorkspaceServices>
  receives ToolContext and typed input
  verifies context workspace_scope matches captured workspace
  calls app-host workflow/model/diagnostic facade
```

The workspace match check is part of app-host's tool boundary. A tool call with a mismatched workspace scope is an authorization/orchestration error, not a workflow command validation error.

The `#[agent_tool]` macro lives in `crates/agent-macros`. The macro generates schema and wrapper code only. It must not bypass policy.

Concrete tool functions live in `app-host` because they need access to workflow, model, runtime, and diagnostic services.

V1 uses explicit registration:

```text
register_app_tools(registry, Arc<WorkspaceServices>)
```

V1 does not use distributed auto-registration such as `inventory`.

`AgentService` receives the frozen registry from `WorkspaceHost` construction. It owns Agent sessions and registry access, but it does not discover concrete app tools itself.

## Agent Turn Orchestration

`AgentService` is the app-host boundary that turns the crate-local Agent loop into a workspace feature. It owns workspace-scoped session lookup, provider lookup, and event sink wiring, then delegates the actual turn lifecycle to `crates/agent::AgentLoop`.

```text
AgentService::run_turn(request)
  -> load AgentSession from workspace-scoped session store
  -> resolve session.provider through AgentProviderCatalog
  -> build AgentLoop(provider, event_sink)
  -> call AgentLoop::run_turn(AgentTurnRequest)
  -> return AgentTurnResult plus any app-host projection needed by host adapters
```

Provider selection is app-host/provider-adapter state, not an `agent` crate concern. V1 may use a deterministic test/mock provider catalog to prove the orchestration path. Real OpenAI-compatible, Anthropic, or Rig-backed provider adapters live in `crates/agent-provider` and register `Arc<dyn AgentProvider>` values behind the same catalog boundary.

The V1 provider catalog should stay minimal:

```text
AgentProviderCatalog
  providers: ProviderName -> Arc<dyn AgentProvider>
```

Unknown provider names are host orchestration errors. They should not panic and should not cause app-host to fabricate a tool observation, because provider lookup happens before the Agent loop starts.

`AgentService` should not call concrete tools directly. Tool execution still goes through the session's frozen `AgentToolRegistry` and the policy gate inside `AgentLoop`. This keeps the provider loop, app-host concrete tools, and workflow command policy in one deterministic path.

`AgentService::run_turn` may return `reimagine_agent::AgentTurnResult` directly in V1. Add an app-host wrapper only if the host must return extra app-host data such as collected events. Do not create a parallel app-host turn lifecycle vocabulary.

Agent session permissions are explicit. `AgentService` should expose a session creation path that accepts `PermissionSet` rather than silently granting broad write access. Host adapters decide which permissions a user/session receives.

Agent-local events are emitted through an injected `AgentEventSink`. App-host may initially use `VecAgentEventSink` in tests and later bridge the same stream through `AgentDomainEventAdapter` into the common host event/report pipeline for Tauri or future Axum. Tool observations remain provider-visible messages; host events and diagnostics remain separate projections.

V1 Agent tools include:

```text
workflow.get
workflow.preview_commands
workflow.propose_commands
workflow.apply_commands
model.list
model.resolve_ref
diagnostics.for_workflow
```

Workflow command tools use app-host workflow sessions:

```text
workflow.preview_commands
  -> WorkflowService.preview_batch(...)
  -> returns CommandResult / diff / diagnostics

workflow.propose_commands
  -> WorkflowService.preview_batch(...)
  -> stores a pending proposal and returns ProposalReceipt
  -> does not mutate workflow

workflow.apply_commands
  -> checks Agent mode and ToolPolicy
  -> requires preview success
  -> checks WorkflowCommandPolicy for auto-apply eligibility
  -> agent mode may auto-apply only low-risk editor-only command batches
  -> build mode does not directly mutate workflow through Agent tools
```

`app-host` is responsible for converting Agent tool input into `core::CommandBatch` values with `CommandActorKind::Agent` and the correct provenance. Core remains responsible for command validation, preview, apply, history, undo/redo, and diagnostics.

Workflow proposal state belongs to `app-host`, not `agent` and not `core`. A proposal stores:

```text
WorkflowProposal
  proposal_id
  workflow_id
  base_version
  agent_session_id
  command_batch
  preview_result
  created_at
  status: pending | accepted | rejected | superseded
```

`workflow.propose_commands` previews the batch and returns a `ProposalReceipt` without mutating the workflow:

```text
ProposalReceipt
  proposal_id
  workflow_id
  base_version
  preview_result
  status: pending
  effective: false
```

V1 stores only pending proposals in the proposal store. One pending proposal per workflow is allowed; a new proposal supersedes the older pending proposal for that workflow.

`workflow.apply_commands` may apply a direct policy-approved agent-mode batch. Build-mode proposal approval is a host/app-host API, not an Agent tool call. When a host/human approves a pending proposal, app-host applies the stored command batch with `CommandProvenance::AgentProposal` and records the approver. V1 accepts or rejects proposals as a whole.

Tool results and host events are distinct:

```text
Tool output
  returned to the Agent loop as model-observable result

DomainEvent / Diagnostic
  emitted for UI, audit, host streams, and future Axum/Tauri adapters
```

Mutation-capable tool outputs include `effective`. `effective = false` means the model received a valid observation, but workflow state did not change.

V1 Agent tools must not expose:

```text
runtime.run_workflow
runtime.cancel_run
shell
arbitrary filesystem write
```

Runtime execution remains a human/host action in V1.

## Tauri and Axum Connection

`src-tauri` binds IPC commands to `app-host` facade methods. It owns Tauri-specific event emission and window integration only.

Future Axum adapters bind routes/WebSocket/SSE to the same `app-host` facade. They should not reimplement workflow/model/runtime orchestration.

## Suggested Module Layout

```text
src/
  lib.rs
  app.rs
  workspace.rs
  error.rs
  services.rs
  workflow.rs
  models.rs
  runtime.rs
  agent.rs
  agent_tools.rs
  agent_tools/
    workflow.rs
    model.rs
    diagnostics.rs
```

Use modern Rust module layout. Do not introduce `mod.rs` files or `#[path = "..."]` attributes.

## Implementation Slices

`app-host` should be implemented in small slices:

```text
app-host/01a
  crate scaffold
  AppHost / WorkspaceHost
  WorkflowService session registry and JSON persistence
  ModelService manifest facade
  AgentService session registry shell

app-host/01b
  readiness snapshot bridge
  run_workflow orchestration
  mock node executor coverage

app-host/01c
  concrete V1 Agent tools
  proposal store
  workflow command policy integration

app-host/02
  AgentService turn orchestration
  workspace-scoped session lookup
  provider catalog / resolver seam
  AgentLoop invocation and event sink wiring
```

This split keeps service ownership, run orchestration, and Agent editing policy independently reviewable.
