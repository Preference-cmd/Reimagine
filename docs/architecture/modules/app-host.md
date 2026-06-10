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
  base_path
  config_service
  workflow_service
  model_service
  runtime_service
  agent_service
  node_catalog
  node_executor_registry
```

`WorkspaceHost` is the shared application state center. Tauri and future Axum hold it through their own host state mechanisms. Agent tools receive a controlled context that can call its facade methods.

## Service Facades

`WorkflowService` owns app-level workflow session management:

```text
WorkflowService
  sessions: WorkflowId -> WorkflowSession
```

It coordinates:

- opening workflow JSON;
- saving workflow JSON;
- applying `WorkflowCommand` batches;
- previewing command batches;
- building readiness plans for a selected target.

`core` still owns the rules for a single workflow/session. `WorkflowService` owns the app-level registry.

`ModelService` wraps `model-manager` operations:

- load/save manifest;
- scan model roots;
- list models;
- resolve `ModelRef`;
- produce readiness diagnostics/snapshots for core readiness.

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

## Agent Tool Boundary

Agent tools use an attribute macro to keep tool metadata close to implementation:

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
  agent_session_id
  mode
  correlation_id
  actor
  permissions

app-host concrete tool
  captures Arc<WorkspaceHost>
  receives ToolContext and typed input
  calls app-host workflow/model/diagnostic facade
```

The `#[agent_tool]` macro lives in `crates/agent-macros`. The macro generates schema and wrapper code only. It must not bypass policy.

Concrete tool functions live in `app-host` because they need access to workflow, model, runtime, and diagnostic services.

V1 uses explicit registration:

```text
register_app_tools(registry)
```

V1 does not use distributed auto-registration such as `inventory`.

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
  -> stores or returns a proposal envelope
  -> does not mutate workflow

workflow.apply_commands
  -> checks Agent mode and ToolPolicy
  -> requires preview success
  -> agent mode may auto-apply only low-risk editor-only command batches
  -> build mode applies only after human/host approval
```

`app-host` is responsible for converting Agent tool input into `core::CommandBatch` values with `CommandActorKind::Agent` and the correct provenance. Core remains responsible for command validation, preview, apply, history, undo/redo, and diagnostics.

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
