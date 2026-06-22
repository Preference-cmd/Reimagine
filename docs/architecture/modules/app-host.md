# App Host Module Architecture

> Status: working draft
> Crate: `crates/app-host`

## Role

`app-host` is the application service and composition layer. It assembles the reusable domain crates into one host-neutral surface for Tauri, Axum, and Agent workflows.

It is not a concrete host adapter. It must not contain Tauri command attributes, Axum routes, React/UI state, or backend inference kernels.

## Responsibilities

- Own the application/workspace service facade.
- Hold workspace-level services and registries.
- Coordinate config, workflow sessions, model manager, runtime, node executors, and Agent service.
- Build model readiness snapshots and pass them to core readiness.
- Call `core` to validate workflows, apply workflow commands, and build execution plans.
- Call `runtime` to run prepared execution plans.
- Load built-in plugin metadata and wire plugin extensions into domain
  registries.
- Provide concrete Agent tools over workflow/model/diagnostic operations.
- Offer a shared API surface for Tauri, Axum, and Agent tool execution.

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
app-host -> inference
app-host -> inference-backends/candle   # concrete backend crate selected by config in V1
app-host -> agent
app-host -> agent-provider
app-host -> agent-macros

src-tauri -> app-host
axum-host -> app-host
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

`WorkspaceHost` is the shared application state center. Tauri and Axum hold it through their own host state mechanisms. Agent sessions are bound to one workspace, not to global `AppHost` state.

`WorkspaceServices` is the app-host service container captured by concrete Agent tools. Tools capture `Arc<WorkspaceServices>`, not `Arc<WorkspaceHost>`, so they cannot recurse back through `AgentService` or mutate the registry while handling a tool call.

V1 can keep `AppHost` single-workspace:

```text
AppHost
  workspace: Arc<WorkspaceHost>
```

The type shape should still make the workspace boundary explicit so future multi-workspace support can route Agent sessions and host requests to the correct `WorkspaceHost`.

`app-host` owns the unified bootstrap entry that assembles `WorkspaceHost`. Tauri and Axum adapters should receive an already-built workspace handle rather than duplicating service composition.

## Host API DTOs

Tauri and Axum are equal host adapters over the same app-host semantics. Shared
request/response DTOs and projection helpers should live in `app-host`, not in
one host adapter and then be copied into the other.

Suggested module shape:

```text
src/
  api.rs
  api/
    health.rs
    nodes.rs
    workflows.rs
    runs.rs
    artifacts.rs
```

The API modules should expose host-neutral request/response shapes for common
operations:

```text
open workflow
run workflow
get run snapshot or summary
list run events or event records
list node definitions
resolve/download artifact references
```

Host adapters remain responsible for transport details:

```text
HTTP extractors/status codes/headers -> axum-host
Tauri command attributes/window events -> src-tauri
shared DTOs/projections/facade calls -> app-host::api
```

V1 does not require Tauri to call Axum over localhost. Tauri may reuse the same
DTO shapes and `WorkspaceHost` facade directly. A future embedded Axum server
inside the desktop app is possible, but it is a host decision rather than a
domain dependency.

`node_catalog` is the workspace handle to the built-in node catalog from
`crates/nodes`. It is the catalog exposed to host adapters, UI DTOs, Agent
tools, validation/readiness flows, and import adapters. `app-host` may project
or serialize the catalog for clients, but it must not maintain a second copy of
node slots, params, output rules, or aliases.

`app-host` exposes the catalog through a single `NodeCatalogService`. The
service is the host-neutral surface for `list_node_defs` / `find_node_def`
operations, implements `core::model::NodeCatalog` so core validation and
readiness can consume it through the host, and owns the catalog/executor
alignment check. UI, Tauri, Axum, and Agent tools must read node metadata
through this service; they must not derive `NodeDef` data from any other
source.

`NodeExecutorRegistry` is assembled alongside the catalog but serves a
different purpose: it maps catalog `NodeTypeId` values to execution behavior.
The registry is not a node metadata source and is not iterated as such.

`NodeCatalogService::check_alignment(&registry)` produces a
`NodeCatalogAlignment` report with two collections:

```text
NodeCatalogAlignment
  backend: BackendSelection
  missing_executors: Vec<NodeTypeId>  # catalog entries with no registered executor
  orphan_executors:  Vec<NodeTypeId>  # executors with no catalog entry
```

The report converts into a stream of host-facing `Diagnostic` values:

```text
APP_HOST/NODE_CATALOG_MISSING_EXECUTOR  Error    — node type is in the catalog
                                                       but has no executor for the
                                                       selected backend
APP_HOST/NODE_CATALOG_ORPHAN_EXECUTOR   Warning  — executor is registered for the
                                                       selected backend but has no
                                                       catalog entry
```

The alignment check is wired into `WorkspaceHost::build_plan` and the
`diagnostics.for_workflow` Agent tool. A missing executor that the workflow
references blocks the run; orphan executors surface as warnings so users
can fix drift without breaking currently-runnable workflows. Diagnostics
reference the `NodeTypeId` and the selected backend profile and do not
leak backend internals.

The readiness `OperationReport` (which now carries the alignment
diagnostics) is returned to host adapters through both
`RunWorkflowResult::Started` and `RunWorkflowResult::Blocked`. `Started`
typically carries an empty or warning-only report; `Blocked` carries
error-severity diagnostics that prevented the run. The Axum DTO mirrors
this with a `diagnostics` field on the `Started` variant so HTTP clients
can surface non-blocking drift.

Axum exposes the same catalog as `GET /nodes`, projecting
`WorkspaceHost::list_node_defs()` into the frontend-friendly node definition
DTO. This route is an adapter projection, not a second catalog. The UI may use
it as the development and HTTP source for node metadata; it must not keep a
parallel hard-coded list of built-in node slots, params, or outputs.

The diagnostic target domain for catalog/executor alignment diagnostics
is `"app-host.node_catalog"`, scoped to the affected `NodeTypeId`. This
is the first dotted sub-domain style in app-host diagnostics; the
dotted convention lets UI and Agent consumers route or de-duplicate
by `app-host.<subsystem>` without parsing the message.

## Plugin And Backend Composition

`app-host` is the composition root for plugin-shaped registration. V1 uses
static built-in plugins rather than runtime third-party loading:

```text
BuiltinPluginLoader
  -> PluginDescriptor
  -> Vec<PluginExtension>
  -> group by HostSurface
  -> construct concrete services/adapters
  -> register into inference / agent / node catalog registries
```

For inference, Candle is treated as a built-in plugin extension:

```text
PluginDescriptor("builtin.candle")
  -> PluginExtension {
       extension: "backend.candle",
       extends: HostSurface::InferenceBackend
     }
  -> app-host constructs Candle backend instance from config
  -> app-host registers BackendInstance with inference router
```

The plugin metadata contract does not construct backends. App-host still owns
factory/wiring decisions because it has config, workspace paths, model
services, and host policy.

## Backend Selection

`app-host` is the composition root for inference backends, but it should not
hard-code Candle inside the runtime path. V1 should model backend selection as
configuration:

```text
backend extension: "backend.candle"
backend instance:  "candle" or "candle:<device-profile>"
```

Backend selection belongs at the app-host/config boundary, where user settings
can select a backend extension or backend instance and workspace bootstrap can
instantiate the matching concrete backend crate.

```text
AppConfig
  inference backend config

WorkspaceHost::new(config)
  -> resolve configured backend extension / backend instance
  -> construct backend adapters from config
  -> construct ModelResolver adapter
  -> build inference::InferenceBackendRegistry
  -> construct inference::InferenceRuntime / router
  -> register_builtin_inference_executors(...)
  -> construct RuntimeService
```

`app-host` assembles the runtime-facing trait chain:

```text
BuiltinNodeCatalog
  -> NodeExecutorRegistry
    -> inference concrete NodeExecutor
      -> Arc<dyn inference::InferenceRuntime>
        -> registry-backed backend selection
        -> concrete inference::InferenceBackend
```

It should not move node orchestration into itself; orchestration belongs to the
concrete inference executors.

Candle may be the V1 default, but the default is still a config value, not a
runtime or executor constant. Runtime receives only the populated executor
registry and optional backend-instance runtime hooks; it does not know which concrete backend
was selected.

V1 may configure only one backend by default, but the app-host composition shape
should still build a registry-backed `inference::InferenceRuntime`.
Backend selection is a router/resource concern based on handle affinity and
capability support, not a runtime scheduler decision:

```text
inference backend config
  -> app-host constructs backend adapters
  -> app-host registers adapters by BackendInstance
  -> app-host constructs BackendSelectionPolicy from config/runtime policy inputs
  -> app-host constructs inference InferenceRuntime/router
  -> app-host registers inference executors with the router
  -> runtime executes the prepared plan with that executor registry
```

The router must be configurable. App-host is the composition point for that
configuration: default backend, allowed backend set, fallback order, disabled
backends, and future budget/priority hints all enter the inference router here.
Runtime may own high-level run policy, but it should not call a concrete
backend directly to load a model or run a capability.

Model loading backend selection is applied through `InferenceRuntime`:

```text
checkpoint_loader
  -> LoadBundleRequest
  -> router chooses backend from BackendSelectionPolicy when no handle affinity exists
  -> backend returns Model / Clip / Vae handles carrying backend affinity
```

After a backend-bound handle exists, app-host configuration must not cause
silent fallback to a different backend. Cross-backend execution requires an
explicit bridge/transfer policy so the router can emit precise diagnostics when
transfer is unsupported.

Implicit cross-backend tensor transfer is not allowed. If the router sees
handles from incompatible backends or devices, it must use an explicit bridge or
return a readiness/runtime diagnostic. Runtime itself should not inspect
backend internals to make that decision.

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

Workspace bootstrap has two construction paths:

```text
WorkspaceHost::with_defaults(...)
  test/backward-compatible helper
  may panic only for programmer errors

WorkspaceHost::try_with_defaults(...).await
  production bootstrap path
  loads config through ConfigHandle<T>
  returns app-host bootstrap error/report on invalid config
```

Missing optional config documents may still load module defaults through
`ConfigHandle<T>`. Invalid JSON, unsupported config schema, or invalid enum
values must not be silently replaced by defaults. App-host should preserve the
config key/path and diagnostic information so Tauri, Axum, and future UI shells
can surface the bootstrap problem. Runtime must not read config directly.

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
