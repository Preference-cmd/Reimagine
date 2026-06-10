# Agent Module Architecture

> Status: working draft
> Crate: `crates/agent`

## Role

`agent` is the Rust-side Agent runtime domain. It manages Agent sessions, mode policy, tool calls, workflow proposals, and provider access through a Reimagine-owned provider abstraction.

Agent sessions are workspace-scoped. An Agent session belongs to one workspace and can only use the tools, workflow sessions, model manifest, config, diagnostics, and policy exposed for that workspace.

```text
WorkspaceHost
  agent_service
    AgentSession
      workspace_scope
      mode
      provider_session
      tool_policy
```

There is no app-global Agent session in V1. `AppHost` may route to a workspace, but the Agent runtime and tool context operate inside the selected workspace boundary.

## V1 Provider Boundary

Reimagine owns:

```text
AgentProvider
  complete(request)
  stream(request)
  list_models()
```

V1 provider support targets OpenAI-compatible endpoints and Anthropic, but `agent/01` only defines the Reimagine-owned provider boundary. It must not bind the runtime crate to a concrete provider framework or SDK.

Provider implementations can be added behind this trait later. Rig is the preferred V1 candidate for that implementation layer because it provides provider/model/streaming abstractions without needing to own Reimagine's workspace, workflow command, or tool policy semantics.

Cersei is a useful reference for complete coding-agent runtime design, especially event streaming, tool execution lifecycle, MCP integration, memory, skills, and sub-agent orchestration. It should not be a V1 runtime dependency for `agent`, because those responsibilities overlap with Reimagine-owned workspace scope, app-host concrete tools, `ToolContext`, workflow proposal policy, and `WorkflowCommand` editing semantics.

```text
crates/agent
  Reimagine-owned AgentProvider trait
  Reimagine-owned tool registry and policy
  no Rig or Cersei dependency in agent/01

future provider adapter
  Rig-backed OpenAI-compatible provider
  Rig-backed Anthropic provider

architecture reference only
  Cersei-style event stream and tool lifecycle patterns
```

The Agent runtime remains owned by Reimagine because workflow command policy, proposal diffs, workspace scoping, and safety rules are app-specific.

## Modes

- `agent`: may auto-apply allowed low-risk edits.
- `build`: creates a full workflow proposal and diff; human acceptance applies it.

V1 accepts or rejects proposals as a whole.

V1 keeps auto-apply conservative:

```text
agent mode
  may auto-apply low-risk, reversible, editor-only WorkflowCommand batches
  must not run workflows, cancel runs, scan models, overwrite files, or perform arbitrary filesystem writes

build mode
  proposes a complete command batch and returns preview/diff/report
  human approval applies the proposal
```

Low-risk editor-only commands are commands that mutate workflow edit state through core command/session semantics and remain undoable through workflow history. V1 policy should treat graph/data edits as candidates for auto-apply only after preview succeeds and policy allows the command kinds. Destructive or external-effect operations require human/host approval.

The Agent runtime does not decide workflow command semantics. It enforces tool/mode/permission policy and delegates command preview/apply behavior to app-host tools.

## Tool Boundary

Agent tools are declared with an attribute macro and executed through a registry/policy layer.

```text
crates/agent
  AgentTool
  AgentToolRegistry
  ToolContext
  ToolPolicy
  ToolSpec

crates/agent-macros
  #[agent_tool]

crates/app-host
  concrete app tools
  register_app_tools(...)
```

The macro derives tool schema and wrapper code only. Tool authorization always goes through `ToolPolicy` before execution.

V1 uses explicit tool registration rather than distributed auto-registration.

Concrete tools live in `app-host` because they need access to workflow, model, runtime, and diagnostic facades. The `agent` crate itself should not depend on `app-host`.

`ToolContext` is defined by `agent` and supplied by `app-host` when a concrete tool is invoked. V1 keeps it generic:

```text
ToolContext
  workspace_scope
  agent_session_id
  mode
  correlation_id
  actor
  permissions
```

`workspace_scope` is an opaque workspace identity/scope value defined by `agent` or shared app domain types, not an `app-host` handle. It is used for policy, audit, and correlation. It does not let the `agent` crate access workspace services directly.

`ToolContext` should not carry an erased app-host capability bag in V1. Concrete app-host tool closures/functions can capture `Arc<WorkspaceHost>` directly because they live in `app-host`. This keeps `agent` independent from app-specific state and avoids turning context into an untyped service locator.

V1 tools:

```text
workflow.get
workflow.preview_commands
workflow.propose_commands
workflow.apply_commands
model.list
model.resolve_ref
diagnostics.for_workflow
```

V1 does not expose runtime run/cancel tools to Agent.

## Tool Macro Shape

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
    // app-host facade call
}
```

Input and output types should derive Serde and JSON Schema traits so provider adapters can expose schemas to OpenAI-compatible, Anthropic, or Rig-backed providers.

V1 macro expansion preserves explicit registration. For a function named `preview_commands`, the macro generates a small `AgentTool` wrapper and a constructor such as `preview_commands_agent_tool()`. App-host registers that wrapper with `AgentToolRegistry`; execution still happens through `AgentToolRegistry::invoke`, so `ToolPolicy` remains the only public execution gate.

The generated wrapper deserializes registry-boundary `ToolInput` (`serde_json::Value`) into the function's typed input, calls the async function, and serializes the typed output back into `ToolOutput`. Schema metadata is derived from `schemars::JsonSchema`-compatible input/output types and attached to `ToolSpec`.

## Code Organization

`crates/agent` should use the same modern Rust module style as the rest of the workspace: no `mod.rs`, no `#[path]`, and no large catch-all file.

Suggested V1 structure:

```text
crates/agent/src/lib.rs
crates/agent/src/error.rs
crates/agent/src/ids.rs
crates/agent/src/mode.rs
crates/agent/src/session.rs
crates/agent/src/context.rs
crates/agent/src/permissions.rs
crates/agent/src/tool.rs
crates/agent/src/registry.rs
crates/agent/src/policy.rs
crates/agent/src/provider.rs
crates/agent/src/event.rs
crates/agent/src/event_adapter.rs
crates/agent/src/report.rs
```

Module responsibilities:

```text
ids.rs
  AgentSessionId, WorkspaceScope, ToolName, ProviderName, ModelName

mode.rs
  AgentMode::Agent, AgentMode::Build

session.rs
  workspace-scoped AgentSession and in-memory V1 session state

context.rs
  ToolContext metadata only; no app-host handles

permissions.rs
  ToolPermission, ToolRiskLevel, PermissionSet

tool.rs
  AgentTool trait, ToolSpec, ToolInput, ToolOutput, ToolResult

registry.rs
  explicit registration, duplicate rejection, deterministic listing, policy-mediated invocation

policy.rs
  mode, permission, and risk checks; no workflow command semantics

provider.rs
  AgentProvider trait, AgentRequest, AgentResponse, AgentStreamEvent, ModelInfo

event.rs
  AgentEvent and agent-local event payloads

event_adapter.rs
  AgentDomainEventAdapter implementing the core event adapter trait when available

report.rs
  AgentReport or ToolInvocationReport for policy/tool/provider outcomes

error.rs
  AgentError, ToolError, ProviderError
```

`lib.rs` should declare private modules and re-export the public surface. Prefer:

```rust
mod context;
mod error;
mod event;

pub use context::ToolContext;
pub use error::{AgentError, ProviderError, ToolError};
pub use event::AgentEvent;
```

Use `serde_json::Value` at the registry boundary for `ToolInput` and `ToolOutput`. Concrete app-host tools and future `#[agent_tool]` wrappers can deserialize into strongly typed input/output structs. This keeps the registry compatible with OpenAI-compatible, Anthropic, and future Rig-backed tool schemas without binding `agent/01` to a schema-generation crate.

`AgentEvent` is the agent-local event model. It should not be replaced by core `DomainEvent`. Instead, `event_adapter.rs` projects `AgentEvent` into core's common event language once the core adapter trait exists.
