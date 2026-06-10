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

V1 should prefer Rig behind this trait. V1 provider support covers OpenAI-compatible endpoints and Anthropic.

The Agent runtime remains owned by Reimagine because workflow command policy, proposal diffs, and safety rules are app-specific.

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
