# Agent Module Architecture

> Status: working draft
> Crate: `crates/agent`

## Role

`agent` is the Rust-side Agent runtime domain. It manages Agent sessions, mode policy, tool calls, workflow proposals, and provider access through a Reimagine-owned provider abstraction.

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

`ToolContext` is defined by `agent` and supplied by `app-host` when a concrete tool is invoked. It may carry controlled handles/capabilities, but it must not require `agent` to import `app-host` types.

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
