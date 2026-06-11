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
      tool_registry
      tool_policy
```

There is no app-global Agent session in V1. `AppHost` may route to a workspace, but the Agent runtime and tool context operate inside the selected workspace boundary.

At session creation time, the session is bound to:

```text
workspace_scope
mode
provider session/config
frozen AgentToolRegistry
ToolPolicy
```

V1 does not dynamically add or remove built-in app tools while a session is running. This mirrors the Codex-style capability-surface model: the Agent loop receives a stable set of tools and policies from the host, then invokes those tools only through the registry.

## Terminology

Reimagine uses Codex-style Agent loop terminology, adapted to workflow editing rather than coding-agent shell/file execution:

```text
AgentSession
  Long-lived workspace-scoped Agent state. Owns workspace_scope, mode,
  provider identity/config, frozen tool registry, policy, and conversation
  continuity.

AgentTurn
  One user/agent request lifecycle inside a session. A turn may call the
  provider multiple times if the model asks to invoke tools.

AgentLoop
  The harness that runs an AgentTurn. It builds provider requests,
  advertises tool specs, executes requested tools through the registry,
  feeds tool observations back to the provider, emits events, and stops
  when a final assistant response or stop condition is reached.

ToolCall
  Provider-requested tool invocation. V1 already represents this through
  provider::ToolCall and provider::ToolCallId.

ToolObservation
  The model-visible result of a tool call. It is serialized into a
  provider::Message::tool_result(...) and fed back into the next provider
  request.

ToolCallResult
  Agent-loop record of a tool call outcome. It carries tool_call_id,
  tool name, status, output or diagnostics, whether the call was
  effective, and correlation data. The provider receives the observation;
  hosts may also receive events/diagnostics.

AgentEvent / DomainEvent / Diagnostic
  Host-facing observation streams for UI, audit, and adapters. They are
  not the provider's conversation state.
```

`AgentSession` is the closest Reimagine concept to Codex's thread continuity. `AgentTurn` is the closest Reimagine concept to a Codex turn. Reimagine does not copy Codex's coding-specific shell, patch, or filesystem tools into V1.

## Agent Loop

The Agent loop is owned by Reimagine, not by a provider SDK. Providers answer model requests; the Agent loop coordinates provider calls, tool calls, tool observations, events, and stop conditions.

```text
Agent loop
  -> collect session context
     - workspace_scope
     - mode
     - tool specs
     - relevant workflow/model/diagnostic context
  -> call AgentProvider
  -> receive assistant text and/or tool call requests
  -> invoke AgentToolRegistry
     - ToolPolicy check
     - concrete app-host tool execution
  -> feed ToolCallResult back to the provider as observation
  -> emit AgentEvent / DomainEvent / diagnostics for host observers
  -> continue until final assistant response or stop condition
```

The provider sees tool schemas and tool observations. It does not receive app-host service handles, event bus handles, workflow sessions, or proposal stores.

Tool observations and domain events are separate channels:

```text
ToolCallResult / ToolOutput
  consumed by the Agent loop and sent back to the provider

AgentEvent / DomainEvent / Diagnostic
  consumed by UI, host adapters, audit, and future Axum/Tauri streams
```

This separation prevents the model from inferring workflow state from host-only events. If a tool creates a proposal, the tool output must explicitly report that the proposal is pending and that the workflow was not mutated.

## Agent Turn Lifecycle

V1 Agent loop execution should be deterministic and testable with a mock provider:

```text
run_turn(session, input)
  -> create AgentTurn
  -> build initial messages from session history + turn input
  -> build AgentToolDefinition list from registry ToolSpec values
  -> call AgentProvider::complete(...)
  -> append assistant message
  -> if assistant message has no tool calls:
       finish turn with final assistant response
  -> if assistant message has tool calls:
       for each tool call:
         invoke AgentToolRegistry with ToolContext
         convert success/failure into ToolCallResult
         append Message::tool_result(...)
         emit AgentEvent::ToolInvoked/ToolCompleted/ToolFailed
       call provider again with appended tool observations
  -> repeat until final response, max tool steps, cancellation, or provider error
```

V1 may execute tool calls sequentially. Parallel tool execution can be considered later, after event ordering, proposal ordering, and UI projection semantics are stable.

Stop conditions:

```text
final_response
  provider returns assistant content with no tool calls

max_tool_steps
  loop reaches the configured V1 guard before final response

provider_error
  AgentProvider returns ProviderError

tool_error
  tool cannot be invoked or returns ToolError; V1 feeds the error
  observation to the provider unless policy marks it terminal

cancelled
  future turn control; agent/02 may leave this as a placeholder
```

V1 should define turn status and result shapes even if streaming, steering, interruption, and real provider adapters are deferred.

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

Approval is a host/app-host concern, not a tool-internal side effect. In build mode, Agent tools create proposals and return proposal receipts. A human or host action later accepts or rejects the proposal as a whole. The approved apply path is represented in workflow provenance, but the approval action itself is not a provider tool call.

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

`ToolContext` should not carry an erased app-host capability bag in V1. Concrete app-host tool wrappers capture app-host service state outside the context, typically through an `Arc<WorkspaceServices>`-style value owned by `app-host`. This keeps `agent` independent from app-specific state and avoids turning context into an untyped service locator.

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

## Tool Results and Effects

Every tool invocation returns an observation that can be fed back into the Agent loop. For mutation-capable tools, the output must distinguish preview/proposal from committed state:

```text
workflow.preview_commands
  returns CommandResult-style preview
  effective = false

workflow.propose_commands
  returns ProposalReceipt with proposal_id, base_version, preview_result
  effective = false
  workflow is not mutated

workflow.apply_commands
  returns apply result
  effective = true only when core apply succeeds
```

`effective` is a tool-output property for the Agent loop. It is not a replacement for core history or workflow versioning. It exists so the model, UI, and tests can tell whether a tool invocation actually changed workflow state.

Rejected tool calls, policy denials, and failed service calls should return structured diagnostics where possible. The Agent loop may feed those diagnostics back to the provider as part of the tool observation, while app-host/event adapters can also project them to the host-facing event stream.

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

`#[agent_tool]` is optional for app-host concrete tools in V1. It is practical for stateless tools, but workspace-bound app-host tools may hand-write `AgentTool` wrappers when they need to capture `Arc<WorkspaceServices>`. The macro must not force app-host state into `ToolContext`.

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

turn.rs
  AgentTurnId, AgentTurnStatus, AgentTurnRequest, AgentTurnResult,
  ToolCallResult, stop condition shapes

loop.rs
  minimal AgentLoop / AgentTurnRunner over AgentProvider and
  AgentToolRegistry

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
