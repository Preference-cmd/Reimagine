# Agent Provider Module Architecture

> Status: working draft
> Crate: `crates/agent-provider`

## Role

`agent-provider` contains concrete provider adapters for the Reimagine-owned
`AgentProvider` trait defined in `crates/agent`.

It translates provider SDKs or provider-framework abstractions into
`AgentRequest`, `AgentResponse`, `AgentStreamEvent`, `Message`, `ToolCall`, and
`AgentToolDefinition` shapes. It does not own Agent loop orchestration,
workspace state, tool policy, workflow commands, or proposal semantics.

## Responsibilities

- Implement `reimagine_agent::AgentProvider` for V1 provider backends.
- Provide a Rig-backed OpenAI-compatible adapter.
- Provide a Rig-backed Anthropic adapter.
- Translate Reimagine tool definitions into provider-native tool schemas.
- Translate provider tool calls into Reimagine `ToolCall` values.
- Translate provider usage / stop reasons / errors into Reimagine provider
  boundary types.
- Expose construction helpers that app-host can register into
  `AgentProviderCatalog`.

## Non-Responsibilities

- Agent loop ownership.
- Tool execution or `AgentToolRegistry::invoke`.
- Tool policy, permission checks, or app-host workflow command policy.
- Workflow/session/proposal mutation.
- App config persistence.
- Tauri IPC, Axum routes, or UI state.
- Runtime workflow execution.

## Dependency Direction

```text
agent-provider -> agent
agent-provider -> config     (only if provider config document helpers live here)
agent-provider -> Rig / provider SDK crates

app-host -> agent
app-host -> agent-provider
```

`agent` must not depend on `agent-provider`.

`app-host` owns provider composition. It reads provider configuration, constructs
provider adapters from `agent-provider`, and registers them into
`AgentProviderCatalog`.

## V1 Provider Targets

V1 targets:

```text
OpenAI-compatible
  base_url
  api_key
  default_model
  optional organization/project-style metadata if the adapter supports it

Anthropic
  api_key
  default_model
```

Rig is the preferred implementation layer for V1 because it can reduce
provider-specific request/stream plumbing while Reimagine retains ownership of
the Agent loop, tool observations, and policy.

If Rig's abstractions are insufficient for a required Reimagine behavior, the
adapter may use direct provider SDK/HTTP code behind the same
`AgentProvider` trait. This should not affect `crates/agent` or app-host tool
policy.

## Config Boundary

Provider configuration is not part of `crates/agent`.

V1 should represent provider config as typed data that app-host can load through
the existing config infrastructure and pass into `agent-provider` constructors:

```text
AgentProviderConfig
  providers: [ProviderConfig]

ProviderConfig
  id: ProviderName
  kind: openai_compatible | anthropic
  base_url?
  api_key_ref or api_key
  default_model?
  enabled
```

V1 provider config and provider secrets are read from and written to the
workspace/app configuration files. Environment variables are not part of the V1
provider configuration path. The provider adapter should not read app config
globally; app-host/config loads the file-backed config and passes resolved
values into constructors.

## App-Host Registration

App-host registers concrete providers:

```text
AppHost / WorkspaceHost construction
  -> load provider config
  -> construct agent-provider adapters
  -> AgentProviderCatalog::register(Arc<dyn AgentProvider>)
  -> AgentService uses catalog during run_turn
```

Unknown provider names are app-host orchestration errors before the Agent loop
starts. Provider API errors after the loop starts are `ProviderError` values
returned from `AgentProvider::complete` or `stream`.

## Streaming Boundary

`agent-provider` should treat streaming as a provider translation capability,
not as Agent loop ownership.

The adapter may expose `AgentProvider::stream` even while V1 `AgentLoop` still
uses `complete` for turn execution. Streaming support must preserve the same
Reimagine-owned shapes as non-streaming completion:

```text
provider stream event
  -> AgentStreamEvent::ContentDelta
  -> AgentStreamEvent::ToolCallDelta
  -> AgentStreamEvent::ToolCall
  -> AgentStreamEvent::Usage
  -> AgentStreamEvent::Done
```

Streaming must not execute tools inside Rig or the provider adapter. Tool-call
delta events are only partial provider output. A tool becomes executable only
after the adapter can emit a complete `ToolCall` with stable id, name, and JSON
arguments. The Reimagine Agent loop remains responsible for executing the tool,
feeding the observation back to the provider, and enforcing tool policy.

OpenAI-compatible streaming commonly emits text and tool-call arguments as
deltas. Anthropic streaming has its own event vocabulary. The adapter should
normalize both into `AgentStreamEvent` without exposing provider-native event
types to `crates/agent` or app-host.

If a provider/backend cannot support streaming, `stream` must return an
explicit `ProviderError` such as `streaming_unsupported`. It must not silently
fall back to `complete`, because callers need to know whether they can show
incremental output.

Future Agent loop streaming execution can consume the same provider stream and
project incremental output through the app-host event pipeline. That future work
belongs in `crates/agent` / `crates/app-host`, not in `agent-provider`.

## Testing

V1 provider adapter tests should cover:

- request translation with messages and tool definitions;
- assistant final response translation;
- tool-call response translation;
- provider error translation;
- model listing shape;
- stream event translation for content deltas, tool-call deltas, complete tool
  calls, usage, terminal done, and explicit unsupported-streaming errors.

Tests should avoid live network calls. Use mock clients, mocked HTTP transport,
or Rig test seams where available.

## Suggested Module Layout

```text
crates/agent-provider/src/lib.rs
crates/agent-provider/src/config.rs
crates/agent-provider/src/error.rs
crates/agent-provider/src/openai_compatible.rs
crates/agent-provider/src/anthropic.rs
crates/agent-provider/src/rig.rs
crates/agent-provider/src/translation.rs
```

Use modern Rust module layout. Do not introduce `mod.rs` files or `#[path]`
attributes.
