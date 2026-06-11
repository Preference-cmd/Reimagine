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

The exact secret-storage strategy can evolve. V1 may use local config values or
environment variables, but the provider adapter should not read app config
globally. Constructors receive resolved config.

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

## Testing

V1 provider adapter tests should cover:

- request translation with messages and tool definitions;
- assistant final response translation;
- tool-call response translation;
- provider error translation;
- model listing shape;
- stream event translation if streaming is included in the slice.

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
