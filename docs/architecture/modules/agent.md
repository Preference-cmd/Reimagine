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
