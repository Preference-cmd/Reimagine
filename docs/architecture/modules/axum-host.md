# Future Axum Host Architecture

> Status: future

## Role

An Axum server host is a future peer host adapter for remote/headless operation. It should reuse the same crates as Tauri.

## Potential Surface

```text
POST /workflows/:id/commands
GET  /workflows/:id
POST /workflows/:id/run
GET  /runs/:id/events
POST /agent/sessions
POST /agent/sessions/:id/messages
```

## Non-Responsibilities

- Workflow mutation logic.
- Agent policy logic.
- Runtime scheduling logic.
- Backend inference logic.
