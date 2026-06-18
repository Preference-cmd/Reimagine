# Axum Host Architecture

> Status: implemented V1 HTTP harness; still working draft for future streaming and remote/headless features

## Role

The Axum host is a peer host adapter for remote / headless operation and
end-to-end workflow testing. It reuses the same `app-host` facade as Tauri
and never reaches into `runtime` or concrete inference backends directly.

## Responsibilities

- Own HTTP routing and request/response serialization.
- Hold shared host state and inject `Arc<WorkspaceHost>`.
- Expose workflow open, workflow run, run snapshot, and run event
  endpoints for test and remote use.
- Bridge HTTP payloads into app-host facade calls.
- Provide a stable JSON wire contract for clients (curl, integration
  tests, future UI shells).

## Suggested Module Layout

```text
src/
  lib.rs
  state.rs
  error.rs
  dto.rs
  router.rs
  server.rs
  routes/
    health.rs
    workflows.rs
    runs.rs
  recorder.rs
```

`lib.rs` stays thin and re-exports the public host surface. `server.rs`
only runs the listener; it must not build `WorkspaceHost`. `recorder.rs`
provides the `RunEventSink` the runtime is wired with so run events
become queryable over HTTP.

## Suggested V1 Surface

```text
GET  /health
POST /workflows/open
POST /workflows/:id/run
GET  /runs/:id
GET  /runs/:id/events
```

The first slice supports workflow run testing against the existing
`sdxl-base-workflow.json` example and the current
`WorkspaceHost::run_workflow` path.

### Request / response shape

- `POST /workflows/open` accepts `{ "id": "..." }` (load from disk),
  `{ "workflow": { ... } }` (register inline), or both. Returns
  `{ "workflow_id", "source": "disk" | "inline" | "existing" }`.
- `POST /workflows/:id/run` accepts an optional body of
  `{ "target_selection": ..., "correlation_id": "..." }`. Returns
  `{ "outcome": "started" | "blocked", ... }`.
- `GET /runs/:id` returns
  `{ "kind": "snapshot" | "summary", ... }` depending on the run's
  terminal state. Snapshot and summary artifact DTOs include the runtime
  artifact id, producing node id, and host-neutral `reference`.
- `GET /runs/:id/events` returns
  `{ "run_id", "events": [ { "kind": "RunQueued", ... } ] }`.

## Non-Responsibilities

- Workflow mutation logic.
- Workflow session registry ownership (delegated to `app-host`).
- Workflow readiness orchestration.
- Agent policy logic.
- Runtime scheduling logic.
- Backend inference logic.
- UI state.
- Tauri-specific types in the public API.

`run` is host-facing in V1 and calls `WorkspaceHost::run_workflow`
rather than touching runtime internals directly. Future agent-facing
routes may be added later, but this slice is only concerned with
workflow execution E2E testing.

## State Shape

```text
AxumHostState
  workspace: Arc<WorkspaceHost>
  event_recorder: Arc<RunEventRecorder>
```

`RunEventRecorder` is an in-memory `RunEventSink`. The app-host
bootstrap path installs the same recorder instance as the runtime's
sink before handing `Arc<WorkspaceHost>` to Axum, so every lifecycle
event lands in a per-run bucket and `GET /runs/:id/events` reads that
same bucket.

Handlers use standard Axum `State` and `Json` extractors to call the
shared app-host facade.

## Integration Notes

- `axum-host` is a peer to `src-tauri`, not a replacement for it.
- It does not duplicate app-host orchestration logic.
- It reads run snapshots and summaries through `WorkspaceHost` facade
  methods rather than through `RuntimeService`.
- It does not depend on concrete inference backends directly. It may depend on
  `reimagine-runtime` for host-neutral DTO projections and for the
  `RunEventSink` adapter boundary.
- The V1 event surface is JSON polling. SSE / WebSocket can be a
  later refinement on top of the same recorder.
