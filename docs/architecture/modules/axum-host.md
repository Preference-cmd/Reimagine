# Axum Host Architecture

> Status: implemented V1 HTTP harness; working draft for dev server,
> artifact access, streaming, and remote/headless features

## Role

The Axum host is a peer host adapter for remote / headless operation,
developer automation, and end-to-end workflow testing. It reuses the same
`app-host` facade as Tauri and never reaches into `runtime` or concrete
inference backends directly.

Axum is not a test-only backdoor. Tauri and Axum are equal interaction
surfaces over the same workspace host:

```text
Tauri command -> app-host API DTO -> WorkspaceHost
HTTP request  -> app-host API DTO -> WorkspaceHost
```

They may share request/response DTOs and projection helpers through
`app-host::api`, while keeping transport-specific details in their own host
adapters.

## Responsibilities

- Own HTTP routing and request/response serialization.
- Hold shared host state and inject `Arc<WorkspaceHost>`.
- Expose workflow open, workflow run, run snapshot, and run event
  endpoints for test and remote use.
- Bridge HTTP payloads into app-host facade calls.
- Provide a stable JSON wire contract for clients (curl, integration
  tests, future UI shells).
- Provide a runnable developer server for workflow E2E testing against a real
  workspace `base_path`.
- Install HTTP tracing/logging middleware for request, route, and run
  observability.

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
    artifacts.rs
  recorder.rs
```

`lib.rs` stays thin and re-exports the public host surface. `server.rs`
only runs the listener; it must not build `WorkspaceHost`. `recorder.rs`
provides the `RunEventSink` the runtime is wired with so run events
become queryable over HTTP.

## Suggested V1 Surface

```text
GET  /health
GET  /nodes
POST /workflows/open
POST /workflows/:id/run
GET  /runs/:id
GET  /runs/:id/events
GET  /artifacts/:artifact_id
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
- `GET /artifacts/:artifact_id` resolves a runtime artifact id from the
  current workspace's run snapshots/summaries and serves the corresponding
  workspace output file. V1 should prefer artifact ids over arbitrary file
  paths. Any file-serving route must prove the resolved path stays under
  `<base_path>/output`.

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

## Developer Server

The Axum host should provide a runnable development entry point:

```text
cargo run -p reimagine-axum-host -- \
  --base-path ~/ReimagineWorkspace \
  --addr 127.0.0.1:7878
```

The server entry point owns only transport/bootstrap concerns:

- parse `--base-path`, `--addr`, and logging options;
- construct `WorkspaceHost::try_with_defaults(...)`;
- install a shared `RunEventRecorder`;
- start `build_router().with_state(...)`;
- print the listening address and the selected workspace directories.

It must not duplicate workflow readiness, model scanning, backend selection,
or runtime orchestration logic. Those stay in `app-host` and its services.

The development workspace uses the existing V1 path layout:

```text
base_path/
  models/
    manifest.json
    checkpoints/
      sdxl_base_1.0.safetensors
  input/
  output/
  workflows/
  config/
    inference_backend.json
    model_series.json
```

`config/inference_backend.json` selects the backend and device. V1 can keep
the persisted shape simple:

```json
{
  "backend": "candle",
  "candle_device": "cpu"
}
```

`models/manifest.json` maps workflow `ModelRef` ids such as
`sdxl-base-1.0` to local model sources. Workflow JSON never stores absolute
model paths.

## Tracing

Axum logging is developer/runtime observability. It complements structured
diagnostics; it does not replace them.

The dev server should initialize `tracing_subscriber` and install HTTP request
tracing middleware such as `tower_http::trace::TraceLayer`. The implementation
should follow current `tower-http` documentation for `TraceLayer`, including
custom span creation and response/failure hooks.

V1 tracing requirements:

- respect `RUST_LOG` and optionally `--log-filter`;
- log startup fields: listening address, workspace `base_path`, selected
  backend, selected backend instance or device profile when available;
- create request spans with method, path, status, and latency;
- include workflow/run context on relevant routes, especially
  `workflow_id`, `run_id`, and `correlation_id` when available;
- record warning/error logs for failed requests and bootstrap errors;
- keep prompts, secrets, API keys, and backend-private payload details out of
  info-level logs;
- keep complete local model paths at debug level or behind explicit filters.

Route handlers should not scatter ad-hoc logging. Shared middleware and small
route-level span helpers should carry the HTTP context, while app-host,
runtime, and inference continue to emit structured diagnostics and run events.

## Integration Notes

- `axum-host` is a peer to `src-tauri`, not a replacement for it.
- It does not duplicate app-host orchestration logic.
- It should share host API DTO/projection shapes through `app-host::api` when
  those shapes are useful to both Tauri and Axum.
- It reads run snapshots and summaries through `WorkspaceHost` facade
  methods rather than through `RuntimeService`.
- It does not depend on concrete inference backends directly. It may depend on
  `reimagine-runtime` for host-neutral DTO projections and for the
  `RunEventSink` adapter boundary.
- The V1 event surface is JSON polling. SSE / WebSocket can be a
  later refinement on top of the same recorder.
