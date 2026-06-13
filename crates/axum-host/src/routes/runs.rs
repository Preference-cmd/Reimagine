//! `GET /runs/:id` and `GET /runs/:id/events` routes.
//!
//! Both routes are read-only views over the app-host run facade and the
//! shared [`RunEventRecorder`]. They are deliberately simple: the
//! runtime already owns the snapshot/summary state machines, so we
//! ask `WorkspaceHost` for host-neutral observations and project them
//! out over HTTP.

use axum::Json;
use axum::extract::{Path, State};
use reimagine_core::model::RunId;

use crate::dto::{RunDto, RunEventsResponse};
use crate::error::{AxumHostError, AxumHostResult};
use crate::state::AxumHostState;

/// `GET /runs/:id` — return the live snapshot if the run is still
/// active, the terminal summary if it has finished, or `404` if the
/// runtime has never seen it.
pub async fn get(
    State(state): State<AxumHostState>,
    Path(id): Path<String>,
) -> AxumHostResult<Json<RunDto>> {
    let run_id = RunId::new(id);
    if let Some(summary) = state.workspace().run_summary(&run_id) {
        return Ok(Json(RunDto::Summary(summary.into())));
    }
    if let Some(snapshot) = state.workspace().run_snapshot(&run_id) {
        return Ok(Json(RunDto::Snapshot(snapshot.into())));
    }
    Err(AxumHostError::UnknownRun { run_id })
}

/// `GET /runs/:id/events` — return every `RunEvent` the recorder has
/// captured for the run. The runtime itself already enforces a
/// single-writer / append-only model; the recorder is the bridge
/// from `Arc<dyn RunEventSink>` to a queryable in-memory log.
pub async fn events(
    State(state): State<AxumHostState>,
    Path(id): Path<String>,
) -> AxumHostResult<Json<RunEventsResponse>> {
    let run_id = RunId::new(id);
    let events = state.event_recorder().events_for(&run_id);
    Ok(Json(RunEventsResponse {
        run_id,
        events: events.into_iter().map(Into::into).collect(),
    }))
}
