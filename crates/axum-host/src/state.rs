//! Shared HTTP state for the Axum host.
//!
//! V1 holds the `Arc<WorkspaceHost>` facade plus the [`RunEventRecorder`]
//! that captures runtime events for `GET /runs/:id/events` polling.
//! Both are required: the host facade is the only path to workflow /
//! model / runtime orchestration, and the recorder is what makes the
//! runtime's event sink reachable from the HTTP layer.

use std::sync::Arc;

use reimagine_app_host::WorkspaceHost;

use crate::recorder::RunEventRecorder;

#[derive(Clone)]
pub struct AxumHostState {
    workspace: Arc<WorkspaceHost>,
    event_recorder: Arc<RunEventRecorder>,
}

impl std::fmt::Debug for AxumHostState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AxumHostState")
            .field(
                "workspace_scope",
                &self.workspace.workspace_scope().as_str(),
            )
            .field("active_runs", &self.event_recorder.active_run_count())
            .finish_non_exhaustive()
    }
}

impl AxumHostState {
    /// Build shared state for the Axum host.
    ///
    /// The recorder is shared with the runtime that drives workflow
    /// execution. Tests may pass a recorder directly; production code
    /// typically uses [`AxumHostState::for_workspace`] which wires a
    /// fresh recorder into the host.
    pub fn new(workspace: Arc<WorkspaceHost>, event_recorder: Arc<RunEventRecorder>) -> Self {
        Self {
            workspace,
            event_recorder,
        }
    }

    /// Borrow the underlying workspace facade.
    pub fn workspace(&self) -> &Arc<WorkspaceHost> {
        &self.workspace
    }

    /// Borrow the shared event recorder.
    pub fn event_recorder(&self) -> &Arc<RunEventRecorder> {
        &self.event_recorder
    }
}
