//! Host-facing run observation helpers.
//!
//! Hosts should read run state through `WorkspaceHost` instead of reaching
//! through to the runtime service directly. The returned snapshot and summary
//! types are still host-neutral runtime observation shapes.

use reimagine_core::model::RunId;
use reimagine_inference::BackendInstanceSnapshot;
use reimagine_runtime::{RunSnapshot, RunSummary};

use crate::WorkspaceHost;

impl WorkspaceHost {
    /// Return the live snapshot for an active run, if the runtime still holds it.
    pub fn run_snapshot(&self, run_id: &RunId) -> Option<RunSnapshot> {
        self.runtime_service().snapshot(run_id)
    }

    /// Return the terminal summary for a completed, failed, or cancelled run.
    pub fn run_summary(&self, run_id: &RunId) -> Option<RunSummary> {
        self.runtime_service().summary(run_id)
    }

    /// Return a snapshot per concrete backend instance visible to the runtime.
    pub async fn backend_instance_snapshots(&self) -> Vec<BackendInstanceSnapshot> {
        self.runtime_service().backend_instance_snapshots().await
    }
}
