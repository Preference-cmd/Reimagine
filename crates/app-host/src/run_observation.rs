//! Host-facing run observation helpers.
//!
//! Hosts should read run state through `WorkspaceHost` instead of reaching
//! through to the runtime service directly. The returned snapshot and summary
//! types are still host-neutral runtime observation shapes.

use reimagine_core::model::{ArtifactId, RunId};
use reimagine_inference::BackendInstanceSnapshot;
use reimagine_runtime::{RunSnapshot, RunSummary};

use crate::WorkspaceHost;
use crate::artifact_access::{
    ArtifactAccess, ArtifactAccessError, media_type_for_reference, resolve_artifact_path,
};

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

    /// Resolve an artifact id to access information.
    ///
    /// Searches active RunSnapshot.artifacts and terminal RunSummary.artifacts.
    /// Validates path safety and returns ArtifactAccess if the file exists.
    pub fn resolve_artifact(&self, id: &ArtifactId) -> Result<ArtifactAccess, ArtifactAccessError> {
        // Search through runtime service for the artifact
        let artifact_ref = self
            .runtime_service()
            .find_artifact(id)
            .ok_or(ArtifactAccessError::UnknownArtifact)?;

        // Validate media type (V1: PNG only)
        let media_type = media_type_for_reference(&artifact_ref.reference)?;

        // Validate path safety and resolve to absolute path
        let path = resolve_artifact_path(&artifact_ref.reference, self.config().paths())?;

        // Check if the file exists
        if !path.exists() {
            return Err(ArtifactAccessError::FileGone);
        }

        Ok(ArtifactAccess {
            artifact_id: artifact_ref.id,
            node_id: artifact_ref.node_id,
            reference: artifact_ref.reference,
            path,
            media_type,
        })
    }
}
