use serde::Serialize;

#[cfg(feature = "aigc")]
use crate::WorkerSwitchError;

/// Machine-readable failure classification carried in IPC error payloads.
///
/// The frontend branches on the snake_case string form of these codes
/// (e.g. `worker_unavailable`) instead of parsing free-form messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AppHostErrorCode {
    /// Workspace or inference bootstrap failed.
    BootstrapFailed,
    /// Generic command failure (no more specific classification applies).
    CommandFailed,
    /// The named provider is not registered. Nothing produces it while no
    /// agent integration is wired in; kept so the code set the frontend
    /// matches on stays stable.
    UnknownProvider,
    /// A model or model manifest is missing or unreadable.
    ModelNotFound,
    /// A model acquisition/download operation failed.
    ModelDownloadFailed,
    /// A run or inference operation failed.
    InferenceError,
    /// No active worker, or the worker is unavailable, failed, or stale.
    WorkerUnavailable,
    /// The submitted workflow (or workflow JSON) is invalid.
    WorkflowInvalid,
    /// The caller supplied an unsafe path or reference.
    PermissionDenied,
    /// The requested entity does not exist.
    NotFound,
    /// Version or state conflict (e.g. stale proposal).
    Conflict,
    /// A filesystem operation failed.
    Io,
    /// AIGC subsystem is unavailable (e.g. inference bootstrap failed).
    AigcUnavailable,
}

impl AppHostErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BootstrapFailed => "bootstrap_failed",
            Self::CommandFailed => "command_failed",
            Self::UnknownProvider => "unknown_provider",
            Self::ModelNotFound => "model_not_found",
            Self::ModelDownloadFailed => "model_download_failed",
            Self::InferenceError => "inference_error",
            Self::WorkerUnavailable => "worker_unavailable",
            Self::WorkflowInvalid => "workflow_invalid",
            Self::PermissionDenied => "permission_denied",
            Self::NotFound => "not_found",
            Self::Conflict => "conflict",
            Self::Io => "io_error",
            Self::AigcUnavailable => "aigc_unavailable",
        }
    }
}

/// Classification for [`WorkerSwitchError`]. Every switch failure means the
/// requested worker is not usable, so all variants map to
/// [`AppHostErrorCode::WorkerUnavailable`].
#[cfg(feature = "aigc")]
pub fn worker_switch_error_code(error: &WorkerSwitchError) -> AppHostErrorCode {
    match error {
        WorkerSwitchError::NoActiveWorker
        | WorkerSwitchError::Startup { .. }
        | WorkerSwitchError::TargetNotReady { .. }
        | WorkerSwitchError::DrainTimeout { .. }
        | WorkerSwitchError::Cancellation { .. }
        | WorkerSwitchError::Shutdown { .. }
        | WorkerSwitchError::StaleHandle { .. } => AppHostErrorCode::WorkerUnavailable,
    }
}

/// Optional structured context for a [`WorkerSwitchError`] error payload.
#[cfg(feature = "aigc")]
pub fn worker_switch_error_details(error: &WorkerSwitchError) -> Option<serde_json::Value> {
    match error {
        WorkerSwitchError::TargetNotReady { instance }
        | WorkerSwitchError::DrainTimeout { instance }
        | WorkerSwitchError::Shutdown { instance, .. }
        | WorkerSwitchError::StaleHandle { instance } => {
            Some(serde_json::json!({ "instance": instance.to_string() }))
        }
        WorkerSwitchError::Cancellation { run_id, .. } => {
            Some(serde_json::json!({ "run_id": run_id.to_string() }))
        }
        WorkerSwitchError::NoActiveWorker | WorkerSwitchError::Startup { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AppHostError;
    use reimagine_core::model::{DocumentVersion, RunId, WorkflowId};

    fn workflow_error() -> AppHostError {
        AppHostError::UnknownWorkflow {
            workflow_id: WorkflowId::new("wf-1"),
        }
    }

    #[test]
    fn unknown_entities_classify_as_not_found() {
        assert_eq!(workflow_error().code(), AppHostErrorCode::NotFound);
        assert_eq!(
            AppHostError::UnknownRun {
                run_id: RunId::new("run_x")
            }
            .code(),
            AppHostErrorCode::NotFound
        );
        assert_eq!(
            AppHostError::NoPendingProposal {
                workflow_id: WorkflowId::new("wf-1")
            }
            .code(),
            AppHostErrorCode::NotFound
        );
    }

    #[test]
    fn stale_and_conflicting_state_classify_as_conflict() {
        assert_eq!(
            AppHostError::ProposalStale {
                workflow_id: WorkflowId::new("wf-1"),
                proposal_base_version: DocumentVersion::new(1),
                current_version: DocumentVersion::new(2),
            }
            .code(),
            AppHostErrorCode::Conflict
        );
        assert_eq!(
            AppHostError::WorkflowVersionConflict {
                workflow_id: WorkflowId::new("wf-1"),
                expected: DocumentVersion::new(1),
                actual: DocumentVersion::new(2),
            }
            .code(),
            AppHostErrorCode::Conflict
        );
    }

    #[test]
    fn workflow_json_classifies_as_workflow_invalid() {
        assert_eq!(
            AppHostError::WorkflowJson {
                path: std::path::PathBuf::new(),
                message: "bad json".to_owned(),
            }
            .code(),
            AppHostErrorCode::WorkflowInvalid
        );
    }

    #[test]
    fn unknown_provider_keeps_its_wire_name() {
        assert_eq!(
            AppHostErrorCode::UnknownProvider.as_str(),
            "unknown_provider"
        );
        assert_eq!(
            serde_json::to_value(AppHostErrorCode::UnknownProvider).expect("serialize"),
            "unknown_provider"
        );
    }

    #[test]
    fn bootstrap_failures_classify_as_bootstrap_failed() {
        assert_eq!(
            AppHostError::InferenceBootstrap {
                message: "boom".to_owned()
            }
            .code(),
            AppHostErrorCode::BootstrapFailed
        );
    }

    #[test]
    fn aigc_unavailable_keeps_its_wire_name() {
        assert_eq!(
            AppHostErrorCode::AigcUnavailable.as_str(),
            "aigc_unavailable"
        );
        assert_eq!(
            serde_json::to_value(AppHostErrorCode::AigcUnavailable).expect("serialize"),
            "aigc_unavailable"
        );
        assert_eq!(
            AppHostError::AigcUnavailable {
                reason: "inference bootstrap failed".to_owned()
            }
            .code(),
            AppHostErrorCode::AigcUnavailable
        );
    }

    #[test]
    fn io_classifies_as_io() {
        assert_eq!(
            AppHostError::Io {
                path: std::path::PathBuf::from("/tmp/x"),
                message: "boom".to_owned(),
            }
            .code(),
            AppHostErrorCode::Io
        );
    }

    #[cfg(feature = "aigc")]
    #[test]
    fn worker_switch_errors_classify_as_worker_unavailable() {
        for error in [
            WorkerSwitchError::NoActiveWorker,
            WorkerSwitchError::StaleHandle {
                instance: reimagine_inference::BackendInstance::new("burn:wgpu:default"),
            },
            WorkerSwitchError::TargetNotReady {
                instance: reimagine_inference::BackendInstance::new("burn:wgpu:default"),
            },
        ] {
            assert_eq!(
                worker_switch_error_code(&error),
                AppHostErrorCode::WorkerUnavailable
            );
        }
    }

    #[cfg(feature = "aigc")]
    #[test]
    fn stale_handle_details_carry_instance() {
        let error = WorkerSwitchError::StaleHandle {
            instance: reimagine_inference::BackendInstance::new("burn:wgpu:default"),
        };
        let details = worker_switch_error_details(&error).expect("details");
        assert_eq!(details["instance"], "burn:wgpu:default");
    }
}
