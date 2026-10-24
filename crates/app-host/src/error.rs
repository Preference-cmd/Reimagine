use reimagine_config::ConfigError;
use reimagine_core::model::{DocumentVersion, ProjectId, RunId, WorkflowId};
use reimagine_runtime::RuntimeServiceError;

use crate::artifact_access::ArtifactAccessError;
use crate::error_code::AppHostErrorCode;

pub type AppHostResult<T> = Result<T, AppHostError>;

#[derive(Debug)]
pub enum AppHostError {
    UnknownProject {
        project_id: ProjectId,
    },
    ProjectAlreadyExists {
        project_id: ProjectId,
    },
    ProjectIdPathUnsafe {
        project_id: String,
    },
    UnknownBoard {
        project_id: ProjectId,
    },
    UnknownWorkflow {
        workflow_id: WorkflowId,
    },
    NoPendingProposal {
        workflow_id: WorkflowId,
    },
    ProposalStale {
        workflow_id: WorkflowId,
        proposal_base_version: DocumentVersion,
        current_version: DocumentVersion,
    },
    UnknownRun {
        run_id: RunId,
    },
    WorkflowIdPathUnsafe {
        workflow_id: WorkflowId,
    },
    WorkflowVersionConflict {
        workflow_id: WorkflowId,
        expected: DocumentVersion,
        actual: DocumentVersion,
    },
    /// The workflow commands applied in memory but their atomic
    /// persistence to `projects/{project_id}/workflows/` failed (AR-08).
    /// The session was rolled back; the caller must not report success.
    WorkflowPersistFailed {
        workflow_id: WorkflowId,
        message: String,
    },
    Io {
        path: std::path::PathBuf,
        message: String,
    },
    WorkflowJson {
        path: std::path::PathBuf,
        message: String,
    },
    BootstrapConfig(ConfigError),
    InferenceBootstrap {
        message: String,
    },
    RebootFailed {
        message: String,
    },
    Runtime(RuntimeServiceError),
    ModelManager(reimagine_model_manager::ModelManagerError),
    ModelAcquisition(reimagine_model_acquisition::ModelAcquisitionError),
    BurnCheckpointImport {
        message: String,
    },
    ArtifactAccess(ArtifactAccessError),
    #[cfg(feature = "aigc")]
    WorkerManagement(aigc::WorkerManagementError),
    AigcUnavailable {
        reason: String,
    },
    #[cfg(feature = "aigc")]
    WorkerSwitch(aigc::WorkerSwitchError),
}

impl std::fmt::Display for AppHostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownProject { project_id } => {
                write!(f, "unknown project `{project_id}`")
            }
            Self::ProjectAlreadyExists { project_id } => {
                write!(f, "project `{project_id}` already exists")
            }
            Self::ProjectIdPathUnsafe { project_id } => {
                write!(
                    f,
                    "project id `{project_id}` is not safe as a project directory name"
                )
            }
            Self::UnknownBoard { project_id } => {
                write!(f, "project `{project_id}` has no board")
            }
            Self::UnknownWorkflow { workflow_id } => {
                write!(f, "unknown workflow `{workflow_id}`")
            }
            Self::NoPendingProposal { workflow_id } => {
                write!(f, "no pending proposal for workflow `{workflow_id}`")
            }
            Self::ProposalStale {
                workflow_id,
                proposal_base_version,
                current_version,
            } => write!(
                f,
                "proposal for workflow `{workflow_id}` is stale: proposal was based on version {proposal_base_version}, but the workflow is now at version {current_version}"
            ),
            Self::UnknownRun { run_id } => {
                write!(f, "unknown run `{run_id}`")
            }
            Self::WorkflowIdPathUnsafe { workflow_id } => {
                write!(f, "workflow id `{workflow_id}` is not safe as a file name")
            }
            Self::WorkflowVersionConflict {
                workflow_id,
                expected,
                actual,
            } => write!(
                f,
                "workflow `{workflow_id}` version conflict: expected {expected}, got {actual}"
            ),
            Self::WorkflowPersistFailed {
                workflow_id,
                message,
            } => write!(
                f,
                "workflow `{workflow_id}` could not be persisted after apply: {message}"
            ),
            Self::Io { path, message } => {
                write!(f, "io error at `{}`: {message}", path.display())
            }
            Self::WorkflowJson { path, message } => {
                write!(f, "workflow json error at `{}`: {message}", path.display())
            }
            Self::BootstrapConfig(error) => write!(f, "config bootstrap failed: {error}"),
            Self::InferenceBootstrap { message } => {
                write!(f, "inference bootstrap failed: {message}")
            }
            Self::RebootFailed { message } => {
                write!(f, "re-bootstrap failed: {message}")
            }
            Self::Runtime(error) => write!(f, "{error}"),
            Self::ModelManager(error) => write!(f, "{error}"),
            Self::ModelAcquisition(error) => {
                write!(
                    f,
                    "{}",
                    reimagine_core::diagnostic::DiagnosticError::user_message(error)
                )
            }
            Self::BurnCheckpointImport { message } => {
                write!(f, "Burn checkpoint import error: {message}")
            }
            Self::ArtifactAccess(error) => write!(f, "artifact access error: {error}"),
            #[cfg(feature = "aigc")]
            Self::WorkerManagement(error) => write!(f, "worker management error: {error}"),
            Self::AigcUnavailable { reason } => {
                write!(f, "AIGC subsystem is unavailable: {reason}")
            }
            #[cfg(feature = "aigc")]
            Self::WorkerSwitch(error) => write!(f, "{error}"),
        }
    }
}

impl AppHostError {
    /// Machine-readable classification for IPC error payloads.
    pub fn code(&self) -> AppHostErrorCode {
        match self {
            Self::UnknownProject { .. } => AppHostErrorCode::NotFound,
            Self::ProjectAlreadyExists { .. } => AppHostErrorCode::Conflict,
            Self::ProjectIdPathUnsafe { .. } => AppHostErrorCode::PermissionDenied,
            Self::UnknownBoard { .. } => AppHostErrorCode::NotFound,
            Self::UnknownWorkflow { .. } => AppHostErrorCode::NotFound,
            Self::NoPendingProposal { .. } => AppHostErrorCode::NotFound,
            Self::ProposalStale { .. } => AppHostErrorCode::Conflict,
            Self::UnknownRun { .. } => AppHostErrorCode::NotFound,
            Self::WorkflowIdPathUnsafe { .. } => AppHostErrorCode::PermissionDenied,
            Self::WorkflowVersionConflict { .. } => AppHostErrorCode::Conflict,
            Self::WorkflowPersistFailed { .. } => AppHostErrorCode::Io,
            Self::Io { .. } => AppHostErrorCode::Io,
            Self::WorkflowJson { .. } => AppHostErrorCode::WorkflowInvalid,
            Self::BootstrapConfig(_)
            | Self::InferenceBootstrap { .. }
            | Self::RebootFailed { .. } => AppHostErrorCode::BootstrapFailed,
            Self::Runtime(_) => AppHostErrorCode::InferenceError,
            Self::ModelManager(_) => AppHostErrorCode::ModelNotFound,
            Self::ModelAcquisition(error) => match error {
                reimagine_model_acquisition::ModelAcquisitionError::ConfigInvalid { .. }
                | reimagine_model_acquisition::ModelAcquisitionError::InvalidRequest { .. } => {
                    AppHostErrorCode::CommandFailed
                }
                _ => AppHostErrorCode::ModelDownloadFailed,
            },
            Self::BurnCheckpointImport { .. } => AppHostErrorCode::CommandFailed,
            Self::ArtifactAccess(error) => match error {
                ArtifactAccessError::UnknownArtifact | ArtifactAccessError::FileGone => {
                    AppHostErrorCode::NotFound
                }
                ArtifactAccessError::UnsafeReference => AppHostErrorCode::PermissionDenied,
                ArtifactAccessError::UnsupportedMedia => AppHostErrorCode::CommandFailed,
            },
            #[cfg(feature = "aigc")]
            Self::WorkerManagement(_) => AppHostErrorCode::WorkerUnavailable,
            Self::AigcUnavailable { .. } => AppHostErrorCode::AigcUnavailable,
            #[cfg(feature = "aigc")]
            Self::WorkerSwitch(error) => crate::error_code::worker_switch_error_code(error),
        }
    }

    /// Optional structured context for the IPC error payload.
    pub fn details(&self) -> Option<serde_json::Value> {
        match self {
            Self::UnknownProject { project_id }
            | Self::ProjectAlreadyExists { project_id }
            | Self::UnknownBoard { project_id } => {
                Some(serde_json::json!({ "project_id": project_id.to_string() }))
            }
            Self::ProjectIdPathUnsafe { project_id } => {
                Some(serde_json::json!({ "project_id": project_id }))
            }
            Self::UnknownWorkflow { workflow_id } => {
                Some(serde_json::json!({ "workflow_id": workflow_id.to_string() }))
            }
            Self::NoPendingProposal { workflow_id } => {
                Some(serde_json::json!({ "workflow_id": workflow_id.to_string() }))
            }
            Self::ProposalStale {
                workflow_id,
                proposal_base_version,
                current_version,
            } => Some(serde_json::json!({
                "workflow_id": workflow_id.to_string(),
                "proposal_base_version": proposal_base_version.to_string(),
                "current_version": current_version.to_string(),
            })),
            Self::UnknownRun { run_id } => {
                Some(serde_json::json!({ "run_id": run_id.to_string() }))
            }
            Self::WorkflowVersionConflict {
                workflow_id,
                expected,
                actual,
            } => Some(serde_json::json!({
                "workflow_id": workflow_id.to_string(),
                "expected": expected.to_string(),
                "actual": actual.to_string(),
            })),
            Self::WorkflowPersistFailed { workflow_id, .. } => {
                Some(serde_json::json!({ "workflow_id": workflow_id.to_string() }))
            }
            Self::Io { path, .. } => {
                Some(serde_json::json!({ "path": path.display().to_string() }))
            }
            Self::WorkflowJson { path, .. } => {
                if path.as_os_str().is_empty() {
                    None
                } else {
                    Some(serde_json::json!({ "path": path.display().to_string() }))
                }
            }
            Self::AigcUnavailable { reason } => Some(serde_json::json!({ "reason": reason })),
            #[cfg(feature = "aigc")]
            Self::WorkerSwitch(error) => crate::error_code::worker_switch_error_details(error),
            _ => None,
        }
    }
}

impl std::error::Error for AppHostError {}

impl From<reimagine_model_acquisition::ModelAcquisitionError> for AppHostError {
    fn from(value: reimagine_model_acquisition::ModelAcquisitionError) -> Self {
        Self::ModelAcquisition(value)
    }
}

impl From<reimagine_model_manager::ModelManagerError> for AppHostError {
    fn from(value: reimagine_model_manager::ModelManagerError) -> Self {
        Self::ModelManager(value)
    }
}

impl From<ConfigError> for AppHostError {
    fn from(error: ConfigError) -> Self {
        Self::BootstrapConfig(error)
    }
}

impl From<RuntimeServiceError> for AppHostError {
    fn from(value: RuntimeServiceError) -> Self {
        Self::Runtime(value)
    }
}

impl From<ArtifactAccessError> for AppHostError {
    fn from(error: ArtifactAccessError) -> Self {
        Self::ArtifactAccess(error)
    }
}

#[cfg(feature = "aigc")]
impl From<aigc::WorkerManagementError> for AppHostError {
    fn from(error: aigc::WorkerManagementError) -> Self {
        Self::WorkerManagement(error)
    }
}

#[cfg(feature = "aigc")]
impl From<aigc::AigcError> for AppHostError {
    fn from(error: aigc::AigcError) -> Self {
        match error {
            aigc::AigcError::Io { path, message } => Self::Io { path, message },
            aigc::AigcError::BootstrapConfig(e) => Self::BootstrapConfig(e),
            aigc::AigcError::InferenceBootstrap { message } => Self::InferenceBootstrap { message },
            aigc::AigcError::ModelManager(e) => Self::ModelManager(e),
            aigc::AigcError::ModelAcquisition(e) => Self::ModelAcquisition(e),
            aigc::AigcError::BurnCheckpointImport { message } => {
                Self::BurnCheckpointImport { message }
            }
            aigc::AigcError::WorkerManagement(e) => Self::WorkerManagement(e),
        }
    }
}

#[cfg(feature = "aigc")]
impl From<aigc::WorkerSwitchError> for AppHostError {
    fn from(error: aigc::WorkerSwitchError) -> Self {
        Self::WorkerSwitch(error)
    }
}
