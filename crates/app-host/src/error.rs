use reimagine_config::ConfigError;
use reimagine_core::model::{ProjectId, RunId, WorkflowId, WorkflowVersion};
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
        proposal_base_version: WorkflowVersion,
        current_version: WorkflowVersion,
    },
    UnknownAgentSession {
        session_id: reimagine_agent_harness::AgentSessionId,
    },
    AgentTurnInProgress {
        session_id: reimagine_agent_harness::AgentSessionId,
    },
    NoActiveAgentTurn {
        session_id: reimagine_agent_harness::AgentSessionId,
    },
    UnknownAgentProvider {
        provider: reimagine_agent_harness::ProviderName,
    },
    UnknownAgentMode {
        mode: String,
    },
    UnknownRun {
        run_id: RunId,
    },
    WorkflowIdPathUnsafe {
        workflow_id: WorkflowId,
    },
    WorkflowVersionConflict {
        workflow_id: WorkflowId,
        expected: WorkflowVersion,
        actual: WorkflowVersion,
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
    #[cfg(feature = "candle")]
    CandleCheckpointImport(reimagine_inference_candle::SdxlCheckpointImportError),
    BurnCheckpointImport {
        message: String,
    },
    ArtifactAccess(ArtifactAccessError),
    WorkerManagement(crate::WorkerManagementError),
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
            Self::UnknownAgentSession { session_id } => {
                write!(f, "unknown agent session `{session_id}`")
            }
            Self::AgentTurnInProgress { session_id } => {
                write!(
                    f,
                    "agent session `{session_id}` already has a turn in progress"
                )
            }
            Self::NoActiveAgentTurn { session_id } => {
                write!(
                    f,
                    "agent session `{session_id}` has no active turn to cancel"
                )
            }
            Self::UnknownAgentProvider { provider } => {
                write!(f, "unknown agent provider `{provider}`")
            }
            Self::UnknownAgentMode { mode } => {
                write!(f, "unknown agent mode `{mode}`")
            }
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
            #[cfg(feature = "candle")]
            Self::CandleCheckpointImport(error) => write!(f, "{error}"),
            Self::BurnCheckpointImport { message } => {
                write!(f, "Burn checkpoint import error: {message}")
            }
            Self::ArtifactAccess(error) => write!(f, "artifact access error: {error}"),
            Self::WorkerManagement(error) => write!(f, "worker management error: {error}"),
        }
    }
}

impl AppHostError {
    /// Machine-readable classification for IPC error payloads.
    pub fn code(&self) -> AppHostErrorCode {
        match self {
            Self::UnknownProject { .. } => AppHostErrorCode::NotFound,
            Self::ProjectAlreadyExists { .. } => AppHostErrorCode::Conflict,
            Self::UnknownBoard { .. } => AppHostErrorCode::NotFound,
            Self::UnknownWorkflow { .. } => AppHostErrorCode::NotFound,
            Self::NoPendingProposal { .. } => AppHostErrorCode::NotFound,
            Self::ProposalStale { .. } => AppHostErrorCode::Conflict,
            Self::UnknownAgentSession { .. } => AppHostErrorCode::NotFound,
            Self::AgentTurnInProgress { .. } => AppHostErrorCode::Conflict,
            Self::NoActiveAgentTurn { .. } => AppHostErrorCode::NotFound,
            Self::UnknownAgentProvider { .. } => AppHostErrorCode::UnknownProvider,
            Self::UnknownAgentMode { .. } => AppHostErrorCode::CommandFailed,
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
            #[cfg(feature = "candle")]
            Self::CandleCheckpointImport(_) => AppHostErrorCode::CommandFailed,
            Self::BurnCheckpointImport { .. } => AppHostErrorCode::CommandFailed,
            Self::ArtifactAccess(error) => match error {
                ArtifactAccessError::UnknownArtifact | ArtifactAccessError::FileGone => {
                    AppHostErrorCode::NotFound
                }
                ArtifactAccessError::UnsafeReference => AppHostErrorCode::PermissionDenied,
                ArtifactAccessError::UnsupportedMedia => AppHostErrorCode::CommandFailed,
            },
            Self::WorkerManagement(_) => AppHostErrorCode::WorkerUnavailable,
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
            Self::UnknownAgentSession { session_id }
            | Self::AgentTurnInProgress { session_id }
            | Self::NoActiveAgentTurn { session_id } => {
                Some(serde_json::json!({ "session_id": session_id.to_string() }))
            }
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

#[cfg(feature = "candle")]
impl From<reimagine_inference_candle::SdxlCheckpointImportError> for AppHostError {
    fn from(value: reimagine_inference_candle::SdxlCheckpointImportError) -> Self {
        Self::CandleCheckpointImport(value)
    }
}

impl From<ArtifactAccessError> for AppHostError {
    fn from(error: ArtifactAccessError) -> Self {
        Self::ArtifactAccess(error)
    }
}

impl From<crate::WorkerManagementError> for AppHostError {
    fn from(error: crate::WorkerManagementError) -> Self {
        Self::WorkerManagement(error)
    }
}
