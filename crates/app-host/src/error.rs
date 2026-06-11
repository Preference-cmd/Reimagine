use reimagine_core::model::{RunId, WorkflowId, WorkflowVersion};
use reimagine_runtime::RuntimeServiceError;

pub type AppHostResult<T> = Result<T, AppHostError>;

#[derive(Debug)]
pub enum AppHostError {
    UnknownWorkflow {
        workflow_id: WorkflowId,
    },
    NoPendingProposal {
        workflow_id: WorkflowId,
    },
    UnknownAgentSession {
        session_id: reimagine_agent::AgentSessionId,
    },
    UnknownAgentProvider {
        provider: reimagine_agent::ProviderName,
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
    Io {
        path: std::path::PathBuf,
        message: String,
    },
    WorkflowJson {
        path: std::path::PathBuf,
        message: String,
    },
    Runtime(RuntimeServiceError),
    ModelManager(reimagine_model_manager::ModelManagerError),
}

impl std::fmt::Display for AppHostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownWorkflow { workflow_id } => {
                write!(f, "unknown workflow `{workflow_id}`")
            }
            Self::NoPendingProposal { workflow_id } => {
                write!(f, "no pending proposal for workflow `{workflow_id}`")
            }
            Self::UnknownAgentSession { session_id } => {
                write!(f, "unknown agent session `{session_id}`")
            }
            Self::UnknownAgentProvider { provider } => {
                write!(f, "unknown agent provider `{provider}`")
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
            Self::Io { path, message } => {
                write!(f, "io error at `{}`: {message}", path.display())
            }
            Self::WorkflowJson { path, message } => {
                write!(f, "workflow json error at `{}`: {message}", path.display())
            }
            Self::Runtime(error) => write!(f, "{error}"),
            Self::ModelManager(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for AppHostError {}

impl From<reimagine_model_manager::ModelManagerError> for AppHostError {
    fn from(value: reimagine_model_manager::ModelManagerError) -> Self {
        Self::ModelManager(value)
    }
}

impl From<RuntimeServiceError> for AppHostError {
    fn from(value: RuntimeServiceError) -> Self {
        Self::Runtime(value)
    }
}
