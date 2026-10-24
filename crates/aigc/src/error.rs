//! Error types for the AIGC domain module.

use std::path::PathBuf;

use reimagine_config::ConfigError;
use reimagine_core::diagnostic::DiagnosticError;

use crate::worker_management::WorkerManagementError;

pub type AigcResult<T> = Result<T, AigcError>;

/// Top-level error type for AIGC operations.
#[derive(Debug, thiserror::Error)]
pub enum AigcError {
    #[error("io error at `{}`: {message}", path.display())]
    Io { path: PathBuf, message: String },
    #[error("config bootstrap failed: {0}")]
    BootstrapConfig(#[from] ConfigError),
    #[error("inference bootstrap failed: {message}")]
    InferenceBootstrap { message: String },
    #[error("{0}")]
    ModelManager(#[from] reimagine_model_manager::ModelManagerError),
    #[error("{}", DiagnosticError::user_message(.0))]
    ModelAcquisition(#[from] reimagine_model_acquisition::ModelAcquisitionError),
    #[cfg(feature = "candle")]
    #[error("{0}")]
    CandleCheckpointImport(#[from] reimagine_inference_candle::SdxlCheckpointImportError),
    #[error("Burn checkpoint import error: {message}")]
    BurnCheckpointImport { message: String },
    #[error("worker management error: {0}")]
    WorkerManagement(#[from] WorkerManagementError),
}
