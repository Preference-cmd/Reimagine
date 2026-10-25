use std::fmt;
use std::path::PathBuf;

use reimagine_backend_worker_protocol::TransportError;

#[derive(Debug)]
pub enum WorkerHostError {
    Spawn {
        path: PathBuf,
        message: String,
    },
    Transport(TransportError),
    /// The worker did not become ready within the startup deadline.
    StartupTimeout,
    /// The worker printed something other than its READY line on stdout.
    StartupOutput {
        line: String,
    },
    IdentityMismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },
    CapabilityMismatch {
        requested: String,
        available: Vec<String>,
    },
    RequestTimeout,
    TransportLost {
        message: String,
    },
    ShutdownTimeout,
    AlreadyStarted,
}

impl fmt::Display for WorkerHostError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn { path, message } => {
                write!(
                    formatter,
                    "failed to spawn worker `{}`: {message}",
                    path.display()
                )
            }
            Self::Transport(error) => write!(formatter, "worker transport failed: {error}"),
            Self::StartupTimeout => write!(formatter, "worker startup timed out"),
            Self::StartupOutput { line } => write!(
                formatter,
                "worker wrote `{line}` on stdout instead of its READY line"
            ),
            Self::IdentityMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "worker identity field `{field}` expected `{expected}` but received `{actual}`"
            ),
            Self::CapabilityMismatch {
                requested,
                available,
            } => write!(
                formatter,
                "worker does not support capability `{requested}`; available: [{}]",
                available.join(", ")
            ),
            Self::RequestTimeout => write!(formatter, "worker request timed out"),
            Self::TransportLost { message } => {
                write!(formatter, "worker transport lost: {message}")
            }
            Self::ShutdownTimeout => write!(formatter, "worker graceful shutdown timed out"),
            Self::AlreadyStarted => write!(formatter, "worker supervisor already owns a process"),
        }
    }
}

impl std::error::Error for WorkerHostError {}
