use std::fmt;
use std::path::PathBuf;

use reimagine_backend_worker_protocol::CodecError;

#[derive(Debug)]
pub enum WorkerHostError {
    Spawn {
        path: PathBuf,
        message: String,
    },
    Io {
        operation: &'static str,
        message: String,
    },
    CleanEof {
        operation: &'static str,
    },
    IncompleteFrame {
        operation: &'static str,
        received: usize,
        expected: usize,
    },
    Protocol(CodecError),
    StartupTimeout,
    UnexpectedStartupMessage {
        kind: &'static str,
    },
    IdentityMismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },
    RequestTimeout {
        request_id: String,
    },
    ControlTimeout {
        control_id: String,
    },
    TransportLost {
        message: String,
    },
    UnexpectedWorkerMessage {
        kind: &'static str,
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
            Self::Io { operation, message } => {
                write!(formatter, "worker {operation} I/O failed: {message}")
            }
            Self::CleanEof { operation } => {
                write!(formatter, "worker closed stdout before {operation}")
            }
            Self::IncompleteFrame {
                operation,
                received,
                expected,
            } => write!(
                formatter,
                "worker closed stdout during {operation}: received {received} of {expected} bytes"
            ),
            Self::Protocol(error) => write!(formatter, "worker protocol failed: {error}"),
            Self::StartupTimeout => write!(formatter, "worker startup handshake timed out"),
            Self::UnexpectedStartupMessage { kind } => {
                write!(formatter, "worker sent `{kind}` instead of worker_hello")
            }
            Self::IdentityMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "worker identity field `{field}` expected `{expected}` but received `{actual}`"
            ),
            Self::RequestTimeout { request_id } => {
                write!(formatter, "worker request `{request_id}` timed out")
            }
            Self::ControlTimeout { control_id } => {
                write!(formatter, "worker control `{control_id}` timed out")
            }
            Self::TransportLost { message } => {
                write!(formatter, "worker transport lost: {message}")
            }
            Self::UnexpectedWorkerMessage { kind } => {
                write!(formatter, "unexpected worker message `{kind}`")
            }
            Self::ShutdownTimeout => write!(formatter, "worker graceful shutdown timed out"),
            Self::AlreadyStarted => write!(formatter, "worker supervisor already owns a process"),
        }
    }
}

impl std::error::Error for WorkerHostError {}

impl From<CodecError> for WorkerHostError {
    fn from(error: CodecError) -> Self {
        Self::Protocol(error)
    }
}
