use std::fmt;

/// Error type for transport-level operations: spawning or reaching a
/// worker process. Protocol-level results (backend errors, cancellations)
/// are not transport failures.
#[derive(Debug)]
pub enum TransportError {
    Io(String),
    ConnectionFailed(String),
    Timeout,
    Closed,
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "transport I/O error: {msg}"),
            Self::ConnectionFailed(msg) => write!(f, "transport connection failed: {msg}"),
            Self::Timeout => write!(f, "transport operation timed out"),
            Self::Closed => write!(f, "transport closed"),
        }
    }
}

impl std::error::Error for TransportError {}

/// The kind of transport used to reach a worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportKind {
    /// A local child process reached over a private socket or named pipe.
    Local,
    Quic,
    Grpc,
    /// In-memory transport used by tests.
    Mock,
}

/// Descriptive metadata about a transport connection.
#[derive(Debug, Clone)]
pub struct TransportDescription {
    pub kind: TransportKind,
    pub endpoint: String,
}
