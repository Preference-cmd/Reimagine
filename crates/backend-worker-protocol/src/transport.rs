use std::fmt;

use async_trait::async_trait;

/// Error type for transport operations.
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

/// The kind of transport used for worker communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportKind {
    Stdio,
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

/// A bidirectional transport for worker communication.
///
/// Implementations provide the raw byte stream over which the
/// length-prefixed JSON worker protocol is exchanged. The protocol
/// layer (FrameCodec, WireMessage) is transport-agnostic and works
/// over any byte stream.
#[async_trait]
pub trait WorkerTransport: Send + Sync + 'static {
    /// Descriptive metadata about this transport.
    fn description(&self) -> TransportDescription;

    /// Gracefully shut down the transport.
    ///
    /// For stdio transports this force-kills the child process.
    /// For network transports this closes the connection.
    async fn shutdown(&self) -> Result<(), TransportError> {
        Ok(())
    }
}
