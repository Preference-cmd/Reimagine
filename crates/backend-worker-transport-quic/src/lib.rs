pub mod listener;
pub mod tls;
pub mod transport;

pub use transport::QuicTransport;

/// Errors specific to the QUIC transport.
#[derive(Debug)]
pub enum Error {
    Certificate(String),
    Tls(String),
    ConnectionFailed(String),
    Io(String),
    Protocol(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Certificate(msg) => write!(f, "certificate error: {msg}"),
            Self::Tls(msg) => write!(f, "TLS error: {msg}"),
            Self::ConnectionFailed(msg) => write!(f, "connection failed: {msg}"),
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
            Self::Protocol(msg) => write!(f, "protocol error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}
