//! The worker contract: the `reimagine.worker.v2` protobuf service, the
//! Rust types the protobuf messages map to, and both transports the
//! contract travels on - a private local socket (Unix domain socket,
//! Windows named pipe) and TLS-protected HTTP/2 for remote workers with
//! TOFU certificate pinning, bearer-token authentication, and mDNS
//! discovery/registration.
//!
//! The protocol version the compatibility checks are written against is
//! [`WORKER_PROTOCOL_VERSION`]; it travels once, in the `Hello` response.

mod error;
mod handshake;
mod identity;
mod progress;
mod request;
mod response;
pub mod transport;
mod worker_profile;

pub mod auth;
pub mod client;
pub mod conversion;
pub mod discovery;
pub mod local;
pub mod remote;
pub mod server;
/// Test-only fake worker service. Not part of the public contract.
#[doc(hidden)]
pub mod testing;
pub mod tls;
pub mod trust;

pub use error::BackendExecutionError;
pub use handshake::{ProtocolRange, ProtocolVersion, WORKER_PROTOCOL_VERSION, WorkerHello};
pub use identity::{BackendInstanceId, WorkerIdentity, WorkerIncarnationId, WorkerInstallationId};
pub use progress::ProgressFrame;
pub use request::request_operation;
pub use response::TerminalOutcome;
pub use transport::{TransportDescription, TransportError, TransportKind};
pub use worker_profile::{WorkerInstanceProfile, WorkerProfile};

pub use auth::{GrpcAuth, TOKEN_ENV};
pub use client::WorkerClient;
pub use local::{LocalEndpoint, LocalListenError};
pub use remote::{ConnectTrust, TrustPolicy};
pub use tls::GrpcTls;
pub use trust::{CertFingerprint, TrustedKeyStore};

/// Include the generated protobuf code for the `reimagine.worker.v2`
/// contract.
pub mod proto {
    tonic::include_proto!("reimagine.worker.v2");
}

/// Maximum size of a single protobuf message in either direction.
///
/// Payloads are JSON documents (tensor data never crosses the wire);
/// the limit mirrors the frame-protocol cap this contract replaced so
/// large JSON responses keep flowing.
pub const MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

/// Re-exports so worker implementations and typed-client consumers do
/// not need tonic as a direct dependency.
pub use tonic::async_trait;
pub use tonic::{Request, Response, Status, Streaming};
