pub mod auth;
pub mod client;
pub mod conversion;
pub mod local;
pub mod server;
/// Test-only fake worker service. Not part of the public contract.
#[doc(hidden)]
pub mod testing;
pub mod tls;

pub use auth::GrpcAuth;
pub use client::WorkerClient;
pub use local::{LocalEndpoint, LocalListenError};
pub use tls::GrpcTls;

/// Include the generated protobuf code for the `reimagine.worker.v2`
/// contract.
pub mod proto {
    tonic::include_proto!("reimagine.worker.v2");
}

/// Maximum size of a single protobuf message in either direction.
///
/// Payloads are JSON documents (tensor data never crosses the wire);
/// the limit mirrors the old frame-protocol cap so large JSON responses
/// keep flowing.
pub const MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

/// Re-exports so worker implementations and typed-client consumers do
/// not need tonic as a direct dependency.
pub use tonic::async_trait;
pub use tonic::{Request, Response, Status, Streaming};
