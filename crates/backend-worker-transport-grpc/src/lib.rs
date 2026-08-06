pub mod auth;
pub mod client;
pub mod conversion;
pub mod server;
pub mod tls;
pub mod transport;

pub use client::{GrpcAuth, GrpcTls};

/// Include the generated protobuf code.
pub mod proto {
    tonic::include_proto!("reimagine.worker.v1");
}
