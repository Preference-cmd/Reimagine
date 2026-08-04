pub mod client;
pub mod conversion;
pub mod server;
pub mod transport;

/// Include the generated protobuf code.
pub mod proto {
    tonic::include_proto!("reimagine.worker.v1");
}
