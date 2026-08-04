pub mod conversion;

/// Include the generated protobuf code.
pub mod proto {
    tonic::include_proto!("reimagine.worker.v1");
}
