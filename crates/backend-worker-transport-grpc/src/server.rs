//! Server-side glue for the `reimagine.worker.v2` contract.
//!
//! Worker binaries implement the generated [`proto::worker_server::Worker`]
//! trait and serve it through [`serve_local`]; token enforcement is a
//! one-line call per RPC via [`require_bearer`].

use tonic::Request;
use tonic::Status;

use crate::auth::check_bearer;
use crate::local::{LocalEndpoint, LocalIncoming};
use crate::proto;
use crate::proto::worker_server::WorkerServer;

/// Build a tonic server for `service` with the worker message-size
/// limits applied.
#[must_use]
pub fn worker_server<S>(service: S) -> WorkerServer<S>
where
    S: proto::worker_server::Worker,
{
    WorkerServer::new(service)
        .max_decoding_message_size(crate::MAX_MESSAGE_BYTES)
        .max_encoding_message_size(crate::MAX_MESSAGE_BYTES)
}

/// Serve the v2 contract on a bound local endpoint.
///
/// This future runs until the incoming stream errors or the runtime is
/// torn down; workers supervise it alongside their stdin lifeline.
pub async fn serve_local<S>(
    incoming: LocalIncoming,
    service: WorkerServer<S>,
) -> Result<(), tonic::transport::Error>
where
    S: proto::worker_server::Worker + Send + Sync + 'static,
{
    crate::local::serve(incoming, service).await
}

/// Reject a request whose bearer token does not match `expected`.
///
/// Every RPC on a token-authenticated worker must call this first.
pub fn require_bearer<T>(request: &Request<T>, expected: Option<&str>) -> Result<(), Status> {
    check_bearer(request.metadata(), expected)
}

/// Parse and validate a local endpoint passed on the worker command
/// line.
pub fn parse_listen_endpoint(endpoint: &str) -> Result<LocalEndpoint, Status> {
    LocalEndpoint::parse(endpoint).map_err(|error| Status::invalid_argument(error.to_string()))
}
