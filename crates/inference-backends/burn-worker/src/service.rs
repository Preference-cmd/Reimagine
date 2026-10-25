//! The Burn worker's implementation of the `reimagine.worker.v2`
//! contract.
//!
//! Every RPC first checks the bearer token. `Execute` dispatches the
//! operation on a worker thread with a per-request cancellation token:
//! dropping the response stream (the gRPC cancellation path) cancels
//! that token, so a host-side cancel stops the running operation.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use reimagine_backend_worker_protocol::{
    BackendExecutionError, TerminalOutcome, WORKER_PROTOCOL_VERSION,
};
use reimagine_backend_worker_transport_grpc::Request;
use reimagine_backend_worker_transport_grpc::Response;
use reimagine_backend_worker_transport_grpc::Status;
use reimagine_backend_worker_transport_grpc::async_trait;
use reimagine_backend_worker_transport_grpc::conversion::outcome_to_proto;
use reimagine_backend_worker_transport_grpc::proto::worker_server::Worker;
use reimagine_backend_worker_transport_grpc::proto::{
    self, CleanupRequest, CleanupResponse, ExecuteEvent, ExecuteRequest, HealthRequest,
    HealthResponse, HelloRequest, HelloResponse, ShutdownRequest, ShutdownResponse,
};
use reimagine_backend_worker_transport_grpc::server::require_bearer;
use reimagine_inference_burn::{BurnBackend, with_request_cancellation};
use tokio::sync::mpsc;
use tokio_stream::Stream;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

use crate::mapping::{self, MappingResult, TokenGenerator};
use crate::probe;
use crate::shutdown;

/// Response stream that cancels the running operation when the client
/// drops the call.
pub struct ExecuteStream {
    inner: ReceiverStream<Result<ExecuteEvent, Status>>,
    cancellation: Arc<CancellationToken>,
}

impl Stream for ExecuteStream {
    type Item = Result<ExecuteEvent, Status>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl Drop for ExecuteStream {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

/// The Burn worker service over the v2 contract.
pub struct BurnWorkerService {
    backend: BurnBackend,
    tokens: Arc<TokenGenerator>,
    runtime: Arc<tokio::runtime::Runtime>,
    token: Option<String>,
    shutdown_requested: Arc<AtomicBool>,
}

impl BurnWorkerService {
    #[must_use]
    pub fn new(
        backend: BurnBackend,
        token: Option<String>,
        shutdown_requested: Arc<AtomicBool>,
    ) -> Self {
        Self {
            backend,
            tokens: Arc::new(TokenGenerator::new()),
            runtime: Arc::new(
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("worker dispatch runtime"),
            ),
            token,
            shutdown_requested,
        }
    }
}

#[async_trait]
impl Worker for BurnWorkerService {
    async fn hello(
        &self,
        request: Request<HelloRequest>,
    ) -> Result<Response<HelloResponse>, Status> {
        require_bearer(&request, self.token.as_deref())?;
        let (identity, profile) = probe::build(&self.backend);
        Ok(Response::new(HelloResponse {
            protocol_version: u32::from(WORKER_PROTOCOL_VERSION.0),
            identity: Some(proto::WorkerIdentity {
                backend_instance_id: identity.backend_instance_id.0,
                installation_id: identity.installation_id.0,
                incarnation_id: identity.incarnation_id.0,
                worker_version: identity.worker_version,
                backend_kind: identity.backend_kind,
                target: identity.target,
                manifest_digest: identity.manifest_digest,
            }),
            profile: Some(
                reimagine_backend_worker_transport_grpc::conversion::profile_to_proto(&profile),
            ),
        }))
    }

    type ExecuteStream = ExecuteStream;

    async fn execute(
        &self,
        request: Request<ExecuteRequest>,
    ) -> Result<Response<Self::ExecuteStream>, Status> {
        require_bearer(&request, self.token.as_deref())?;
        let request = request.into_inner();
        let payload: serde_json::Value = serde_json::from_slice(&request.payload)
            .map_err(|error| Status::invalid_argument(error.to_string()))?;

        let (tx, rx) = mpsc::channel::<Result<ExecuteEvent, Status>>(4);
        let cancellation = Arc::new(CancellationToken::new());
        let stream_cancellation = Arc::clone(&cancellation);

        let backend = self.backend.clone();
        let tokens = Arc::clone(&self.tokens);
        let runtime = Arc::clone(&self.runtime);
        let operation = request.operation;
        let progress_tx = tx.clone();
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                with_request_cancellation(Arc::clone(&cancellation), || {
                    mapping::dispatch(&runtime, &backend, &tokens, &operation, &payload)
                })
            }))
            .unwrap_or_else(|_| {
                MappingResult::BackendError(BackendExecutionError {
                    code: "dispatch_panic".to_owned(),
                    message: format!("operation `{operation}` panicked"),
                    retryable: false,
                })
            });
            let outcome = match result {
                MappingResult::Success(output) => TerminalOutcome::Success { output },
                MappingResult::Cancelled => TerminalOutcome::Cancelled,
                MappingResult::BackendError(error) => TerminalOutcome::BackendError { error },
                MappingResult::NotImplemented => TerminalOutcome::BackendError {
                    error: BackendExecutionError {
                        code: "not_implemented".to_owned(),
                        message: "operation not implemented in this worker".to_owned(),
                        retryable: false,
                    },
                },
            };
            let _ = progress_tx.blocking_send(Ok(ExecuteEvent {
                event: Some(proto::execute_event::Event::Result(outcome_to_proto(
                    &outcome,
                ))),
            }));
        });

        Ok(Response::new(ExecuteStream {
            inner: ReceiverStream::new(rx),
            cancellation: stream_cancellation,
        }))
    }

    async fn health(
        &self,
        request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        require_bearer(&request, self.token.as_deref())?;
        Ok(Response::new(HealthResponse {
            healthy: true,
            message: None,
        }))
    }

    async fn cleanup(
        &self,
        request: Request<CleanupRequest>,
    ) -> Result<Response<CleanupResponse>, Status> {
        require_bearer(&request, self.token.as_deref())?;
        let request = request.into_inner();
        let mut released = request
            .run_id
            .map(|run_id| {
                self.backend
                    .store()
                    .cleanup_run(&reimagine_core::model::RunId::new(&run_id))
            })
            .unwrap_or(0);
        for object_id in &request.object_ids {
            released += usize::from(
                self.backend
                    .store()
                    .release_payload(&reimagine_inference::BackendPayloadKey::new(object_id)),
            );
        }
        Ok(Response::new(CleanupResponse {
            released_objects: released as u64,
        }))
    }

    async fn shutdown(
        &self,
        request: Request<ShutdownRequest>,
    ) -> Result<Response<ShutdownResponse>, Status> {
        require_bearer(&request, self.token.as_deref())?;
        shutdown::cleanup(&self.backend);
        // The main-thread monitor watches this flag and exits after a
        // short grace period, once the acknowledgement has flushed.
        self.shutdown_requested.store(true, Ordering::Release);
        Ok(Response::new(ShutdownResponse {}))
    }
}
