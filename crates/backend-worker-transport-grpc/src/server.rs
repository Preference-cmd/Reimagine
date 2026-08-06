use std::pin::Pin;
use std::sync::Arc;

use futures_core::Stream;
use tokio::sync::mpsc;
use tonic::{Request, Response, Status, Streaming};

use crate::auth::check_bearer;
use crate::proto;
use crate::proto::worker_service_server::WorkerService;

/// A handler callback invoked for each incoming `HostToWorker` message.
///
/// The handler receives the message and returns an optional
/// `WorkerToHost` response. Returning `None` means no response is
/// needed (e.g. for one-way messages).
pub type MessageHandler = Arc<
    dyn Fn(
            proto::HostToWorker,
        ) -> std::pin::Pin<Box<dyn Future<Output = Option<proto::WorkerToHost>> + Send>>
        + Send
        + Sync,
>;

use std::future::Future;

/// tonic `WorkerService` implementation for the worker side.
///
/// Accepts a bidirectional `Communication` stream, reads
/// `HostToWorker` messages, dispatches them through the provided
/// handler, and sends `WorkerToHost` responses back.
///
/// When a bearer token is configured, every RPC (including
/// `HealthCheck`) must carry `authorization: Bearer <token>`; requests
/// without a matching token are rejected with `Status::unauthenticated`.
pub struct GrpcWorkerService {
    handler: MessageHandler,
    token: Option<String>,
}

impl GrpcWorkerService {
    /// Create a new server with the given message handler and no token
    /// requirement (open/plain mode, backward compatible).
    pub fn new(handler: MessageHandler) -> Self {
        Self::with_token(handler, None)
    }

    /// Create a server that requires `authorization: Bearer <token>` on
    /// every RPC. `token == None` keeps the open (plain) mode.
    #[must_use]
    pub fn with_token(handler: MessageHandler, token: Option<String>) -> Self {
        Self { handler, token }
    }

    /// Authorize a request against the configured token.
    fn authorize(&self, metadata: &tonic::metadata::MetadataMap) -> Result<(), Status> {
        check_bearer(metadata, self.token.as_deref())
    }
}

type ResponseStream = Pin<Box<dyn Stream<Item = Result<proto::WorkerToHost, Status>> + Send>>;

#[tonic::async_trait]
impl WorkerService for GrpcWorkerService {
    type CommunicationStream = ResponseStream;

    async fn communication(
        &self,
        request: Request<Streaming<proto::HostToWorker>>,
    ) -> Result<Response<Self::CommunicationStream>, Status> {
        self.authorize(request.metadata())?;
        let mut inbound = request.into_inner();
        let handler = Arc::clone(&self.handler);
        let (tx, rx) = mpsc::channel(64);

        // Spawn a task that reads from the inbound stream, calls the
        // handler, and forwards responses to the outbound channel.
        tokio::spawn(async move {
            while let Some(msg_result) = inbound.message().await.transpose() {
                match msg_result {
                    Ok(msg) => {
                        let response = handler(msg).await;
                        if let Some(resp) = response
                            && tx.send(Ok(resp)).await.is_err()
                        {
                            break; // receiver dropped
                        }
                    }
                    Err(status) => {
                        let _ = tx.send(Err(status)).await;
                        break;
                    }
                }
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream)))
    }

    async fn health_check(
        &self,
        request: Request<proto::HealthRequest>,
    ) -> Result<Response<proto::HealthResponse>, Status> {
        self.authorize(request.metadata())?;
        Ok(Response::new(proto::HealthResponse {
            healthy: true,
            message: "ok".into(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::worker_service_server::WorkerServiceServer;

    /// Simple echo handler for testing.
    fn echo_handler() -> MessageHandler {
        Arc::new(|msg| {
            Box::pin(async move {
                // Echo back any request as a simple progress message
                match msg.message {
                    Some(proto::host_to_worker::Message::Request(r)) => Some(proto::WorkerToHost {
                        message: Some(proto::worker_to_host::Message::Terminal(proto::Terminal {
                            protocol_version: r.protocol_version,
                            incarnation_id: r.incarnation_id,
                            request_id: r.request_id,
                            correlation_id: r.correlation_id,
                            outcome: Some(proto::TerminalOutcome {
                                outcome: Some(proto::terminal_outcome::Outcome::Success(
                                    proto::Success { output: r.payload },
                                )),
                            }),
                        })),
                    }),
                    _ => None,
                }
            })
        })
    }

    #[tokio::test]
    async fn server_starts_and_handles_health_check() {
        let service = GrpcWorkerService::new(echo_handler());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tonic::transport::Server::builder()
            .add_service(WorkerServiceServer::new(service))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener));

        tokio::spawn(server);

        let mut client = crate::proto::worker_service_client::WorkerServiceClient::connect(
            format!("http://{addr}"),
        )
        .await
        .unwrap();

        let resp = client
            .health_check(proto::HealthRequest {})
            .await
            .unwrap()
            .into_inner();

        assert!(resp.healthy);
        assert_eq!(resp.message, "ok");
    }
}
