use tonic::transport::Channel;

use crate::proto::worker_service_client::WorkerServiceClient;
use crate::transport::GrpcTransport;

/// Connect to a remote gRPC worker and perform the initial handshake.
///
/// Returns a `GrpcTransport` ready for protocol communication.
///
/// `endpoint` should be a URI like `http://127.0.0.1:50051`.
pub async fn connect(endpoint: &str) -> Result<GrpcTransport, tonic::Status> {
    let channel = Channel::from_shared(endpoint.to_owned())
        .map_err(|e| tonic::Status::internal(format!("invalid endpoint: {e}")))?
        .connect()
        .await
        .map_err(|e| tonic::Status::internal(format!("connect failed: {e}")))?;

    let mut client = WorkerServiceClient::new(channel);

    let (tx, rx) = GrpcTransport::channel();
    let rx_stream = tokio_stream::wrappers::ReceiverStream::new(rx);

    let response = client.communication(rx_stream).await?;
    let stream = response.into_inner();

    Ok(GrpcTransport::new(tx, stream, endpoint.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto;
    use crate::proto::worker_service_server::WorkerServiceServer;
    use crate::server::{GrpcWorkerService, MessageHandler};
    use reimagine_backend_worker_protocol::WorkerTransport;
    use std::sync::Arc;

    fn echo_handler() -> MessageHandler {
        Arc::new(|msg| {
            Box::pin(async move {
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
    async fn client_connects_and_performs_handshake() {
        let service = GrpcWorkerService::new(echo_handler());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tonic::transport::Server::builder()
            .add_service(WorkerServiceServer::new(service))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener));

        tokio::spawn(server);

        let transport = connect(&format!("http://{addr}")).await.unwrap();

        assert_eq!(
            transport.description().kind,
            reimagine_backend_worker_protocol::TransportKind::Grpc
        );

        // Send a request and receive a terminal response
        transport
            .send(proto::HostToWorker {
                message: Some(proto::host_to_worker::Message::Request(proto::Request {
                    protocol_version: 1,
                    incarnation_id: "inc-1".into(),
                    request_id: "req-1".into(),
                    correlation_id: "corr-1".into(),
                    operation: "echo".into(),
                    payload: serde_json::to_vec(&serde_json::json!({"hello": "world"})).unwrap(),
                })),
            })
            .await
            .unwrap();

        let response = transport.recv().await.unwrap().unwrap();
        assert!(matches!(
            response.message,
            Some(proto::worker_to_host::Message::Terminal(_))
        ));
    }
}
