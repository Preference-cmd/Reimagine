use std::fmt;

use async_trait::async_trait;
use reimagine_backend_worker_protocol::{
    TransportDescription, TransportError, TransportKind, WorkerTransport,
};
use tokio::sync::{Mutex, mpsc};

use crate::proto;

/// gRPC bidirectional stream transport for worker communication.
///
/// Wraps a tonic bidirectional `Communication` stream. The host sends
/// `HostToWorker` messages via an mpsc channel and receives
/// `WorkerToHost` messages from a `tonic::Streaming` receiver.
pub struct GrpcTransport {
    sender: mpsc::Sender<proto::HostToWorker>,
    receiver: Mutex<tonic::Streaming<proto::WorkerToHost>>,
    endpoint: String,
}

impl GrpcTransport {
    /// The capacity of the internal send channel.
    const SEND_CHANNEL_CAPACITY: usize = 64;

    /// Create a new `GrpcTransport` from a tonic streaming pair.
    ///
    /// `sender` feeds `HostToWorker` messages into the outbound stream.
    /// `receiver` yields `WorkerToHost` messages from the inbound stream.
    /// `endpoint` is a human-readable description of the remote address.
    pub fn new(
        sender: mpsc::Sender<proto::HostToWorker>,
        receiver: tonic::Streaming<proto::WorkerToHost>,
        endpoint: String,
    ) -> Self {
        Self {
            sender,
            receiver: Mutex::new(receiver),
            endpoint,
        }
    }

    /// Return a new mpsc sender pair whose receiver is suitable for
    /// passing to `tonic::IntoStreamingRequest`.
    pub fn channel() -> (
        mpsc::Sender<proto::HostToWorker>,
        mpsc::Receiver<proto::HostToWorker>,
    ) {
        mpsc::channel(Self::SEND_CHANNEL_CAPACITY)
    }

    /// Send a `HostToWorker` message to the worker.
    pub async fn send(&self, msg: proto::HostToWorker) -> Result<(), TransportError> {
        self.sender
            .send(msg)
            .await
            .map_err(|e| TransportError::Io(format!("send failed: {e}")))
    }

    /// Receive the next `WorkerToHost` message from the worker.
    ///
    /// Returns `Ok(None)` when the stream is closed.
    pub async fn recv(&self) -> Result<Option<proto::WorkerToHost>, TransportError> {
        let mut guard = self.receiver.lock().await;
        guard
            .message()
            .await
            .map_err(|e| TransportError::Io(format!("recv failed: {e}")))
    }
}

impl fmt::Debug for GrpcTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GrpcTransport")
            .field("endpoint", &self.endpoint)
            .finish()
    }
}

#[async_trait]
impl WorkerTransport for GrpcTransport {
    fn description(&self) -> TransportDescription {
        TransportDescription {
            kind: TransportKind::Grpc,
            endpoint: self.endpoint.clone(),
        }
    }

    async fn shutdown(&self) -> Result<(), TransportError> {
        // Dropping the sender signals the server that no more messages
        // are coming, which will cause the server's read loop to end.
        // We can't drop the sender here since we only have &self, so
        // we close the channel by cloning and dropping the clone.
        // Actually, mpsc::Sender::close is not available, so we
        // just return Ok. The caller should drop the transport.
        Ok(())
    }
}
