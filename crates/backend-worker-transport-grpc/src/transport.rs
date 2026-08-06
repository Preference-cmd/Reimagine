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
///
/// The send half is held as `Mutex<Option<_>>` so [`WorkerTransport::shutdown`]
/// can actually close the stream (dropping the sender ends the outbound
/// stream, which ends the worker's read loop and the HTTP/2 connection);
/// sends after shutdown fail with a closed-transport error.
pub struct GrpcTransport {
    sender: Mutex<Option<mpsc::Sender<proto::HostToWorker>>>,
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
            sender: Mutex::new(Some(sender)),
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
        let guard = self.sender.lock().await;
        let sender = guard
            .as_ref()
            .ok_or_else(|| TransportError::Io("transport is closed".to_owned()))?;
        sender
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
        // Dropping the mpsc sender ends the outbound stream; the
        // worker-side read loop observes the stream end and exits,
        // closing the HTTP/2 connection. The transport refuses further
        // sends (see [`Self::send`]).
        self.sender.lock().await.take();
        Ok(())
    }
}
