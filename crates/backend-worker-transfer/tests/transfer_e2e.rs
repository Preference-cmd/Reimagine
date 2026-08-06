//! End-to-end tensor transfer between QUIC workers (T17).
//!
//! Two in-process workers talk over QUIC: worker A (source) plans a
//! transfer via `TransferPlanner`, executes it through `TransferExecutor`,
//! and worker B (target) stores and echoes the bytes. The executor's
//! QUIC channel speaks the T14 tensor protocol frames
//! (`TensorTransferRequest` → `TensorTransferAck(Accepted)` →
//! `TensorData`* → `TensorTransferAck(Complete)`).

use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use reimagine_backend_worker_protocol::transport::TransportKind;
use reimagine_backend_worker_protocol::{
    CorrelationId, HostHello, ProtocolRange, TensorDataFrame, TensorMetadata,
    TensorTransferAckFrame, TensorTransferRequestFrame, TransferStatus, WireMessage,
};
use reimagine_backend_worker_transfer::executor::{
    TransferChannel, TransferExecutor, seal_envelope,
};
use reimagine_backend_worker_transfer::{
    ConfigurableCostModel, TransferPlan, TransferPlanner, WorkerSpec,
};
use reimagine_backend_worker_transport_quic::QuicTransport;
use reimagine_backend_worker_transport_quic::listener::QuicWorkerListener;
use reimagine_backend_worker_transport_quic::tls::SelfSignedCert;

/// Test tensor size: a 64 KB latent-like payload.
const PAYLOAD_SIZE: usize = 64 * 1024;
/// Per-transfer `TensorData` chunk size on the wire.
const DATA_CHUNK_SIZE: usize = 256 * 1024;

static NEXT_CORRELATION: AtomicU64 = AtomicU64::new(0);

fn next_correlation_id() -> CorrelationId {
    CorrelationId(format!(
        "t17-{}",
        NEXT_CORRELATION.fetch_add(1, Ordering::Relaxed)
    ))
}

fn patterned(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i.wrapping_mul(31) % 251) as u8).collect()
}

fn tensor_metadata(size_bytes: u64) -> TensorMetadata {
    TensorMetadata {
        dtype: "f32".to_owned(),
        shape: vec![1, 4, 64, 64],
        size_bytes,
        backend_format: "burn::nchw".to_owned(),
    }
}

async fn read_wire_message(recv: &mut quinn::RecvStream) -> Result<WireMessage, String> {
    let mut prefix = [0u8; 4];
    recv.read_exact(&mut prefix)
        .await
        .map_err(|e| format!("read prefix: {e}"))?;
    let len = u32::from_be_bytes(prefix) as usize;
    let mut payload = vec![0u8; len];
    recv.read_exact(&mut payload)
        .await
        .map_err(|e| format!("read payload: {e}"))?;
    serde_json::from_slice(&payload).map_err(|e| format!("deserialize: {e}"))
}

async fn write_wire_message(
    send: &mut quinn::SendStream,
    message: &WireMessage,
) -> Result<(), String> {
    let json = serde_json::to_vec(message).map_err(|e| format!("serialize: {e}"))?;
    send.write_all(&(json.len() as u32).to_be_bytes())
        .await
        .map_err(|e| format!("write length: {e}"))?;
    send.write_all(&json)
        .await
        .map_err(|e| format!("write payload: {e}"))?;
    Ok(())
}

/// QUIC-backed [`TransferChannel`] speaking the T14 tensor protocol.
///
/// Holds the bi-stream that the listener's handshake completed on
/// (`accept()` reads the transfer frames on that same stream). `send`
/// negotiates the transfer (request → ack → chunked data frames →
/// complete ack) and parks the stream; `recv` reads the echo data frame
/// the target writes after the complete ack.
struct QuicChannel {
    target_id: String,
    key: String,
    metadata: TensorMetadata,
    stream: Mutex<Option<(quinn::SendStream, quinn::RecvStream)>>,
}

#[async_trait]
impl TransferChannel for QuicChannel {
    fn target_id(&self) -> &str {
        &self.target_id
    }

    async fn send(&self, bytes: &[u8]) -> Result<(), String> {
        let (mut send, mut recv) = self
            .stream
            .lock()
            .map_err(|_| "stream lock poisoned".to_owned())?
            .take()
            .ok_or_else(|| "no stream available: call send() once".to_owned())?;
        let correlation_id = next_correlation_id();
        write_wire_message(
            &mut send,
            &WireMessage::TensorTransferRequest(TensorTransferRequestFrame {
                source_token: self.key.clone(),
                target_worker_id: self.target_id.clone(),
                tensor_metadata: self.metadata.clone(),
            }),
        )
        .await?;
        let status = match read_wire_message(&mut recv).await? {
            WireMessage::TensorTransferAck(ack) => ack.status,
            other => {
                return Err(format!(
                    "expected TensorTransferAck, got {:?}",
                    other.kind()
                ));
            }
        };
        if !matches!(status, TransferStatus::Accepted) {
            return Err(format!("transfer rejected by target: {status:?}"));
        }
        let mut sequence = 0u64;
        let mut remaining = bytes;
        loop {
            let (chunk, is_final) = if remaining.len() <= DATA_CHUNK_SIZE {
                (remaining, true)
            } else {
                (&remaining[..DATA_CHUNK_SIZE], false)
            };
            write_wire_message(
                &mut send,
                &WireMessage::TensorData(TensorDataFrame {
                    correlation_id: correlation_id.clone(),
                    sequence,
                    data: chunk.to_vec(),
                    is_final,
                }),
            )
            .await?;
            sequence += 1;
            if is_final {
                break;
            }
            remaining = &remaining[DATA_CHUNK_SIZE..];
        }
        match read_wire_message(&mut recv).await? {
            WireMessage::TensorTransferAck(ack) if ack.status == TransferStatus::Complete => {}
            WireMessage::TensorTransferAck(ack) => {
                return Err(format!("expected Complete ack, got {:?}", ack.status));
            }
            other => {
                return Err(format!(
                    "expected TensorTransferAck, got {:?}",
                    other.kind()
                ));
            }
        }
        let mut stream = self
            .stream
            .lock()
            .map_err(|_| "stream lock poisoned".to_owned())?;
        *stream = Some((send, recv));
        Ok(())
    }

    async fn recv(&self) -> Result<Vec<u8>, String> {
        let (_send, mut recv) = self
            .stream
            .lock()
            .map_err(|_| "stream lock poisoned".to_owned())?
            .take()
            .ok_or_else(|| "no stream available: call send() first".to_owned())?;
        match read_wire_message(&mut recv).await? {
            WireMessage::TensorData(data) if data.is_final => Ok(data.data),
            WireMessage::TensorData(data) => Err(format!(
                "expected final echo frame, got sequence {}",
                data.sequence
            )),
            other => Err(format!("expected echo TensorData, got {:?}", other.kind())),
        }
    }
}

/// In-memory channel: stores what `send` pushes, returns it on `recv`.
/// Used by the IPC-plan test and the benchmark (no transport noise).
struct MemoryChannel {
    target_id: String,
    stored: Mutex<Vec<u8>>,
}

#[async_trait]
impl TransferChannel for MemoryChannel {
    fn target_id(&self) -> &str {
        &self.target_id
    }

    async fn send(&self, bytes: &[u8]) -> Result<(), String> {
        let mut stored = self
            .stored
            .lock()
            .map_err(|_| "stored lock poisoned".to_owned())?;
        *stored = bytes.to_vec();
        Ok(())
    }

    async fn recv(&self) -> Result<Vec<u8>, String> {
        let stored = self
            .stored
            .lock()
            .map_err(|_| "stored lock poisoned".to_owned())?;
        Ok(stored.clone())
    }
}

#[tokio::test]
async fn transfer_tensor_over_quic_between_workers() {
    let cert = SelfSignedCert::generate("localhost").unwrap();
    let listener =
        QuicWorkerListener::start(SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0), &cert).unwrap();
    let listen_addr = listener.local_addr().unwrap();

    // Worker B ("worker-b"): accept one connection, serve one transfer,
    // store the bytes, ack Complete, and echo them back for the source's
    // integrity verification.
    let stored: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    let server_stored = Arc::clone(&stored);
    let server = tokio::spawn(async move {
        let (connection, mut send, mut recv, hello) =
            listener.accept().await.map_err(|e| e.to_string())?;
        assert_eq!(hello.identity.backend_kind, "fake");

        let (key, metadata) = match read_wire_message(&mut recv).await? {
            WireMessage::TensorTransferRequest(req) => (req.source_token, req.tensor_metadata),
            other => {
                return Err(format!(
                    "expected TensorTransferRequest, got {:?}",
                    other.kind()
                ));
            }
        };
        assert_eq!(metadata.size_bytes, PAYLOAD_SIZE as u64);

        write_wire_message(
            &mut send,
            &WireMessage::TensorTransferAck(TensorTransferAckFrame {
                correlation_id: next_correlation_id(),
                status: TransferStatus::Accepted,
                target_token: Some(format!("tok-{key}")),
            }),
        )
        .await?;

        let mut expected_correlation: Option<CorrelationId> = None;
        let mut received: Vec<u8> = Vec::with_capacity(metadata.size_bytes as usize);
        loop {
            match read_wire_message(&mut recv).await? {
                WireMessage::TensorData(data) => {
                    let expected =
                        expected_correlation.get_or_insert_with(|| data.correlation_id.clone());
                    if &data.correlation_id != expected {
                        return Err("correlation id mismatch".to_owned());
                    }
                    received.extend_from_slice(&data.data);
                    if data.is_final {
                        break;
                    }
                }
                other => {
                    return Err(format!("expected TensorData, got {:?}", other.kind()));
                }
            }
        }
        let correlation_id = expected_correlation.expect("at least one data frame");
        *server_stored
            .lock()
            .map_err(|_| "stored lock poisoned".to_owned())? = received.clone();

        write_wire_message(
            &mut send,
            &WireMessage::TensorTransferAck(TensorTransferAckFrame {
                correlation_id: correlation_id.clone(),
                status: TransferStatus::Complete,
                target_token: None,
            }),
        )
        .await?;
        write_wire_message(
            &mut send,
            &WireMessage::TensorData(TensorDataFrame {
                correlation_id,
                sequence: 0,
                data: received,
                is_final: true,
            }),
        )
        .await?;

        // Keep the connection alive until the source closes it.
        let _ = connection.closed().await;
        Ok::<(), String>(())
    });

    // Worker A ("worker-a"): connect, complete the host handshake on
    // the first bi-stream (the listener's `accept()` returns this same
    // stream pair), plan the transfer, execute it.
    let client_addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0);
    let transport = QuicTransport::connect(client_addr, listen_addr, "localhost", &cert)
        .await
        .unwrap();
    let (mut hello_send, mut hello_recv) = transport.open_bi().await.unwrap();
    write_wire_message(
        &mut hello_send,
        &WireMessage::HostHello(HostHello {
            supported_protocols: ProtocolRange::new(1, 1),
        }),
    )
    .await
    .unwrap();
    match read_wire_message(&mut hello_recv).await.unwrap() {
        WireMessage::WorkerHello(hello) => {
            assert_eq!(hello.identity.backend_kind, "fake");
        }
        other => panic!("expected WorkerHello, got {:?}", other.kind()),
    }

    let channel: Arc<dyn TransferChannel> = Arc::new(QuicChannel {
        target_id: "worker-b".to_owned(),
        key: "latent-1".to_owned(),
        metadata: tensor_metadata(PAYLOAD_SIZE as u64),
        stream: Mutex::new(Some((hello_send, hello_recv))),
    });
    let executor = TransferExecutor::new(HashMap::from([("worker-b".to_owned(), channel)]));
    let planner = TransferPlanner::new(
        vec![
            WorkerSpec {
                id: "worker-a".to_owned(),
                transport_kind: TransportKind::Quic,
            },
            WorkerSpec {
                id: "worker-b".to_owned(),
                transport_kind: TransportKind::Quic,
            },
        ],
        ConfigurableCostModel::default(),
    );

    let payload = patterned(PAYLOAD_SIZE);
    let plan = planner.plan_transfer(
        "worker-a",
        "worker-b",
        &tensor_metadata(PAYLOAD_SIZE as u64),
    );
    match &plan {
        TransferPlan::Network {
            target,
            estimated_bytes,
            ..
        } => {
            assert_eq!(target, "worker-b");
            assert_eq!(*estimated_bytes, PAYLOAD_SIZE as u64);
        }
        other => panic!("expected Network plan, got {other:?}"),
    }

    let moved = executor.execute(&plan, &payload).await.unwrap();
    assert_eq!(moved, payload, "source-side integrity verification failed");

    // Dropping the executor closes the QUIC connection, letting the
    // server task finish.
    drop(executor);
    server.await.unwrap().unwrap();

    let target_received = stored.lock().unwrap();
    assert_eq!(
        &*target_received,
        &seal_envelope(&payload),
        "target worker received corrupted bytes"
    );
}

#[tokio::test]
async fn local_plan_and_unknown_worker() {
    let planner = TransferPlanner::new(
        vec![WorkerSpec {
            id: "a".to_owned(),
            transport_kind: TransportKind::Quic,
        }],
        ConfigurableCostModel::default(),
    );
    let executor = TransferExecutor::new(HashMap::new());
    let payload = patterned(64);

    let local = planner.plan_transfer("a", "a", &tensor_metadata(64));
    let moved = executor.execute(&local, &payload).await.unwrap();
    assert_eq!(moved, payload);

    let unknown = planner.plan_transfer("a", "nope", &tensor_metadata(64));
    match executor.execute(&unknown, &payload).await {
        Err(reason) => assert!(reason.contains("nope"), "{reason}"),
        Ok(_) => panic!("expected an error for an unknown target worker"),
    }
}

#[tokio::test]
async fn ipc_plan_moves_bytes_through_in_memory_channel() {
    let planner = TransferPlanner::new(
        vec![
            WorkerSpec {
                id: "a".to_owned(),
                transport_kind: TransportKind::Stdio,
            },
            WorkerSpec {
                id: "b".to_owned(),
                transport_kind: TransportKind::Stdio,
            },
        ],
        ConfigurableCostModel::default(),
    );
    let channel = Arc::new(MemoryChannel {
        target_id: "b".to_owned(),
        stored: Mutex::new(Vec::new()),
    });
    let executor = TransferExecutor::new(HashMap::from([(
        "b".to_owned(),
        channel.clone() as Arc<dyn TransferChannel>,
    )]));

    let payload = patterned(4096);
    let plan = planner.plan_transfer("a", "b", &tensor_metadata(4096));
    assert!(matches!(plan, TransferPlan::Ipc { .. }), "{plan:?}");

    let moved = executor.execute(&plan, &payload).await.unwrap();
    assert_eq!(moved, payload);
    let stored = channel.stored.lock().unwrap();
    assert_eq!(&*stored, &seal_envelope(&payload));
}

/// Transfer-overhead benchmark; run with `cargo test -- --ignored`.
///
/// Uses an in-memory channel so the numbers measure executor overhead
/// (envelope seal + integrity verification) with no transport noise.
/// Roadmap acceptance sizes: 128 KB latent, 3 MB image.
#[ignore]
#[tokio::test]
async fn transfer_overhead_benchmark() {
    let cases: [(&str, usize, u32); 2] = [
        ("128 KB latent", 128 * 1024, 256),
        ("3 MB image", 3 * 1024 * 1024, 32),
    ];
    for (label, size, iterations) in cases {
        let payload = patterned(size);
        let channel = Arc::new(MemoryChannel {
            target_id: "b".to_owned(),
            stored: Mutex::new(Vec::new()),
        });
        let executor = TransferExecutor::new(HashMap::from([(
            "b".to_owned(),
            channel.clone() as Arc<dyn TransferChannel>,
        )]));
        let plan = TransferPlan::Network {
            key: "a:b".to_owned(),
            target: "b".to_owned(),
            estimated_bytes: size as u64,
        };

        executor.execute(&plan, &payload).await.unwrap();
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let moved = executor.execute(&plan, &payload).await.unwrap();
            assert_eq!(moved, payload);
        }
        let elapsed = start.elapsed();
        println!(
            "[bench] {label}: {iterations} iters, {:.3} ms/transfer \
             (in-memory channel, executor overhead only)",
            elapsed.as_secs_f64() * 1000.0 / f64::from(iterations)
        );
    }
}
