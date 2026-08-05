use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::{Arc, Mutex, mpsc};

use reimagine_backend_worker_protocol::{
    BackendExecutionError, CorrelationId, FrameCodec, MessageSender, ProtocolVersion, RequestId,
    TerminalOutcome, WireMessage, WorkerIncarnationId, validate_message_direction,
};
use reimagine_inference_burn::{BurnBackend, with_request_cancellation};
use tokio_util::sync::CancellationToken;

use crate::mapping::{self, MappingResult, TokenGenerator};
use crate::shutdown;

/// Event multiplexed from the reader thread and the dispatch threads
/// onto the single serve-loop channel.
enum ServeEvent {
    Frame(WireMessage),
    Outcome(DispatchOutcome),
    TransportClosed(String),
}

/// Terminal-relevant metadata for a finished request.
struct DispatchOutcome {
    request_id: RequestId,
    correlation_id: CorrelationId,
    protocol_version: ProtocolVersion,
    incarnation_id: WorkerIncarnationId,
    result: MappingResult,
}

/// Enter the request dispatch loop.
///
/// Returns on EOF, protocol error, or after a clean shutdown
/// handshake. The function writes only protocol messages to
/// `writer`; diagnostics go to stderr.
///
/// A dedicated reader thread owns `reader` and keeps forwarding
/// incoming frames while requests run; each request is dispatched on
/// its own thread with a per-request cancellation token registered
/// in `in_flight`, so `Cancel` frames are processed while an
/// operation is still executing. The serve loop is the only writer
/// to `writer`, preserving frame ordering.
///
/// All incoming frames are validated against the incarnation ID
/// established during the handshake. Frames from a mismatched or
/// stale incarnation are rejected with a protocol error.
pub fn serve_loop(
    rt: tokio::runtime::Runtime,
    backend: &BurnBackend,
    codec: FrameCodec,
    reader: impl Read + Send + 'static,
    writer: &mut impl Write,
    protocol_version: ProtocolVersion,
    process_incarnation: &WorkerIncarnationId,
) {
    let tokens = Arc::new(TokenGenerator::new());
    let rt = Arc::new(rt);
    let (event_tx, event_rx) = mpsc::channel::<ServeEvent>();

    let reader_tx = event_tx.clone();
    let _reader_thread = match std::thread::Builder::new()
        .name("worker-reader".to_owned())
        .spawn(move || {
            let mut reader = reader;
            loop {
                match codec.read(&mut reader) {
                    Ok(msg) => {
                        if reader_tx.send(ServeEvent::Frame(msg)).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("serve loop: read error: {e}");
                        let _ = reader_tx.send(ServeEvent::TransportClosed(e.to_string()));
                        break;
                    }
                }
            }
        }) {
        Ok(thread) => thread,
        Err(e) => {
            eprintln!("serve loop: failed to spawn reader thread: {e}");
            return;
        }
    };

    let in_flight: Arc<Mutex<HashMap<RequestId, Arc<CancellationToken>>>> =
        Arc::new(Mutex::new(HashMap::new()));

    loop {
        let message = match event_rx.recv() {
            Ok(ServeEvent::Frame(msg)) => msg,
            Ok(ServeEvent::TransportClosed(reason)) => {
                eprintln!("serve loop: transport closed: {reason}");
                break;
            }
            Ok(ServeEvent::Outcome(outcome)) => {
                in_flight
                    .lock()
                    .expect("in-flight registry poisoned")
                    .remove(&outcome.request_id);
                if !write_terminal(writer, &codec, outcome) {
                    break;
                }
                continue;
            }
            Err(_) => break,
        };

        if validate_message_direction(&message, MessageSender::Host).is_err() {
            eprintln!("serve loop: host sent wrong-direction `{}`", message.kind());
            break;
        }
        if let Some((frame_protocol, frame_incarnation)) = extract_session(&message) {
            if frame_protocol != protocol_version || frame_incarnation != process_incarnation {
                eprintln!("serve loop: session mismatch for `{}`", message.kind());
                break;
            }
        } else if !matches!(message, WireMessage::Ping { .. }) {
            eprintln!("serve loop: unexpected post-handshake `{}`", message.kind());
            break;
        }

        match message {
            WireMessage::Request(frame) => {
                eprintln!(
                    "serve loop: request {} (op={}, req={})",
                    frame.correlation_id.0, frame.operation, frame.request_id.0
                );

                let token = Arc::new(CancellationToken::new());
                in_flight
                    .lock()
                    .expect("in-flight registry poisoned")
                    .insert(frame.request_id.clone(), token.clone());

                let backend = backend.clone();
                let tokens = Arc::clone(&tokens);
                let rt = Arc::clone(&rt);
                let outcome_tx = event_tx.clone();
                let request_id = frame.request_id.clone();
                let correlation_id = frame.correlation_id.clone();
                let incarnation_id = frame.incarnation_id.clone();
                let operation = frame.operation.clone();
                let payload = frame.payload.clone();
                let protocol_version = frame.protocol_version;

                let spawned = std::thread::Builder::new()
                    .name(format!("dispatch-{}", request_id.0))
                    .spawn(move || {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            with_request_cancellation(token, || {
                                mapping::dispatch(&rt, &backend, &tokens, &operation, &payload)
                            })
                        }))
                        .unwrap_or_else(|_| {
                            MappingResult::BackendError(BackendExecutionError {
                                code: "dispatch_panic".to_string(),
                                message: format!("operation `{operation}` panicked"),
                                retryable: false,
                            })
                        });
                        let _ = outcome_tx.send(ServeEvent::Outcome(DispatchOutcome {
                            request_id,
                            correlation_id,
                            protocol_version,
                            incarnation_id,
                            result,
                        }));
                    });

                if let Err(e) = spawned {
                    eprintln!(
                        "serve loop: failed to spawn dispatch for req {}: {e}",
                        frame.request_id.0
                    );
                    in_flight
                        .lock()
                        .expect("in-flight registry poisoned")
                        .remove(&frame.request_id);
                    let outcome = DispatchOutcome {
                        request_id: frame.request_id.clone(),
                        correlation_id: frame.correlation_id.clone(),
                        protocol_version: frame.protocol_version,
                        incarnation_id: frame.incarnation_id.clone(),
                        result: MappingResult::BackendError(BackendExecutionError {
                            code: "dispatch_failed".to_string(),
                            message: e.to_string(),
                            retryable: false,
                        }),
                    };
                    if !write_terminal(writer, &codec, outcome) {
                        break;
                    }
                }
            }

            WireMessage::Progress(frame) => {
                // Worker-initiated progress is sent by the dispatch
                // handler; incoming progress from the host is ignored.
                eprintln!(
                    "serve loop: unexpected Progress from host (req={})",
                    frame.request_id.0
                );
            }

            WireMessage::Cancel(frame) => {
                let token = in_flight
                    .lock()
                    .expect("in-flight registry poisoned")
                    .remove(&frame.request_id);
                match token {
                    Some(token) => {
                        token.cancel();
                        eprintln!("serve loop: cancel req={} accepted", frame.request_id.0);
                        let ack = reimagine_backend_worker_protocol::CancelAckFrame {
                            protocol_version: frame.protocol_version,
                            incarnation_id: frame.incarnation_id.clone(),
                            request_id: frame.request_id.clone(),
                            correlation_id: frame.correlation_id,
                            accepted: true,
                            already_terminal: false,
                        };
                        if let Err(e) = codec.write(writer, &WireMessage::CancelAck(ack)) {
                            eprintln!("serve loop: write error: {e}");
                            break;
                        }
                    }
                    None => {
                        eprintln!(
                            "serve loop: cancel req={} already terminal",
                            frame.request_id.0
                        );
                        let ack = reimagine_backend_worker_protocol::CancelAckFrame {
                            protocol_version: frame.protocol_version,
                            incarnation_id: frame.incarnation_id.clone(),
                            request_id: frame.request_id.clone(),
                            correlation_id: frame.correlation_id,
                            accepted: false,
                            already_terminal: true,
                        };
                        if let Err(e) = codec.write(writer, &WireMessage::CancelAck(ack)) {
                            eprintln!("serve loop: write error: {e}");
                            break;
                        }
                    }
                }
            }

            WireMessage::Health(frame) => {
                let ack = reimagine_backend_worker_protocol::HealthAckFrame {
                    protocol_version: frame.protocol_version,
                    incarnation_id: frame.incarnation_id.clone(),
                    control_id: frame.control_id,
                    healthy: true,
                    message: None,
                };
                if let Err(e) = codec.write(writer, &WireMessage::HealthAck(ack)) {
                    eprintln!("serve loop: write error: {e}");
                    break;
                }
            }

            WireMessage::Cleanup(frame) => {
                let mut released = frame
                    .run_id
                    .as_deref()
                    .map(|run_id| {
                        backend
                            .store()
                            .cleanup_run(&reimagine_core::model::RunId::new(run_id))
                    })
                    .unwrap_or(0);
                for object_id in &frame.object_ids {
                    released +=
                        usize::from(backend.store().release_payload(
                            &reimagine_inference::BackendPayloadKey::new(object_id),
                        ));
                }
                let ack = reimagine_backend_worker_protocol::CleanupAckFrame {
                    protocol_version: frame.protocol_version,
                    incarnation_id: frame.incarnation_id.clone(),
                    control_id: frame.control_id,
                    released_objects: released as u64,
                };
                if let Err(e) = codec.write(writer, &WireMessage::CleanupAck(ack)) {
                    eprintln!("serve loop: write error: {e}");
                    break;
                }
            }

            WireMessage::Shutdown(frame) => {
                eprintln!("serve loop: shutdown request received");
                shutdown::cleanup(backend, &frame.incarnation_id);
                let ack = reimagine_backend_worker_protocol::ShutdownAckFrame {
                    protocol_version: frame.protocol_version,
                    incarnation_id: frame.incarnation_id.clone(),
                    control_id: frame.control_id,
                };
                if let Err(e) = codec.write(writer, &WireMessage::ShutdownAck(ack)) {
                    eprintln!("serve loop: write error: {e}");
                }
                let _ = writer.flush();
                break;
            }

            WireMessage::Ping { nonce } => {
                if let Err(e) = codec.write(writer, &WireMessage::Ping { nonce }) {
                    eprintln!("serve loop: write error: {e}");
                    break;
                }
            }

            WireMessage::HostHello(_) | WireMessage::WorkerHello(_) => {
                eprintln!("serve loop: unexpected hello message");
            }

            WireMessage::HealthAck(_)
            | WireMessage::CleanupAck(_)
            | WireMessage::ShutdownAck(_)
            | WireMessage::CancelAck(_)
            | WireMessage::Terminal(_)
            | WireMessage::TensorTransferRequest(_)
            | WireMessage::TensorTransferAck(_)
            | WireMessage::TensorData(_) => {
                eprintln!(
                    "serve loop: unexpected worker-directed message: {:?}",
                    message.kind()
                );
            }
        }
    }

    drop(event_tx);
}

/// Write the terminal frame for a finished request. Returns `false`
/// if the write failed and the serve loop must terminate.
fn write_terminal(writer: &mut impl Write, codec: &FrameCodec, outcome: DispatchOutcome) -> bool {
    let terminal = match outcome.result {
        MappingResult::Success(output) => reimagine_backend_worker_protocol::TerminalFrame {
            protocol_version: outcome.protocol_version,
            incarnation_id: outcome.incarnation_id,
            request_id: outcome.request_id,
            correlation_id: outcome.correlation_id,
            outcome: TerminalOutcome::Success { output },
        },
        MappingResult::Cancelled => reimagine_backend_worker_protocol::TerminalFrame {
            protocol_version: outcome.protocol_version,
            incarnation_id: outcome.incarnation_id,
            request_id: outcome.request_id,
            correlation_id: outcome.correlation_id,
            outcome: TerminalOutcome::Cancelled,
        },
        MappingResult::BackendError(error) => reimagine_backend_worker_protocol::TerminalFrame {
            protocol_version: outcome.protocol_version,
            incarnation_id: outcome.incarnation_id,
            request_id: outcome.request_id,
            correlation_id: outcome.correlation_id,
            outcome: TerminalOutcome::BackendError { error },
        },
        MappingResult::NotImplemented => {
            let error = BackendExecutionError {
                code: "not_implemented".to_string(),
                message: "operation not implemented in MB03 worker".to_string(),
                retryable: false,
            };
            reimagine_backend_worker_protocol::TerminalFrame {
                protocol_version: outcome.protocol_version,
                incarnation_id: outcome.incarnation_id,
                request_id: outcome.request_id,
                correlation_id: outcome.correlation_id,
                outcome: TerminalOutcome::BackendError { error },
            }
        }
    };
    if let Err(e) = codec.write(writer, &WireMessage::Terminal(terminal)) {
        eprintln!("serve loop: write error: {e}");
        return false;
    }
    if let Err(e) = writer.flush() {
        eprintln!("serve loop: flush error: {e}");
        return false;
    }
    true
}

/// Extract the incarnation ID from a wire message, if the message
/// type carries one.
fn extract_session(message: &WireMessage) -> Option<(ProtocolVersion, &WorkerIncarnationId)> {
    match message {
        WireMessage::Request(frame) => Some((frame.protocol_version, &frame.incarnation_id)),
        WireMessage::Progress(frame) => Some((frame.protocol_version, &frame.incarnation_id)),
        WireMessage::Cancel(frame) => Some((frame.protocol_version, &frame.incarnation_id)),
        WireMessage::Health(frame) => Some((frame.protocol_version, &frame.incarnation_id)),
        WireMessage::Cleanup(frame) => Some((frame.protocol_version, &frame.incarnation_id)),
        WireMessage::Shutdown(frame) => Some((frame.protocol_version, &frame.incarnation_id)),
        WireMessage::HostHello(_)
        | WireMessage::WorkerHello(_)
        | WireMessage::Ping { .. }
        | WireMessage::Terminal(_)
        | WireMessage::HealthAck(_)
        | WireMessage::CleanupAck(_)
        | WireMessage::ShutdownAck(_)
        | WireMessage::CancelAck(_)
        | WireMessage::TensorTransferRequest(_)
        | WireMessage::TensorTransferAck(_)
        | WireMessage::TensorData(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::io::{self, Read, Write};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Condvar, Mutex, mpsc};
    use std::time::Duration;

    use reimagine_backend_worker_protocol::{
        CancelFrame, ControlId, HealthFrame, RequestFrame, ShutdownFrame,
    };
    use reimagine_inference_burn::{BurnBackend, BurnBackendConfig};

    use super::*;

    /// In-process duplex pipe: the serve loop reads/writes its half,
    /// the test drives the other half. Blocking reads wake on bytes
    /// or close, so frame reads are deterministic.
    #[derive(Clone)]
    struct TestPipe {
        bytes: Arc<Mutex<VecDeque<u8>>>,
        closed: Arc<AtomicBool>,
        wake: Arc<Condvar>,
    }

    impl TestPipe {
        fn new() -> Self {
            Self {
                bytes: Arc::new(Mutex::new(VecDeque::new())),
                closed: Arc::new(AtomicBool::new(false)),
                wake: Arc::new(Condvar::new()),
            }
        }

        fn close(&self) {
            self.closed.store(true, Ordering::SeqCst);
            self.wake.notify_all();
        }
    }

    impl Read for TestPipe {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let mut bytes = self.bytes.lock().expect("pipe poisoned");
            loop {
                if !bytes.is_empty() {
                    let n = buf.len().min(bytes.len());
                    for byte in buf.iter_mut().take(n) {
                        *byte = bytes.pop_front().expect("non-empty deque");
                    }
                    return Ok(n);
                }
                if self.closed.load(Ordering::SeqCst) {
                    return Ok(0);
                }
                bytes = self.wake.wait(bytes).expect("pipe poisoned");
            }
        }
    }

    impl Write for TestPipe {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let mut bytes = self.bytes.lock().expect("pipe poisoned");
            bytes.extend(buf.iter().copied());
            self.wake.notify_all();
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn test_runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime")
    }

    fn test_backend() -> BurnBackend {
        BurnBackend::new(BurnBackendConfig::new(
            "/tmp/reimagine-burn-worker-test-models",
            "/tmp/reimagine-burn-worker-test-output",
        ))
        .expect("test backend")
    }

    const INCARNATION: &str = "test-incarnation";
    const CORRELATION: &str = "test-correlation";

    fn request_frame(request_id: &str, operation: &str, payload: serde_json::Value) -> WireMessage {
        WireMessage::Request(RequestFrame {
            protocol_version: ProtocolVersion(1),
            incarnation_id: INCARNATION.into(),
            request_id: RequestId(request_id.to_owned()),
            correlation_id: CORRELATION.into(),
            operation: operation.to_owned(),
            payload,
        })
    }

    fn cancel_frame(request_id: &str) -> WireMessage {
        WireMessage::Cancel(CancelFrame {
            protocol_version: ProtocolVersion(1),
            incarnation_id: INCARNATION.into(),
            request_id: RequestId(request_id.to_owned()),
            correlation_id: CORRELATION.into(),
        })
    }

    fn write_frame(pipe: &TestPipe, codec: &FrameCodec, message: &WireMessage) {
        let mut pipe = pipe.clone();
        codec.write(&mut pipe, message).expect("write frame");
    }

    fn read_frame(pipe: &TestPipe, codec: &FrameCodec) -> WireMessage {
        read_frame_with_timeout(pipe, codec, Duration::from_secs(10))
    }

    fn read_frame_with_timeout(
        pipe: &TestPipe,
        codec: &FrameCodec,
        timeout: Duration,
    ) -> WireMessage {
        let pipe = pipe.clone();
        let codec = *codec;
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut pipe = pipe;
            let result = codec.read(&mut pipe);
            let _ = tx.send(result);
        });
        match rx.recv_timeout(timeout) {
            Ok(Ok(message)) => message,
            Ok(Err(e)) => panic!("failed to decode frame: {e}"),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                panic!("timed out after {timeout:?} waiting for a frame")
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                panic!("frame reader thread disconnected")
            }
        }
    }

    /// Spawn a serve loop over in-process pipes and return the test
    /// ends plus the server join handle.
    fn spawn_server() -> (TestPipe, TestPipe, FrameCodec, std::thread::JoinHandle<()>) {
        let backend = test_backend();
        let codec = FrameCodec::new(64 * 1024 * 1024);
        let stdin = TestPipe::new();
        let stdout = TestPipe::new();
        let incarnation: WorkerIncarnationId = INCARNATION.into();
        let server_stdin = stdin.clone();
        let mut server_stdout = stdout.clone();
        let server = std::thread::spawn(move || {
            serve_loop(
                test_runtime(),
                &backend,
                codec,
                server_stdin,
                &mut server_stdout,
                ProtocolVersion(1),
                &incarnation,
            );
        });
        (stdin, stdout, codec, server)
    }

    fn shutdown_and_join(
        stdin: &TestPipe,
        stdout: &TestPipe,
        codec: &FrameCodec,
        server: std::thread::JoinHandle<()>,
    ) {
        write_frame(
            stdin,
            codec,
            &WireMessage::Shutdown(ShutdownFrame {
                protocol_version: ProtocolVersion(1),
                incarnation_id: INCARNATION.into(),
                control_id: ControlId("shutdown-1".into()),
            }),
        );
        match read_frame(stdout, codec) {
            WireMessage::ShutdownAck(_) => {}
            other => panic!("expected shutdown ack, got {:?}", other.kind()),
        }
        stdin.close();
        server.join().expect("serve loop thread");
    }

    #[test]
    fn cancel_of_in_flight_request_acks_accepted_and_terminates_cancelled() {
        let (stdin, stdout, codec, server) = spawn_server();

        // Start a long-running request; the dispatch thread registers
        // it in the in-flight registry before the spin loop begins.
        write_frame(
            &stdin,
            &codec,
            &request_frame(
                "req-1",
                "test.spin",
                serde_json::json!({ "duration_ms": 60000 }),
            ),
        );
        // The ping echo is only written after the request was
        // registered and dispatched, so the request is provably in
        // flight once it arrives.
        write_frame(&stdin, &codec, &WireMessage::Ping { nonce: "n1".into() });
        assert_eq!(
            read_frame(&stdout, &codec),
            WireMessage::Ping { nonce: "n1".into() }
        );

        // Cancel while the operation is still spinning.
        write_frame(&stdin, &codec, &cancel_frame("req-1"));
        match read_frame(&stdout, &codec) {
            WireMessage::CancelAck(frame) => {
                assert!(frame.accepted, "cancel must be accepted");
                assert!(!frame.already_terminal, "request was still in flight");
            }
            other => panic!("expected cancel ack, got {:?}", other.kind()),
        }

        // The cancelled request must terminate with Cancelled.
        match read_frame(&stdout, &codec) {
            WireMessage::Terminal(frame) => {
                assert_eq!(frame.outcome, TerminalOutcome::Cancelled);
                assert_eq!(frame.request_id, RequestId("req-1".into()));
            }
            other => panic!("expected terminal, got {:?}", other.kind()),
        }

        shutdown_and_join(&stdin, &stdout, &codec, server);
    }

    #[test]
    fn control_frames_are_served_while_request_is_in_flight() {
        let (stdin, stdout, codec, server) = spawn_server();

        write_frame(
            &stdin,
            &codec,
            &request_frame(
                "req-2",
                "test.spin",
                serde_json::json!({ "duration_ms": 60000 }),
            ),
        );
        write_frame(&stdin, &codec, &WireMessage::Ping { nonce: "n2".into() });
        assert_eq!(
            read_frame(&stdout, &codec),
            WireMessage::Ping { nonce: "n2".into() }
        );

        // Health is answered while the request is still running.
        write_frame(
            &stdin,
            &codec,
            &WireMessage::Health(HealthFrame {
                protocol_version: ProtocolVersion(1),
                incarnation_id: INCARNATION.into(),
                control_id: ControlId("health-1".into()),
            }),
        );
        match read_frame(&stdout, &codec) {
            WireMessage::HealthAck(frame) => {
                assert!(frame.healthy);
                assert_eq!(frame.control_id, ControlId("health-1".into()));
            }
            other => panic!("expected health ack, got {:?}", other.kind()),
        }

        write_frame(&stdin, &codec, &cancel_frame("req-2"));
        match read_frame(&stdout, &codec) {
            WireMessage::CancelAck(frame) => assert!(frame.accepted),
            other => panic!("expected cancel ack, got {:?}", other.kind()),
        }
        match read_frame(&stdout, &codec) {
            WireMessage::Terminal(frame) => {
                assert_eq!(frame.outcome, TerminalOutcome::Cancelled);
            }
            other => panic!("expected terminal, got {:?}", other.kind()),
        }

        shutdown_and_join(&stdin, &stdout, &codec, server);
    }

    #[test]
    fn cancel_after_completion_acks_already_terminal() {
        let (stdin, stdout, codec, server) = spawn_server();

        write_frame(
            &stdin,
            &codec,
            &request_frame(
                "req-3",
                "test.spin",
                serde_json::json!({ "duration_ms": 10 }),
            ),
        );
        match read_frame(&stdout, &codec) {
            WireMessage::Terminal(frame) => {
                assert!(matches!(frame.outcome, TerminalOutcome::Success { .. }));
            }
            other => panic!("expected terminal, got {:?}", other.kind()),
        }

        // The request already finished, so the cancel is a no-op ack.
        write_frame(&stdin, &codec, &cancel_frame("req-3"));
        match read_frame(&stdout, &codec) {
            WireMessage::CancelAck(frame) => {
                assert!(!frame.accepted);
                assert!(frame.already_terminal, "request already finished");
            }
            other => panic!("expected cancel ack, got {:?}", other.kind()),
        }

        shutdown_and_join(&stdin, &stdout, &codec, server);
    }

    #[test]
    fn unknown_operation_reports_backend_error_terminal() {
        let (stdin, stdout, codec, server) = spawn_server();

        write_frame(
            &stdin,
            &codec,
            &request_frame("req-4", "no.such_op", serde_json::json!({})),
        );
        match read_frame(&stdout, &codec) {
            WireMessage::Terminal(frame) => {
                assert!(matches!(
                    frame.outcome,
                    TerminalOutcome::BackendError { .. }
                ));
            }
            other => panic!("expected terminal, got {:?}", other.kind()),
        }

        shutdown_and_join(&stdin, &stdout, &codec, server);
    }
}
