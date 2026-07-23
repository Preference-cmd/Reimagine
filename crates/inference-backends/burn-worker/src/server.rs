use std::io::{Read, Write};

use reimagine_backend_worker_protocol::{
    BackendExecutionError, FrameCodec, MessageSender, ProtocolVersion, TerminalOutcome,
    WireMessage, WorkerIncarnationId, validate_message_direction,
};
use reimagine_inference_burn::BurnBackend;

use crate::mapping::{self, MappingResult, TokenGenerator};
use crate::shutdown;

/// Enter the request dispatch loop.
///
/// Returns on EOF, protocol error, or after a clean shutdown
/// handshake. The function writes only protocol messages to
/// `writer`; diagnostics go to stderr.
///
/// All incoming frames are validated against the incarnation ID
/// established during the handshake. Frames from a mismatched or
/// stale incarnation are rejected with a protocol error.
pub fn serve_loop(
    rt: &tokio::runtime::Runtime,
    backend: &BurnBackend,
    codec: &FrameCodec,
    reader: &mut impl Read,
    writer: &mut impl Write,
    protocol_version: ProtocolVersion,
    process_incarnation: &WorkerIncarnationId,
) {
    let tokens = TokenGenerator::new();

    loop {
        let message = match codec.read(reader) {
            Ok(msg) => msg,
            Err(e) => {
                eprintln!("serve loop: read error: {e}");
                break;
            }
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

                let result =
                    mapping::dispatch(rt, backend, &tokens, &frame.operation, &frame.payload);

                match result {
                    MappingResult::Success(output) => {
                        let terminal = reimagine_backend_worker_protocol::TerminalFrame {
                            protocol_version: frame.protocol_version,
                            incarnation_id: frame.incarnation_id,
                            request_id: frame.request_id,
                            correlation_id: frame.correlation_id,
                            outcome: TerminalOutcome::Success { output },
                        };
                        if let Err(e) = codec.write(writer, &WireMessage::Terminal(terminal)) {
                            eprintln!("serve loop: write error: {e}");
                            break;
                        }
                    }
                    MappingResult::BackendError(error) => {
                        let terminal = reimagine_backend_worker_protocol::TerminalFrame {
                            protocol_version: frame.protocol_version,
                            incarnation_id: frame.incarnation_id,
                            request_id: frame.request_id,
                            correlation_id: frame.correlation_id,
                            outcome: TerminalOutcome::BackendError { error },
                        };
                        if let Err(e) = codec.write(writer, &WireMessage::Terminal(terminal)) {
                            eprintln!("serve loop: write error: {e}");
                            break;
                        }
                    }
                    MappingResult::NotImplemented => {
                        let error = BackendExecutionError {
                            code: "not_implemented".to_string(),
                            message: format!(
                                "operation `{}` not implemented in MB03 worker",
                                frame.operation
                            ),
                            retryable: false,
                        };
                        let terminal = reimagine_backend_worker_protocol::TerminalFrame {
                            protocol_version: frame.protocol_version,
                            incarnation_id: frame.incarnation_id,
                            request_id: frame.request_id,
                            correlation_id: frame.correlation_id,
                            outcome: TerminalOutcome::BackendError { error },
                        };
                        if let Err(e) = codec.write(writer, &WireMessage::Terminal(terminal)) {
                            eprintln!("serve loop: write error: {e}");
                            break;
                        }
                    }
                }

                if let Err(e) = writer.flush() {
                    eprintln!("serve loop: flush error: {e}");
                    break;
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
                eprintln!(
                    "serve loop: cancel req={} (ack only, no-op in MB03)",
                    frame.request_id.0
                );
                let ack = reimagine_backend_worker_protocol::CancelAckFrame {
                    protocol_version: frame.protocol_version,
                    incarnation_id: frame.incarnation_id.clone(),
                    request_id: frame.request_id.clone(),
                    correlation_id: frame.correlation_id,
                    accepted: false,
                    already_terminal: false,
                };
                if let Err(e) = codec.write(writer, &WireMessage::CancelAck(ack)) {
                    eprintln!("serve loop: write error: {e}");
                    break;
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
            | WireMessage::Terminal(_) => {
                eprintln!(
                    "serve loop: unexpected worker-directed message: {:?}",
                    message.kind()
                );
            }
        }
    }
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
        | WireMessage::CancelAck(_) => None,
    }
}
