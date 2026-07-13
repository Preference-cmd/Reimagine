use std::collections::HashMap;
use std::io::{self, Write};
use std::process;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reimagine_backend_worker_protocol::{
    BackendExecutionError, BackendInstanceId, CancelAckFrame, CleanupAckFrame, FrameCodec,
    HealthAckFrame, ProgressFrame, ProtocolRange, RequestFrame, RequestId, ShutdownAckFrame,
    TerminalFrame, TerminalOutcome, WireMessage, WorkerHello, WorkerIdentity, WorkerIncarnationId,
    WorkerInstallationId, WorkerInstanceProfile, WorkerProfile, negotiate_protocol,
};
use serde_json::json;

const MAXIMUM_FRAME_BYTES: u32 = 1024 * 1024;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("host fake worker fixture started");
    let codec = FrameCodec::new(MAXIMUM_FRAME_BYTES);
    let stdout = Arc::new(Mutex::new(io::stdout()));
    let cancellations = Arc::new(Mutex::new(HashMap::<RequestId, Arc<AtomicBool>>::new()));
    let mut stdin = io::stdin().lock();
    let WireMessage::HostHello(host) = codec.read(&mut stdin)? else {
        return Err("first frame must be host_hello".into());
    };
    let mode = std::env::var("FAKE_WORKER_MODE").unwrap_or_default();
    if mode == "hang" {
        thread::sleep(Duration::from_secs(5));
    }
    if mode == "malformed_hello" {
        let mut stdout = io::stdout().lock();
        stdout.write_all(&1_u32.to_be_bytes())?;
        stdout.write_all(b"{")?;
        stdout.flush()?;
        return Ok(());
    }
    if mode == "partial_hello" {
        let mut stdout = io::stdout().lock();
        stdout.write_all(&20_u32.to_be_bytes())?;
        stdout.write_all(b"{")?;
        stdout.flush()?;
        return Ok(());
    }
    if mode == "clean_eof_hello" {
        return Ok(());
    }
    if mode == "partial_prefix_hello" {
        let mut stdout = io::stdout().lock();
        stdout.write_all(&20_u32.to_be_bytes()[..2])?;
        stdout.flush()?;
        return Ok(());
    }
    if mode == "stderr_flood" {
        let mut stderr = io::stderr().lock();
        stderr.write_all(&vec![b'x'; 256 * 1024])?;
        stderr.flush()?;
    }
    let selected = negotiate_protocol(host.supported_protocols, ProtocolRange::new(1, 1))?;
    let incarnation = WorkerIncarnationId(format!(
        "fixture-{}-{}",
        process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    let mut capabilities = vec!["echo".to_owned(), "delay".to_owned(), "progress".to_owned()];
    if mode != "profile_without_latent" {
        capabilities.push("latent.create_empty".to_owned());
    }
    write_message(
        &stdout,
        &WireMessage::WorkerHello(WorkerHello {
            selected_protocol: selected,
            identity: WorkerIdentity {
                backend_instance_id: BackendInstanceId::from("fake:cpu:default"),
                installation_id: WorkerInstallationId::from("fake-installation"),
                incarnation_id: incarnation.clone(),
                worker_version: env!("CARGO_PKG_VERSION").to_owned(),
                backend_kind: "fake".to_owned(),
                target: std::env::consts::ARCH.to_owned(),
                manifest_digest: if mode == "wrong_identity" {
                    "wrong-manifest".to_owned()
                } else {
                    "test-manifest".to_owned()
                },
            },
            profile: WorkerProfile {
                instances: vec![WorkerInstanceProfile {
                    backend_instance_id: BackendInstanceId::from(if mode == "wrong_profile" {
                        "fake:cpu:other"
                    } else {
                        "fake:cpu:default"
                    }),
                    device_label: "cpu".to_owned(),
                    capabilities,
                    operation_options: json!({}),
                }],
            },
        }),
    )?;

    loop {
        match codec.read(&mut stdin) {
            Ok(WireMessage::Request(request)) => {
                validate_session(
                    request.protocol_version,
                    &request.incarnation_id,
                    selected,
                    &incarnation,
                )?;
                spawn_request(request, Arc::clone(&stdout), Arc::clone(&cancellations));
            }
            Ok(WireMessage::Cancel(frame)) => {
                validate_session(
                    frame.protocol_version,
                    &frame.incarnation_id,
                    selected,
                    &incarnation,
                )?;
                let cancellation = cancellations
                    .lock()
                    .unwrap()
                    .get(&frame.request_id)
                    .cloned();
                let accepted = cancellation.is_some();
                if let Some(cancellation) = cancellation {
                    cancellation.store(true, Ordering::Release);
                }
                write_message(
                    &stdout,
                    &WireMessage::CancelAck(CancelAckFrame {
                        protocol_version: selected,
                        incarnation_id: incarnation.clone(),
                        request_id: frame.request_id,
                        correlation_id: frame.correlation_id,
                        accepted,
                        already_terminal: !accepted,
                    }),
                )?;
            }
            Ok(WireMessage::Health(frame)) => {
                validate_session(
                    frame.protocol_version,
                    &frame.incarnation_id,
                    selected,
                    &incarnation,
                )?;
                write_message(
                    &stdout,
                    &WireMessage::HealthAck(HealthAckFrame {
                        protocol_version: selected,
                        incarnation_id: incarnation.clone(),
                        control_id: frame.control_id,
                        healthy: true,
                        message: None,
                    }),
                )?;
            }
            Ok(WireMessage::Cleanup(frame)) => {
                validate_session(
                    frame.protocol_version,
                    &frame.incarnation_id,
                    selected,
                    &incarnation,
                )?;
                write_message(
                    &stdout,
                    &WireMessage::CleanupAck(CleanupAckFrame {
                        protocol_version: selected,
                        incarnation_id: incarnation.clone(),
                        control_id: frame.control_id,
                        released_objects: frame.object_ids.len() as u64,
                    }),
                )?;
            }
            Ok(WireMessage::Shutdown(frame)) => {
                validate_session(
                    frame.protocol_version,
                    &frame.incarnation_id,
                    selected,
                    &incarnation,
                )?;
                if mode != "ignore_shutdown" {
                    write_message(
                        &stdout,
                        &WireMessage::ShutdownAck(ShutdownAckFrame {
                            protocol_version: selected,
                            incarnation_id: incarnation.clone(),
                            control_id: frame.control_id,
                        }),
                    )?;
                    break;
                }
            }
            Err(reimagine_backend_worker_protocol::CodecError::Io(error))
                if error.kind() == io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Ok(message) => return Err(format!("unexpected host frame `{}`", message.kind()).into()),
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn validate_session(
    actual_protocol: reimagine_backend_worker_protocol::ProtocolVersion,
    actual_incarnation: &WorkerIncarnationId,
    expected_protocol: reimagine_backend_worker_protocol::ProtocolVersion,
    expected_incarnation: &WorkerIncarnationId,
) -> Result<(), Box<dyn std::error::Error>> {
    if actual_protocol != expected_protocol || actual_incarnation != expected_incarnation {
        return Err("stale protocol session".into());
    }
    Ok(())
}

fn spawn_request(
    request: RequestFrame,
    stdout: Arc<Mutex<io::Stdout>>,
    cancellations: Arc<Mutex<HashMap<RequestId, Arc<AtomicBool>>>>,
) {
    let cancellation = Arc::new(AtomicBool::new(false));
    cancellations
        .lock()
        .unwrap()
        .insert(request.request_id.clone(), Arc::clone(&cancellation));
    thread::spawn(move || {
        if request.operation == "crash" {
            process::exit(17);
        }
        let outcome = match request.operation.as_str() {
            "echo" => TerminalOutcome::Success {
                output: request.payload.clone(),
            },
            "delay" => {
                let milliseconds = request
                    .payload
                    .get("milliseconds")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(25);
                let steps = (milliseconds / 5).max(1);
                for _ in 0..steps {
                    thread::sleep(Duration::from_millis(5));
                    if cancellation.load(Ordering::Acquire) {
                        break;
                    }
                }
                if cancellation.load(Ordering::Acquire) {
                    TerminalOutcome::Cancelled
                } else {
                    TerminalOutcome::Success {
                        output: json!({ "delayed_milliseconds": milliseconds }),
                    }
                }
            }
            "progress" => {
                for sequence in 1..=3 {
                    write_message(
                        &stdout,
                        &WireMessage::Progress(ProgressFrame {
                            protocol_version: request.protocol_version,
                            incarnation_id: request.incarnation_id.clone(),
                            request_id: request.request_id.clone(),
                            correlation_id: request.correlation_id.clone(),
                            sequence,
                            completed: sequence,
                            total: Some(3),
                            message: Some(format!("step {sequence}")),
                        }),
                    )
                    .unwrap();
                }
                TerminalOutcome::Success {
                    output: json!(null),
                }
            }
            "latent.create_empty" => TerminalOutcome::Success {
                output: json!({
                    "worker_token": format!("latent:{}", request.request_id.0),
                    "width": request.payload["width"],
                    "height": request.payload["height"],
                    "batch_size": request.payload["batch_size"]
                }),
            },
            operation => TerminalOutcome::BackendError {
                error: BackendExecutionError {
                    code: "unsupported_operation".to_owned(),
                    message: format!("unsupported fake operation `{operation}`"),
                    retryable: false,
                },
            },
        };
        cancellations.lock().unwrap().remove(&request.request_id);
        write_message(
            &stdout,
            &WireMessage::Terminal(TerminalFrame {
                protocol_version: request.protocol_version,
                incarnation_id: request.incarnation_id,
                request_id: request.request_id,
                correlation_id: request.correlation_id,
                outcome,
            }),
        )
        .unwrap();
    });
}

fn write_message(
    stdout: &Arc<Mutex<io::Stdout>>,
    message: &WireMessage,
) -> Result<(), reimagine_backend_worker_protocol::CodecError> {
    FrameCodec::new(MAXIMUM_FRAME_BYTES).write(&mut *stdout.lock().unwrap(), message)
}
