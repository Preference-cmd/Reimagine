use std::collections::HashMap;
use std::io;
use std::process;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reimagine_backend_worker_protocol::{
    BackendExecutionError, BackendInstanceId, CancelAckFrame, FrameCodec, ProtocolRange,
    RequestFrame, RequestId, TerminalFrame, TerminalOutcome, WireMessage, WorkerHello,
    WorkerIdentity, WorkerIncarnationId, WorkerInstallationId, negotiate_protocol,
};
use serde_json::json;

const MAXIMUM_FRAME_BYTES: u32 = 1024 * 1024;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("fake backend worker fixture started");
    let codec = FrameCodec::new(MAXIMUM_FRAME_BYTES);
    let stdout = Arc::new(Mutex::new(io::stdout()));
    let cancellations = Arc::new(Mutex::new(HashMap::<RequestId, Arc<AtomicBool>>::new()));
    let mut stdin = io::stdin().lock();

    let WireMessage::HostHello(hello) = codec.read(&mut stdin)? else {
        return Err("first frame must be host_hello".into());
    };
    let selected = negotiate_protocol(hello.supported_protocols, ProtocolRange::new(1, 1))?;
    let incarnation = format!(
        "fake-{}-{}",
        process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    );
    write_message(
        &stdout,
        &WireMessage::WorkerHello(WorkerHello {
            selected_protocol: selected,
            identity: WorkerIdentity {
                backend_instance_id: BackendInstanceId::from("fake:cpu:default"),
                installation_id: WorkerInstallationId::from("fake-installation"),
                incarnation_id: WorkerIncarnationId(incarnation),
                worker_version: env!("CARGO_PKG_VERSION").to_owned(),
                backend_kind: "fake".to_owned(),
                target: std::env::consts::ARCH.to_owned(),
                manifest_digest: "test-manifest".to_owned(),
            },
        }),
    )?;

    loop {
        match codec.read(&mut stdin) {
            Ok(WireMessage::Request(request)) => {
                spawn_request(request, Arc::clone(&stdout), Arc::clone(&cancellations));
            }
            Ok(WireMessage::Cancel(cancel)) => {
                let cancellation = cancellations
                    .lock()
                    .unwrap()
                    .get(&cancel.request_id)
                    .cloned();
                let accepted = cancellation.is_some();
                if let Some(cancellation) = cancellation {
                    cancellation.store(true, Ordering::Release);
                }
                write_message(
                    &stdout,
                    &WireMessage::CancelAck(CancelAckFrame {
                        request_id: cancel.request_id,
                        correlation_id: cancel.correlation_id,
                        accepted,
                        already_terminal: !accepted,
                    }),
                )?;
            }
            Err(reimagine_backend_worker_protocol::CodecError::Io(error))
                if error.kind() == io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Ok(_) => return Err("unexpected host frame".into()),
            Err(error) => return Err(error.into()),
        }
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
                    if cancellation.load(Ordering::Acquire) {
                        break;
                    }
                    write_message(
                        &stdout,
                        &WireMessage::Progress(reimagine_backend_worker_protocol::ProgressFrame {
                            request_id: request.request_id.clone(),
                            correlation_id: request.correlation_id.clone(),
                            sequence,
                            completed: sequence,
                            total: Some(3),
                            message: Some(format!("step {sequence}")),
                        }),
                    )
                    .unwrap();
                    thread::sleep(Duration::from_millis(2));
                }
                if cancellation.load(Ordering::Acquire) {
                    TerminalOutcome::Cancelled
                } else {
                    TerminalOutcome::Success {
                        output: json!(null),
                    }
                }
            }
            _ => TerminalOutcome::BackendError {
                error: BackendExecutionError {
                    code: "unsupported_operation".to_owned(),
                    message: format!("unsupported fake operation `{}`", request.operation),
                    retryable: false,
                },
            },
        };
        cancellations.lock().unwrap().remove(&request.request_id);
        write_message(
            &stdout,
            &WireMessage::Terminal(TerminalFrame {
                request_id: request.request_id.clone(),
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
