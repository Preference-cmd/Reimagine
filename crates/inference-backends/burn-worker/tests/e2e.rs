//! End-to-end process test for the Burn worker binary.
//!
//! Spawns the real `reimagine-inference-burn-worker` binary,
//! performs a handshake, sends a `latent.create_empty` request,
//! and verifies the terminal response. Uses the
//! `CARGO_BIN_EXE_reimagine-inference-burn-worker` env var that
//! Cargo sets at test build time.

use std::io::{BufReader, Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Arc;

use reimagine_backend_worker_host::{
    ExpectedWorkerIdentity, ProcessInferenceBackend, WorkerLaunchSpec, WorkerLimits,
    WorkerSupervisor,
};
use reimagine_backend_worker_protocol::{
    BackendInstanceId, CleanupFrame, ControlId, CorrelationId, FrameCodec, HostHello,
    ProtocolRange, ProtocolVersion, RequestFrame, RequestId, ShutdownFrame, TerminalFrame,
    TerminalOutcome, WireMessage, WorkerHello, WorkerInstallationId,
};
use reimagine_core::model::NodeId;
use reimagine_inference::{
    CreateEmptyLatentRequest, DiffusionSampleRequest, ImagePreviewRequest, ImageSaveRequest,
    InferenceBackend, LatentDecodeRequest, SamplerName, SchedulerName,
};

mod tiny_fixture {
    include!("../../burn/tests/tiny_sdxl_e2e.rs");
}

const MAXIMUM_FRAME_BYTES: u32 = 64 * 1024 * 1024;

struct BurnWorkerProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    hello: WorkerHello,
}

impl BurnWorkerProcess {
    fn spawn() -> Self {
        let bin = env!("CARGO_BIN_EXE_reimagine-inference-burn-worker");
        let mut child = Command::new(bin)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env_clear()
            .env("REIMAGINE_MODELS_DIR", "/tmp")
            .env("REIMAGINE_OUTPUT_DIR", "/tmp")
            .env("REIMAGINE_ALLOWED_MODEL_ROOTS", "/tmp")
            .env("REIMAGINE_ALLOWED_OUTPUT_ROOTS", "/tmp")
            .spawn()
            .expect("failed to spawn burn worker process");

        let mut stdin = child.stdin.take().expect("stdin not piped");
        let mut stdout = BufReader::new(child.stdout.take().expect("stdout not piped"));

        let codec = FrameCodec::new(MAXIMUM_FRAME_BYTES);

        // Send HostHello
        codec
            .write(
                &mut stdin,
                &WireMessage::HostHello(HostHello {
                    supported_protocols: ProtocolRange::new(1, 1),
                }),
            )
            .expect("failed to write HostHello");
        // Write is buffered; must flush through the ChildStdin directly.
        // FrameCodec::write for BufWriter<&mut dyn Write> — but here we
        // have ChildStdin. Use write_all via the codec's encode_payload.
        // Actually FrameCodec has a write(&mut self, writer: &mut impl Write, ...)
        // which already flushes? Let's just use the standard approach.
        // The codec write calls writer.write_all which on a ChildStdin does
        // unbuffered writes. We need to ensure everything is flushed.

        // Read WorkerHello using a manual frame protocol
        let mut prefix = [0u8; 4];
        stdout
            .read_exact(&mut prefix)
            .expect("failed to read frame prefix");
        let declared = u32::from_be_bytes(prefix);
        let mut payload = vec![0u8; declared as usize];
        stdout
            .read_exact(&mut payload)
            .expect("failed to read frame payload");
        let WireMessage::WorkerHello(hello) = codec
            .decode_payload(&payload)
            .expect("failed to decode WorkerHello")
        else {
            panic!("expected WorkerHello as first response");
        };

        Self {
            child,
            stdin,
            stdout,
            hello,
        }
    }

    fn request(&mut self, operation: &str, payload: serde_json::Value) -> WireMessage {
        let codec = FrameCodec::new(MAXIMUM_FRAME_BYTES);
        // Encode and write the request frame
        let message = WireMessage::Request(RequestFrame {
            protocol_version: ProtocolVersion(1),
            incarnation_id: self.hello.identity.incarnation_id.clone(),
            request_id: RequestId::from("e2e-test"),
            correlation_id: CorrelationId::from("e2e-correlation"),
            operation: operation.to_owned(),
            payload,
        });
        let raw = codec
            .encode_payload(&message)
            .expect("failed to encode request");
        let len = u32::try_from(raw.len()).expect("payload too large");
        self.stdin
            .write_all(&len.to_be_bytes())
            .expect("failed to write frame prefix");
        self.stdin
            .write_all(&raw)
            .expect("failed to write frame payload");
        self.stdin.flush().expect("failed to flush request");

        // Read the response frame
        let mut prefix = [0u8; 4];
        self.stdout
            .read_exact(&mut prefix)
            .expect("failed to read response prefix");
        let declared = u32::from_be_bytes(prefix);
        let mut payload = vec![0u8; declared as usize];
        self.stdout
            .read_exact(&mut payload)
            .expect("failed to read response payload");
        codec
            .decode_payload(&payload)
            .expect("failed to decode response")
    }

    fn send(&mut self, message: &WireMessage) {
        let codec = FrameCodec::new(MAXIMUM_FRAME_BYTES);
        codec.write(&mut self.stdin, message).unwrap();
        self.stdin.flush().unwrap();
    }

    fn read(&mut self) -> WireMessage {
        FrameCodec::new(MAXIMUM_FRAME_BYTES)
            .read(&mut self.stdout)
            .unwrap()
    }

    fn exits_within(&mut self, deadline: std::time::Duration) -> bool {
        let started = std::time::Instant::now();
        while started.elapsed() < deadline {
            if self.child.try_wait().unwrap().is_some() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        false
    }
}

impl Drop for BurnWorkerProcess {
    fn drop(&mut self) {
        // Send shutdown for clean termination
        let codec = FrameCodec::new(MAXIMUM_FRAME_BYTES);
        if let Ok(raw) = codec.encode_payload(&WireMessage::Shutdown(ShutdownFrame {
            protocol_version: ProtocolVersion(1),
            incarnation_id: self.hello.identity.incarnation_id.clone(),
            control_id: reimagine_backend_worker_protocol::ControlId::from("e2e-shutdown"),
        })) {
            let len = u32::try_from(raw.len()).unwrap_or(0);
            let _ = self.stdin.write_all(&len.to_be_bytes());
            let _ = self.stdin.write_all(&raw);
            let _ = self.stdin.flush();
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn process_launch_spec(
    models_dir: &std::path::Path,
    output_dir: &std::path::Path,
) -> WorkerLaunchSpec {
    #[cfg(feature = "wgpu")]
    let backend_instance_id = "burn:wgpu:default";
    #[cfg(all(not(feature = "wgpu"), feature = "flex"))]
    let backend_instance_id = "burn:flex:cpu";

    WorkerLaunchSpec {
        executable: env!("CARGO_BIN_EXE_reimagine-inference-burn-worker").into(),
        expected: ExpectedWorkerIdentity {
            backend_instance_id: BackendInstanceId::from(backend_instance_id),
            installation_id: WorkerInstallationId::from("dev"),
            backend_kind: "burn".to_owned(),
            target: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
            manifest_digest: "dev".to_owned(),
        },
        supported_protocols: ProtocolRange::new(1, 1),
        limits: WorkerLimits {
            request_timeout: std::time::Duration::from_secs(120),
            ..WorkerLimits::default()
        },
        environment: vec![
            (
                "REIMAGINE_MODELS_DIR".to_owned(),
                models_dir.display().to_string(),
            ),
            (
                "REIMAGINE_OUTPUT_DIR".to_owned(),
                output_dir.display().to_string(),
            ),
            (
                "REIMAGINE_ALLOWED_MODEL_ROOTS".to_owned(),
                models_dir.display().to_string(),
            ),
            (
                "REIMAGINE_ALLOWED_OUTPUT_ROOTS".to_owned(),
                output_dir.display().to_string(),
            ),
        ],
    }
}

#[tokio::test]
async fn process_adapter_runs_tiny_sdxl_chain() {
    let root = tempfile::tempdir().expect("package root");
    let output = tempfile::tempdir().expect("output dir");
    let worker = Arc::new(
        WorkerSupervisor::new(process_launch_spec(root.path(), output.path()))
            .start()
            .await
            .expect("start real Burn worker"),
    );
    let backend = ProcessInferenceBackend::new(worker);

    let loaded = backend
        .load_bundle(tiny_fixture::load_request(root.path()))
        .await
        .expect("load tiny fixture through process");
    let positive = backend
        .text_encode(tiny_fixture::text_request(
            loaded.clip().clone(),
            "small bright city at sunrise",
            "process-positive",
        ))
        .await
        .expect("positive text.encode through process")
        .into_conditioning();
    let negative = backend
        .text_encode(tiny_fixture::text_request(
            loaded.clip().clone(),
            "low quality blur",
            "process-negative",
        ))
        .await
        .expect("negative text.encode through process")
        .into_conditioning();
    let latent = backend
        .create_empty_latent(CreateEmptyLatentRequest::new(
            64,
            64,
            1,
            tiny_fixture::run_id(),
            tiny_fixture::workflow_id(),
            tiny_fixture::workflow_version(),
            NodeId::new("process-empty"),
        ))
        .await
        .expect("empty latent through process")
        .into_latent();
    let sampled = backend
        .diffusion_sample(DiffusionSampleRequest::new(
            loaded.model().clone(),
            positive,
            negative,
            latent,
            1234,
            1,
            1.0,
            SamplerName::Euler,
            SchedulerName::Normal,
            1.0,
            tiny_fixture::run_id(),
            tiny_fixture::workflow_id(),
            tiny_fixture::workflow_version(),
            NodeId::new("process-diffusion"),
        ))
        .await
        .expect("diffusion through process")
        .into_latent();
    let image = backend
        .latent_decode(LatentDecodeRequest::new(
            loaded.vae().clone(),
            sampled,
            tiny_fixture::run_id(),
            tiny_fixture::workflow_id(),
            tiny_fixture::workflow_version(),
            NodeId::new("process-decode"),
        ))
        .await
        .expect("decode through process")
        .into_image();
    let preview = backend
        .image_preview(ImagePreviewRequest::new(
            image.clone(),
            tiny_fixture::run_id(),
            tiny_fixture::workflow_id(),
            tiny_fixture::workflow_version(),
            NodeId::new("process-preview"),
        ))
        .await
        .expect("preview through process")
        .into_artifact();
    let saved = backend
        .image_save(
            ImageSaveRequest::new(
                image,
                tiny_fixture::run_id(),
                tiny_fixture::workflow_id(),
                tiny_fixture::workflow_version(),
                NodeId::new("process-save"),
            )
            .with_filename_prefix("process-tiny-sdxl"),
        )
        .await
        .expect("save through process")
        .into_artifact();

    tiny_fixture::assert_png_artifact(output.path(), preview.as_str());
    tiny_fixture::assert_png_artifact(output.path(), saved.as_str());
}

#[test]
fn worker_handshake_returns_valid_hello() {
    let worker = BurnWorkerProcess::spawn();

    assert_eq!(
        worker.hello.selected_protocol,
        ProtocolVersion(1),
        "selected protocol must be 1"
    );
    assert_eq!(
        worker.hello.identity.backend_kind, "burn",
        "backend kind must be 'burn'"
    );
    assert!(
        !worker.hello.identity.incarnation_id.0.is_empty(),
        "incarnation id must be non-empty"
    );
    assert!(
        !worker.hello.identity.worker_version.is_empty(),
        "worker version must be non-empty"
    );
    assert!(
        !worker.hello.profile.instances.is_empty(),
        "profile must have at least one instance"
    );
}

#[test]
fn worker_create_empty_latent_returns_expected_response() {
    let mut worker = BurnWorkerProcess::spawn();

    let response = worker.request(
        "latent.create_empty",
        serde_json::json!({
            "width": 64,
            "height": 64,
            "batch_size": 1,
        }),
    );

    match response {
        WireMessage::Terminal(terminal) => match terminal.outcome {
            TerminalOutcome::Success { ref output } => {
                assert!(
                    output
                        .get("worker_token")
                        .and_then(|v| v.as_str())
                        .is_some(),
                    "response must contain worker_token: {output}"
                );
                assert_eq!(
                    output.get("width").and_then(|v| v.as_u64()),
                    Some(64),
                    "response width must echo request"
                );
                assert_eq!(
                    output.get("height").and_then(|v| v.as_u64()),
                    Some(64),
                    "response height must echo request"
                );
                assert_eq!(
                    output.get("batch_size").and_then(|v| v.as_u64()),
                    Some(1),
                    "response batch_size must echo request"
                );
            }
            TerminalOutcome::BackendError { ref error } => {
                panic!(
                    "create_empty_latent returned backend error: {}: {}",
                    error.code, error.message
                );
            }
            TerminalOutcome::Cancelled => {
                panic!("create_empty_latent was cancelled");
            }
        },
        other => {
            panic!("expected Terminal response, got: {:?}", other.kind());
        }
    }
}

#[test]
fn worker_unknown_operation_returns_error() {
    let mut worker = BurnWorkerProcess::spawn();

    let response = worker.request("nonexistent_op", serde_json::json!({}));

    match response {
        WireMessage::Terminal(terminal) => match terminal.outcome {
            TerminalOutcome::BackendError { ref error } => {
                assert_eq!(
                    error.code, "unknown_operation",
                    "error code must be 'unknown_operation'"
                );
            }
            _ => panic!("expected backend error for unknown operation"),
        },
        other => panic!("expected Terminal response, got: {:?}", other.kind()),
    }
}

#[test]
fn cleanup_reports_actual_run_removals_once() {
    let mut worker = BurnWorkerProcess::spawn();
    let response = worker.request(
        "latent.create_empty",
        serde_json::json!({
            "width": 64,
            "height": 64,
            "batch_size": 1,
            "run_id": "cleanup-run",
        }),
    );
    assert!(matches!(response, WireMessage::Terminal(_)));

    for expected in [1, 0] {
        worker.send(&WireMessage::Cleanup(CleanupFrame {
            protocol_version: ProtocolVersion(1),
            incarnation_id: worker.hello.identity.incarnation_id.clone(),
            control_id: ControlId::from("cleanup-control"),
            run_id: Some("cleanup-run".to_owned()),
            object_ids: Vec::new(),
        }));
        let WireMessage::CleanupAck(ack) = worker.read() else {
            panic!("expected cleanup ack");
        };
        assert_eq!(ack.released_objects, expected);
    }
}

#[test]
fn wrong_protocol_version_terminates_session() {
    let mut worker = BurnWorkerProcess::spawn();
    worker.send(&WireMessage::Request(RequestFrame {
        protocol_version: ProtocolVersion(99),
        incarnation_id: worker.hello.identity.incarnation_id.clone(),
        request_id: RequestId::from("wrong-version"),
        correlation_id: CorrelationId::from("wrong-version"),
        operation: "latent.create_empty".to_owned(),
        payload: serde_json::json!({ "width": 64, "height": 64, "batch_size": 1 }),
    }));
    assert!(worker.exits_within(std::time::Duration::from_secs(1)));
}

#[test]
fn stale_incarnation_terminates_session() {
    let mut worker = BurnWorkerProcess::spawn();
    worker.send(&WireMessage::Request(RequestFrame {
        protocol_version: ProtocolVersion(1),
        incarnation_id: reimagine_backend_worker_protocol::WorkerIncarnationId::from("stale"),
        request_id: RequestId::from("stale-incarnation"),
        correlation_id: CorrelationId::from("stale-incarnation"),
        operation: "latent.create_empty".to_owned(),
        payload: serde_json::json!({ "width": 64, "height": 64, "batch_size": 1 }),
    }));
    assert!(worker.exits_within(std::time::Duration::from_secs(1)));
}

#[test]
fn wrong_direction_terminates_session() {
    let mut worker = BurnWorkerProcess::spawn();
    worker.send(&WireMessage::Terminal(TerminalFrame {
        protocol_version: ProtocolVersion(1),
        incarnation_id: worker.hello.identity.incarnation_id.clone(),
        request_id: RequestId::from("wrong-direction"),
        correlation_id: CorrelationId::from("wrong-direction"),
        outcome: TerminalOutcome::Cancelled,
    }));
    assert!(worker.exits_within(std::time::Duration::from_secs(1)));
}
