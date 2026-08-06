//! Adversarial and property tests for [`FrameCodec`] (B5-3 / BE-24).
//!
//! # Why deterministic property tests instead of `cargo-fuzz`
//!
//! The ticket's acceptance criteria ask for fuzz-style coverage. We
//! deliberately use **seeded, deterministic property tests** instead of a
//! `cargo-fuzz`/`libfuzzer` target:
//!
//! - `cargo-fuzz` requires a nightly toolchain and its own corpus/build
//!   pipeline, which the workspace CI gate
//!   (`cargo check --workspace --all-targets` + clippy + fmt) cannot run.
//! - A fixed-seed PRNG (here `rand::rngs::StdRng`) produces byte-identical
//!   inputs on every run and every platform, so failures are reproducible
//!   in CI without a fuzz harness.
//! - The codec surface is small: a u32 length prefix + JSON. Bounded
//!   random exploration of chunk splits, plus the adversarial fixtures
//!   below (huge declared lengths, truncation, deep nesting, non-UTF8),
//!   covers the same failure classes as a fuzzer would.
//!
//! Every test asserts the codec either decodes a well-formed frame or
//! returns a [`CodecError`] — never panics and never allocates more than
//! `maximum_frame_bytes` per frame.

use std::io::{self, Cursor, Read};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use reimagine_backend_worker_protocol::{
    BackendExecutionError, BackendInstanceId, CancelAckFrame, CancelFrame, CleanupAckFrame,
    CleanupFrame, CodecError, ControlId, CorrelationId, FrameCodec, HealthAckFrame, HealthFrame,
    HostHello, ProgressFrame, ProtocolRange, ProtocolVersion, RequestFrame, RequestId,
    ShutdownAckFrame, ShutdownFrame, TensorDataFrame, TensorMetadata, TensorTransferAckFrame,
    TensorTransferRequestFrame, TerminalFrame, TerminalOutcome, TransferStatus, WireMessage,
    WorkerHello, WorkerIdentity, WorkerIncarnationId, WorkerInstallationId, WorkerInstanceProfile,
    WorkerProfile,
};
use serde_json::json;

/// Reader that returns at most `chunk` bytes per `read` call, so tests
/// can slice a frame across arbitrary chunk boundaries.
struct ChunkedReader {
    bytes: Vec<u8>,
    position: usize,
    chunk: usize,
}

impl ChunkedReader {
    fn new(bytes: Vec<u8>, chunk: usize) -> Self {
        Self {
            bytes,
            position: 0,
            chunk: chunk.max(1),
        }
    }
}

impl Read for ChunkedReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.position == self.bytes.len() {
            return Ok(0);
        }
        let count = buffer
            .len()
            .min(self.chunk)
            .min(self.bytes.len() - self.position);
        buffer[..count].copy_from_slice(&self.bytes[self.position..self.position + count]);
        self.position += count;
        Ok(count)
    }
}

/// Reader that returns a random (seeded) 1..=`max_chunk` bytes per call,
/// deterministically splitting frames across arbitrary boundaries.
struct RandomChunkReader {
    bytes: Vec<u8>,
    position: usize,
    max_chunk: usize,
    rng: StdRng,
}

impl RandomChunkReader {
    fn new(bytes: Vec<u8>, max_chunk: usize, seed: u64) -> Self {
        Self {
            bytes,
            position: 0,
            max_chunk: max_chunk.max(1),
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Read for RandomChunkReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.position == self.bytes.len() {
            return Ok(0);
        }
        let chunk = self
            .rng
            .gen_range(1..=self.max_chunk)
            .min(buffer.len())
            .min(self.bytes.len() - self.position);
        buffer[..chunk].copy_from_slice(&self.bytes[self.position..self.position + chunk]);
        self.position += chunk;
        Ok(chunk)
    }
}

/// Encode `message` into a wire frame (prefix + payload).
fn encode_frame(codec: &FrameCodec, message: &WireMessage) -> Vec<u8> {
    let mut encoded = Vec::new();
    codec.write(&mut encoded, message).unwrap();
    encoded
}

/// One sample of every `WireMessage` kind, so property tests explore the
/// full message surface.
fn message_pool() -> Vec<WireMessage> {
    let request_id = RequestId::from("r1");
    let correlation_id = CorrelationId::from("c1");
    vec![
        WireMessage::Ping {
            nonce: "n1".to_owned(),
        },
        WireMessage::HostHello(HostHello {
            supported_protocols: ProtocolRange::new(1, 2),
        }),
        WireMessage::WorkerHello(WorkerHello {
            selected_protocol: ProtocolVersion(2),
            identity: WorkerIdentity {
                backend_instance_id: BackendInstanceId::from("burn:wgpu:default"),
                installation_id: WorkerInstallationId::from("install-1"),
                incarnation_id: WorkerIncarnationId::from("inc-1"),
                worker_version: "0.1.0".to_owned(),
                backend_kind: "burn-wgpu".to_owned(),
                target: "aarch64-apple-darwin".to_owned(),
                manifest_digest: "sha256:abc".to_owned(),
            },
            profile: WorkerProfile {
                instances: vec![WorkerInstanceProfile {
                    backend_instance_id: BackendInstanceId::from("burn:wgpu:default"),
                    device_label: "wgpu:default".to_owned(),
                    capabilities: vec!["load_bundle".to_owned()],
                    operation_options: json!({}),
                }],
            },
        }),
        WireMessage::Request(RequestFrame {
            protocol_version: ProtocolVersion(2),
            incarnation_id: WorkerIncarnationId::from("inc-1"),
            request_id: request_id.clone(),
            correlation_id: correlation_id.clone(),
            operation: "echo".to_owned(),
            payload: json!({ "input": 1 }),
        }),
        WireMessage::Progress(ProgressFrame {
            protocol_version: ProtocolVersion(2),
            incarnation_id: WorkerIncarnationId::from("inc-1"),
            request_id: request_id.clone(),
            correlation_id: correlation_id.clone(),
            sequence: 1,
            completed: 1,
            total: Some(2),
            message: Some("halfway".to_owned()),
        }),
        WireMessage::Cancel(CancelFrame {
            protocol_version: ProtocolVersion(2),
            incarnation_id: WorkerIncarnationId::from("inc-1"),
            request_id: request_id.clone(),
            correlation_id: correlation_id.clone(),
        }),
        WireMessage::CancelAck(CancelAckFrame {
            protocol_version: ProtocolVersion(2),
            incarnation_id: WorkerIncarnationId::from("inc-1"),
            request_id: request_id.clone(),
            correlation_id: correlation_id.clone(),
            accepted: true,
            already_terminal: false,
        }),
        WireMessage::Terminal(TerminalFrame {
            protocol_version: ProtocolVersion(2),
            incarnation_id: WorkerIncarnationId::from("inc-1"),
            request_id,
            correlation_id,
            outcome: TerminalOutcome::BackendError {
                error: BackendExecutionError {
                    code: "execution_failed".to_owned(),
                    message: "failure".to_owned(),
                    retryable: false,
                },
            },
        }),
        WireMessage::Health(HealthFrame {
            protocol_version: ProtocolVersion(2),
            incarnation_id: WorkerIncarnationId::from("inc-1"),
            control_id: ControlId::from("health-1"),
        }),
        WireMessage::HealthAck(HealthAckFrame {
            protocol_version: ProtocolVersion(2),
            incarnation_id: WorkerIncarnationId::from("inc-1"),
            control_id: ControlId::from("health-1"),
            healthy: true,
            message: None,
        }),
        WireMessage::Cleanup(CleanupFrame {
            protocol_version: ProtocolVersion(2),
            incarnation_id: WorkerIncarnationId::from("inc-1"),
            control_id: ControlId::from("cleanup-1"),
            run_id: Some("run-1".to_owned()),
            object_ids: vec!["object-1".to_owned()],
        }),
        WireMessage::CleanupAck(CleanupAckFrame {
            protocol_version: ProtocolVersion(2),
            incarnation_id: WorkerIncarnationId::from("inc-1"),
            control_id: ControlId::from("cleanup-1"),
            released_objects: 1,
        }),
        WireMessage::Shutdown(ShutdownFrame {
            protocol_version: ProtocolVersion(2),
            incarnation_id: WorkerIncarnationId::from("inc-1"),
            control_id: ControlId::from("shutdown-1"),
        }),
        WireMessage::ShutdownAck(ShutdownAckFrame {
            protocol_version: ProtocolVersion(2),
            incarnation_id: WorkerIncarnationId::from("inc-1"),
            control_id: ControlId::from("shutdown-1"),
        }),
        WireMessage::TensorTransferRequest(TensorTransferRequestFrame {
            source_token: "src-tok-1".to_owned(),
            target_worker_id: "worker-b".to_owned(),
            tensor_metadata: TensorMetadata {
                dtype: "f16".to_owned(),
                shape: vec![1, 3, 512, 512],
                size_bytes: 1_572_864,
                backend_format: "burn::nchw".to_owned(),
            },
        }),
        WireMessage::TensorTransferAck(TensorTransferAckFrame {
            correlation_id: CorrelationId::from("c1"),
            status: TransferStatus::Accepted,
            target_token: Some("tgt-tok-1".to_owned()),
        }),
        WireMessage::TensorData(TensorDataFrame {
            correlation_id: CorrelationId::from("c1"),
            sequence: 0,
            data: vec![0u8; 64],
            is_final: true,
        }),
    ]
}

// ── Malformed frames ────────────────────────────────────────────────

#[test]
fn huge_declared_lengths_are_rejected_from_the_prefix_without_allocation() {
    let codec = FrameCodec::new(1024);

    // Declared exactly at the maximum is rejected when the codec's cap is
    // lower, before any payload allocation happens.
    for declared in [1025_u32, u32::MAX] {
        let mut reader = Cursor::new(declared.to_be_bytes());
        let error = codec.read(&mut reader).unwrap_err();
        assert!(
            matches!(
                error,
                CodecError::FrameTooLarge { declared: d, maximum: 1024 } if d == declared
            ),
            "declared {declared} must be rejected without allocation, got {error:?}"
        );
        // Nothing beyond the 4-byte prefix was consumed.
        assert_eq!(reader.position(), 4);
    }
}

#[test]
fn truncated_frame_body_reports_io_error_without_panicking() {
    let codec = FrameCodec::new(1024);
    // Declares 512 bytes but only 4 are present.
    let mut bytes = 512_u32.to_be_bytes().to_vec();
    bytes.extend_from_slice(b"ab");
    let error = codec.read(&mut Cursor::new(bytes)).unwrap_err();
    assert!(
        matches!(error, CodecError::Io(ref io_error) if io_error.kind() == io::ErrorKind::UnexpectedEof),
        "truncated body must surface as an I/O error, got {error:?}"
    );
}

#[test]
fn truncated_json_payload_is_malformed_json() {
    let codec = FrameCodec::new(1024);
    let payload = b"{";
    let mut encoded = (payload.len() as u32).to_be_bytes().to_vec();
    encoded.extend_from_slice(payload);
    assert!(matches!(
        codec.read(&mut Cursor::new(encoded)),
        Err(CodecError::MalformedJson(_))
    ));
}

#[test]
fn empty_frame_is_rejected_as_malformed() {
    let codec = FrameCodec::new(1024);
    let encoded = 0_u32.to_be_bytes().to_vec();
    assert!(matches!(
        codec.read(&mut Cursor::new(encoded)),
        Err(CodecError::MalformedJson(_))
    ));
}

#[test]
fn declared_length_smaller_than_payload_does_not_overread() {
    let codec = FrameCodec::new(1024);
    // Declare 3 bytes; supply 12. The codec must read exactly 3 bytes and
    // fail on the JSON, leaving the rest unread in the stream.
    let mut encoded = 3_u32.to_be_bytes().to_vec();
    encoded.extend_from_slice(b"{\"kind\":\"ping\",\"body\":{\"nonce\":\"x\"}}");
    let mut cursor = Cursor::new(encoded);
    assert!(matches!(
        codec.read(&mut cursor),
        Err(CodecError::MalformedJson(_))
    ));
    assert_eq!(
        cursor.position(),
        4 + 3,
        "codec must not consume beyond declared length"
    );
}

#[test]
fn garbage_after_a_valid_frame_is_not_silently_accepted() {
    let codec = FrameCodec::new(1024);
    let mut stream = encode_frame(
        &codec,
        &WireMessage::Ping {
            nonce: "frame-1".to_owned(),
        },
    );
    stream.extend_from_slice(b"not a frame");

    let mut cursor = Cursor::new(stream);
    assert_eq!(
        codec.read(&mut cursor).unwrap(),
        WireMessage::Ping {
            nonce: "frame-1".to_owned()
        },
        "the valid frame must decode cleanly"
    );
    assert!(
        codec.read(&mut cursor).is_err(),
        "trailing garbage must be rejected, never silently ignored"
    );
}

#[test]
fn unknown_message_kind_with_extra_fields_is_rejected() {
    let codec = FrameCodec::new(1024);
    let payload = br#"{"kind":"alien_signal","body":{"blob":123},"extra":true}"#;
    let mut encoded = (payload.len() as u32).to_be_bytes().to_vec();
    encoded.extend_from_slice(payload);
    assert!(matches!(
        codec.read(&mut Cursor::new(encoded)),
        Err(CodecError::UnknownMessageKind(kind)) if kind == "alien_signal"
    ));
}

// ── Partial reads across chunk boundaries ───────────────────────────

#[test]
fn frame_roundtrips_byte_by_byte() {
    let codec = FrameCodec::new(4096);
    let expected = WireMessage::Ping {
        nonce: "byte-by-byte".to_owned(),
    };
    let encoded = encode_frame(&codec, &expected);
    let mut reader = ChunkedReader::new(encoded, 1);
    assert_eq!(codec.read(&mut reader).unwrap(), expected);
}

#[test]
fn concatenated_frames_survive_random_chunk_splitting() {
    let mut rng = StdRng::seed_from_u64(0xBEE24);
    let codec = FrameCodec::new(4096);
    let pool = message_pool();

    let mut stream = Vec::new();
    let expected: Vec<WireMessage> = (0..50)
        .map(|_| pool[rng.gen_range(0..pool.len())].clone())
        .collect();
    for message in &expected {
        stream.extend(encode_frame(&codec, message));
    }

    // Feed the whole stream in random-sized chunks (1..=64 bytes).
    let mut reader = RandomChunkReader::new(stream, 64, 0xBEE24);
    let mut decoded = Vec::new();
    while decoded.len() < expected.len() {
        let message = codec
            .read(&mut reader)
            .expect("chunked stream decode failed");
        decoded.push(message);
    }
    assert_eq!(
        decoded, expected,
        "random chunking must preserve frame boundaries"
    );
    // The final read must see end-of-stream, not garbage.
    assert!(codec.read(&mut reader).is_err());
}

// ── Adversarial payloads ────────────────────────────────────────────

#[test]
fn deeply_nested_json_is_rejected_without_panicking() {
    let codec = FrameCodec::new(1024 * 1024);
    // 10_000 levels of nesting: serde_json's recursion limit must reject
    // this with an error, never a stack overflow or allocation blow-up.
    let mut payload = vec![b'['; 10_001];
    payload.extend_from_slice(b"0");
    payload.extend(std::iter::repeat_n(b']', 10_000));

    let mut encoded = (payload.len() as u32).to_be_bytes().to_vec();
    encoded.extend_from_slice(&payload);
    assert!(
        matches!(
            codec.read(&mut Cursor::new(encoded)),
            Err(CodecError::MalformedJson(_))
        ),
        "deeply nested JSON must be rejected by the recursion limit"
    );
}

#[test]
fn non_utf8_payload_is_rejected_as_malformed_json() {
    let codec = FrameCodec::new(1024);
    let payload = [0xff, 0xfe, 0x00, 0x81, 0x80];
    let mut encoded = (payload.len() as u32).to_be_bytes().to_vec();
    encoded.extend_from_slice(&payload);
    assert!(matches!(
        codec.read(&mut Cursor::new(encoded)),
        Err(CodecError::MalformedJson(_))
    ));
}

#[test]
fn oversized_payloads_are_rejected_at_encode_time() {
    let codec = FrameCodec::new(16);
    let message = WireMessage::Ping {
        nonce: "this payload exceeds the 16-byte maximum".to_owned(),
    };
    let error = codec.encode_payload(&message).unwrap_err();
    assert!(
        matches!(error, CodecError::FrameTooLarge { .. }),
        "encode_payload must reject oversized payloads, got {error:?}"
    );
    // And a full write must fail the same way, not write a corrupt frame.
    let mut buffer = Vec::new();
    assert!(codec.write(&mut buffer, &message).is_err());
    assert!(
        buffer.is_empty(),
        "failed write must not leave partial frames"
    );
}

#[test]
fn payload_at_the_maximum_boundary_roundtrips() {
    let codec = FrameCodec::new(1024);
    // A JSON payload close to the boundary (data carries a tensor blob;
    // JSON array expansion makes 200 bytes land just under 1024).
    let mut data = vec![0u8; 200];
    for (index, byte) in data.iter_mut().enumerate() {
        *byte = (index % 251) as u8;
    }
    let message = WireMessage::TensorData(TensorDataFrame {
        correlation_id: CorrelationId::from("c-boundary"),
        sequence: 0,
        data,
        is_final: false,
    });
    let encoded = encode_frame(&codec, &message);
    assert!(encoded.len() as u32 <= 1024);
    assert_eq!(codec.read(&mut Cursor::new(encoded)).unwrap(), message);
}

// ── Seeded roundtrip property ───────────────────────────────────────

#[test]
fn seeded_random_messages_roundtrip_through_chunked_streams() {
    let mut rng = StdRng::seed_from_u64(0x5EED_BE24);
    let codec = FrameCodec::new(4096);
    let pool = message_pool();

    for iteration in 0..200 {
        let message = pool[rng.gen_range(0..pool.len())].clone();
        let encoded = encode_frame(&codec, &message);

        // Decode through a randomly chunked reader (1..=32 bytes).
        let seed = 0x5EED_BE24 ^ (iteration as u64) << 16;
        let mut reader = RandomChunkReader::new(encoded.clone(), 32, seed);
        let decoded = codec.read(&mut reader).expect("roundtrip decode failed");
        assert_eq!(decoded, message, "iteration {iteration} mismatch");

        // Sanity: the reader consumed the whole frame and no more.
        assert_eq!(reader.position, encoded.len());
    }
}

#[test]
fn seeded_random_multi_message_stream_roundtrips() {
    let mut rng = StdRng::seed_from_u64(0xC0DE_BE24);
    let codec = FrameCodec::new(4096);
    let pool = message_pool();

    let messages: Vec<WireMessage> = (0..100)
        .map(|_| pool[rng.gen_range(0..pool.len())].clone())
        .collect();
    let mut stream = Vec::new();
    for message in &messages {
        stream.extend(encode_frame(&codec, message));
    }

    let mut reader = Cursor::new(stream);
    for (index, expected) in messages.iter().enumerate() {
        let decoded = codec
            .read(&mut reader)
            .unwrap_or_else(|error| panic!("stream frame {index} failed: {error:?}"));
        assert_eq!(&decoded, expected, "stream frame {index} mismatch");
    }
    assert_eq!(
        reader.position() as usize,
        reader.get_ref().len(),
        "no trailing bytes"
    );
}
