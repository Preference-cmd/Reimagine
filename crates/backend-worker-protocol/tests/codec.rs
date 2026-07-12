use std::io::{self, Read};

use reimagine_backend_worker_protocol::{
    BackendExecutionError, BackendInstanceId, CancelAckFrame, CancelFrame, CodecError,
    CorrelationId, FrameCodec, HostHello, ProgressFrame, ProtocolRange, ProtocolVersion,
    RequestFrame, RequestId, TerminalFrame, TerminalOutcome, WireMessage, WorkerHello,
    WorkerIdentity, WorkerIncarnationId, WorkerInstallationId,
};
use serde_json::json;

struct FragmentedReader {
    bytes: Vec<u8>,
    position: usize,
}

impl Read for FragmentedReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.position == self.bytes.len() {
            return Ok(0);
        }
        let count = buffer.len().min(1);
        buffer[..count].copy_from_slice(&self.bytes[self.position..self.position + count]);
        self.position += count;
        Ok(count)
    }
}

#[test]
fn frame_roundtrips_across_partial_reads() {
    let codec = FrameCodec::new(1024);
    let expected = WireMessage::Ping {
        nonce: "frame-1".to_owned(),
    };
    let mut encoded = Vec::new();
    codec.write(&mut encoded, &expected).unwrap();

    let mut reader = FragmentedReader {
        bytes: encoded,
        position: 0,
    };
    assert_eq!(codec.read(&mut reader).unwrap(), expected);
}

#[test]
fn oversized_frame_is_rejected_from_its_prefix() {
    let codec = FrameCodec::new(16);
    let mut reader = io::Cursor::new(17_u32.to_be_bytes());

    let error = codec.read(&mut reader).unwrap_err();
    assert!(matches!(
        error,
        CodecError::FrameTooLarge {
            declared: 17,
            maximum: 16
        }
    ));
}

#[test]
fn malformed_json_is_distinct_from_transport_io() {
    let codec = FrameCodec::new(1024);
    let payload = b"{";
    let mut encoded = (payload.len() as u32).to_be_bytes().to_vec();
    encoded.extend_from_slice(payload);

    assert!(matches!(
        codec.read(&mut io::Cursor::new(encoded)),
        Err(CodecError::MalformedJson(_))
    ));
}

#[test]
fn unknown_fields_are_accepted() {
    let codec = FrameCodec::new(1024);
    let payload = br#"{"kind":"ping","body":{"nonce":"n-1","future_body_extension":1},"future_extension":true}"#;
    let mut encoded = (payload.len() as u32).to_be_bytes().to_vec();
    encoded.extend_from_slice(payload);

    assert_eq!(
        codec.read(&mut io::Cursor::new(encoded)).unwrap(),
        WireMessage::Ping {
            nonce: "n-1".to_owned()
        }
    );
}

#[test]
fn unknown_message_kind_is_rejected_precisely() {
    let codec = FrameCodec::new(1024);
    let payload = br#"{"kind":"future_message"}"#;
    let mut encoded = (payload.len() as u32).to_be_bytes().to_vec();
    encoded.extend_from_slice(payload);

    assert!(matches!(
        codec.read(&mut io::Cursor::new(encoded)),
        Err(CodecError::UnknownMessageKind(kind)) if kind == "future_message"
    ));
}

#[test]
fn every_wire_message_kind_roundtrips() {
    let request_id = RequestId::from("r1");
    let correlation_id = CorrelationId::from("c1");
    let messages = vec![
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
        }),
        WireMessage::Request(RequestFrame {
            protocol_version: ProtocolVersion(2),
            request_id: request_id.clone(),
            correlation_id: correlation_id.clone(),
            operation: "echo".to_owned(),
            payload: json!({ "input": 1 }),
        }),
        WireMessage::Progress(ProgressFrame {
            request_id: request_id.clone(),
            correlation_id: correlation_id.clone(),
            sequence: 1,
            completed: 1,
            total: Some(2),
            message: Some("halfway".to_owned()),
        }),
        WireMessage::Cancel(CancelFrame {
            request_id: request_id.clone(),
            correlation_id: correlation_id.clone(),
        }),
        WireMessage::CancelAck(CancelAckFrame {
            request_id: request_id.clone(),
            correlation_id: correlation_id.clone(),
            accepted: true,
            already_terminal: false,
        }),
        WireMessage::Terminal(TerminalFrame {
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
    ];
    let codec = FrameCodec::new(4096);
    for expected in messages {
        let mut encoded = Vec::new();
        codec.write(&mut encoded, &expected).unwrap();
        assert_eq!(codec.read(&mut io::Cursor::new(encoded)).unwrap(), expected);
    }
}
