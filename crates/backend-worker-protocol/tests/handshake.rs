use reimagine_backend_worker_protocol::{
    BackendInstanceId, CodecError, FrameCodec, HandshakeError, ProtocolRange, ProtocolVersion,
    WireMessage, WorkerHello, WorkerIdentity, WorkerIncarnationId, WorkerInstallationId,
    WorkerInstanceProfile, WorkerProfile, negotiate_protocol,
};
use std::io;

#[test]
fn negotiation_selects_highest_mutually_supported_version() {
    assert_eq!(
        negotiate_protocol(ProtocolRange::new(1, 4), ProtocolRange::new(2, 3)).unwrap(),
        ProtocolVersion(3)
    );
}

#[test]
fn negotiation_reports_both_incompatible_ranges() {
    let host = ProtocolRange::new(1, 2);
    let worker = ProtocolRange::new(3, 4);
    assert_eq!(
        negotiate_protocol(host, worker),
        Err(HandshakeError::NoCompatibleVersion { host, worker })
    );
}

#[test]
fn negotiation_rejects_an_inverted_range() {
    let invalid = ProtocolRange::new(3, 2);
    assert_eq!(
        negotiate_protocol(invalid, ProtocolRange::new(1, 3)),
        Err(HandshakeError::InvalidRange(invalid))
    );
}

#[test]
fn worker_hello_requires_authoritative_profile() {
    let hello = WireMessage::WorkerHello(WorkerHello {
        selected_protocol: ProtocolVersion(1),
        identity: WorkerIdentity {
            backend_instance_id: BackendInstanceId::from("fake:cpu:default"),
            installation_id: WorkerInstallationId::from("install-1"),
            incarnation_id: WorkerIncarnationId::from("inc-1"),
            worker_version: "0.1.0".to_owned(),
            backend_kind: "fake".to_owned(),
            target: "test-target".to_owned(),
            manifest_digest: "sha256:test".to_owned(),
        },
        profile: WorkerProfile {
            instances: vec![WorkerInstanceProfile {
                backend_instance_id: BackendInstanceId::from("fake:cpu:default"),
                device_label: "cpu".to_owned(),
                capabilities: vec!["echo".to_owned()],
                operation_options: serde_json::json!({}),
            }],
        },
    });
    let mut value = serde_json::to_value(hello).unwrap();
    value
        .get_mut("body")
        .and_then(serde_json::Value::as_object_mut)
        .unwrap()
        .remove("profile");
    let payload = serde_json::to_vec(&value).unwrap();
    let mut encoded = (payload.len() as u32).to_be_bytes().to_vec();
    encoded.extend_from_slice(&payload);

    assert!(matches!(
        FrameCodec::new(2048).read(&mut io::Cursor::new(encoded)),
        Err(CodecError::MalformedJson(_))
    ));
}
