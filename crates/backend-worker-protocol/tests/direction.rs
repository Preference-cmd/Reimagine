use reimagine_backend_worker_protocol::{
    BackendInstanceId, HostHello, MessageSender, ProtocolRange, ProtocolVersion, ProtocolViolation,
    WireMessage, WorkerHello, WorkerIdentity, WorkerIncarnationId, WorkerInstallationId,
    WorkerInstanceProfile, WorkerProfile, validate_message_direction,
};

fn worker_hello() -> WireMessage {
    WireMessage::WorkerHello(WorkerHello {
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
    })
}

#[test]
fn worker_hello_is_rejected_when_sent_by_host() {
    assert_eq!(
        validate_message_direction(&worker_hello(), MessageSender::Host),
        Err(ProtocolViolation::InvalidDirection {
            kind: "worker_hello",
            sender: MessageSender::Host,
        })
    );
}

#[test]
fn host_hello_is_accepted_from_host() {
    assert_eq!(
        validate_message_direction(
            &WireMessage::HostHello(HostHello {
                supported_protocols: ProtocolRange::new(1, 1),
            }),
            MessageSender::Host,
        ),
        Ok(())
    );
}
