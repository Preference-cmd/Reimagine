//! End-to-end integration tests for gRPC transport.
//!
//! Verifies that all 14 WireMessage types flow correctly through
//! the gRPC conversion layer and transport.

use std::sync::Arc;

use reimagine_backend_worker_protocol::{
    BackendExecutionError, BackendInstanceId, CancelAckFrame, CancelFrame, CleanupAckFrame,
    CleanupFrame, ControlId, CorrelationId, HealthAckFrame, HealthFrame, HostHello, ProgressFrame,
    ProtocolRange, ProtocolVersion, RequestFrame, RequestId, ShutdownAckFrame, ShutdownFrame,
    TerminalFrame, TerminalOutcome, WireMessage, WorkerIdentity, WorkerIncarnationId,
    WorkerInstallationId, WorkerInstanceProfile, WorkerProfile, WorkerTransport as _,
};
use reimagine_backend_worker_transport_grpc::client;
use reimagine_backend_worker_transport_grpc::proto;
use reimagine_backend_worker_transport_grpc::proto::worker_service_server::WorkerServiceServer;
use reimagine_backend_worker_transport_grpc::server::{GrpcWorkerService, MessageHandler};

/// Start a gRPC server with the given handler and return the endpoint URL.
async fn start_server(handler: MessageHandler) -> String {
    let service = GrpcWorkerService::new(handler);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server = tonic::transport::Server::builder()
        .add_service(WorkerServiceServer::new(service))
        .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener));

    tokio::spawn(server);
    format!("http://{addr}")
}

/// A worker handler that responds to all host-to-worker message types
/// with appropriate worker-to-host responses.
fn comprehensive_handler() -> MessageHandler {
    Arc::new(|msg| {
        Box::pin(async move {
            match msg.message {
                // HostHello -> WorkerHello
                Some(proto::host_to_worker::Message::HostHello(h)) => Some(proto::WorkerToHost {
                    message: Some(proto::worker_to_host::Message::WorkerHello(
                        proto::WorkerHello {
                            selected_protocol: std::cmp::min(h.protocol_max, 1),
                            identity: Some(proto::WorkerIdentity {
                                backend_instance_id: "burn:cuda:0".into(),
                                installation_id: "install-1".into(),
                                incarnation_id: "inc-1".into(),
                                worker_version: "0.1.0".into(),
                                backend_kind: "burn".into(),
                                target: "aarch64".into(),
                                manifest_digest: "abc123".into(),
                            }),
                            profile: Some(proto::WorkerProfile {
                                instances: vec![proto::WorkerInstanceProfile {
                                    backend_instance_id: "burn:cuda:0".into(),
                                    device_label: "cuda:0".into(),
                                    capabilities: vec!["echo".into()],
                                    operation_options: serde_json::to_vec(
                                        &serde_json::json!({"batch_size": 1}),
                                    )
                                    .unwrap(),
                                }],
                            }),
                        },
                    )),
                }),
                // Request -> Terminal (echo)
                Some(proto::host_to_worker::Message::Request(r)) => Some(proto::WorkerToHost {
                    message: Some(proto::worker_to_host::Message::Terminal(proto::Terminal {
                        protocol_version: r.protocol_version,
                        incarnation_id: r.incarnation_id,
                        request_id: r.request_id,
                        correlation_id: r.correlation_id,
                        outcome: Some(proto::TerminalOutcome {
                            outcome: Some(proto::terminal_outcome::Outcome::Success(
                                proto::Success { output: r.payload },
                            )),
                        }),
                    })),
                }),
                // Cancel -> CancelAck
                Some(proto::host_to_worker::Message::Cancel(c)) => Some(proto::WorkerToHost {
                    message: Some(proto::worker_to_host::Message::CancelAck(
                        proto::CancelAck {
                            protocol_version: c.protocol_version,
                            incarnation_id: c.incarnation_id,
                            request_id: c.request_id,
                            correlation_id: c.correlation_id,
                            accepted: true,
                            already_terminal: false,
                        },
                    )),
                }),
                // Health -> HealthAck
                Some(proto::host_to_worker::Message::Health(h)) => Some(proto::WorkerToHost {
                    message: Some(proto::worker_to_host::Message::HealthAck(
                        proto::HealthAck {
                            protocol_version: h.protocol_version,
                            incarnation_id: h.incarnation_id,
                            control_id: h.control_id,
                            healthy: true,
                            message: Some("all good".into()),
                        },
                    )),
                }),
                // Cleanup -> CleanupAck
                Some(proto::host_to_worker::Message::Cleanup(c)) => Some(proto::WorkerToHost {
                    message: Some(proto::worker_to_host::Message::CleanupAck(
                        proto::CleanupAck {
                            protocol_version: c.protocol_version,
                            incarnation_id: c.incarnation_id,
                            control_id: c.control_id,
                            released_objects: c.object_ids.len() as u64,
                        },
                    )),
                }),
                // Shutdown -> ShutdownAck
                Some(proto::host_to_worker::Message::Shutdown(s)) => Some(proto::WorkerToHost {
                    message: Some(proto::worker_to_host::Message::ShutdownAck(
                        proto::ShutdownAck {
                            protocol_version: s.protocol_version,
                            incarnation_id: s.incarnation_id,
                            control_id: s.control_id,
                        },
                    )),
                }),
                // TensorTransferRequest -> TensorTransferAck (accepted)
                Some(proto::host_to_worker::Message::TensorTransferRequest(t)) => {
                    Some(proto::WorkerToHost {
                        message: Some(proto::worker_to_host::Message::TensorTransferAck(
                            proto::TensorTransferAck {
                                correlation_id: "corr-transfer".into(),
                                status: proto::TransferStatus::Accepted as i32,
                                reason: None,
                                target_token: Some(t.source_token),
                            },
                        )),
                    })
                }
                None => None,
            }
        })
    })
}

#[tokio::test]
async fn grpc_transport_worker_transport_trait() {
    let endpoint = start_server(comprehensive_handler()).await;
    let transport = client::connect(&endpoint).await.unwrap();

    assert_eq!(
        transport.description().kind,
        reimagine_backend_worker_protocol::TransportKind::Grpc
    );
    assert!(transport.description().endpoint.contains("http://"));

    transport.shutdown().await.unwrap();
}

#[tokio::test]
async fn grpc_request_terminal_roundtrip() {
    let endpoint = start_server(comprehensive_handler()).await;
    let transport = client::connect(&endpoint).await.unwrap();

    // Send Request, receive Terminal
    transport
        .send(proto::HostToWorker {
            message: Some(proto::host_to_worker::Message::Request(proto::Request {
                protocol_version: 1,
                incarnation_id: "inc-1".into(),
                request_id: "req-1".into(),
                correlation_id: "corr-1".into(),
                operation: "echo".into(),
                payload: serde_json::to_vec(&serde_json::json!({"value": 42})).unwrap(),
            })),
        })
        .await
        .unwrap();

    let response = transport.recv().await.unwrap().unwrap();
    match response.message {
        Some(proto::worker_to_host::Message::Terminal(t)) => {
            assert_eq!(t.protocol_version, 1);
            assert_eq!(t.request_id, "req-1");
            assert_eq!(t.correlation_id, "corr-1");
            match t.outcome {
                Some(proto::TerminalOutcome {
                    outcome: Some(proto::terminal_outcome::Outcome::Success(s)),
                }) => {
                    let output: serde_json::Value = serde_json::from_slice(&s.output).unwrap();
                    assert_eq!(output, serde_json::json!({"value": 42}));
                }
                other => panic!("expected Success outcome, got {other:?}"),
            }
        }
        other => panic!("expected Terminal, got {other:?}"),
    }
}

#[tokio::test]
async fn grpc_cancel_ack_roundtrip() {
    let endpoint = start_server(comprehensive_handler()).await;
    let transport = client::connect(&endpoint).await.unwrap();

    transport
        .send(proto::HostToWorker {
            message: Some(proto::host_to_worker::Message::Cancel(proto::Cancel {
                protocol_version: 1,
                incarnation_id: "inc-1".into(),
                request_id: "req-1".into(),
                correlation_id: "corr-1".into(),
            })),
        })
        .await
        .unwrap();

    let response = transport.recv().await.unwrap().unwrap();
    match response.message {
        Some(proto::worker_to_host::Message::CancelAck(c)) => {
            assert!(c.accepted);
            assert!(!c.already_terminal);
            assert_eq!(c.request_id, "req-1");
        }
        other => panic!("expected CancelAck, got {other:?}"),
    }
}

#[tokio::test]
async fn grpc_health_ack_roundtrip() {
    let endpoint = start_server(comprehensive_handler()).await;
    let transport = client::connect(&endpoint).await.unwrap();

    transport
        .send(proto::HostToWorker {
            message: Some(proto::host_to_worker::Message::Health(proto::Health {
                protocol_version: 1,
                incarnation_id: "inc-1".into(),
                control_id: "health-1".into(),
            })),
        })
        .await
        .unwrap();

    let response = transport.recv().await.unwrap().unwrap();
    match response.message {
        Some(proto::worker_to_host::Message::HealthAck(h)) => {
            assert!(h.healthy);
            assert_eq!(h.message.as_deref(), Some("all good"));
            assert_eq!(h.control_id, "health-1");
        }
        other => panic!("expected HealthAck, got {other:?}"),
    }
}

#[tokio::test]
async fn grpc_cleanup_ack_roundtrip() {
    let endpoint = start_server(comprehensive_handler()).await;
    let transport = client::connect(&endpoint).await.unwrap();

    transport
        .send(proto::HostToWorker {
            message: Some(proto::host_to_worker::Message::Cleanup(proto::Cleanup {
                protocol_version: 1,
                incarnation_id: "inc-1".into(),
                control_id: "cleanup-1".into(),
                run_id: Some("run-1".into()),
                object_ids: vec!["obj-1".into(), "obj-2".into()],
            })),
        })
        .await
        .unwrap();

    let response = transport.recv().await.unwrap().unwrap();
    match response.message {
        Some(proto::worker_to_host::Message::CleanupAck(c)) => {
            assert_eq!(c.released_objects, 2);
            assert_eq!(c.control_id, "cleanup-1");
        }
        other => panic!("expected CleanupAck, got {other:?}"),
    }
}

#[tokio::test]
async fn grpc_shutdown_ack_roundtrip() {
    let endpoint = start_server(comprehensive_handler()).await;
    let transport = client::connect(&endpoint).await.unwrap();

    transport
        .send(proto::HostToWorker {
            message: Some(proto::host_to_worker::Message::Shutdown(proto::Shutdown {
                protocol_version: 1,
                incarnation_id: "inc-1".into(),
                control_id: "shutdown-1".into(),
            })),
        })
        .await
        .unwrap();

    let response = transport.recv().await.unwrap().unwrap();
    match response.message {
        Some(proto::worker_to_host::Message::ShutdownAck(s)) => {
            assert_eq!(s.control_id, "shutdown-1");
            assert_eq!(s.protocol_version, 1);
        }
        other => panic!("expected ShutdownAck, got {other:?}"),
    }
}

#[tokio::test]
async fn grpc_multiple_requests_sequential() {
    let endpoint = start_server(comprehensive_handler()).await;
    let transport = client::connect(&endpoint).await.unwrap();

    // Send 3 requests, receive 3 terminals
    for i in 0..3 {
        transport
            .send(proto::HostToWorker {
                message: Some(proto::host_to_worker::Message::Request(proto::Request {
                    protocol_version: 1,
                    incarnation_id: "inc-1".into(),
                    request_id: format!("req-{i}"),
                    correlation_id: format!("corr-{i}"),
                    operation: "echo".into(),
                    payload: serde_json::to_vec(&serde_json::json!({"i": i})).unwrap(),
                })),
            })
            .await
            .unwrap();
    }

    for i in 0..3 {
        let response = transport.recv().await.unwrap().unwrap();
        match response.message {
            Some(proto::worker_to_host::Message::Terminal(t)) => {
                assert_eq!(t.request_id, format!("req-{i}"));
                assert_eq!(t.correlation_id, format!("corr-{i}"));
            }
            other => panic!("expected Terminal for request {i}, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn grpc_all_host_to_worker_wiremessage_types() {
    // Verifies that all 6 host-to-worker WireMessage types can be
    // converted to proto and sent over the transport.
    let endpoint = start_server(comprehensive_handler()).await;
    let transport = client::connect(&endpoint).await.unwrap();

    // 1. HostHello
    let wire = WireMessage::HostHello(HostHello {
        supported_protocols: ProtocolRange::new(1, 2),
    });
    let proto_msg: proto::HostToWorker = (&wire).try_into().unwrap();
    transport.send(proto_msg).await.unwrap();
    let resp = transport.recv().await.unwrap().unwrap();
    assert!(matches!(
        resp.message,
        Some(proto::worker_to_host::Message::WorkerHello(_))
    ));

    // 2. Request
    let wire = WireMessage::Request(RequestFrame {
        protocol_version: ProtocolVersion(1),
        incarnation_id: WorkerIncarnationId("inc-1".into()),
        request_id: RequestId("req-1".into()),
        correlation_id: CorrelationId("corr-1".into()),
        operation: "echo".into(),
        payload: serde_json::json!({"test": true}),
    });
    let proto_msg: proto::HostToWorker = (&wire).try_into().unwrap();
    transport.send(proto_msg).await.unwrap();
    let resp = transport.recv().await.unwrap().unwrap();
    assert!(matches!(
        resp.message,
        Some(proto::worker_to_host::Message::Terminal(_))
    ));

    // 3. Cancel
    let wire = WireMessage::Cancel(CancelFrame {
        protocol_version: ProtocolVersion(1),
        incarnation_id: WorkerIncarnationId("inc-1".into()),
        request_id: RequestId("req-2".into()),
        correlation_id: CorrelationId("corr-2".into()),
    });
    let proto_msg: proto::HostToWorker = (&wire).try_into().unwrap();
    transport.send(proto_msg).await.unwrap();
    let resp = transport.recv().await.unwrap().unwrap();
    assert!(matches!(
        resp.message,
        Some(proto::worker_to_host::Message::CancelAck(_))
    ));

    // 4. Health
    let wire = WireMessage::Health(HealthFrame {
        protocol_version: ProtocolVersion(1),
        incarnation_id: WorkerIncarnationId("inc-1".into()),
        control_id: ControlId("health-1".into()),
    });
    let proto_msg: proto::HostToWorker = (&wire).try_into().unwrap();
    transport.send(proto_msg).await.unwrap();
    let resp = transport.recv().await.unwrap().unwrap();
    assert!(matches!(
        resp.message,
        Some(proto::worker_to_host::Message::HealthAck(_))
    ));

    // 5. Cleanup
    let wire = WireMessage::Cleanup(CleanupFrame {
        protocol_version: ProtocolVersion(1),
        incarnation_id: WorkerIncarnationId("inc-1".into()),
        control_id: ControlId("cleanup-1".into()),
        run_id: Some("run-1".into()),
        object_ids: vec!["obj-1".into()],
    });
    let proto_msg: proto::HostToWorker = (&wire).try_into().unwrap();
    transport.send(proto_msg).await.unwrap();
    let resp = transport.recv().await.unwrap().unwrap();
    assert!(matches!(
        resp.message,
        Some(proto::worker_to_host::Message::CleanupAck(_))
    ));

    // 6. Shutdown
    let wire = WireMessage::Shutdown(ShutdownFrame {
        protocol_version: ProtocolVersion(1),
        incarnation_id: WorkerIncarnationId("inc-1".into()),
        control_id: ControlId("shutdown-1".into()),
    });
    let proto_msg: proto::HostToWorker = (&wire).try_into().unwrap();
    transport.send(proto_msg).await.unwrap();
    let resp = transport.recv().await.unwrap().unwrap();
    assert!(matches!(
        resp.message,
        Some(proto::worker_to_host::Message::ShutdownAck(_))
    ));
}

#[tokio::test]
async fn grpc_all_worker_to_host_wiremessage_types() {
    // Verifies that all 7 worker-to-host WireMessage types can be
    // converted to proto and deserialized back.
    //
    // We test this via the conversion layer directly, then verify the
    // round-trip preserves all fields.

    // 1. WorkerHello
    let wire = WireMessage::WorkerHello(reimagine_backend_worker_protocol::WorkerHello {
        selected_protocol: ProtocolVersion(1),
        identity: WorkerIdentity {
            backend_instance_id: BackendInstanceId("burn:cuda:0".into()),
            installation_id: WorkerInstallationId("install-1".into()),
            incarnation_id: WorkerIncarnationId("inc-1".into()),
            worker_version: "0.1.0".into(),
            backend_kind: "burn".into(),
            target: "aarch64".into(),
            manifest_digest: "abc123".into(),
        },
        profile: WorkerProfile {
            instances: vec![WorkerInstanceProfile {
                backend_instance_id: BackendInstanceId("burn:cuda:0".into()),
                device_label: "cuda:0".into(),
                capabilities: vec!["echo".into()],
                operation_options: serde_json::json!({"batch_size": 1}),
            }],
        },
    });
    let proto_msg: proto::WorkerToHost = (&wire).try_into().unwrap();
    let back: WireMessage = proto_msg.try_into().unwrap();
    assert_eq!(format!("{wire:?}"), format!("{back:?}"));

    // 2. Progress
    let wire = WireMessage::Progress(ProgressFrame {
        protocol_version: ProtocolVersion(1),
        incarnation_id: WorkerIncarnationId("inc-1".into()),
        request_id: RequestId("req-1".into()),
        correlation_id: CorrelationId("corr-1".into()),
        sequence: 1,
        completed: 5,
        total: Some(10),
        message: Some("step 1".into()),
    });
    let proto_msg: proto::WorkerToHost = (&wire).try_into().unwrap();
    let back: WireMessage = proto_msg.try_into().unwrap();
    assert_eq!(format!("{wire:?}"), format!("{back:?}"));

    // 3. CancelAck
    let wire = WireMessage::CancelAck(CancelAckFrame {
        protocol_version: ProtocolVersion(1),
        incarnation_id: WorkerIncarnationId("inc-1".into()),
        request_id: RequestId("req-1".into()),
        correlation_id: CorrelationId("corr-1".into()),
        accepted: true,
        already_terminal: false,
    });
    let proto_msg: proto::WorkerToHost = (&wire).try_into().unwrap();
    let back: WireMessage = proto_msg.try_into().unwrap();
    assert_eq!(format!("{wire:?}"), format!("{back:?}"));

    // 4. Terminal (Success)
    let wire = WireMessage::Terminal(TerminalFrame {
        protocol_version: ProtocolVersion(1),
        incarnation_id: WorkerIncarnationId("inc-1".into()),
        request_id: RequestId("req-1".into()),
        correlation_id: CorrelationId("corr-1".into()),
        outcome: TerminalOutcome::Success {
            output: serde_json::json!({"ok": true}),
        },
    });
    let proto_msg: proto::WorkerToHost = (&wire).try_into().unwrap();
    let back: WireMessage = proto_msg.try_into().unwrap();
    assert_eq!(format!("{wire:?}"), format!("{back:?}"));

    // 5. Terminal (Cancelled)
    let wire = WireMessage::Terminal(TerminalFrame {
        protocol_version: ProtocolVersion(1),
        incarnation_id: WorkerIncarnationId("inc-1".into()),
        request_id: RequestId("req-1".into()),
        correlation_id: CorrelationId("corr-1".into()),
        outcome: TerminalOutcome::Cancelled,
    });
    let proto_msg: proto::WorkerToHost = (&wire).try_into().unwrap();
    let back: WireMessage = proto_msg.try_into().unwrap();
    assert_eq!(format!("{wire:?}"), format!("{back:?}"));

    // 6. Terminal (BackendError)
    let wire = WireMessage::Terminal(TerminalFrame {
        protocol_version: ProtocolVersion(1),
        incarnation_id: WorkerIncarnationId("inc-1".into()),
        request_id: RequestId("req-1".into()),
        correlation_id: CorrelationId("corr-1".into()),
        outcome: TerminalOutcome::BackendError {
            error: BackendExecutionError {
                code: "gpu_oom".into(),
                message: "out of memory".into(),
                retryable: true,
            },
        },
    });
    let proto_msg: proto::WorkerToHost = (&wire).try_into().unwrap();
    let back: WireMessage = proto_msg.try_into().unwrap();
    assert_eq!(format!("{wire:?}"), format!("{back:?}"));

    // 7. HealthAck
    let wire = WireMessage::HealthAck(HealthAckFrame {
        protocol_version: ProtocolVersion(1),
        incarnation_id: WorkerIncarnationId("inc-1".into()),
        control_id: ControlId("health-1".into()),
        healthy: true,
        message: Some("all good".into()),
    });
    let proto_msg: proto::WorkerToHost = (&wire).try_into().unwrap();
    let back: WireMessage = proto_msg.try_into().unwrap();
    assert_eq!(format!("{wire:?}"), format!("{back:?}"));

    // 8. CleanupAck
    let wire = WireMessage::CleanupAck(CleanupAckFrame {
        protocol_version: ProtocolVersion(1),
        incarnation_id: WorkerIncarnationId("inc-1".into()),
        control_id: ControlId("cleanup-1".into()),
        released_objects: 5,
    });
    let proto_msg: proto::WorkerToHost = (&wire).try_into().unwrap();
    let back: WireMessage = proto_msg.try_into().unwrap();
    assert_eq!(format!("{wire:?}"), format!("{back:?}"));

    // 9. ShutdownAck
    let wire = WireMessage::ShutdownAck(ShutdownAckFrame {
        protocol_version: ProtocolVersion(1),
        incarnation_id: WorkerIncarnationId("inc-1".into()),
        control_id: ControlId("shutdown-1".into()),
    });
    let proto_msg: proto::WorkerToHost = (&wire).try_into().unwrap();
    let back: WireMessage = proto_msg.try_into().unwrap();
    assert_eq!(format!("{wire:?}"), format!("{back:?}"));
}
