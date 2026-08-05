use reimagine_backend_worker_protocol::{
    BackendExecutionError, BackendInstanceId, CancelAckFrame, CancelFrame, CleanupAckFrame,
    CleanupFrame, ControlId, CorrelationId, HealthAckFrame, HealthFrame, HostHello, ProgressFrame,
    ProtocolVersion, RequestFrame, RequestId, ShutdownAckFrame, ShutdownFrame, TerminalFrame,
    TerminalOutcome, WireMessage, WorkerHello, WorkerIdentity, WorkerIncarnationId,
    WorkerInstallationId, WorkerInstanceProfile, WorkerProfile,
};

use crate::proto;

fn u16_to_u32(v: u16) -> u32 {
    v as u32
}
fn u32_to_u16(v: u32) -> u16 {
    v as u16
}

// ── WireMessage → Proto ──────────────────────────────────────────

impl TryFrom<&WireMessage> for proto::HostToWorker {
    type Error = String;

    fn try_from(msg: &WireMessage) -> Result<Self, Self::Error> {
        let message = match msg {
            WireMessage::HostHello(h) => {
                proto::host_to_worker::Message::HostHello(proto::HostHello {
                    protocol_min: u16_to_u32(h.supported_protocols.minimum.0),
                    protocol_max: u16_to_u32(h.supported_protocols.maximum.0),
                })
            }
            WireMessage::Request(r) => proto::host_to_worker::Message::Request(proto::Request {
                protocol_version: u16_to_u32(r.protocol_version.0),
                incarnation_id: r.incarnation_id.0.clone(),
                request_id: r.request_id.0.clone(),
                correlation_id: r.correlation_id.0.clone(),
                operation: r.operation.clone(),
                payload: serde_json::to_vec(&r.payload).map_err(|e| e.to_string())?,
            }),
            WireMessage::Cancel(c) => proto::host_to_worker::Message::Cancel(proto::Cancel {
                protocol_version: u16_to_u32(c.protocol_version.0),
                incarnation_id: c.incarnation_id.0.clone(),
                request_id: c.request_id.0.clone(),
                correlation_id: c.correlation_id.0.clone(),
            }),
            WireMessage::Health(h) => proto::host_to_worker::Message::Health(proto::Health {
                protocol_version: u16_to_u32(h.protocol_version.0),
                incarnation_id: h.incarnation_id.0.clone(),
                control_id: h.control_id.0.clone(),
            }),
            WireMessage::Cleanup(c) => proto::host_to_worker::Message::Cleanup(proto::Cleanup {
                protocol_version: u16_to_u32(c.protocol_version.0),
                incarnation_id: c.incarnation_id.0.clone(),
                control_id: c.control_id.0.clone(),
                run_id: c.run_id.clone(),
                object_ids: c.object_ids.clone(),
            }),
            WireMessage::Shutdown(s) => proto::host_to_worker::Message::Shutdown(proto::Shutdown {
                protocol_version: u16_to_u32(s.protocol_version.0),
                incarnation_id: s.incarnation_id.0.clone(),
                control_id: s.control_id.0.clone(),
            }),
            _ => return Err(format!("not a host-to-worker message: {:?}", msg.kind())),
        };
        Ok(proto::HostToWorker {
            message: Some(message),
        })
    }
}

impl TryFrom<&WireMessage> for proto::WorkerToHost {
    type Error = String;

    fn try_from(msg: &WireMessage) -> Result<Self, Self::Error> {
        let message = match msg {
            WireMessage::WorkerHello(h) => {
                proto::worker_to_host::Message::WorkerHello(proto::WorkerHello {
                    selected_protocol: u16_to_u32(h.selected_protocol.0),
                    identity: Some(proto::WorkerIdentity {
                        backend_instance_id: h.identity.backend_instance_id.0.clone(),
                        installation_id: h.identity.installation_id.0.clone(),
                        incarnation_id: h.identity.incarnation_id.0.clone(),
                        worker_version: h.identity.worker_version.clone(),
                        backend_kind: h.identity.backend_kind.clone(),
                        target: h.identity.target.clone(),
                        manifest_digest: h.identity.manifest_digest.clone(),
                    }),
                    profile: Some(proto::WorkerProfile {
                        instances: h
                            .profile
                            .instances
                            .iter()
                            .map(|i| proto::WorkerInstanceProfile {
                                backend_instance_id: i.backend_instance_id.0.clone(),
                                device_label: i.device_label.clone(),
                                capabilities: i.capabilities.clone(),
                                operation_options: serde_json::to_vec(&i.operation_options)
                                    .unwrap_or_default(),
                            })
                            .collect(),
                    }),
                })
            }
            WireMessage::Progress(p) => proto::worker_to_host::Message::Progress(proto::Progress {
                protocol_version: u16_to_u32(p.protocol_version.0),
                incarnation_id: p.incarnation_id.0.clone(),
                request_id: p.request_id.0.clone(),
                correlation_id: p.correlation_id.0.clone(),
                sequence: p.sequence,
                completed: p.completed,
                total: p.total,
                message: p.message.clone(),
            }),
            WireMessage::CancelAck(c) => {
                proto::worker_to_host::Message::CancelAck(proto::CancelAck {
                    protocol_version: u16_to_u32(c.protocol_version.0),
                    incarnation_id: c.incarnation_id.0.clone(),
                    request_id: c.request_id.0.clone(),
                    correlation_id: c.correlation_id.0.clone(),
                    accepted: c.accepted,
                    already_terminal: c.already_terminal,
                })
            }
            WireMessage::Terminal(t) => {
                let outcome = Some(match &t.outcome {
                    TerminalOutcome::Success { output } => {
                        proto::terminal_outcome::Outcome::Success(proto::Success {
                            output: serde_json::to_vec(output).unwrap_or_default(),
                        })
                    }
                    TerminalOutcome::Cancelled => {
                        proto::terminal_outcome::Outcome::Cancelled(vec![])
                    }
                    TerminalOutcome::BackendError { error } => {
                        proto::terminal_outcome::Outcome::BackendError(proto::BackendError {
                            code: error.code.clone(),
                            message: error.message.clone(),
                            retryable: error.retryable,
                        })
                    }
                });
                proto::worker_to_host::Message::Terminal(proto::Terminal {
                    protocol_version: u16_to_u32(t.protocol_version.0),
                    incarnation_id: t.incarnation_id.0.clone(),
                    request_id: t.request_id.0.clone(),
                    correlation_id: t.correlation_id.0.clone(),
                    outcome: Some(proto::TerminalOutcome { outcome }),
                })
            }
            WireMessage::HealthAck(h) => {
                proto::worker_to_host::Message::HealthAck(proto::HealthAck {
                    protocol_version: u16_to_u32(h.protocol_version.0),
                    incarnation_id: h.incarnation_id.0.clone(),
                    control_id: h.control_id.0.clone(),
                    healthy: h.healthy,
                    message: h.message.clone(),
                })
            }
            WireMessage::CleanupAck(c) => {
                proto::worker_to_host::Message::CleanupAck(proto::CleanupAck {
                    protocol_version: u16_to_u32(c.protocol_version.0),
                    incarnation_id: c.incarnation_id.0.clone(),
                    control_id: c.control_id.0.clone(),
                    released_objects: c.released_objects,
                })
            }
            WireMessage::ShutdownAck(s) => {
                proto::worker_to_host::Message::ShutdownAck(proto::ShutdownAck {
                    protocol_version: u16_to_u32(s.protocol_version.0),
                    incarnation_id: s.incarnation_id.0.clone(),
                    control_id: s.control_id.0.clone(),
                })
            }
            _ => return Err(format!("not a worker-to-host message: {:?}", msg.kind())),
        };
        Ok(proto::WorkerToHost {
            message: Some(message),
        })
    }
}

// ── Proto → WireMessage ──────────────────────────────────────────

impl TryFrom<proto::HostToWorker> for WireMessage {
    type Error = String;

    fn try_from(msg: proto::HostToWorker) -> Result<Self, Self::Error> {
        let inner = msg.message.ok_or("missing message")?;
        match inner {
            proto::host_to_worker::Message::HostHello(h) => Ok(WireMessage::HostHello(HostHello {
                supported_protocols: reimagine_backend_worker_protocol::ProtocolRange::new(
                    u32_to_u16(h.protocol_min),
                    u32_to_u16(h.protocol_max),
                ),
            })),
            proto::host_to_worker::Message::Request(r) => {
                let payload = serde_json::from_slice(&r.payload).map_err(|e| e.to_string())?;
                Ok(WireMessage::Request(RequestFrame {
                    protocol_version: ProtocolVersion(u32_to_u16(r.protocol_version)),
                    incarnation_id: WorkerIncarnationId(r.incarnation_id),
                    request_id: RequestId(r.request_id),
                    correlation_id: CorrelationId(r.correlation_id),
                    operation: r.operation,
                    payload,
                }))
            }
            proto::host_to_worker::Message::Cancel(c) => Ok(WireMessage::Cancel(CancelFrame {
                protocol_version: ProtocolVersion(u32_to_u16(c.protocol_version)),
                incarnation_id: WorkerIncarnationId(c.incarnation_id),
                request_id: RequestId(c.request_id),
                correlation_id: CorrelationId(c.correlation_id),
            })),
            proto::host_to_worker::Message::Health(h) => Ok(WireMessage::Health(HealthFrame {
                protocol_version: ProtocolVersion(u32_to_u16(h.protocol_version)),
                incarnation_id: WorkerIncarnationId(h.incarnation_id),
                control_id: ControlId(h.control_id),
            })),
            proto::host_to_worker::Message::Cleanup(c) => Ok(WireMessage::Cleanup(CleanupFrame {
                protocol_version: ProtocolVersion(u32_to_u16(c.protocol_version)),
                incarnation_id: WorkerIncarnationId(c.incarnation_id),
                control_id: ControlId(c.control_id),
                run_id: c.run_id,
                object_ids: c.object_ids,
            })),
            proto::host_to_worker::Message::Shutdown(s) => {
                Ok(WireMessage::Shutdown(ShutdownFrame {
                    protocol_version: ProtocolVersion(u32_to_u16(s.protocol_version)),
                    incarnation_id: WorkerIncarnationId(s.incarnation_id),
                    control_id: ControlId(s.control_id),
                }))
            }
        }
    }
}

impl TryFrom<proto::WorkerToHost> for WireMessage {
    type Error = String;

    fn try_from(msg: proto::WorkerToHost) -> Result<Self, Self::Error> {
        let inner = msg.message.ok_or("missing message")?;
        match inner {
            proto::worker_to_host::Message::WorkerHello(h) => {
                let identity = h.identity.ok_or("missing identity")?;
                let profile = h.profile.ok_or("missing profile")?;
                Ok(WireMessage::WorkerHello(WorkerHello {
                    selected_protocol: ProtocolVersion(u32_to_u16(h.selected_protocol)),
                    identity: WorkerIdentity {
                        backend_instance_id: BackendInstanceId(identity.backend_instance_id),
                        installation_id: WorkerInstallationId(identity.installation_id),
                        incarnation_id: WorkerIncarnationId(identity.incarnation_id),
                        worker_version: identity.worker_version,
                        backend_kind: identity.backend_kind,
                        target: identity.target,
                        manifest_digest: identity.manifest_digest,
                    },
                    profile: WorkerProfile {
                        instances: profile
                            .instances
                            .into_iter()
                            .map(|i| {
                                let operation_options =
                                    serde_json::from_slice(&i.operation_options)
                                        .unwrap_or(serde_json::Value::Null);
                                WorkerInstanceProfile {
                                    backend_instance_id: BackendInstanceId(i.backend_instance_id),
                                    device_label: i.device_label,
                                    capabilities: i.capabilities,
                                    operation_options,
                                }
                            })
                            .collect(),
                    },
                }))
            }
            proto::worker_to_host::Message::Progress(p) => {
                Ok(WireMessage::Progress(ProgressFrame {
                    protocol_version: ProtocolVersion(u32_to_u16(p.protocol_version)),
                    incarnation_id: WorkerIncarnationId(p.incarnation_id),
                    request_id: RequestId(p.request_id),
                    correlation_id: CorrelationId(p.correlation_id),
                    sequence: p.sequence,
                    completed: p.completed,
                    total: p.total,
                    message: p.message,
                }))
            }
            proto::worker_to_host::Message::CancelAck(c) => {
                Ok(WireMessage::CancelAck(CancelAckFrame {
                    protocol_version: ProtocolVersion(u32_to_u16(c.protocol_version)),
                    incarnation_id: WorkerIncarnationId(c.incarnation_id),
                    request_id: RequestId(c.request_id),
                    correlation_id: CorrelationId(c.correlation_id),
                    accepted: c.accepted,
                    already_terminal: c.already_terminal,
                }))
            }
            proto::worker_to_host::Message::Terminal(t) => {
                let outcome_proto = t.outcome.ok_or("missing outcome")?;
                let outcome = match outcome_proto.outcome {
                    Some(proto::terminal_outcome::Outcome::Success(s)) => {
                        let output =
                            serde_json::from_slice(&s.output).unwrap_or(serde_json::Value::Null);
                        TerminalOutcome::Success { output }
                    }
                    Some(proto::terminal_outcome::Outcome::Cancelled(_)) => {
                        TerminalOutcome::Cancelled
                    }
                    Some(proto::terminal_outcome::Outcome::BackendError(e)) => {
                        TerminalOutcome::BackendError {
                            error: BackendExecutionError {
                                code: e.code,
                                message: e.message,
                                retryable: e.retryable,
                            },
                        }
                    }
                    None => return Err("missing terminal outcome".into()),
                };
                Ok(WireMessage::Terminal(TerminalFrame {
                    protocol_version: ProtocolVersion(u32_to_u16(t.protocol_version)),
                    incarnation_id: WorkerIncarnationId(t.incarnation_id),
                    request_id: RequestId(t.request_id),
                    correlation_id: CorrelationId(t.correlation_id),
                    outcome,
                }))
            }
            proto::worker_to_host::Message::HealthAck(h) => {
                Ok(WireMessage::HealthAck(HealthAckFrame {
                    protocol_version: ProtocolVersion(u32_to_u16(h.protocol_version)),
                    incarnation_id: WorkerIncarnationId(h.incarnation_id),
                    control_id: ControlId(h.control_id),
                    healthy: h.healthy,
                    message: h.message,
                }))
            }
            proto::worker_to_host::Message::CleanupAck(c) => {
                Ok(WireMessage::CleanupAck(CleanupAckFrame {
                    protocol_version: ProtocolVersion(u32_to_u16(c.protocol_version)),
                    incarnation_id: WorkerIncarnationId(c.incarnation_id),
                    control_id: ControlId(c.control_id),
                    released_objects: c.released_objects,
                }))
            }
            proto::worker_to_host::Message::ShutdownAck(s) => {
                Ok(WireMessage::ShutdownAck(ShutdownAckFrame {
                    protocol_version: ProtocolVersion(u32_to_u16(s.protocol_version)),
                    incarnation_id: WorkerIncarnationId(s.incarnation_id),
                    control_id: ControlId(s.control_id),
                }))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn host_hello_roundtrip() {
        let original = WireMessage::HostHello(HostHello {
            supported_protocols: reimagine_backend_worker_protocol::ProtocolRange::new(1, 2),
        });
        let proto_msg: proto::HostToWorker = (&original).try_into().unwrap();
        let back: WireMessage = proto_msg.try_into().unwrap();
        assert_eq!(format!("{:?}", original), format!("{:?}", back));
    }

    #[test]
    fn request_roundtrip() {
        let original = WireMessage::Request(RequestFrame {
            protocol_version: ProtocolVersion(1),
            incarnation_id: WorkerIncarnationId("inc-1".into()),
            request_id: RequestId("req-1".into()),
            correlation_id: CorrelationId("corr-1".into()),
            operation: "echo".into(),
            payload: json!({"value": 42}),
        });
        let proto_msg: proto::HostToWorker = (&original).try_into().unwrap();
        let back: WireMessage = proto_msg.try_into().unwrap();
        assert_eq!(format!("{:?}", original), format!("{:?}", back));
    }

    #[test]
    fn terminal_success_roundtrip() {
        let original = WireMessage::Terminal(TerminalFrame {
            protocol_version: ProtocolVersion(1),
            incarnation_id: WorkerIncarnationId("inc-1".into()),
            request_id: RequestId("req-1".into()),
            correlation_id: CorrelationId("corr-1".into()),
            outcome: TerminalOutcome::Success {
                output: json!({"ok": true}),
            },
        });
        let proto_msg: proto::WorkerToHost = (&original).try_into().unwrap();
        let back: WireMessage = proto_msg.try_into().unwrap();
        assert_eq!(format!("{:?}", original), format!("{:?}", back));
    }

    #[test]
    fn terminal_cancelled_roundtrip() {
        let original = WireMessage::Terminal(TerminalFrame {
            protocol_version: ProtocolVersion(1),
            incarnation_id: WorkerIncarnationId("inc-1".into()),
            request_id: RequestId("req-1".into()),
            correlation_id: CorrelationId("corr-1".into()),
            outcome: TerminalOutcome::Cancelled,
        });
        let proto_msg: proto::WorkerToHost = (&original).try_into().unwrap();
        let back: WireMessage = proto_msg.try_into().unwrap();
        assert_eq!(format!("{:?}", original), format!("{:?}", back));
    }

    #[test]
    fn terminal_error_roundtrip() {
        let original = WireMessage::Terminal(TerminalFrame {
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
        let proto_msg: proto::WorkerToHost = (&original).try_into().unwrap();
        let back: WireMessage = proto_msg.try_into().unwrap();
        assert_eq!(format!("{:?}", original), format!("{:?}", back));
    }

    #[test]
    fn progress_roundtrip() {
        let original = WireMessage::Progress(ProgressFrame {
            protocol_version: ProtocolVersion(1),
            incarnation_id: WorkerIncarnationId("inc-1".into()),
            request_id: RequestId("req-1".into()),
            correlation_id: CorrelationId("corr-1".into()),
            sequence: 1,
            completed: 1,
            total: Some(10),
            message: Some("step 1".into()),
        });
        let proto_msg: proto::WorkerToHost = (&original).try_into().unwrap();
        let back: WireMessage = proto_msg.try_into().unwrap();
        assert_eq!(format!("{:?}", original), format!("{:?}", back));
    }

    #[test]
    fn cancel_roundtrip() {
        let original = WireMessage::Cancel(CancelFrame {
            protocol_version: ProtocolVersion(1),
            incarnation_id: WorkerIncarnationId("inc-1".into()),
            request_id: RequestId("req-1".into()),
            correlation_id: CorrelationId("corr-1".into()),
        });
        let proto_msg: proto::HostToWorker = (&original).try_into().unwrap();
        let back: WireMessage = proto_msg.try_into().unwrap();
        assert_eq!(format!("{:?}", original), format!("{:?}", back));
    }

    #[test]
    fn health_roundtrip() {
        let original = WireMessage::Health(HealthFrame {
            protocol_version: ProtocolVersion(1),
            incarnation_id: WorkerIncarnationId("inc-1".into()),
            control_id: ControlId("health-1".into()),
        });
        let proto_msg: proto::HostToWorker = (&original).try_into().unwrap();
        let back: WireMessage = proto_msg.try_into().unwrap();
        assert_eq!(format!("{:?}", original), format!("{:?}", back));
    }

    #[test]
    fn health_ack_roundtrip() {
        let original = WireMessage::HealthAck(HealthAckFrame {
            protocol_version: ProtocolVersion(1),
            incarnation_id: WorkerIncarnationId("inc-1".into()),
            control_id: ControlId("health-1".into()),
            healthy: true,
            message: Some("all good".into()),
        });
        let proto_msg: proto::WorkerToHost = (&original).try_into().unwrap();
        let back: WireMessage = proto_msg.try_into().unwrap();
        assert_eq!(format!("{:?}", original), format!("{:?}", back));
    }

    #[test]
    fn shutdown_roundtrip() {
        let original = WireMessage::Shutdown(ShutdownFrame {
            protocol_version: ProtocolVersion(1),
            incarnation_id: WorkerIncarnationId("inc-1".into()),
            control_id: ControlId("shutdown-1".into()),
        });
        let proto_msg: proto::HostToWorker = (&original).try_into().unwrap();
        let back: WireMessage = proto_msg.try_into().unwrap();
        assert_eq!(format!("{:?}", original), format!("{:?}", back));
    }

    #[test]
    fn shutdown_ack_roundtrip() {
        let original = WireMessage::ShutdownAck(ShutdownAckFrame {
            protocol_version: ProtocolVersion(1),
            incarnation_id: WorkerIncarnationId("inc-1".into()),
            control_id: ControlId("shutdown-1".into()),
        });
        let proto_msg: proto::WorkerToHost = (&original).try_into().unwrap();
        let back: WireMessage = proto_msg.try_into().unwrap();
        assert_eq!(format!("{:?}", original), format!("{:?}", back));
    }

    #[test]
    fn cleanup_roundtrip() {
        let original = WireMessage::Cleanup(CleanupFrame {
            protocol_version: ProtocolVersion(1),
            incarnation_id: WorkerIncarnationId("inc-1".into()),
            control_id: ControlId("cleanup-1".into()),
            run_id: Some("run-1".into()),
            object_ids: vec!["obj-1".into(), "obj-2".into()],
        });
        let proto_msg: proto::HostToWorker = (&original).try_into().unwrap();
        let back: WireMessage = proto_msg.try_into().unwrap();
        assert_eq!(format!("{:?}", original), format!("{:?}", back));
    }

    #[test]
    fn cleanup_ack_roundtrip() {
        let original = WireMessage::CleanupAck(CleanupAckFrame {
            protocol_version: ProtocolVersion(1),
            incarnation_id: WorkerIncarnationId("inc-1".into()),
            control_id: ControlId("cleanup-1".into()),
            released_objects: 1,
        });
        let proto_msg: proto::WorkerToHost = (&original).try_into().unwrap();
        let back: WireMessage = proto_msg.try_into().unwrap();
        assert_eq!(format!("{:?}", original), format!("{:?}", back));
    }

    #[test]
    fn cancel_ack_roundtrip() {
        let original = WireMessage::CancelAck(CancelAckFrame {
            protocol_version: ProtocolVersion(1),
            incarnation_id: WorkerIncarnationId("inc-1".into()),
            request_id: RequestId("req-1".into()),
            correlation_id: CorrelationId("corr-1".into()),
            accepted: true,
            already_terminal: false,
        });
        let proto_msg: proto::WorkerToHost = (&original).try_into().unwrap();
        let back: WireMessage = proto_msg.try_into().unwrap();
        assert_eq!(format!("{:?}", original), format!("{:?}", back));
    }

    #[test]
    fn worker_hello_roundtrip() {
        let original = WireMessage::WorkerHello(WorkerHello {
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
                    capabilities: vec!["load_bundle".into(), "text_encode".into()],
                    operation_options: json!({"batch_size": 1}),
                }],
            },
        });
        let proto_msg: proto::WorkerToHost = (&original).try_into().unwrap();
        let back: WireMessage = proto_msg.try_into().unwrap();
        assert_eq!(format!("{:?}", original), format!("{:?}", back));
    }
}
