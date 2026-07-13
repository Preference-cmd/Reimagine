use serde::{Deserialize, Serialize};

use crate::{
    CancelAckFrame, CancelFrame, CleanupAckFrame, CleanupFrame, HealthAckFrame, HealthFrame,
    HostHello, ProgressFrame, RequestFrame, ShutdownAckFrame, ShutdownFrame, TerminalFrame,
    WorkerHello,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "body", rename_all = "snake_case")]
pub enum WireMessage {
    Ping { nonce: String },
    HostHello(HostHello),
    WorkerHello(WorkerHello),
    Request(RequestFrame),
    Progress(ProgressFrame),
    Cancel(CancelFrame),
    CancelAck(CancelAckFrame),
    Terminal(TerminalFrame),
    Health(HealthFrame),
    HealthAck(HealthAckFrame),
    Cleanup(CleanupFrame),
    CleanupAck(CleanupAckFrame),
    Shutdown(ShutdownFrame),
    ShutdownAck(ShutdownAckFrame),
}

impl WireMessage {
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Ping { .. } => "ping",
            Self::HostHello(_) => "host_hello",
            Self::WorkerHello(_) => "worker_hello",
            Self::Request(_) => "request",
            Self::Progress(_) => "progress",
            Self::Cancel(_) => "cancel",
            Self::CancelAck(_) => "cancel_ack",
            Self::Terminal(_) => "terminal",
            Self::Health(_) => "health",
            Self::HealthAck(_) => "health_ack",
            Self::Cleanup(_) => "cleanup",
            Self::CleanupAck(_) => "cleanup_ack",
            Self::Shutdown(_) => "shutdown",
            Self::ShutdownAck(_) => "shutdown_ack",
        }
    }

    pub(crate) fn is_known_kind(kind: &str) -> bool {
        matches!(
            kind,
            "ping"
                | "host_hello"
                | "worker_hello"
                | "request"
                | "progress"
                | "cancel"
                | "cancel_ack"
                | "terminal"
                | "health"
                | "health_ack"
                | "cleanup"
                | "cleanup_ack"
                | "shutdown"
                | "shutdown_ack"
        )
    }
}
