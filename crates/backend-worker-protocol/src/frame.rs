use serde::{Deserialize, Serialize};

use crate::{
    CancelAckFrame, CancelFrame, HostHello, ProgressFrame, RequestFrame, TerminalFrame, WorkerHello,
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
}

impl WireMessage {
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
        )
    }
}
