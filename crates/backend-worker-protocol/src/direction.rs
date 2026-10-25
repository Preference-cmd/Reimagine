use std::fmt;

use crate::WireMessage;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MessageSender {
    Host,
    Worker,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProtocolViolation {
    InvalidDirection {
        kind: &'static str,
        sender: MessageSender,
    },
}

impl fmt::Display for ProtocolViolation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "worker protocol violation: {self:?}")
    }
}

impl std::error::Error for ProtocolViolation {}

pub fn validate_message_direction(
    message: &WireMessage,
    sender: MessageSender,
) -> Result<(), ProtocolViolation> {
    let allowed = match sender {
        MessageSender::Host => matches!(
            message,
            WireMessage::Ping { .. }
                | WireMessage::HostHello(_)
                | WireMessage::Request(_)
                | WireMessage::Cancel(_)
                | WireMessage::Health(_)
                | WireMessage::Cleanup(_)
                | WireMessage::Shutdown(_)
        ),
        MessageSender::Worker => matches!(
            message,
            WireMessage::Ping { .. }
                | WireMessage::WorkerHello(_)
                | WireMessage::Progress(_)
                | WireMessage::CancelAck(_)
                | WireMessage::Terminal(_)
                | WireMessage::HealthAck(_)
                | WireMessage::CleanupAck(_)
                | WireMessage::ShutdownAck(_)
        ),
    };
    if allowed {
        return Ok(());
    }
    Err(ProtocolViolation::InvalidDirection {
        kind: message.kind(),
        sender,
    })
}
