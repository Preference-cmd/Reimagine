use std::fmt;

use serde::{Deserialize, Serialize};

use crate::WorkerIdentity;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProtocolVersion(pub u16);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProtocolRange {
    pub minimum: ProtocolVersion,
    pub maximum: ProtocolVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HostHello {
    pub supported_protocols: ProtocolRange,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkerHello {
    pub selected_protocol: ProtocolVersion,
    pub identity: WorkerIdentity,
}

impl ProtocolRange {
    #[must_use]
    pub const fn new(minimum: u16, maximum: u16) -> Self {
        Self {
            minimum: ProtocolVersion(minimum),
            maximum: ProtocolVersion(maximum),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HandshakeError {
    InvalidRange(ProtocolRange),
    NoCompatibleVersion {
        host: ProtocolRange,
        worker: ProtocolRange,
    },
}

impl fmt::Display for HandshakeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRange(range) => write!(
                formatter,
                "invalid protocol range {}..={}",
                range.minimum.0, range.maximum.0
            ),
            Self::NoCompatibleVersion { host, worker } => write!(
                formatter,
                "no compatible protocol version between host {}..={} and worker {}..={}",
                host.minimum.0, host.maximum.0, worker.minimum.0, worker.maximum.0
            ),
        }
    }
}

impl std::error::Error for HandshakeError {}

pub fn negotiate_protocol(
    host: ProtocolRange,
    worker: ProtocolRange,
) -> Result<ProtocolVersion, HandshakeError> {
    if host.minimum > host.maximum {
        return Err(HandshakeError::InvalidRange(host));
    }
    if worker.minimum > worker.maximum {
        return Err(HandshakeError::InvalidRange(worker));
    }
    let minimum = host.minimum.max(worker.minimum);
    let maximum = host.maximum.min(worker.maximum);
    if minimum > maximum {
        return Err(HandshakeError::NoCompatibleVersion { host, worker });
    }
    Ok(maximum)
}
