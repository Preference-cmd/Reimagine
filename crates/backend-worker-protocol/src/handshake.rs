//! Protocol versioning for the worker contract.
//!
//! The version travels once, in the `Hello` response. A host that cannot
//! work with the reported version rejects the worker instead of
//! negotiating down: there is no cross-version wire compatibility.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{WorkerIdentity, WorkerProfile};

/// The protocol version this workspace speaks.
pub const WORKER_PROTOCOL_VERSION: ProtocolVersion = ProtocolVersion(2);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProtocolVersion(pub u16);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProtocolRange {
    pub minimum: ProtocolVersion,
    pub maximum: ProtocolVersion,
}

impl ProtocolRange {
    #[must_use]
    pub const fn new(minimum: u16, maximum: u16) -> Self {
        Self {
            minimum: ProtocolVersion(minimum),
            maximum: ProtocolVersion(maximum),
        }
    }

    /// Whether `version` lies inside this range, inclusive on both ends.
    #[must_use]
    pub fn contains(&self, version: ProtocolVersion) -> bool {
        version >= self.minimum && version <= self.maximum
    }
}

impl fmt::Display for ProtocolRange {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}..={}", self.minimum.0, self.maximum.0)
    }
}

/// The worker's self-description returned by the `Hello` RPC: the wire
/// protocol version it speaks, its identity, and the instance profile it
/// serves.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkerHello {
    pub protocol: ProtocolVersion,
    pub identity: WorkerIdentity,
    pub profile: WorkerProfile,
}

#[cfg(test)]
mod tests {
    use super::{ProtocolRange, ProtocolVersion, WORKER_PROTOCOL_VERSION};

    #[test]
    fn range_contains_is_inclusive_on_both_ends() {
        let range = ProtocolRange::new(2, 2);
        assert!(range.contains(ProtocolVersion(2)));
        assert!(!range.contains(ProtocolVersion(1)));
        assert!(!range.contains(ProtocolVersion(3)));
    }

    #[test]
    fn workspace_speaks_protocol_v2() {
        assert_eq!(WORKER_PROTOCOL_VERSION.0, 2);
    }
}
