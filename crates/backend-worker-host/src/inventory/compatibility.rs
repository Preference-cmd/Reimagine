//! Compatibility of installed worker packages with this host's worker
//! protocol.
//!
//! The protocol version travels in the package manifest (and therefore
//! in every installation record). Packages written before the field
//! existed default to v1, which the host rejects with a reinstall
//! diagnostic instead of failing later at launch.

use std::fmt;

use reimagine_backend_worker_protocol::ProtocolRange;

use crate::inventory::records::InstallationRecord;

/// Why an installed worker package fails the host's protocol check.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProtocolIncompatibility {
    /// Protocol range the package declares (or defaulted to).
    pub package_range: (u16, u16),
    /// Protocol range this host requires.
    pub host_range: (u16, u16),
}

impl fmt::Display for ProtocolIncompatibility {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "installed worker speaks protocol {}..={} but this host requires protocol {}..={}: reinstall the worker",
            self.package_range.0, self.package_range.1, self.host_range.0, self.host_range.1
        )
    }
}

impl std::error::Error for ProtocolIncompatibility {}

/// Check an installed worker package's declared protocol range against
/// the host's range. The ranges must overlap.
pub fn check_installed_protocol(
    package_min: u16,
    package_max: u16,
    host: ProtocolRange,
) -> Result<(), ProtocolIncompatibility> {
    let overlap_min = package_min.max(host.minimum.0);
    let overlap_max = package_max.min(host.maximum.0);
    if overlap_min <= overlap_max {
        Ok(())
    } else {
        Err(ProtocolIncompatibility {
            package_range: (package_min, package_max),
            host_range: (host.minimum.0, host.maximum.0),
        })
    }
}

/// Convenience wrapper over [`check_installed_protocol`] for a durable
/// installation record.
pub fn check_record_protocol(
    record: &InstallationRecord,
    host: ProtocolRange,
) -> Result<(), ProtocolIncompatibility> {
    check_installed_protocol(
        record.protocol_version_min,
        record.protocol_version_max,
        host,
    )
}

#[cfg(test)]
mod tests {
    use super::{check_installed_protocol, check_record_protocol};
    use crate::inventory::records::InstallationRecord;
    use reimagine_backend_worker_protocol::ProtocolRange;

    #[test]
    fn v1_only_package_is_rejected_with_reinstall_diagnostic() {
        let error =
            check_installed_protocol(1, 1, ProtocolRange::new(2, 2)).expect_err("v1 rejected");
        assert_eq!(error.package_range, (1, 1));
        assert_eq!(error.host_range, (2, 2));
        assert!(
            error.to_string().contains("reinstall the worker"),
            "diagnostic must tell the operator to reinstall: {error}"
        );
    }

    #[test]
    fn overlapping_ranges_are_accepted() {
        check_installed_protocol(1, 3, ProtocolRange::new(2, 2)).expect("overlap");
        check_installed_protocol(2, 2, ProtocolRange::new(2, 2)).expect("exact match");
    }

    #[test]
    fn record_defaults_parse_as_v1_and_are_rejected() {
        // A record written before the protocol fields existed deserializes
        // with the v1 default.
        let legacy = serde_json::from_value::<InstallationRecord>(serde_json::json!({
            "installation_id": "worker-v1",
            "version": "1.0.0",
            "identity": {
                "backend_instance_id": "burn:wgpu:default",
                "installation_id": "worker-v1",
                "backend_kind": "burn",
                "target": "test-target",
                "manifest_digest": "digest",
            },
            "installed_at": "2026-01-01T00:00:00Z",
            "install_path": "/installed/worker-v1",
            "manifest_profile": null,
        }))
        .expect("legacy record parses");
        assert_eq!(legacy.protocol_version_min, 1);
        assert_eq!(legacy.protocol_version_max, 1);
        let error = check_record_protocol(&legacy, ProtocolRange::new(2, 2))
            .expect_err("legacy record rejected");
        assert!(
            error.to_string().contains("reinstall the worker"),
            "got: {error}"
        );
    }
}
