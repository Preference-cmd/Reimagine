use serde::{Deserialize, Serialize};

/// A single target exposed by the catalog after TUF verification
/// and compatibility filtering.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CatalogTarget {
    /// The target path in the TUF targets metadata.
    pub path: String,
    /// SHA-256 hex digest of the target content.
    pub sha256: String,
    /// Length of the target in bytes.
    pub length: u64,
    /// Custom metadata from TUF targets (OS, arch, kind, protocol, etc.).
    pub custom: TargetCustomMetadata,
    /// Download URL (resolved from the base catalog URL and target path).
    pub download_url: String,
}

impl CatalogTarget {
    #[must_use]
    pub fn expected_identity(&self) -> crate::ExpectedWorkerIdentity {
        crate::ExpectedWorkerIdentity {
            backend_instance_id: crate::BackendInstanceId(self.custom.backend_instance_id.clone()),
            installation_id: crate::WorkerInstallationId(self.custom.installation_id.clone()),
            backend_kind: self.custom.worker_kind.clone(),
            target: self.custom.target.clone(),
            manifest_digest: self.custom.manifest_digest.clone(),
        }
    }
}

/// Compatibility metadata attached to each target in the TUF targets metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetCustomMetadata {
    pub version: String,
    pub installation_id: String,
    pub backend_instance_id: String,
    pub os: String,
    pub arch: String,
    pub worker_kind: String,
    pub protocol_version_min: u16,
    pub protocol_version_max: u16,
    pub package_format: String,
    pub min_runtime_version: Option<String>,
    pub target: String,
    pub manifest_digest: String,
}

/// Host information used for compatibility filtering.
#[derive(Clone, Debug)]
pub struct HostInfo {
    pub os: String,
    pub arch: String,
    pub supported_protocol_range: (u16, u16),
}

/// Compatibility filter that checks whether a target matches host constraints.
///
/// Unknown fields are treated as incompatible (fail-closed).
#[derive(Clone, Debug)]
pub struct CompatibilityFilter {
    host: HostInfo,
}

impl CompatibilityFilter {
    #[must_use]
    pub fn new(host: HostInfo) -> Self {
        Self { host }
    }

    /// Returns `true` if the target is compatible with this host.
    ///
    /// All of the following must match:
    /// - OS (exact match, case-insensitive)
    /// - Architecture (exact match, case-insensitive)
    /// - Worker kind
    /// - Protocol range overlap with host
    /// - Package format
    /// - Minimum runtime version (if specified)
    #[must_use]
    pub fn is_compatible(&self, custom: &TargetCustomMetadata) -> bool {
        if !self.os_matches(&custom.os) {
            return false;
        }
        if !self.arch_matches(&custom.arch) {
            return false;
        }
        if !self.worker_kind_matches(&custom.worker_kind) {
            return false;
        }
        if !self.protocol_overlaps(custom.protocol_version_min, custom.protocol_version_max) {
            return false;
        }
        if !self.package_format_matches(&custom.package_format) {
            return false;
        }
        if !self.runtime_version_matches(custom.min_runtime_version.as_deref()) {
            return false;
        }
        true
    }

    fn os_matches(&self, target_os: &str) -> bool {
        self.host.os.eq_ignore_ascii_case(target_os)
    }

    fn arch_matches(&self, target_arch: &str) -> bool {
        self.host.arch.eq_ignore_ascii_case(target_arch)
    }

    fn worker_kind_matches(&self, target_kind: &str) -> bool {
        // Currently only "burn" workers are supported
        target_kind.eq_ignore_ascii_case("burn")
    }

    fn protocol_overlaps(&self, target_min: u16, target_max: u16) -> bool {
        let (host_min, host_max) = self.host.supported_protocol_range;
        let min = host_min.max(target_min);
        let max = host_max.min(target_max);
        min <= max
    }

    fn package_format_matches(&self, target_format: &str) -> bool {
        // Currently only tar.gz archives
        target_format.eq_ignore_ascii_case("tar.gz")
    }

    fn runtime_version_matches(&self, minimum: Option<&str>) -> bool {
        let Some(minimum) = minimum else {
            return true;
        };
        version_tuple(env!("CARGO_PKG_VERSION")) >= version_tuple(minimum)
    }
}

fn version_tuple(version: &str) -> (u64, u64, u64) {
    let mut parts = version
        .split('.')
        .map(|part| part.parse::<u64>().unwrap_or(u64::MAX));
    (
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
    )
}

/// Common TUF metadata role names.
#[allow(dead_code)]
pub(crate) mod roles {
    pub const ROOT: &str = "root";
    pub const TIMESTAMP: &str = "timestamp";
    pub const SNAPSHOT: &str = "snapshot";
    pub const TARGETS: &str = "targets";
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_filter(os: &str, arch: &str) -> CompatibilityFilter {
        CompatibilityFilter::new(HostInfo {
            os: os.to_string(),
            arch: arch.to_string(),
            supported_protocol_range: (1, 3),
        })
    }

    fn make_custom() -> TargetCustomMetadata {
        TargetCustomMetadata {
            version: "1.0.0".to_owned(),
            installation_id: "burn-wgpu-v1".to_owned(),
            backend_instance_id: "burn:wgpu:default".to_owned(),
            os: "darwin".to_string(),
            arch: "aarch64".to_string(),
            worker_kind: "burn".to_string(),
            protocol_version_min: 1,
            protocol_version_max: 3,
            package_format: "tar.gz".to_string(),
            min_runtime_version: None,
            target: "aarch64-apple-darwin".to_owned(),
            manifest_digest: "digest".to_owned(),
        }
    }

    #[test]
    fn compatible_os_arch_matches() {
        let filter = make_filter("darwin", "aarch64");
        assert!(filter.is_compatible(&make_custom()));
    }

    #[test]
    fn incompatible_os_rejected() {
        let filter = make_filter("linux", "aarch64");
        assert!(!filter.is_compatible(&make_custom()));
    }

    #[test]
    fn incompatible_arch_rejected() {
        let filter = make_filter("darwin", "x86_64");
        assert!(!filter.is_compatible(&make_custom()));
    }

    #[test]
    fn case_insensitive_os_matches() {
        let filter = make_filter("Darwin", "aarch64");
        assert!(filter.is_compatible(&make_custom()));
    }

    #[test]
    fn protocol_no_overlap_rejected() {
        let filter = CompatibilityFilter::new(HostInfo {
            os: "darwin".to_string(),
            arch: "aarch64".to_string(),
            supported_protocol_range: (4, 5),
        });
        assert!(!filter.is_compatible(&make_custom()));
    }

    #[test]
    fn wrong_worker_kind_rejected() {
        let filter = make_filter("darwin", "aarch64");
        let custom = TargetCustomMetadata {
            worker_kind: "candle".to_string(),
            ..make_custom()
        };
        assert!(!filter.is_compatible(&custom));
    }

    #[test]
    fn wrong_package_format_rejected() {
        let filter = make_filter("darwin", "aarch64");
        let custom = TargetCustomMetadata {
            package_format: "zip".to_string(),
            ..make_custom()
        };
        assert!(!filter.is_compatible(&custom));
    }

    #[test]
    fn unsupported_minimum_runtime_is_rejected() {
        let filter = make_filter("darwin", "aarch64");
        let custom = TargetCustomMetadata {
            min_runtime_version: Some("999.0.0".to_owned()),
            ..make_custom()
        };
        assert!(!filter.is_compatible(&custom));
    }

    #[test]
    fn unknown_custom_metadata_field_is_rejected() {
        let json = serde_json::json!({
            "os": "darwin",
            "version": "1.0.0",
            "installation_id": "burn-wgpu-v1",
            "backend_instance_id": "burn:wgpu:default",
            "arch": "aarch64",
            "worker_kind": "burn",
            "protocol_version_min": 1,
            "protocol_version_max": 3,
            "package_format": "tar.gz",
            "min_runtime_version": null,
            "target": "aarch64-apple-darwin",
            "manifest_digest": "digest",
            "mystery_compatibility": true
        });
        assert!(serde_json::from_value::<TargetCustomMetadata>(json).is_err());
    }
}
