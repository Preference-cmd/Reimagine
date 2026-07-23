use serde::{Deserialize, Serialize};

use crate::ExpectedWorkerIdentity;

/// Limits for archive extraction safety.
#[derive(Clone, Debug)]
pub struct ExtractionLimits {
    /// Maximum number of entries in the archive.
    pub max_entries: usize,
    /// Maximum total expanded size in bytes.
    pub max_expanded_size: u64,
    /// Maximum size of a single entry in bytes.
    pub max_entry_size: u64,
}

impl Default for ExtractionLimits {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            max_expanded_size: 2 * 1024 * 1024 * 1024, // 2 GiB
            max_entry_size: 512 * 1024 * 1024,         // 512 MiB
        }
    }
}

/// The manifest embedded inside a worker package archive.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackageManifest {
    /// Schema version for forward compatibility.
    pub schema_version: u16,
    /// Kind identifier (e.g., "burn-worker").
    pub package_kind: String,
    /// Human-readable package version (e.g., "0.1.0").
    #[serde(default = "default_version")]
    pub version: String,
    /// The expected worker identity for this package.
    pub identity: ExpectedWorkerIdentity,
    /// File entries in the package.
    pub files: Vec<PackageFileEntry>,
    /// Required disk space in bytes (extracted).
    pub required_size: u64,
    /// Number of entries in the archive.
    pub required_entries: usize,
}

fn default_version() -> String {
    "0.0.0".to_string()
}

/// A single file entry in the package manifest.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackageFileEntry {
    /// Relative path within the package.
    pub path: String,
    /// SHA-256 hex digest.
    pub sha256: String,
    /// File size in bytes.
    pub size: u64,
    /// POSIX permissions (octal, e.g., 0o755).
    pub mode: u32,
    /// Whether this is the main executable.
    #[serde(default)]
    pub executable: bool,
}
