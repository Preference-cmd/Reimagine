use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{ExpectedWorkerIdentity, WorkerInstallationId};

/// A durable record of a successfully installed worker.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstallationRecord {
    /// Unique installation identifier (stable across versions).
    pub installation_id: WorkerInstallationId,
    /// Version string from the package.
    pub version: String,
    /// Expected worker identity (matches the identity in the package manifest).
    pub identity: ExpectedWorkerIdentity,
    /// When the installation completed.
    pub installed_at: DateTime<Utc>,
    /// Filesystem path to the installed worker directory.
    pub install_path: String,
    /// Profile information from the package manifest (optional).
    pub manifest_profile: Option<crate::WorkerInstanceProfile>,
}

/// A snapshot of the entire inventory at a point in time.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InventorySnapshot {
    /// All installation records.
    pub records: Vec<InstallationRecord>,
}

/// Index file format for the inventory.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InventoryIndex {
    /// Version stamp for concurrent-modification detection.
    pub version: u64,
    /// List of installed installation IDs.
    pub installation_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VersionPolicy {
    Exact(String),
    Latest,
}
