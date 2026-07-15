use std::fs;
use std::io::Write;

use tracing;

use super::error::{InventoryError, InventoryResult};
use super::records::{InstallationRecord, InventoryIndex, InventorySnapshot, VersionPolicy};
use crate::WorkerStorePaths;

/// Durable inventory store for worker installation records.
///
/// Storage layout:
/// ```text
/// {workers}/inventory/
///   index.json     — version-stamped list of installation IDs
///   {id}.json      — one file per installation record
/// ```
///
/// The index is the authoritative entry point. Individual record
/// files are the source of truth for each installation's metadata.
/// If the index is corrupt, it can be rebuilt from record files.
pub struct InventoryStore {
    store_paths: WorkerStorePaths,
}

impl InventoryStore {
    /// Create a new inventory store backed by the given paths.
    #[must_use]
    pub fn new(store_paths: WorkerStorePaths) -> Self {
        Self { store_paths }
    }

    #[must_use]
    pub fn store_paths(&self) -> &WorkerStorePaths {
        &self.store_paths
    }

    /// Return a snapshot of all installed records.
    pub fn list(&self) -> InventoryResult<InventorySnapshot> {
        let ids = self.read_index()?;
        let mut records = Vec::with_capacity(ids.installation_ids.len());
        let mut needs_rebuild = false;

        for id in &ids.installation_ids {
            match self.read_record(id) {
                Ok(Some(record)) => records.push(record),
                Ok(None) => {
                    // Record file missing — skip (will be cleaned up on next rebuild)
                    tracing::warn!("inventory record `{id}` referenced in index but file missing");
                    needs_rebuild = true;
                }
                Err(e) => {
                    tracing::warn!("failed to read inventory record `{id}`: {e}");
                    self.quarantine_record(id)?;
                    needs_rebuild = true;
                }
            }
        }

        if needs_rebuild {
            self.rebuild_from_records()?;
        }

        Ok(InventorySnapshot { records })
    }

    /// Return a single installation record.
    pub fn get(&self, installation_id: &str) -> InventoryResult<InstallationRecord> {
        match self.read_record(installation_id)? {
            Some(record) => Ok(record),
            None => Err(InventoryError::NotFound {
                installation_id: installation_id.to_string(),
            }),
        }
    }

    pub fn resolve_preference(
        &self,
        backend_instance_id: &str,
        policy: &VersionPolicy,
    ) -> InventoryResult<InstallationRecord> {
        let mut candidates = self
            .list()?
            .records
            .into_iter()
            .filter(|record| record.identity.backend_instance_id.0 == backend_instance_id)
            .filter(|record| match policy {
                VersionPolicy::Exact(version) => &record.version == version,
                VersionPolicy::Latest => true,
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|record| version_tuple(&record.version));
        candidates.pop().ok_or_else(|| InventoryError::NotFound {
            installation_id: format!("{backend_instance_id}:{policy:?}"),
        })
    }

    /// Add a new installation record.
    ///
    /// Atomically writes the record file, then updates the index.
    pub fn add(&self, record: &InstallationRecord) -> InventoryResult<()> {
        let id = &record.installation_id.0;
        self.ensure_inventory_dir()?;

        // Write individual record file
        self.write_record(id, record)?;

        // Update index
        let mut index = self.read_index_allow_missing()?;
        if !index.installation_ids.contains(id) {
            index.installation_ids.push(id.clone());
        }
        index.version += 1;
        self.write_index(&index)?;

        tracing::info!("added inventory record for `{id}`");
        Ok(())
    }

    /// Remove an installation record.
    ///
    /// Removes the record file and updates the index.
    pub fn remove(&self, installation_id: &str) -> InventoryResult<()> {
        let mut index = self.read_index()?;

        // Remove from index
        index.installation_ids.retain(|id| id != installation_id);
        index.version += 1;

        // Remove record file (best-effort)
        let record_path = self.store_paths.inventory_record_path(installation_id);
        let _ = fs::remove_file(&record_path);

        // Write updated index
        self.write_index(&index)?;

        tracing::info!("removed inventory record for `{installation_id}`");
        Ok(())
    }

    /// Rebuild the inventory index from individual record files.
    ///
    /// Called when the index is detected as corrupt. Never silently
    /// treats corrupt as empty — instead, recovers as many records
    /// as possible from the individual record files.
    pub fn rebuild_from_records(&self) -> InventoryResult<InventoryIndex> {
        self.ensure_inventory_dir()?;

        let inventory_dir = self.store_paths.inventory_dir();
        let mut ids = Vec::new();

        if inventory_dir.exists() {
            for entry in fs::read_dir(&inventory_dir).map_err(|e| InventoryError::Io {
                path: inventory_dir.clone(),
                message: format!("failed to read inventory directory: {e}"),
            })? {
                let entry = entry.map_err(|e| InventoryError::Io {
                    path: inventory_dir.clone(),
                    message: format!("failed to read directory entry: {e}"),
                })?;
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "json")
                    && path.file_stem().is_some_and(|stem| stem != "index")
                    && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                {
                    ids.push(stem.to_string());
                }
            }
        }

        let index = InventoryIndex {
            version: 1,
            installation_ids: ids,
        };

        // Validate each record
        let mut valid_ids = Vec::new();
        for id in &index.installation_ids {
            match self.read_record(id) {
                Ok(Some(_)) => valid_ids.push(id.clone()),
                Ok(None) => {} // record file missing from disk
                Err(_) => {
                    self.quarantine_record(id)?;
                }
            }
        }

        let rebuilt = InventoryIndex {
            version: 1,
            installation_ids: valid_ids,
        };

        self.write_index(&rebuilt)?;

        tracing::info!(
            "rebuilt inventory index with {} records",
            rebuilt.installation_ids.len()
        );

        Ok(rebuilt)
    }

    // ── Internal helpers ──────────────────────────────────────────

    fn ensure_inventory_dir(&self) -> InventoryResult<()> {
        let dir = self.store_paths.inventory_dir();
        fs::create_dir_all(&dir).map_err(|e| InventoryError::Io {
            path: dir,
            message: format!("failed to create inventory directory: {e}"),
        })
    }

    fn read_index(&self) -> InventoryResult<InventoryIndex> {
        let path = self.store_paths.inventory_index_path();
        if !path.exists() {
            // If no index exists, try to rebuild from record files
            return self.rebuild_from_records();
        }

        let content = fs::read_to_string(&path).map_err(|e| InventoryError::Io {
            path: path.clone(),
            message: e.to_string(),
        })?;

        match serde_json::from_str::<InventoryIndex>(&content) {
            Ok(index) => Ok(index),
            Err(e) => {
                // Index is corrupt — try to rebuild
                tracing::warn!(
                    "inventory index corrupt at `{}`: {e}; rebuilding",
                    path.display()
                );
                self.rebuild_from_records()
            }
        }
    }

    fn read_index_allow_missing(&self) -> InventoryResult<InventoryIndex> {
        let path = self.store_paths.inventory_index_path();
        if !path.exists() {
            return Ok(InventoryIndex {
                version: 0,
                installation_ids: Vec::new(),
            });
        }

        let content = fs::read_to_string(&path).map_err(|e| InventoryError::Io {
            path: path.clone(),
            message: e.to_string(),
        })?;

        serde_json::from_str(&content).map_err(|_e| InventoryError::Corrupt { path: path.clone() })
    }

    fn write_index(&self, index: &InventoryIndex) -> InventoryResult<()> {
        self.ensure_inventory_dir()?;
        let path = self.store_paths.inventory_index_path();
        let tmp_path = path.with_extension("tmp");

        let content = serde_json::to_string_pretty(index).map_err(|e| InventoryError::Json {
            path: Some(path.clone()),
            message: e.to_string(),
        })?;

        atomic_write(&tmp_path, &path, content.as_bytes())
    }

    fn read_record(&self, id: &str) -> InventoryResult<Option<InstallationRecord>> {
        let path = self.store_paths.inventory_record_path(id);
        if !path.exists() {
            return Ok(None);
        }

        let content = fs::read_to_string(&path).map_err(|e| InventoryError::Io {
            path: path.clone(),
            message: e.to_string(),
        })?;

        serde_json::from_str(&content)
            .map(Some)
            .map_err(|_e| InventoryError::Corrupt { path })
    }

    fn write_record(&self, id: &str, record: &InstallationRecord) -> InventoryResult<()> {
        self.ensure_inventory_dir()?;
        let path = self.store_paths.inventory_record_path(id);
        let tmp_path = path.with_extension("tmp");

        let content = serde_json::to_string_pretty(record).map_err(|e| InventoryError::Json {
            path: Some(path.clone()),
            message: e.to_string(),
        })?;

        atomic_write(&tmp_path, &path, content.as_bytes())
    }

    fn quarantine_record(&self, id: &str) -> InventoryResult<()> {
        let path = self.store_paths.inventory_record_path(id);
        if !path.exists() {
            return Ok(());
        }
        let quarantine_path = path.with_extension(format!(
            "corrupt-{}",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        fs::rename(&path, &quarantine_path).map_err(|error| InventoryError::Io {
            path: path.clone(),
            message: format!("failed to quarantine corrupt record: {error}"),
        })?;
        sync_parent(&quarantine_path)?;
        tracing::warn!(
            "quarantined corrupt inventory record `{}` at `{}`",
            id,
            quarantine_path.display()
        );
        Ok(())
    }
}

fn atomic_write(
    tmp_path: &std::path::Path,
    path: &std::path::Path,
    content: &[u8],
) -> InventoryResult<()> {
    let mut file = fs::File::create(tmp_path).map_err(|error| InventoryError::Io {
        path: tmp_path.to_path_buf(),
        message: error.to_string(),
    })?;
    file.write_all(content)
        .map_err(|error| InventoryError::Io {
            path: tmp_path.to_path_buf(),
            message: error.to_string(),
        })?;
    file.sync_all().map_err(|error| InventoryError::Io {
        path: tmp_path.to_path_buf(),
        message: error.to_string(),
    })?;
    fs::rename(tmp_path, path).map_err(|error| InventoryError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    sync_parent(path)
}

fn sync_parent(path: &std::path::Path) -> InventoryResult<()> {
    #[cfg(not(unix))]
    {
        let _ = path;
        return Ok(());
    }
    #[cfg(unix)]
    {
        let Some(parent) = path.parent() else {
            return Ok(());
        };
        fs::File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| InventoryError::Io {
                path: parent.to_path_buf(),
                message: format!("failed to sync inventory directory: {error}"),
            })
    }
}

fn version_tuple(version: &str) -> (u64, u64, u64) {
    let mut parts = version
        .split('.')
        .map(|part| part.parse::<u64>().unwrap_or(0));
    (
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BackendInstanceId;
    use crate::ExpectedWorkerIdentity;
    use crate::WorkerInstallationId;
    use crate::WorkerStorePaths;

    fn test_record(id: &str) -> InstallationRecord {
        InstallationRecord {
            installation_id: WorkerInstallationId(id.to_string()),
            version: "1.0.0".to_string(),
            identity: ExpectedWorkerIdentity {
                backend_instance_id: BackendInstanceId("burn:wgpu:default".to_string()),
                installation_id: WorkerInstallationId(id.to_string()),
                backend_kind: "burn".to_string(),
                target: "aarch64-apple-darwin".to_string(),
                manifest_digest: "abc123".to_string(),
            },
            installed_at: chrono::Utc::now(),
            install_path: format!("/workers/installed/{id}"),
            manifest_profile: None,
        }
    }

    #[test]
    fn add_and_list() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = WorkerStorePaths::new(tmp.path().to_path_buf());
        let store = InventoryStore::new(paths);

        store.add(&test_record("burn-wgpu-v1")).unwrap();

        let snapshot = store.list().unwrap();
        assert_eq!(snapshot.records.len(), 1);
        assert_eq!(snapshot.records[0].installation_id.0, "burn-wgpu-v1");
    }

    #[test]
    fn get_existing_record() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = WorkerStorePaths::new(tmp.path().to_path_buf());
        let store = InventoryStore::new(paths);

        store.add(&test_record("burn-wgpu-v1")).unwrap();
        let record = store.get("burn-wgpu-v1").unwrap();
        assert_eq!(record.installation_id.0, "burn-wgpu-v1");
    }

    #[test]
    fn get_missing_record_returns_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = WorkerStorePaths::new(tmp.path().to_path_buf());
        let store = InventoryStore::new(paths);

        let result = store.get("nonexistent");
        assert!(result.is_err());
        match result.unwrap_err() {
            InventoryError::NotFound { .. } => {}
            _ => panic!("expected NotFound"),
        }
    }

    #[test]
    fn remove_record() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = WorkerStorePaths::new(tmp.path().to_path_buf());
        let store = InventoryStore::new(paths);

        store.add(&test_record("burn-wgpu-v1")).unwrap();
        store.remove("burn-wgpu-v1").unwrap();

        let snapshot = store.list().unwrap();
        assert!(snapshot.records.is_empty());
    }

    #[test]
    fn rebuild_from_records_after_index_corruption() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = WorkerStorePaths::new(tmp.path().to_path_buf());
        let store = InventoryStore::new(paths);

        store.add(&test_record("burn-wgpu-v1")).unwrap();
        store.add(&test_record("burn-cuda-v1")).unwrap();

        // Corrupt the index
        let index_path = store.store_paths.inventory_index_path();
        fs::write(&index_path, "corrupt garbage").unwrap();

        // List should rebuild and find all records
        let snapshot = store.list().unwrap();
        assert_eq!(snapshot.records.len(), 2);
    }

    #[test]
    fn corrupt_record_is_quarantined_and_removed_from_index() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = WorkerStorePaths::new(tmp.path().to_path_buf());
        let store = InventoryStore::new(paths.clone());
        store.add(&test_record("burn-wgpu-v1")).unwrap();
        fs::write(paths.inventory_record_path("burn-wgpu-v1"), "not json").unwrap();

        assert!(store.list().unwrap().records.is_empty());
        assert!(store.list().unwrap().records.is_empty());
        assert!(fs::read_dir(paths.inventory_dir()).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("burn-wgpu-v1.corrupt-")
        }));
    }

    #[test]
    fn empty_inventory_returns_empty_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = WorkerStorePaths::new(tmp.path().to_path_buf());
        let store = InventoryStore::new(paths);

        let snapshot = store.list().unwrap();
        assert!(snapshot.records.is_empty());
    }

    #[test]
    fn preference_resolves_exact_or_latest_installed_version() {
        let tmp = tempfile::tempdir().unwrap();
        let store = InventoryStore::new(WorkerStorePaths::new(tmp.path()));
        let mut v1 = test_record("burn-wgpu-v1");
        v1.version = "1.0.0".to_owned();
        let mut v2 = test_record("burn-wgpu-v2");
        v2.version = "2.0.0".to_owned();
        store.add(&v1).unwrap();
        store.add(&v2).unwrap();

        assert_eq!(
            store
                .resolve_preference("burn:wgpu:default", &VersionPolicy::Latest)
                .unwrap()
                .version,
            "2.0.0"
        );
        assert_eq!(
            store
                .resolve_preference(
                    "burn:wgpu:default",
                    &VersionPolicy::Exact("1.0.0".to_owned())
                )
                .unwrap()
                .installation_id
                .0,
            "burn-wgpu-v1"
        );
    }
}
