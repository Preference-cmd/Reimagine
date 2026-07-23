pub mod error;
pub mod promote;
pub mod self_check;
pub mod staging;

pub use error::{InstallError, InstallResult};
pub use promote::{
    InstallJournal, InstallLock, JournalManager, JournalState, PromotionEngine, RecoveryAction,
};
pub use self_check::{SelfCheckConfig, SelfCheckRunner};
pub use staging::StagingManager;

use tracing;

use crate::WorkerStorePaths;
use crate::catalog::{CatalogClient, CatalogTarget};
use crate::inventory::{InstallationRecord, InventoryStore};
use crate::package::{ExtractionLimits, PackageExtractor};

/// Configuration for the install engine.
#[derive(Clone, Debug)]
pub struct InstallConfig {
    pub extraction_limits: ExtractionLimits,
    pub self_check_timeout: std::time::Duration,
}

impl Default for InstallConfig {
    fn default() -> Self {
        Self {
            extraction_limits: ExtractionLimits::default(),
            self_check_timeout: std::time::Duration::from_secs(5),
        }
    }
}

/// High-level installation engine that composes catalog, package,
/// staging, self-check, promotion, and inventory into one pipeline.
pub struct InstallEngine {
    config: InstallConfig,
    store_paths: WorkerStorePaths,
    inventory: InventoryStore,
}

impl InstallEngine {
    pub fn new(
        config: InstallConfig,
        store_paths: WorkerStorePaths,
        inventory: InventoryStore,
    ) -> Self {
        Self {
            config,
            store_paths,
            inventory,
        }
    }

    /// Install a worker from a catalog target.
    ///
    /// Pipeline:
    /// 1. Acquire install lock (exclusive)
    /// 2. Create staging directory
    /// 3. Write Staging journal entry
    /// 4. Download target from catalog
    /// 5. Extract archive into staging
    /// 6. Run self-check on the extracted worker binary
    /// 7. Mark journal as Verified
    /// 8. Find the main executable in the extracted package
    /// 9. Atomic promote staging → installed directory
    /// 10. Mark journal as Complete
    /// 11. Write inventory record
    /// 12. Release lock and clean up journal
    pub async fn install(
        &self,
        catalog: &CatalogClient,
        target: &CatalogTarget,
    ) -> Result<InstallationRecord, InstallError> {
        let expected_identity = target.expected_identity();
        let installation_id = &expected_identity.installation_id.0;

        let archive_data = catalog.download_target(target).await?;

        // 1. Acquire lock
        let _lock = InstallLock::acquire(&self.store_paths)?;
        self.recover_pending_locked()?;
        if let Ok(existing) = self.inventory.get(installation_id) {
            return Ok(existing);
        }

        // 2. Create staging
        let staging = StagingManager::create_staging(&self.store_paths, installation_id)?;
        let target_path = self.store_paths.installed_dir(installation_id);

        // 3. Write journal
        let mut journal =
            JournalManager::begin(&self.store_paths, installation_id, &staging, &target_path)?;

        // 5. Extract
        let extractor = PackageExtractor::new(self.config.extraction_limits.clone());
        let manifest = extractor.extract(&archive_data, &staging, Some(&expected_identity))?;

        // 6. Run self-check
        let executable_path = self.find_executable(&staging, &manifest)?;
        let self_check = SelfCheckRunner::new(self::self_check::SelfCheckConfig {
            timeout: self.config.self_check_timeout,
            ..self::self_check::SelfCheckConfig::default()
        });
        self_check.run_check(&executable_path, &expected_identity)?;

        let now = chrono::Utc::now();
        let record = InstallationRecord {
            installation_id: expected_identity.installation_id.clone(),
            version: target.custom.version.clone(),
            identity: expected_identity.clone(),
            installed_at: now,
            install_path: target_path.to_string_lossy().to_string(),
            manifest_profile: None,
        };

        // 7. Mark verified, including the durable inventory intent.
        JournalManager::mark_verified(&self.store_paths, &mut journal, record.clone())?;

        // 8. Atomic promote
        JournalManager::mark_promoting(&self.store_paths, &mut journal)?;
        PromotionEngine::promote(&staging, &target_path)?;

        // 9. Publish inventory before marking the transaction complete.
        self.inventory
            .add(&record)
            .map_err(InstallError::Inventory)?;

        // 10. Mark complete
        JournalManager::mark_complete(&self.store_paths, &mut journal)?;

        // Clean up journal
        let journal_path = self.store_paths.intent_path(installation_id);
        let _ = std::fs::remove_file(&journal_path);

        tracing::info!(
            "worker `{}` installation complete at `{}`",
            installation_id,
            target_path.display()
        );

        Ok(record)
    }

    pub fn recover_pending(&self) -> Result<(), InstallError> {
        let _lock = InstallLock::acquire(&self.store_paths)?;
        self.recover_pending_locked()
    }

    fn recover_pending_locked(&self) -> Result<(), InstallError> {
        let intent_dir = self.store_paths.intent_dir();
        if !intent_dir.exists() {
            return Ok(());
        }
        for entry in std::fs::read_dir(&intent_dir).map_err(|error| InstallError::Io {
            path: intent_dir.clone(),
            message: error.to_string(),
        })? {
            let path = entry
                .map_err(|error| InstallError::Io {
                    path: intent_dir.clone(),
                    message: error.to_string(),
                })?
                .path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let installation_id = path
                .file_stem()
                .and_then(|value| value.to_str())
                .ok_or_else(|| InstallError::JournalCorrupt { path: path.clone() })?;
            let Some(mut journal) =
                JournalManager::read_journal(&self.store_paths, installation_id)?
            else {
                continue;
            };
            match JournalManager::recover(&self.store_paths, &journal)? {
                RecoveryAction::CleanupStaging | RecoveryAction::Failed => {
                    let _ = std::fs::remove_dir_all(&journal.staging_path);
                }
                RecoveryAction::RetryPromote => {
                    PromotionEngine::promote(&journal.staging_path, &journal.target_path)?;
                    self.publish_recovered_record(&journal)?;
                    JournalManager::mark_complete(&self.store_paths, &mut journal)?;
                }
                RecoveryAction::Complete | RecoveryAction::AlreadyComplete => {
                    self.publish_recovered_record(&journal)?;
                }
            }
            let _ = std::fs::remove_file(&path);
        }
        Ok(())
    }

    fn publish_recovered_record(&self, journal: &InstallJournal) -> Result<(), InstallError> {
        let record = journal
            .record
            .as_ref()
            .ok_or_else(|| InstallError::JournalCorrupt {
                path: self.store_paths.intent_path(&journal.installation_id),
            })?;
        self.inventory.add(record).map_err(InstallError::Inventory)
    }

    /// Remove a worker installation from inventory (not filesystem files).
    ///
    /// Filesystem cleanup is best-effort per Windows lock conventions.
    pub fn remove(&self, installation_id: &str) -> Result<(), InstallError> {
        let _lock = InstallLock::acquire(&self.store_paths)?;

        let target_path = self.store_paths.installed_dir(installation_id);
        let _ = PromotionEngine::remove_installed(&target_path);
        self.inventory
            .remove(installation_id)
            .map_err(InstallError::Inventory)
    }

    fn find_executable(
        &self,
        staging: &std::path::Path,
        manifest: &crate::package::PackageManifest,
    ) -> Result<std::path::PathBuf, InstallError> {
        let executable_entry = manifest
            .files
            .iter()
            .find(|f| f.executable)
            .or_else(|| manifest.files.first());

        match executable_entry {
            Some(entry) => Ok(staging.join(&entry.path)),
            None => Err(InstallError::SelfCheckFailed {
                message: "no executable entry found in package manifest".to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovery_promotes_and_publishes_verified_inventory_intent() {
        let temp = tempfile::tempdir().unwrap();
        let paths = WorkerStorePaths::new(temp.path());
        let inventory = InventoryStore::new(paths.clone());
        let engine = InstallEngine::new(
            InstallConfig::default(),
            paths.clone(),
            InventoryStore::new(paths.clone()),
        );
        let staging = StagingManager::create_staging(&paths, "worker-v1").unwrap();
        std::fs::write(staging.join("worker"), b"verified").unwrap();
        let target = paths.installed_dir("worker-v1");
        let record = InstallationRecord {
            installation_id: crate::WorkerInstallationId("worker-v1".to_owned()),
            version: "1.0.0".to_owned(),
            identity: crate::ExpectedWorkerIdentity {
                backend_instance_id: crate::BackendInstanceId("burn:wgpu:default".to_owned()),
                installation_id: crate::WorkerInstallationId("worker-v1".to_owned()),
                backend_kind: "burn".to_owned(),
                target: "test-target".to_owned(),
                manifest_digest: "digest".to_owned(),
            },
            installed_at: chrono::Utc::now(),
            install_path: target.to_string_lossy().into_owned(),
            manifest_profile: None,
        };
        let mut journal = JournalManager::begin(&paths, "worker-v1", &staging, &target).unwrap();
        JournalManager::mark_verified(&paths, &mut journal, record).unwrap();
        JournalManager::mark_promoting(&paths, &mut journal).unwrap();

        engine.recover_pending().unwrap();

        assert_eq!(std::fs::read(target.join("worker")).unwrap(), b"verified");
        assert_eq!(inventory.get("worker-v1").unwrap().version, "1.0.0");
        assert!(!paths.intent_path("worker-v1").exists());
    }
}
