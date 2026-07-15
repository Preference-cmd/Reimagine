use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use fs2::FileExt;
use serde::{Deserialize, Serialize};
use tracing;

use super::error::{InstallError, InstallResult};
use crate::WorkerStorePaths;

/// Journal states for the installation transaction.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum JournalState {
    /// Staging directory created, download/extraction in progress.
    Staging,
    /// Package verified and self-check passed.
    Verified,
    /// Atomic promotion in progress.
    Promoting,
    /// Installation complete.
    Complete,
    /// Installation failed and should be rolled back.
    Failed(String),
}

/// An intent journal entry for tracking installation progress.
///
/// Written before any destructive operation so that crash recovery
/// can determine what state the installation is in.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstallJournal {
    /// Unique installation identifier.
    pub installation_id: String,
    /// Path to the staging directory.
    pub staging_path: PathBuf,
    /// Path to the final install directory.
    pub target_path: PathBuf,
    /// Current state of the transaction.
    pub state: JournalState,
    /// ISO-8601 timestamp of when the journal was created.
    pub created_at: String,
    /// ISO-8601 timestamp of the last state update.
    pub updated_at: String,
    #[serde(default)]
    pub record: Option<crate::InstallationRecord>,
}

/// Crude filesystem-based lock for mutual exclusion among concurrent
/// installer processes.
pub struct InstallLock {
    file: fs::File,
}

impl InstallLock {
    /// Attempt to acquire the install lock.
    ///
    /// Returns `LockContended` if another process holds the lock.
    pub fn acquire(store_paths: &WorkerStorePaths) -> InstallResult<Self> {
        let lock_path = store_paths.install_lock_path();

        // Ensure parent directory exists
        if let Some(parent) = lock_path.parent() {
            fs::create_dir_all(parent).map_err(|e| InstallError::Io {
                path: parent.to_path_buf(),
                message: format!("failed to create lock directory: {e}"),
            })?;
        }

        // Try to create the lock file atomically
        let file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|e| InstallError::Io {
                path: lock_path.clone(),
                message: format!("failed to open install lock: {e}"),
            })?;
        match file.try_lock_exclusive() {
            Ok(()) => {
                tracing::info!("acquired install lock at `{}`", lock_path.display());
                Ok(Self { file })
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                Err(InstallError::LockContended)
            }
            Err(e) => Err(InstallError::Io {
                path: lock_path.clone(),
                message: format!("failed to acquire lock: {e}"),
            }),
        }
    }
}

impl Drop for InstallLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

/// Journal manager for crash-recoverable installation transactions.
pub struct JournalManager;

impl JournalManager {
    /// Create a new journal entry in the `Staging` state.
    pub fn begin(
        store_paths: &WorkerStorePaths,
        installation_id: &str,
        staging_path: &Path,
        target_path: &Path,
    ) -> InstallResult<InstallJournal> {
        let now = iso_now();
        let journal = InstallJournal {
            installation_id: installation_id.to_string(),
            staging_path: staging_path.to_path_buf(),
            target_path: target_path.to_path_buf(),
            state: JournalState::Staging,
            created_at: now.clone(),
            updated_at: now,
            record: None,
        };

        Self::write_journal(store_paths, &journal)?;
        Ok(journal)
    }

    /// Transition journal to `Verified`.
    pub fn mark_verified(
        store_paths: &WorkerStorePaths,
        journal: &mut InstallJournal,
        record: crate::InstallationRecord,
    ) -> InstallResult<()> {
        journal.state = JournalState::Verified;
        journal.record = Some(record);
        journal.updated_at = iso_now();
        Self::write_journal(store_paths, journal)
    }

    /// Transition journal to `Promoting`.
    pub fn mark_promoting(
        store_paths: &WorkerStorePaths,
        journal: &mut InstallJournal,
    ) -> InstallResult<()> {
        journal.state = JournalState::Promoting;
        journal.updated_at = iso_now();
        Self::write_journal(store_paths, journal)
    }

    /// Transition journal to `Complete`.
    pub fn mark_complete(
        store_paths: &WorkerStorePaths,
        journal: &mut InstallJournal,
    ) -> InstallResult<()> {
        journal.state = JournalState::Complete;
        journal.updated_at = iso_now();
        Self::write_journal(store_paths, journal)
    }

    /// Mark the journal as failed.
    pub fn mark_failed(
        store_paths: &WorkerStorePaths,
        journal: &mut InstallJournal,
        reason: &str,
    ) -> InstallResult<()> {
        journal.state = JournalState::Failed(reason.to_string());
        journal.updated_at = iso_now();
        Self::write_journal(store_paths, journal)
    }

    /// Read the journal for a given installation.
    pub fn read_journal(
        store_paths: &WorkerStorePaths,
        installation_id: &str,
    ) -> InstallResult<Option<InstallJournal>> {
        let path = store_paths.intent_path(installation_id);
        if !path.exists() {
            return Ok(None);
        }

        let content = fs::read_to_string(&path)
            .map_err(|_e| InstallError::JournalCorrupt { path: path.clone() })?;

        serde_json::from_str(&content)
            .map(Some)
            .map_err(|_e| InstallError::JournalCorrupt { path })
    }

    /// Recover from a previous incomplete installation.
    ///
    /// Based on the journal state:
    /// - `Staging`: discard the staging directory
    /// - `Verified`: retry promotion, then mark complete
    /// - `Promoting`: check if target exists, promote if not, mark complete
    /// - `Complete`: nothing to do
    /// - `Failed`: discard staging
    pub fn recover(
        _store_paths: &WorkerStorePaths,
        journal: &InstallJournal,
    ) -> InstallResult<RecoveryAction> {
        match &journal.state {
            JournalState::Staging => {
                tracing::info!("recovering from Staging: cleaning up staging");
                let _ = fs::remove_dir_all(&journal.staging_path);
                Ok(RecoveryAction::CleanupStaging)
            }
            JournalState::Verified => {
                tracing::info!("recovering from Verified: retrying promotion");
                Ok(RecoveryAction::RetryPromote)
            }
            JournalState::Promoting => {
                if journal.target_path.exists() {
                    tracing::info!("recovering from Promoting: target exists, completing");
                    Ok(RecoveryAction::Complete)
                } else if journal.staging_path.exists() {
                    tracing::info!("recovering from Promoting: retrying promotion");
                    Ok(RecoveryAction::RetryPromote)
                } else {
                    tracing::info!("recovering from Promoting: nothing to promote, marking failed");
                    Ok(RecoveryAction::Failed)
                }
            }
            JournalState::Complete => {
                tracing::info!("recovering from Complete: installation already done");
                Ok(RecoveryAction::AlreadyComplete)
            }
            JournalState::Failed(_) => {
                tracing::info!("recovering from Failed: cleaning up staging");
                let _ = fs::remove_dir_all(&journal.staging_path);
                Ok(RecoveryAction::CleanupStaging)
            }
        }
    }

    fn write_journal(
        store_paths: &WorkerStorePaths,
        journal: &InstallJournal,
    ) -> InstallResult<()> {
        let intent_dir = store_paths.intent_dir();
        fs::create_dir_all(&intent_dir).map_err(|e| InstallError::Io {
            path: intent_dir,
            message: format!("failed to create intent directory: {e}"),
        })?;

        let path = store_paths.intent_path(&journal.installation_id);
        let content = serde_json::to_string_pretty(journal)
            .map_err(|_e| InstallError::JournalCorrupt { path: path.clone() })?;

        // Atomic write: write to temp file, then rename
        let tmp_path = path.with_extension("tmp");
        let mut file = fs::File::create(&tmp_path).map_err(|e| InstallError::Io {
            path: tmp_path.clone(),
            message: format!("failed to write journal: {e}"),
        })?;
        file.write_all(content.as_bytes())
            .and_then(|()| file.sync_all())
            .map_err(|e| InstallError::Io {
                path: tmp_path.clone(),
                message: format!("failed to persist journal: {e}"),
            })?;
        fs::rename(&tmp_path, &path).map_err(|e| InstallError::Io {
            path: path.clone(),
            message: format!("failed to commit journal: {e}"),
        })?;
        sync_parent(&path)?;

        Ok(())
    }
}

/// Action to take after journal recovery.
#[derive(Debug, PartialEq)]
pub enum RecoveryAction {
    /// Clean up the staging directory (installation failed).
    CleanupStaging,
    /// Retry promotion from staging to target.
    RetryPromote,
    /// Mark the installation as complete.
    Complete,
    /// Already complete, nothing to do.
    AlreadyComplete,
    /// Mark the installation as failed.
    Failed,
}

/// Atomic promotion engine.
///
/// Uses same-filesystem rename for atomicity. The promotion is a
/// two-step process: staging → target (atomic rename), then journal
/// update → Complete.
pub struct PromotionEngine;

impl PromotionEngine {
    /// Promote the staged worker to the target installation directory.
    ///
    /// Uses `rename()` which is atomic on the same filesystem. The
    /// caller must hold the install lock and have written the `Promoting`
    /// journal state before calling this.
    pub fn promote(staging_path: &Path, target_path: &Path) -> InstallResult<()> {
        // Ensure target parent exists
        if let Some(parent) = target_path.parent() {
            fs::create_dir_all(parent).map_err(|e| InstallError::Io {
                path: parent.to_path_buf(),
                message: format!("failed to create target parent: {e}"),
            })?;
        }

        // Remove existing target if present (back up first)
        let backup = target_path.with_extension("bak");
        if target_path.exists() {
            fs::rename(target_path, &backup).map_err(|e| InstallError::PromoteFailed {
                from: staging_path.to_path_buf(),
                to: target_path.to_path_buf(),
                message: format!("failed to back up existing target: {e}"),
            })?;
        }

        // Persist the staged tree before publishing its directory entry.
        sync_tree(staging_path)?;

        // Atomic rename
        fs::rename(staging_path, target_path).map_err(|e| {
            // Try to restore from backup
            if backup.exists() {
                let _ = fs::rename(&backup, target_path);
            }
            InstallError::PromoteFailed {
                from: staging_path.to_path_buf(),
                to: target_path.to_path_buf(),
                message: format!("rename failed: {e}"),
            }
        })?;
        sync_parent(target_path)?;

        // Clean up backup
        let _ = fs::remove_dir_all(&backup);

        tracing::info!(
            "promoted worker from `{}` to `{}`",
            staging_path.display(),
            target_path.display()
        );

        Ok(())
    }

    /// Remove an installed worker directory.
    ///
    /// This is best-effort: on Windows, executable locks and antivirus
    /// races may prevent deletion. Failed cleanup is deferred, not
    /// forced through deletion of an active path.
    pub fn remove_installed(target_path: &Path) -> InstallResult<()> {
        if !target_path.exists() {
            return Ok(());
        }

        match fs::remove_dir_all(target_path) {
            Ok(()) => {
                tracing::info!("removed installed worker at `{}`", target_path.display());
                Ok(())
            }
            Err(e) => {
                // Defer cleanup — don't force deletion on Windows
                tracing::warn!(
                    "failed to remove installed worker at `{}`: {e}",
                    target_path.display()
                );
                Err(InstallError::Io {
                    path: target_path.to_path_buf(),
                    message: e.to_string(),
                })
            }
        }
    }
}

fn sync_tree(path: &Path) -> InstallResult<()> {
    for entry in fs::read_dir(path).map_err(|error| InstallError::Io {
        path: path.to_path_buf(),
        message: format!("failed to read staging tree for sync: {error}"),
    })? {
        let entry = entry.map_err(|error| InstallError::Io {
            path: path.to_path_buf(),
            message: format!("failed to read staging entry for sync: {error}"),
        })?;
        let entry_path = entry.path();
        if entry
            .file_type()
            .map_err(|error| InstallError::Io {
                path: entry_path.clone(),
                message: error.to_string(),
            })?
            .is_dir()
        {
            sync_tree(&entry_path)?;
        } else {
            fs::File::open(&entry_path)
                .and_then(|file| file.sync_all())
                .map_err(|error| InstallError::Io {
                    path: entry_path,
                    message: format!("failed to sync staged file: {error}"),
                })?;
        }
    }
    sync_directory(path)
}

fn sync_parent(path: &Path) -> InstallResult<()> {
    path.parent().map_or(Ok(()), sync_directory)
}

fn sync_directory(path: &Path) -> InstallResult<()> {
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
    #[cfg(unix)]
    {
        fs::File::open(path)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| InstallError::Io {
                path: path.to_path_buf(),
                message: format!("failed to sync directory: {error}"),
            })
    }
}

fn iso_now() -> String {
    chrono::Utc::now().to_rfc3339()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_paths(tmp: &tempfile::TempDir) -> WorkerStorePaths {
        WorkerStorePaths::new(tmp.path().to_path_buf())
    }

    fn test_record() -> crate::InstallationRecord {
        crate::InstallationRecord {
            installation_id: crate::WorkerInstallationId("test-worker".to_owned()),
            version: "1.0.0".to_owned(),
            identity: crate::ExpectedWorkerIdentity {
                backend_instance_id: crate::BackendInstanceId("burn:wgpu:default".to_owned()),
                installation_id: crate::WorkerInstallationId("test-worker".to_owned()),
                backend_kind: "burn".to_owned(),
                target: "test-target".to_owned(),
                manifest_digest: "digest".to_owned(),
            },
            installed_at: chrono::Utc::now(),
            install_path: "/installed/test-worker".to_owned(),
            manifest_profile: None,
        }
    }

    #[test]
    fn lock_acquire_and_release() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = test_paths(&tmp);

        {
            let _lock = InstallLock::acquire(&paths).expect("acquire lock");
            assert!(paths.install_lock_path().exists());

            // Second acquire should fail
            let second = InstallLock::acquire(&paths);
            assert!(second.is_err());
            // Drop lock
        }

        // After drop, lock should be released
        let lock = InstallLock::acquire(&paths).expect("re-acquire lock after drop");
        drop(lock);
    }

    #[test]
    fn journal_begin_and_complete() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = test_paths(&tmp);

        let mut journal = JournalManager::begin(
            &paths,
            "test-worker",
            Path::new("/staging/test-worker"),
            Path::new("/installed/test-worker"),
        )
        .expect("begin journal");

        assert_eq!(journal.state, JournalState::Staging);

        JournalManager::mark_verified(&paths, &mut journal, test_record()).expect("mark verified");
        assert_eq!(journal.state, JournalState::Verified);

        JournalManager::mark_complete(&paths, &mut journal).expect("mark complete");
        assert_eq!(journal.state, JournalState::Complete);

        // Verify persistence
        let loaded = JournalManager::read_journal(&paths, "test-worker")
            .expect("read journal")
            .expect("journal exists");
        assert_eq!(loaded.state, JournalState::Complete);
    }

    #[test]
    fn promote_creates_target() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join("staging");
        let target = tmp.path().join("installed");

        fs::create_dir_all(&staging).unwrap();
        fs::write(staging.join("burn-worker"), b"binary").unwrap();

        PromotionEngine::promote(&staging, &target).expect("promote");
        assert!(target.join("burn-worker").exists());
        assert!(!staging.exists());
    }

    #[test]
    fn journal_recovery_staging_cleans_up() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = test_paths(&tmp);

        let staging = tmp.path().join("staging");
        fs::create_dir_all(&staging).unwrap();
        fs::write(staging.join("test.bin"), b"data").unwrap();

        let journal = InstallJournal {
            installation_id: "test".to_string(),
            staging_path: staging.clone(),
            target_path: tmp.path().join("target"),
            state: JournalState::Staging,
            created_at: iso_now(),
            updated_at: iso_now(),
            record: None,
        };

        let action = JournalManager::recover(&paths, &journal).expect("recover");
        assert_eq!(action, RecoveryAction::CleanupStaging);
        assert!(!staging.exists());
    }
}
