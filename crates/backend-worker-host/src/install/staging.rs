use std::path::{Path, PathBuf};

use tracing;

use super::error::{InstallError, InstallResult};
use crate::WorkerStorePaths;

/// Manages staging directory lifecycle for worker installation.
pub struct StagingManager;

impl StagingManager {
    /// Create a new staging directory for the given installation.
    ///
    /// Returns the path to the created staging directory.
    /// Fails if the staging directory already exists.
    pub fn create_staging(
        store_paths: &WorkerStorePaths,
        installation_id: &str,
    ) -> InstallResult<PathBuf> {
        let staging = store_paths.staging_dir(installation_id);

        if staging.exists() {
            return Err(InstallError::StagingExists { path: staging });
        }

        std::fs::create_dir_all(&staging).map_err(|e| InstallError::Io {
            path: staging.clone(),
            message: format!("failed to create staging directory: {e}"),
        })?;

        tracing::info!("created staging directory at `{}`", staging.display());
        Ok(staging)
    }

    /// Remove a staging directory and all its contents.
    pub fn cleanup_staging(staging_path: &Path) -> InstallResult<()> {
        if !staging_path.exists() {
            return Ok(());
        }

        std::fs::remove_dir_all(staging_path).map_err(|e| InstallError::StagingCleanup {
            path: staging_path.to_path_buf(),
            message: format!("failed to remove staging directory: {e}"),
        })?;

        tracing::info!(
            "cleaned up staging directory at `{}`",
            staging_path.display()
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WorkerStorePaths;

    #[test]
    fn create_and_cleanup_staging() {
        let tmp = tempfile::tempdir().unwrap();
        let store = WorkerStorePaths::new(tmp.path().to_path_buf());

        let staging =
            StagingManager::create_staging(&store, "test-worker").expect("create staging");
        assert!(staging.exists());

        // Second create should fail
        let duplicate = StagingManager::create_staging(&store, "test-worker");
        assert!(duplicate.is_err());

        StagingManager::cleanup_staging(&staging).expect("cleanup staging");
        assert!(!staging.exists());
    }
}
