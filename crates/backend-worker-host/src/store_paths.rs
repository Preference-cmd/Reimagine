use std::path::{Path, PathBuf};

/// Path derivation for the application-global worker store.
///
/// All paths are derived deterministically beneath an injected
/// application-data root that is distinct from the workspace
/// `AppPaths` hierarchy.
///
/// ```text
/// {app_data_root}/workers/
///   .install.lock
///   .intent/{installation_id}.json
///   staging/{installation_id}/
///   installed/{installation_id}/
///   inventory/
///     index.json
///     {installation_id}.json
/// ```
#[derive(Clone, Debug)]
pub struct WorkerStorePaths {
    app_data_root: PathBuf,
}

impl WorkerStorePaths {
    /// Create a new path derivation from an application-data root.
    ///
    /// `app_data_root` must be a writable directory that is not the
    /// workspace base path and not beneath any workspace path. The
    /// caller is responsible for this invariant.
    #[must_use]
    pub fn new(app_data_root: impl Into<PathBuf>) -> Self {
        Self {
            app_data_root: app_data_root.into(),
        }
    }

    /// The root of the worker store hierarchy.
    #[must_use]
    pub fn root(&self) -> PathBuf {
        self.app_data_root.join("workers")
    }

    /// Lock file path for exclusive-installer mutual exclusion.
    #[must_use]
    pub fn install_lock_path(&self) -> PathBuf {
        self.root().join(".install.lock")
    }

    /// Directory for intent journal files.
    #[must_use]
    pub fn intent_dir(&self) -> PathBuf {
        self.root().join(".intent")
    }

    /// Path for a specific intent journal entry.
    #[must_use]
    pub fn intent_path(&self, installation_id: &str) -> PathBuf {
        self.intent_dir().join(format!("{installation_id}.json"))
    }

    /// Base directory for staging areas.
    #[must_use]
    pub fn staging_base(&self) -> PathBuf {
        self.root().join("staging")
    }

    /// Path for a specific staging directory.
    #[must_use]
    pub fn staging_dir(&self, installation_id: &str) -> PathBuf {
        self.staging_base().join(installation_id)
    }

    /// Base directory for installed workers (post-promotion).
    #[must_use]
    pub fn installed_base(&self) -> PathBuf {
        self.root().join("installed")
    }

    /// Path for a specific installed worker directory.
    #[must_use]
    pub fn installed_dir(&self, installation_id: &str) -> PathBuf {
        self.installed_base().join(installation_id)
    }

    /// Directory for inventory record files.
    #[must_use]
    pub fn inventory_dir(&self) -> PathBuf {
        self.root().join("inventory")
    }

    /// Path for a specific inventory record file.
    #[must_use]
    pub fn inventory_record_path(&self, installation_id: &str) -> PathBuf {
        self.inventory_dir().join(format!("{installation_id}.json"))
    }

    /// Path for the inventory index.
    #[must_use]
    pub fn inventory_index_path(&self) -> PathBuf {
        self.inventory_dir().join("index.json")
    }

    /// The application data root (read-only accessor).
    #[must_use]
    pub fn app_data_root(&self) -> &Path {
        &self.app_data_root
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_are_deterministic_under_worker_root() {
        let paths = WorkerStorePaths::new("/app-data");
        assert_eq!(paths.root(), std::path::Path::new("/app-data/workers"));
        assert_eq!(
            paths.intent_dir(),
            std::path::Path::new("/app-data/workers/.intent")
        );
        assert_eq!(
            paths.staging_base(),
            std::path::Path::new("/app-data/workers/staging")
        );
        assert_eq!(
            paths.installed_base(),
            std::path::Path::new("/app-data/workers/installed")
        );
        assert_eq!(
            paths.inventory_dir(),
            std::path::Path::new("/app-data/workers/inventory")
        );
    }

    #[test]
    fn staging_path_uses_installation_id() {
        let paths = WorkerStorePaths::new("/data");
        assert_eq!(
            paths.staging_dir("burn-wgpu-v1"),
            std::path::Path::new("/data/workers/staging/burn-wgpu-v1")
        );
    }

    #[test]
    fn installed_path_uses_installation_id() {
        let paths = WorkerStorePaths::new("/data");
        assert_eq!(
            paths.installed_dir("burn-wgpu-v1"),
            std::path::Path::new("/data/workers/installed/burn-wgpu-v1")
        );
    }

    #[test]
    fn inventory_record_path_uses_installation_id() {
        let paths = WorkerStorePaths::new("/data");
        assert_eq!(
            paths.inventory_record_path("burn-wgpu-v1"),
            std::path::Path::new("/data/workers/inventory/burn-wgpu-v1.json")
        );
    }

    #[test]
    fn lock_path_is_under_root() {
        let paths = WorkerStorePaths::new("/data");
        assert_eq!(
            paths.install_lock_path(),
            std::path::Path::new("/data/workers/.install.lock")
        );
    }
}
