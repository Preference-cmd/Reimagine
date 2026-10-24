use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use reimagine_backend_worker_host::catalog::tuf::{self, RootMetadata, TufKey};
use reimagine_backend_worker_host::{
    CatalogClient, CatalogError, CatalogTarget, CompatibilityFilter, HostInfo, InstallConfig,
    InstallEngine, InstallError, InventoryError, InventoryStore, WorkerStorePaths,
};

const EMBEDDED_ROOT: &str = include_str!("../assets/worker-catalog-root.json");

/// DTO representing a catalog item for the UI.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerCatalogItemDto {
    pub path: String,
    pub version: String,
    pub installation_id: String,
    pub backend_instance_id: String,
    pub os: String,
    pub arch: String,
    pub worker_kind: String,
    pub package_format: String,
    pub length: u64,
    pub sha256: String,
    pub target: String,
    pub manifest_digest: String,
}

/// DTO representing an installed worker for the UI.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerInstallationDto {
    pub installation_id: String,
    pub version: String,
    pub backend_instance_id: String,
    pub backend_kind: String,
    pub target: String,
    pub installed_at: String,
    pub install_path: String,
    pub manifest_digest: String,
}

impl From<reimagine_backend_worker_host::InstallationRecord> for WorkerInstallationDto {
    fn from(record: reimagine_backend_worker_host::InstallationRecord) -> Self {
        Self {
            installation_id: record.installation_id.0,
            version: record.version,
            backend_instance_id: record.identity.backend_instance_id.0,
            backend_kind: record.identity.backend_kind,
            target: record.identity.target,
            installed_at: record.installed_at.to_rfc3339(),
            install_path: record.install_path,
            manifest_digest: record.identity.manifest_digest,
        }
    }
}

impl From<&CatalogTarget> for WorkerCatalogItemDto {
    fn from(target: &CatalogTarget) -> Self {
        Self {
            path: target.path.clone(),
            version: target.custom.version.clone(),
            installation_id: target.custom.installation_id.clone(),
            backend_instance_id: target.custom.backend_instance_id.clone(),
            os: target.custom.os.clone(),
            arch: target.custom.arch.clone(),
            worker_kind: target.custom.worker_kind.clone(),
            package_format: target.custom.package_format.clone(),
            length: target.length,
            sha256: target.sha256.clone(),
            target: target.custom.target.clone(),
            manifest_digest: target.custom.manifest_digest.clone(),
        }
    }
}

/// Errors from worker management operations.
#[derive(Debug)]
pub enum WorkerManagementError {
    Catalog(CatalogError),
    Install(InstallError),
    Inventory(InventoryError),
    NotFound { installation_id: String },
    UnverifiedCatalogTarget { path: String },
    StatePoisoned,
}

impl std::fmt::Display for WorkerManagementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Catalog(error) => write!(f, "catalog error: {error}"),
            Self::Install(error) => write!(f, "install error: {error}"),
            Self::Inventory(error) => write!(f, "inventory error: {error}"),
            Self::NotFound { installation_id } => {
                write!(f, "installation `{installation_id}` not found")
            }
            Self::UnverifiedCatalogTarget { path } => {
                write!(
                    f,
                    "catalog target `{path}` is not in the verified catalog snapshot"
                )
            }
            Self::StatePoisoned => write!(f, "worker management state lock was poisoned"),
        }
    }
}

impl std::error::Error for WorkerManagementError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Catalog(error) => Some(error),
            Self::Install(error) => Some(error),
            Self::Inventory(error) => Some(error),
            Self::NotFound { .. } | Self::UnverifiedCatalogTarget { .. } | Self::StatePoisoned => {
                None
            }
        }
    }
}

impl From<CatalogError> for WorkerManagementError {
    fn from(error: CatalogError) -> Self {
        Self::Catalog(error)
    }
}

impl From<InstallError> for WorkerManagementError {
    fn from(error: InstallError) -> Self {
        Self::Install(error)
    }
}

impl From<InventoryError> for WorkerManagementError {
    fn from(error: InventoryError) -> Self {
        Self::Inventory(error)
    }
}

/// Application-global facade for official worker discovery and installation.
///
/// Only targets retained from a verified catalog snapshot can enter the
/// installer. Installation and removal never activate a worker process.
pub struct WorkerManagementService {
    inventory: InventoryStore,
    install_engine: InstallEngine,
    catalog_client: Option<CatalogClient>,
    trusted_root: RootMetadata,
    trusted_root_keys: HashMap<String, TufKey>,
    verified_catalog: Mutex<Option<Vec<CatalogTarget>>>,
}

impl std::fmt::Debug for WorkerManagementService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerManagementService")
            .field("catalog_configured", &self.catalog_client.is_some())
            .finish_non_exhaustive()
    }
}

impl WorkerManagementService {
    /// Configure the official catalog beneath an application-global data root.
    pub fn new(
        app_data_root: impl Into<PathBuf>,
        catalog_base_url: impl Into<String>,
    ) -> Result<Self, WorkerManagementError> {
        Self::build(app_data_root.into(), Some(catalog_base_url.into()))
    }

    /// Configure durable inventory without network catalog access.
    ///
    /// This is used when the composition root has no catalog endpoint policy;
    /// installed workers remain available and pending transactions recover.
    pub fn offline(app_data_root: impl Into<PathBuf>) -> Result<Self, WorkerManagementError> {
        Self::build(app_data_root.into(), None)
    }

    fn build(
        app_data_root: PathBuf,
        catalog_base_url: Option<String>,
    ) -> Result<Self, WorkerManagementError> {
        let trusted_root: RootMetadata =
            serde_json::from_str(EMBEDDED_ROOT).map_err(|error| CatalogError::RootLoad {
                message: format!("embedded worker catalog root is invalid JSON: {error}"),
            })?;
        let trusted_root_keys = tuf::verify_root(&trusted_root, None)?;

        let store_paths = WorkerStorePaths::new(app_data_root);
        let inventory = InventoryStore::new(store_paths.clone());
        let install_engine = InstallEngine::new(
            InstallConfig::default(),
            store_paths.clone(),
            InventoryStore::new(store_paths),
        );
        install_engine.recover_pending()?;

        let catalog_client = catalog_base_url.map(|base_url| {
            CatalogClient::new(
                base_url,
                CompatibilityFilter::new(HostInfo {
                    os: std::env::consts::OS.to_owned(),
                    arch: std::env::consts::ARCH.to_owned(),
                    supported_protocol_range: (1, 1),
                }),
            )
        });

        Ok(Self {
            inventory,
            install_engine,
            catalog_client,
            trusted_root,
            trusted_root_keys,
            verified_catalog: Mutex::new(None),
        })
    }

    /// List all installed workers from the durable inventory.
    pub fn list_installed(&self) -> Result<Vec<WorkerInstallationDto>, WorkerManagementError> {
        Ok(self
            .inventory
            .list()?
            .records
            .into_iter()
            .map(WorkerInstallationDto::from)
            .collect())
    }

    /// Get one installed worker by stable installation id.
    pub fn get_installed(
        &self,
        installation_id: &str,
    ) -> Result<WorkerInstallationDto, WorkerManagementError> {
        let record = self.inventory.get(installation_id).map_err(|error| {
            if matches!(error, InventoryError::NotFound { .. }) {
                WorkerManagementError::NotFound {
                    installation_id: installation_id.to_owned(),
                }
            } else {
                WorkerManagementError::Inventory(error)
            }
        })?;
        Ok(record.into())
    }

    /// Fetch and cache the TUF-verified compatible catalog.
    pub async fn list_catalog(&self) -> Result<Vec<WorkerCatalogItemDto>, WorkerManagementError> {
        if let Some(targets) = self
            .verified_catalog
            .lock()
            .map_err(|_| WorkerManagementError::StatePoisoned)?
            .as_ref()
        {
            return Ok(targets.iter().map(WorkerCatalogItemDto::from).collect());
        }

        let Some(client) = &self.catalog_client else {
            return Ok(Vec::new());
        };
        let catalog = client
            .fetch_catalog(&self.trusted_root, &self.trusted_root_keys, 0)
            .await?;
        let items = catalog
            .targets
            .iter()
            .map(WorkerCatalogItemDto::from)
            .collect();
        *self
            .verified_catalog
            .lock()
            .map_err(|_| WorkerManagementError::StatePoisoned)? = Some(catalog.targets);
        Ok(items)
    }

    /// Install a target retained from the verified catalog snapshot.
    pub async fn install(
        &self,
        requested: &WorkerCatalogItemDto,
    ) -> Result<WorkerInstallationDto, WorkerManagementError> {
        let target = self
            .verified_catalog
            .lock()
            .map_err(|_| WorkerManagementError::StatePoisoned)?
            .as_ref()
            .and_then(|targets| {
                targets
                    .iter()
                    .find(|target| WorkerCatalogItemDto::from(*target) == *requested)
                    .cloned()
            })
            .ok_or_else(|| WorkerManagementError::UnverifiedCatalogTarget {
                path: requested.path.clone(),
            })?;
        let client = self.catalog_client.as_ref().ok_or_else(|| {
            WorkerManagementError::UnverifiedCatalogTarget {
                path: requested.path.clone(),
            }
        })?;

        Ok(self.install_engine.install(client, &target).await?.into())
    }

    /// Remove a worker without activating or terminating any process.
    pub fn remove(&self, installation_id: &str) -> Result<(), WorkerManagementError> {
        self.install_engine.remove(installation_id)?;
        Ok(())
    }

    #[must_use]
    pub fn inventory(&self) -> &InventoryStore {
        &self.inventory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "reimagine-worker-management-{label}-{}",
            std::process::id()
        ))
    }

    #[test]
    fn embedded_root_is_valid_and_worker_store_is_application_global() {
        let app_data_root = temp_dir("app-data");
        let workspace_root = temp_dir("workspace");
        let service = WorkerManagementService::offline(&app_data_root).expect("service");

        assert!(tuf::verify_root(&service.trusted_root, None).is_ok());
        assert!(
            service
                .inventory
                .store_paths()
                .root()
                .starts_with(&app_data_root)
        );
        assert!(
            !service
                .inventory
                .store_paths()
                .root()
                .starts_with(&workspace_root)
        );
    }

    #[tokio::test]
    async fn install_rejects_a_dto_not_retained_from_verified_metadata() {
        let service = WorkerManagementService::offline(temp_dir("unsigned")).expect("service");
        let requested = WorkerCatalogItemDto {
            path: "arbitrary.tar.gz".to_owned(),
            version: "1.0.0".to_owned(),
            installation_id: "arbitrary".to_owned(),
            backend_instance_id: "burn:wgpu:default".to_owned(),
            os: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
            worker_kind: "burn".to_owned(),
            package_format: "tar.gz".to_owned(),
            length: 1,
            sha256: "00".repeat(32),
            target: "arbitrary".to_owned(),
            manifest_digest: "unsigned".to_owned(),
        };

        assert!(matches!(
            service.install(&requested).await,
            Err(WorkerManagementError::UnverifiedCatalogTarget { .. })
        ));
    }
}
