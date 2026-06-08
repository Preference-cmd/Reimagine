use std::sync::Arc;

use crate::{AppPaths, ConfigDocument, ConfigHandle, ConfigKey, ConfigResult, ConfigStore};

/// Workspace-scoped configuration entry point.
#[derive(Debug, Clone)]
pub struct AppConfig {
    paths: AppPaths,
    store: Arc<ConfigStore>,
}

impl AppConfig {
    pub fn new(paths: AppPaths) -> Self {
        let store = Arc::new(ConfigStore::new(paths.config_dir()));
        Self { paths, store }
    }

    pub fn with_store(paths: AppPaths, store: Arc<ConfigStore>) -> Self {
        Self { paths, store }
    }

    pub fn paths(&self) -> &AppPaths {
        &self.paths
    }

    pub fn store(&self) -> &Arc<ConfigStore> {
        &self.store
    }

    pub fn config<T>(&self) -> ConfigResult<ConfigHandle<T>>
    where
        T: ConfigDocument,
    {
        let key = ConfigKey::new(T::KEY)?;
        Ok(ConfigHandle::new(Arc::clone(&self.store), key))
    }
}
