use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::{ConfigDocument, ConfigKey, ConfigReport, ConfigResult, ConfigStore};

/// Typed accessor for one JSON config document.
#[derive(Debug, Clone)]
pub struct ConfigHandle<T> {
    store: Arc<ConfigStore>,
    key: ConfigKey,
    path: PathBuf,
    _marker: PhantomData<T>,
}

impl<T> ConfigHandle<T>
where
    T: ConfigDocument,
{
    pub fn new(store: Arc<ConfigStore>, key: ConfigKey) -> Self {
        let path = store.path_for_key(&key);
        Self {
            store,
            key,
            path,
            _marker: PhantomData,
        }
    }

    pub async fn load(&self) -> ConfigResult<(T, ConfigReport)> {
        self.store.load_json(&self.key).await
    }

    pub async fn save(&self, value: &T) -> ConfigResult<ConfigReport> {
        self.store.save_json(&self.key, value).await
    }

    pub async fn update(&self, mutator: impl FnOnce(&mut T)) -> ConfigResult<(T, ConfigReport)> {
        self.store.update_json(&self.key, mutator).await
    }

    pub fn key(&self) -> &ConfigKey {
        &self.key
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}
