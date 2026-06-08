use std::path::{Path, PathBuf};

use crate::{
    ConfigDocument, ConfigError, ConfigKey, ConfigReport, ConfigResult, ConfigValidationContext,
    atomic_write,
};

/// Filesystem-backed JSON config store rooted at `<base_path>/config`.
#[derive(Debug, Clone)]
pub struct ConfigStore {
    config_dir: PathBuf,
}

impl ConfigStore {
    pub fn new(config_dir: impl Into<PathBuf>) -> Self {
        Self {
            config_dir: config_dir.into(),
        }
    }

    pub async fn load_json<T>(&self, key: &ConfigKey) -> ConfigResult<(T, ConfigReport)>
    where
        T: ConfigDocument,
    {
        let path = self.path_for_key(key);
        let value = match tokio::fs::read_to_string(&path).await {
            Ok(contents) => {
                serde_json::from_str(&contents).map_err(|error| ConfigError::JsonInvalid {
                    key: Some(key.as_str().to_owned()),
                    path: path.clone(),
                    message: error.to_string(),
                })?
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => T::default(),
            Err(error) => {
                return Err(ConfigError::ReadFailed {
                    path,
                    message: error.to_string(),
                });
            }
        };
        let report = self.report_for(key, &path, &value);
        Ok((value, report))
    }

    pub async fn save_json<T>(&self, key: &ConfigKey, value: &T) -> ConfigResult<ConfigReport>
    where
        T: ConfigDocument,
    {
        let path = self.path_for_key(key);
        let report = self.report_for(key, &path, value);
        let bytes = serde_json::to_vec_pretty(value).map_err(|error| ConfigError::JsonInvalid {
            key: Some(key.as_str().to_owned()),
            path: path.clone(),
            message: error.to_string(),
        })?;
        atomic_write(&path, bytes).await?;
        Ok(report)
    }

    pub async fn update_json<T>(
        &self,
        key: &ConfigKey,
        mutator: impl FnOnce(&mut T),
    ) -> ConfigResult<(T, ConfigReport)>
    where
        T: ConfigDocument,
    {
        let (mut value, _) = self.load_json::<T>(key).await?;
        mutator(&mut value);
        let report = self.save_json(key, &value).await?;
        Ok((value, report))
    }

    pub fn config_dir(&self) -> &Path {
        &self.config_dir
    }

    pub fn path_for_key(&self, key: &ConfigKey) -> PathBuf {
        self.config_dir.join(key.as_str())
    }

    fn report_for<T>(&self, key: &ConfigKey, path: &Path, value: &T) -> ConfigReport
    where
        T: ConfigDocument,
    {
        let context = ConfigValidationContext::new(key.clone(), path.to_path_buf());
        let diagnostics = value.validate(&context);
        ConfigReport::new(key.clone(), path.to_path_buf(), diagnostics)
    }
}
