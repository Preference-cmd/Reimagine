use std::path::{Path, PathBuf};

use crate::{ConfigError, ConfigResult};

/// Workspace directory layout for V1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppPaths {
    base_path: PathBuf,
    models_dir: PathBuf,
    input_dir: PathBuf,
    output_dir: PathBuf,
    workflows_dir: PathBuf,
    config_dir: PathBuf,
}

impl AppPaths {
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        let base_path = base_path.into();
        Self {
            models_dir: base_path.join("models"),
            input_dir: base_path.join("input"),
            output_dir: base_path.join("output"),
            workflows_dir: base_path.join("workflows"),
            config_dir: base_path.join("config"),
            base_path,
        }
    }

    pub async fn ensure_all(&self) -> ConfigResult<()> {
        for dir in [
            &self.base_path,
            &self.models_dir,
            &self.input_dir,
            &self.output_dir,
            &self.workflows_dir,
            &self.config_dir,
        ] {
            tokio::fs::create_dir_all(dir)
                .await
                .map_err(|error| ConfigError::WriteFailed {
                    path: dir.clone(),
                    message: error.to_string(),
                })?;
        }
        Ok(())
    }

    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    pub fn input_dir(&self) -> &Path {
        &self.input_dir
    }

    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }

    pub fn workflows_dir(&self) -> &Path {
        &self.workflows_dir
    }

    pub fn config_dir(&self) -> &Path {
        &self.config_dir
    }
}
