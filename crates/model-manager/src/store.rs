use std::path::{Path, PathBuf};

use reimagine_config::{AppPaths, atomic_write};

use crate::{
    ManifestValidationReport, ModelManagerError, ModelManagerResult, ModelManifest, ModelRoot,
    validate_manifest,
};

const MANIFEST_RELATIVE_PATH: &str = "models/manifest.json";

pub struct ModelManifestStore {
    app_paths: AppPaths,
    manifest_path: PathBuf,
}

impl ModelManifestStore {
    pub fn new(app_paths: AppPaths) -> Self {
        let manifest_path = app_paths.models_dir().join("manifest.json");
        Self {
            app_paths,
            manifest_path,
        }
    }

    pub fn path(&self) -> &Path {
        &self.manifest_path
    }

    pub async fn load(&self) -> ModelManagerResult<(ModelManifest, ManifestValidationReport)> {
        let bytes = match tokio::fs::read(&self.manifest_path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let manifest = default_manifest();
                let report = validate_manifest(&manifest, self.app_paths.models_dir()).await;
                return Ok((manifest, report));
            }
            Err(error) => {
                return Err(ModelManagerError::ReadFailed {
                    path: MANIFEST_RELATIVE_PATH.to_owned(),
                    message: error.to_string(),
                });
            }
        };

        let manifest = serde_json::from_slice::<ModelManifest>(&bytes).map_err(|error| {
            ModelManagerError::ManifestInvalid {
                path: MANIFEST_RELATIVE_PATH.to_owned(),
                message: error.to_string(),
            }
        })?;
        let report = validate_manifest(&manifest, self.app_paths.models_dir()).await;

        Ok((manifest, report))
    }

    pub async fn save(
        &self,
        manifest: &ModelManifest,
    ) -> ModelManagerResult<ManifestValidationReport> {
        let bytes = serde_json::to_vec_pretty(manifest).map_err(|error| {
            ModelManagerError::ManifestInvalid {
                path: MANIFEST_RELATIVE_PATH.to_owned(),
                message: error.to_string(),
            }
        })?;
        atomic_write(&self.manifest_path, &bytes)
            .await
            .map_err(|error| ModelManagerError::WriteFailed {
                path: MANIFEST_RELATIVE_PATH.to_owned(),
                message: error.to_string(),
            })?;

        Ok(validate_manifest(manifest, self.app_paths.models_dir()).await)
    }

    pub async fn update<F>(
        &self,
        mutator: F,
    ) -> ModelManagerResult<(ModelManifest, ManifestValidationReport)>
    where
        F: FnOnce(&mut ModelManifest),
    {
        let (mut manifest, _) = self.load().await?;
        mutator(&mut manifest);
        let report = self.save(&manifest).await?;
        Ok((manifest, report))
    }

    pub async fn remove_model(
        &self,
        model_id: &reimagine_core::model::ModelId,
    ) -> ModelManagerResult<(ModelManifest, ManifestValidationReport)> {
        self.update(|manifest| manifest.remove_model(model_id))
            .await
    }
}

pub async fn load_model_manifest(
    app_paths: AppPaths,
) -> ModelManagerResult<(ModelManifest, ManifestValidationReport)> {
    ModelManifestStore::new(app_paths).load().await
}

fn default_manifest() -> ModelManifest {
    ModelManifest::new().with_root(ModelRoot::base_models())
}
