//! Serializable model manifest schema.

mod descriptor;
mod fingerprint;
mod format;
mod root;
mod source;
mod source_resolution;
mod status;
mod validation;

use serde::{Deserialize, Serialize};

pub use descriptor::ModelDescriptor;
pub use fingerprint::Fingerprint;
pub use format::ModelFormat;
pub use root::{ModelRoot, ModelRootId, ModelRootKind};
pub use source::ModelSource;
pub(crate) use source_resolution::{resolve_root_path, resolve_source_path};
pub use status::ModelSourceStatus;
pub use validation::{
    ManifestValidationReport, validate_manifest, validate_manifest_with_series_config,
};

pub const MODEL_MANIFEST_SCHEMA_VERSION: &str = "reimagine.model_manifest.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    schema_version: String,
    model_roots: Vec<ModelRoot>,
    models: Vec<ModelDescriptor>,
}

impl ModelManifest {
    pub fn new() -> Self {
        Self {
            schema_version: MODEL_MANIFEST_SCHEMA_VERSION.to_owned(),
            model_roots: Vec::new(),
            models: Vec::new(),
        }
    }

    pub fn with_root(mut self, root: ModelRoot) -> Self {
        self.model_roots.push(root);
        self
    }

    pub fn with_schema_version(mut self, schema_version: impl Into<String>) -> Self {
        self.schema_version = schema_version.into();
        self
    }

    pub fn with_model(mut self, model: ModelDescriptor) -> Self {
        self.models.push(model);
        self
    }

    pub fn schema_version(&self) -> &str {
        &self.schema_version
    }

    pub fn model_roots(&self) -> &[ModelRoot] {
        &self.model_roots
    }

    pub fn models(&self) -> &[ModelDescriptor] {
        &self.models
    }

    pub(crate) fn models_mut(&mut self) -> &mut Vec<ModelDescriptor> {
        &mut self.models
    }

    pub fn remove_model(&mut self, model_id: &reimagine_core::model::ModelId) {
        self.models.retain(|model| model.id() != model_id);
    }
}

impl Default for ModelManifest {
    fn default() -> Self {
        Self::new()
    }
}
