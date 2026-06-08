//! Serializable model manifest schema.

#[path = "manifest/descriptor.rs"]
mod descriptor;
#[path = "manifest/fingerprint.rs"]
mod fingerprint;
#[path = "manifest/format.rs"]
mod format;
#[path = "manifest/root.rs"]
mod root;
#[path = "manifest/source.rs"]
mod source;
#[path = "manifest/status.rs"]
mod status;

use serde::{Deserialize, Serialize};

pub use descriptor::ModelDescriptor;
pub use fingerprint::Fingerprint;
pub use format::ModelFormat;
pub use root::{ModelRoot, ModelRootId, ModelRootKind};
pub use source::ModelSource;
pub use status::ModelSourceStatus;

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
}

impl Default for ModelManifest {
    fn default() -> Self {
        Self::new()
    }
}
