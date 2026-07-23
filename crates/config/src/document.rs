use std::path::{Path, PathBuf};

use reimagine_core::diagnostic::{CorrelationId, Diagnostic};
use serde::{Serialize, de::DeserializeOwned};

use crate::ConfigKey;

/// Typed JSON config contract implemented by module crates.
pub trait ConfigDocument:
    Serialize + DeserializeOwned + Default + Send + Sync + Sized + 'static
{
    const KEY: &'static str;
    const SCHEMA_VERSION: &'static str;

    fn validate(&self, context: &ConfigValidationContext) -> Vec<Diagnostic>;
}

/// Context available during document-internal validation.
#[derive(Debug, Clone)]
pub struct ConfigValidationContext {
    key: ConfigKey,
    path: PathBuf,
    correlation_id: Option<CorrelationId>,
}

impl ConfigValidationContext {
    pub fn new(key: ConfigKey, path: impl Into<PathBuf>) -> Self {
        Self {
            key,
            path: path.into(),
            correlation_id: None,
        }
    }

    pub fn with_correlation_id(mut self, correlation_id: CorrelationId) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    pub fn key(&self) -> &ConfigKey {
        &self.key
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn correlation_id(&self) -> Option<&CorrelationId> {
        self.correlation_id.as_ref()
    }
}
