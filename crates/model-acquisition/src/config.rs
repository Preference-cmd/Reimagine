use std::path::PathBuf;

use reimagine_config::{ConfigDocument, ConfigValidationContext};
use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::model::DiagnosticId;
use serde::{Deserialize, Serialize};

/// Configuration for the model-acquisition subsystem.
///
/// Persisted as `<config_dir>/model_acquisition.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAcquisitionConfig {
    #[serde(rename = "$schema_version")]
    schema_version: String,
    #[serde(default)]
    pub huggingface: HuggingFaceConfig,
}

impl Default for ModelAcquisitionConfig {
    fn default() -> Self {
        Self {
            schema_version: Self::SCHEMA_VERSION.to_owned(),
            huggingface: HuggingFaceConfig::default(),
        }
    }
}

impl ConfigDocument for ModelAcquisitionConfig {
    const KEY: &'static str = "model_acquisition.json";
    const SCHEMA_VERSION: &'static str = "1";

    fn validate(&self, _context: &ConfigValidationContext) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        if self.schema_version != Self::SCHEMA_VERSION {
            diagnostics.push(Diagnostic::new(
                DiagnosticId::new("model-acquisition:config:schema_version"),
                DiagnosticCode::new("MODEL_ACQUISITION/CONFIG_INVALID"),
                DiagnosticSeverity::Warning,
                DiagnosticSourceName::new("config"),
                format!(
                    "schema version mismatch: expected {} got {}",
                    Self::SCHEMA_VERSION,
                    self.schema_version
                ),
                DiagnosticTarget::new(DiagnosticTargetDomain::new("model-acquisition"))
                    .with_id(Self::KEY),
            ));
        }

        diagnostics
    }
}

/// HuggingFace-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceConfig {
    /// Explicit token for authenticated downloads.
    /// When `None`, only public repos are accessible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,

    /// Custom HuggingFace Hub endpoint (e.g., `https://huggingface.co`).
    #[serde(default = "default_endpoint")]
    pub endpoint: String,

    /// Custom cache directory for hf-hub.
    /// When `None`, uses the default hf-hub cache (`~/.cache/huggingface/`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_dir: Option<PathBuf>,

    /// Whether to use the hf-hub cache. Defaults to `true`.
    #[serde(default = "default_cache_enabled")]
    pub cache_enabled: bool,

    /// Maximum number of concurrent download workers.
    #[serde(default = "default_max_workers")]
    pub max_workers: usize,
}

impl Default for HuggingFaceConfig {
    fn default() -> Self {
        Self {
            token: None,
            endpoint: default_endpoint(),
            cache_dir: None,
            cache_enabled: default_cache_enabled(),
            max_workers: default_max_workers(),
        }
    }
}

fn default_endpoint() -> String {
    "https://huggingface.co".to_owned()
}

fn default_cache_enabled() -> bool {
    true
}

fn default_max_workers() -> usize {
    4
}
