use reimagine_core::diagnostic::Diagnostic;
use serde::{Deserialize, Serialize};

use crate::{ConfigDocument, ConfigValidationContext};

/// Supported inference backend kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceBackendKind {
    Candle,
}

impl Default for InferenceBackendKind {
    fn default() -> Self {
        Self::Candle
    }
}

/// Persisted inference backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceBackendConfig {
    #[serde(default = "default_schema_version")]
    pub schema_version: String,

    #[serde(default)]
    pub backend: InferenceBackendKind,

    #[serde(default = "default_candle_device")]
    pub candle_device: String,
}

fn default_schema_version() -> String {
    InferenceBackendConfig::SCHEMA_VERSION.to_string()
}

fn default_candle_device() -> String {
    "cpu".to_string()
}

impl Default for InferenceBackendConfig {
    fn default() -> Self {
        Self {
            schema_version: default_schema_version(),
            backend: InferenceBackendKind::default(),
            candle_device: default_candle_device(),
        }
    }
}

impl ConfigDocument for InferenceBackendConfig {
    const KEY: &'static str = "inference_backend.json";
    const SCHEMA_VERSION: &'static str = "1";

    fn validate(&self, _context: &ConfigValidationContext) -> Vec<Diagnostic> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_candle() {
        let cfg = InferenceBackendConfig::default();
        assert_eq!(cfg.backend, InferenceBackendKind::Candle);
        assert_eq!(cfg.candle_device, "cpu");
        assert_eq!(cfg.schema_version, "1");
    }

    #[test]
    fn serialize_has_snake_case_candle() {
        let cfg = InferenceBackendConfig::default();
        let json = serde_json::to_value(&cfg).unwrap();
        assert_eq!(json["backend"], "candle");
    }

    #[test]
    fn missing_backend_defaults_to_candle() {
        let json = r#"{"candle_device": "cpu"}"#;
        let cfg: InferenceBackendConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.backend, InferenceBackendKind::Candle);
    }

    #[test]
    fn missing_device_defaults_to_cpu() {
        let json = r#"{"backend": "candle"}"#;
        let cfg: InferenceBackendConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.candle_device, "cpu");
    }

    #[test]
    fn custom_device_deserializes() {
        let json = r#"{"backend": "candle", "candle_device": "metal"}"#;
        let cfg: InferenceBackendConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.candle_device, "metal");
        assert_eq!(cfg.backend, InferenceBackendKind::Candle);
    }

    #[test]
    fn empty_json_defaults() {
        let cfg: InferenceBackendConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(cfg.backend, InferenceBackendKind::Candle);
        assert_eq!(cfg.candle_device, "cpu");
        assert_eq!(cfg.schema_version, "1");
    }
}
