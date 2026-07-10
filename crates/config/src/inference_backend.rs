use reimagine_core::diagnostic::Diagnostic;
use serde::{Deserialize, Serialize};

use crate::{ConfigDocument, ConfigValidationContext};

/// Supported inference backend kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum InferenceBackendKind {
    #[default]
    Candle,
    Burn,
}

/// Persisted inference backend configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceBackendConfig {
    #[serde(default = "default_schema_version")]
    pub schema_version: String,

    #[serde(default)]
    pub backend: InferenceBackendKind,

    #[serde(default = "default_candle_device")]
    pub candle_device: String,

    /// Open backend instance selected by config, e.g. `"candle:cpu"`.
    ///
    /// This is intentionally a string rather than an inference-owned
    /// `BackendInstance` so the config crate remains independent of the
    /// inference/router crates. App-host validates the identity against live
    /// backend profiles during workspace bootstrap.
    #[serde(default)]
    pub selected_instance: Option<String>,

    /// Optional ordered backend-instance preference list.
    ///
    /// Empty means app-host uses the resolved selected instance first.
    #[serde(default)]
    pub priority_order: Vec<String>,

    /// Backend instances that should not be selected by the app-host policy.
    #[serde(default)]
    pub disabled_instances: Vec<String>,
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
            selected_instance: None,
            priority_order: Vec::new(),
            disabled_instances: Vec::new(),
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
        assert_eq!(cfg.selected_instance, None);
        assert!(cfg.priority_order.is_empty());
        assert!(cfg.disabled_instances.is_empty());
    }

    #[test]
    fn open_selected_instance_deserializes_without_backend_enum_variant() {
        let json = r#"{
            "selected_instance": "stub:cpu",
            "priority_order": ["stub:cpu", "candle:cpu"],
            "disabled_instances": ["candle:metal"]
        }"#;
        let cfg: InferenceBackendConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.backend, InferenceBackendKind::Candle);
        assert_eq!(cfg.candle_device, "cpu");
        assert_eq!(cfg.selected_instance.as_deref(), Some("stub:cpu"));
        assert_eq!(cfg.priority_order, vec!["stub:cpu", "candle:cpu"]);
        assert_eq!(cfg.disabled_instances, vec!["candle:metal"]);
    }
}
