use reimagine_core::diagnostic::Diagnostic;
use serde::{Deserialize, Serialize};

use crate::{ConfigDocument, ConfigValidationContext};

/// Supported inference backend kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum InferenceBackendKind {
    #[default]
    Burn,
    Candle,
}

/// A manually configured static worker endpoint (T12 `ConfigDiscovery`).
///
/// This is also the natural home for pre-shared certificate fingerprints
/// from the T19 trust model: a `fingerprint` turns the endpoint into a
/// *trusted* one (verifiable identity) without interactive confirmation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerEndpointConfig {
    /// Stable worker id, unique across the topology pool.
    pub id: String,

    /// Transport kind string, e.g. `"quic"`.
    ///
    /// Intentionally a string rather than a protocol-owned enum so the
    /// config crate remains independent of worker-protocol types (same
    /// pattern as `selected_instance`).
    #[serde(default = "default_worker_transport")]
    pub transport: String,

    /// Connectable address, e.g. `"quic://192.168.1.10:9100"`.
    pub address: String,

    /// Advertised worker capabilities.
    #[serde(default)]
    pub capabilities: Vec<String>,

    /// Human-readable device label, e.g. `"cuda:0"`.
    #[serde(default)]
    pub device_label: String,

    /// Pre-shared certificate fingerprint (SHA-256 hex, T19).
    ///
    /// `Some` pins the worker's identity, so the endpoint is trusted
    /// without a TOFU round-trip. `None` means the endpoint still needs
    /// the T19 trust flow before connecting.
    #[serde(default)]
    pub fingerprint: Option<String>,
}

fn default_worker_transport() -> String {
    "quic".to_string()
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

    #[serde(default = "default_burn_device")]
    pub burn_device: String,

    /// Open backend instance selected by config, e.g. `"burn:cpu"`.
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

    /// Static worker endpoints for discovery (T12 `ConfigDiscovery`).
    ///
    /// Manual/static configuration of remote workers; discovered via the
    /// discovery orchestrator alongside mDNS. A `fingerprint` per endpoint
    /// pre-shares the T19 cert pin, marking the endpoint trusted.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub workers: Vec<WorkerEndpointConfig>,
}

fn default_schema_version() -> String {
    InferenceBackendConfig::SCHEMA_VERSION.to_string()
}

fn default_candle_device() -> String {
    "cpu".to_string()
}

fn default_burn_device() -> String {
    "cpu".to_string()
}

impl Default for InferenceBackendConfig {
    fn default() -> Self {
        Self {
            schema_version: default_schema_version(),
            backend: InferenceBackendKind::default(),
            candle_device: default_candle_device(),
            burn_device: default_burn_device(),
            selected_instance: None,
            priority_order: Vec::new(),
            disabled_instances: Vec::new(),
            workers: Vec::new(),
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
    fn default_is_burn() {
        let cfg = InferenceBackendConfig::default();
        assert_eq!(cfg.backend, InferenceBackendKind::Burn);
        assert_eq!(cfg.candle_device, "cpu");
        assert_eq!(cfg.burn_device, "cpu");
        assert_eq!(cfg.schema_version, "1");
    }

    #[test]
    fn serialize_has_snake_case_burn() {
        let cfg = InferenceBackendConfig::default();
        let json = serde_json::to_value(&cfg).unwrap();
        assert_eq!(json["backend"], "burn");
    }

    #[test]
    fn missing_backend_defaults_to_burn() {
        let json = r#"{"candle_device": "cpu"}"#;
        let cfg: InferenceBackendConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.backend, InferenceBackendKind::Burn);
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
    fn burn_device_deserializes() {
        let json = r#"{"backend": "burn", "burn_device": "wgpu:default"}"#;
        let cfg: InferenceBackendConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.burn_device, "wgpu:default");
        assert_eq!(cfg.backend, InferenceBackendKind::Burn);
    }

    #[test]
    fn empty_json_defaults() {
        let cfg: InferenceBackendConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(cfg.backend, InferenceBackendKind::Burn);
        assert_eq!(cfg.candle_device, "cpu");
        assert_eq!(cfg.burn_device, "cpu");
        assert_eq!(cfg.schema_version, "1");
        assert_eq!(cfg.selected_instance, None);
        assert!(cfg.priority_order.is_empty());
        assert!(cfg.disabled_instances.is_empty());
        assert!(cfg.workers.is_empty());
    }

    #[test]
    fn static_workers_deserialize_with_defaults() {
        let json = r#"{
            "workers": [
                { "id": "worker-a", "address": "quic://192.168.1.10:9100" },
                {
                    "id": "worker-b",
                    "transport": "quic",
                    "address": "quic://192.168.1.11:9100",
                    "capabilities": ["load_bundle", "text_encode"],
                    "device_label": "cuda:0",
                    "fingerprint": "aabbcc"
                }
            ]
        }"#;
        let cfg: InferenceBackendConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.workers.len(), 2);
        assert_eq!(cfg.workers[0].id, "worker-a");
        assert_eq!(cfg.workers[0].transport, "quic");
        assert_eq!(cfg.workers[0].address, "quic://192.168.1.10:9100");
        assert!(cfg.workers[0].capabilities.is_empty());
        assert_eq!(cfg.workers[0].device_label, "");
        assert_eq!(cfg.workers[0].fingerprint, None);
        assert_eq!(
            cfg.workers[1].capabilities,
            vec!["load_bundle", "text_encode"]
        );
        assert_eq!(cfg.workers[1].device_label, "cuda:0");
        assert_eq!(cfg.workers[1].fingerprint.as_deref(), Some("aabbcc"));
    }

    #[test]
    fn static_workers_default_to_quic_transport() {
        let json = r#"{ "workers": [{ "id": "w", "address": "quic://10.0.0.2:9100" }] }"#;
        let cfg: InferenceBackendConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.workers[0].transport, "quic");
    }

    #[test]
    fn open_selected_instance_deserializes_without_backend_enum_variant() {
        let json = r#"{
            "selected_instance": "stub:cpu",
            "priority_order": ["stub:cpu", "candle:cpu"],
            "disabled_instances": ["candle:metal"]
        }"#;
        let cfg: InferenceBackendConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.backend, InferenceBackendKind::Burn);
        assert_eq!(cfg.candle_device, "cpu");
        assert_eq!(cfg.selected_instance.as_deref(), Some("stub:cpu"));
        assert_eq!(cfg.priority_order, vec!["stub:cpu", "candle:cpu"]);
        assert_eq!(cfg.disabled_instances, vec!["candle:metal"]);
    }
}
