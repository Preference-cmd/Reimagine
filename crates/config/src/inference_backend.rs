use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::model::DiagnosticId;
use serde::{Deserialize, Serialize};

use crate::{ConfigDocument, ConfigValidationContext};

/// Supported inference backend kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum InferenceBackendKind {
    #[default]
    Burn,
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

impl Default for WorkerEndpointConfig {
    fn default() -> Self {
        Self {
            id: String::new(),
            transport: default_worker_transport(),
            address: String::new(),
            capabilities: Vec::new(),
            device_label: String::new(),
            fingerprint: None,
        }
    }
}

/// Persisted inference backend configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InferenceBackendConfig {
    #[serde(default = "default_schema_version")]
    pub schema_version: String,

    #[serde(default)]
    pub backend: InferenceBackendKind,

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

    /// Set on load when the file named the removed Candle backend; it
    /// is read as Burn and reported as a warning. Never persisted.
    #[serde(skip)]
    pub legacy_candle_backend: bool,

    /// `candle:*` instance ids dropped on load, each reported as a
    /// warning. Never persisted.
    #[serde(skip)]
    pub dropped_candle_instances: Vec<String>,
}

#[derive(Deserialize)]
struct RawInferenceBackendConfig {
    #[serde(default = "default_schema_version")]
    schema_version: String,
    #[serde(default)]
    backend: Option<String>,
    #[serde(default = "default_burn_device")]
    burn_device: String,
    #[serde(default)]
    selected_instance: Option<String>,
    #[serde(default)]
    priority_order: Vec<String>,
    #[serde(default)]
    disabled_instances: Vec<String>,
    #[serde(default)]
    workers: Vec<WorkerEndpointConfig>,
}

impl<'de> Deserialize<'de> for InferenceBackendConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawInferenceBackendConfig::deserialize(deserializer)?;
        let mut legacy_candle_backend = false;
        let backend = match raw.backend.as_deref() {
            Some("candle") => {
                legacy_candle_backend = true;
                InferenceBackendKind::Burn
            }
            Some("burn") | None => InferenceBackendKind::Burn,
            Some(other) => {
                return Err(serde::de::Error::custom(format!(
                    "unknown inference backend `{other}`"
                )));
            }
        };

        let mut dropped_candle_instances = Vec::new();
        let selected_instance = match raw.selected_instance {
            Some(inst) if inst.starts_with("candle:") => {
                dropped_candle_instances.push(inst);
                None
            }
            other => other,
        };

        let mut priority_order = Vec::with_capacity(raw.priority_order.len());
        for inst in raw.priority_order {
            if inst.starts_with("candle:") {
                dropped_candle_instances.push(inst);
            } else {
                priority_order.push(inst);
            }
        }

        let mut disabled_instances = Vec::with_capacity(raw.disabled_instances.len());
        for inst in raw.disabled_instances {
            if inst.starts_with("candle:") {
                dropped_candle_instances.push(inst);
            } else {
                disabled_instances.push(inst);
            }
        }

        Ok(Self {
            schema_version: raw.schema_version,
            backend,
            burn_device: raw.burn_device,
            selected_instance,
            priority_order,
            disabled_instances,
            workers: raw.workers,
            legacy_candle_backend,
            dropped_candle_instances,
        })
    }
}

fn default_schema_version() -> String {
    InferenceBackendConfig::SCHEMA_VERSION.to_string()
}

fn default_burn_device() -> String {
    "cpu".to_string()
}

impl Default for InferenceBackendConfig {
    fn default() -> Self {
        Self {
            schema_version: default_schema_version(),
            backend: InferenceBackendKind::default(),
            burn_device: default_burn_device(),
            selected_instance: None,
            priority_order: Vec::new(),
            disabled_instances: Vec::new(),
            workers: Vec::new(),
            legacy_candle_backend: false,
            dropped_candle_instances: Vec::new(),
        }
    }
}

impl ConfigDocument for InferenceBackendConfig {
    const KEY: &'static str = "inference_backend.json";
    const SCHEMA_VERSION: &'static str = "1";

    fn validate(&self, _context: &ConfigValidationContext) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        if self.legacy_candle_backend {
            diagnostics.push(Diagnostic::new(
                DiagnosticId::new("config:inference_backend:legacy_candle_backend"),
                DiagnosticCode::new("CONFIG/INFERENCE_BACKEND_CANDLE_REMOVED"),
                DiagnosticSeverity::Warning,
                DiagnosticSourceName::new("config"),
                "Candle backend was removed; using Burn",
                DiagnosticTarget::new(DiagnosticTargetDomain::new("config")).with_id(Self::KEY),
            ));
        }

        for instance in &self.dropped_candle_instances {
            diagnostics.push(Diagnostic::new(
                DiagnosticId::new(format!(
                    "config:inference_backend:dropped_instance:{instance}"
                )),
                DiagnosticCode::new("CONFIG/INFERENCE_BACKEND_CANDLE_REMOVED"),
                DiagnosticSeverity::Warning,
                DiagnosticSourceName::new("config"),
                format!("Candle backend was removed; dropped `{instance}` from configuration"),
                DiagnosticTarget::new(DiagnosticTargetDomain::new("config")).with_id(Self::KEY),
            ));
        }

        let mut seen = std::collections::HashSet::new();
        for worker in &self.workers {
            if worker.id.trim().is_empty() {
                diagnostics.push(worker_diagnostic(
                    "<unknown>",
                    "worker entry has an empty id",
                ));
            } else if !seen.insert(&worker.id) {
                diagnostics.push(worker_diagnostic(&worker.id, "duplicate worker id"));
            }
            if worker.address.trim().is_empty() {
                let id = if worker.id.is_empty() {
                    "<unknown>"
                } else {
                    &worker.id
                };
                diagnostics.push(worker_diagnostic(id, "worker has an empty address"));
            }
        }
        diagnostics
    }
}

fn worker_diagnostic(id: &str, message: &str) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new(format!("config:inference_backend:worker:{id}")),
        DiagnosticCode::new("CONFIG/INFERENCE_BACKEND_WORKER_INVALID"),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("config"),
        format!("worker `{id}`: {message}"),
        DiagnosticTarget::new(DiagnosticTargetDomain::new("config"))
            .with_id(InferenceBackendConfig::KEY),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConfigKey;

    #[test]
    fn default_is_burn() {
        let cfg = InferenceBackendConfig::default();
        assert_eq!(cfg.backend, InferenceBackendKind::Burn);
        assert_eq!(cfg.burn_device, "cpu");
        assert_eq!(cfg.schema_version, "1");
    }

    #[test]
    fn serialize_has_snake_case_burn() {
        let cfg = InferenceBackendConfig::default();
        let json = serde_json::to_value(&cfg).unwrap();
        assert_eq!(json["backend"], "burn");
        assert!(json.get("candle_device").is_none());
    }

    #[test]
    fn missing_backend_defaults_to_burn() {
        let json = r#"{"burn_device": "cpu"}"#;
        let cfg: InferenceBackendConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.backend, InferenceBackendKind::Burn);
    }

    #[test]
    fn old_settings_with_candle_backend_and_device_loads_as_burn_with_diagnostics() {
        let json = r#"{
            "backend": "candle",
            "candle_device": "metal",
            "burn_device": "wgpu:default",
            "selected_instance": "candle:cpu",
            "priority_order": ["candle:cpu", "burn:cpu"],
            "disabled_instances": ["candle:metal"]
        }"#;
        let cfg: InferenceBackendConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.backend, InferenceBackendKind::Burn);
        assert_eq!(cfg.burn_device, "wgpu:default");
        assert_eq!(cfg.selected_instance, None);
        assert_eq!(cfg.priority_order, vec!["burn:cpu"]);
        assert!(cfg.disabled_instances.is_empty());

        let context = ConfigValidationContext::new(
            ConfigKey::new(InferenceBackendConfig::KEY).unwrap(),
            "test/inference_backend.json",
        );
        let diagnostics = cfg.validate(&context);
        assert_eq!(diagnostics.len(), 4);
        assert!(
            diagnostics
                .iter()
                .any(|d| d.message() == "Candle backend was removed; using Burn")
        );
        assert!(
            diagnostics
                .iter()
                .any(|d| d.message().contains("candle:cpu"))
        );
        assert!(
            diagnostics
                .iter()
                .any(|d| d.message().contains("candle:metal"))
        );
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
    fn validate_rejects_empty_and_duplicate_worker_ids() {
        let cfg = InferenceBackendConfig {
            workers: vec![
                WorkerEndpointConfig {
                    id: String::new(),
                    ..WorkerEndpointConfig::default()
                },
                WorkerEndpointConfig {
                    id: "w".to_owned(),
                    ..WorkerEndpointConfig::default()
                },
                WorkerEndpointConfig {
                    id: "w".to_owned(),
                    address: "quic://10.0.0.2:9100".to_owned(),
                    ..WorkerEndpointConfig::default()
                },
            ],
            ..InferenceBackendConfig::default()
        };
        let context = ConfigValidationContext::new(
            ConfigKey::new("inference_backend.json").unwrap(),
            "test/inference_backend.json",
        );
        let diagnostics = cfg.validate(&context);
        assert_eq!(diagnostics.len(), 4);
        assert_eq!(
            diagnostics
                .iter()
                .filter(|d| d.message().contains("empty id"))
                .count(),
            1
        );
        assert_eq!(
            diagnostics
                .iter()
                .filter(|d| d.message().contains("duplicate worker id"))
                .count(),
            1
        );
        assert_eq!(
            diagnostics
                .iter()
                .filter(|d| d.message().contains("empty address"))
                .count(),
            2
        );
    }

    #[test]
    fn validate_rejects_empty_address() {
        let cfg = InferenceBackendConfig {
            workers: vec![WorkerEndpointConfig {
                id: "w".to_owned(),
                ..WorkerEndpointConfig::default()
            }],
            ..InferenceBackendConfig::default()
        };
        let context = ConfigValidationContext::new(
            ConfigKey::new("inference_backend.json").unwrap(),
            "test/inference_backend.json",
        );
        let diagnostics = cfg.validate(&context);
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].message().contains("empty address"));
    }

    #[test]
    fn validate_passes_well_formed_workers() {
        let cfg = InferenceBackendConfig {
            workers: vec![WorkerEndpointConfig {
                id: "w".to_owned(),
                address: "quic://10.0.0.2:9100".to_owned(),
                ..WorkerEndpointConfig::default()
            }],
            ..InferenceBackendConfig::default()
        };
        let context = ConfigValidationContext::new(
            ConfigKey::new("inference_backend.json").unwrap(),
            "test/inference_backend.json",
        );
        assert!(cfg.validate(&context).is_empty());
    }

    #[test]
    fn open_selected_instance_deserializes_without_backend_enum_variant() {
        let json = r#"{
            "selected_instance": "stub:cpu",
            "priority_order": ["stub:cpu", "other:cpu"],
            "disabled_instances": ["other:gpu"]
        }"#;
        let cfg: InferenceBackendConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.backend, InferenceBackendKind::Burn);
        assert_eq!(cfg.selected_instance.as_deref(), Some("stub:cpu"));
        assert_eq!(cfg.priority_order, vec!["stub:cpu", "other:cpu"]);
        assert_eq!(cfg.disabled_instances, vec!["other:gpu"]);
    }
}
