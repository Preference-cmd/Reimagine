use std::collections::BTreeMap;

use reimagine_core::model::ModelRole;
use serde::{Deserialize, Serialize};

use super::{ModelFormat, ModelSource};

/// A single role-keyed component source for a [`ModelDescriptor`](super::ModelDescriptor).
///
/// Components let a stable model id resolve to multiple local files (for
/// example a split SDXL base that has separate UNet, CLIP-L, CLIP-G, and
/// VAE weights). The component shape mirrors the existing
/// [`ModelSource`] / [`ModelFormat`] descriptors so the manifest layer
/// can resolve each component to an absolute path the same way it
/// resolves the primary descriptor source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelComponentSource {
    role: ModelRole,
    source: ModelSource,
    format: ModelFormat,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    metadata: BTreeMap<String, String>,
}

impl ModelComponentSource {
    /// Build a new component source entry.
    pub fn new(role: ModelRole, source: ModelSource, format: ModelFormat) -> Self {
        Self {
            role,
            source,
            format,
            metadata: BTreeMap::new(),
        }
    }

    /// Attach a single metadata `key=value` entry. The component map is
    /// rendered into a backend metadata string by the app-host
    /// projection; keys such as `component=unet` are parsed by the
    /// Candle backend's split-source helpers.
    ///
    /// Subsequent calls with the same `key` overwrite the previous
    /// value (the underlying map is a [`BTreeMap`]). The serialised
    /// output renders entries in sorted key order.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    pub fn role(&self) -> ModelRole {
        self.role
    }

    pub fn source(&self) -> &ModelSource {
        &self.source
    }

    pub fn format(&self) -> ModelFormat {
        self.format
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// Render a stable, human-readable label for diagnostics.
    ///
    /// Prefers the `component=...` metadata key (e.g. `unet`, `clip_l`)
    /// when present and falls back to the lowercase role name. Used by
    /// the validator and resolver to compose per-component diagnostic
    /// messages such as "model component `unet` source file is missing".
    pub fn label(&self) -> String {
        let role = format!("{:?}", self.role);
        match self.metadata.get("component") {
            Some(value) => format!("{role}:{value}"),
            None => role.to_ascii_lowercase(),
        }
    }
}
