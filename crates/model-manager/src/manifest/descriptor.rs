use std::collections::BTreeMap;

use reimagine_core::model::{ModelId, ModelRole, ModelSeries, ModelVariant};
use serde::{Deserialize, Serialize};

use super::{Fingerprint, ModelFormat, ModelSource, ModelSourceStatus};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelDescriptor {
    id: ModelId,
    model_series: ModelSeries,
    variant: ModelVariant,
    roles: Vec<ModelRole>,
    source: ModelSource,
    source_status: ModelSourceStatus,
    format: ModelFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    size_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    observed_size_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    observed_modified_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fingerprint: Option<Fingerprint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    verified_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    discovered_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    updated_at: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    metadata: BTreeMap<String, String>,
}

impl ModelDescriptor {
    pub fn new(
        id: ModelId,
        model_series: ModelSeries,
        variant: ModelVariant,
        roles: Vec<ModelRole>,
        source: ModelSource,
        format: ModelFormat,
    ) -> Self {
        Self {
            id,
            model_series,
            variant,
            roles,
            source,
            source_status: ModelSourceStatus::Unverified,
            format,
            size_bytes: None,
            observed_size_bytes: None,
            observed_modified_at: None,
            fingerprint: None,
            verified_at: None,
            discovered_at: None,
            updated_at: None,
            metadata: BTreeMap::new(),
        }
    }

    pub fn with_source_status(mut self, source_status: ModelSourceStatus) -> Self {
        self.source_status = source_status;
        self
    }

    pub fn with_size_bytes(mut self, size_bytes: u64) -> Self {
        self.size_bytes = Some(size_bytes);
        self
    }

    pub fn with_observed_size_bytes(mut self, observed_size_bytes: u64) -> Self {
        self.observed_size_bytes = Some(observed_size_bytes);
        self
    }

    pub fn with_observed_modified_at(mut self, observed_modified_at: impl Into<String>) -> Self {
        self.observed_modified_at = Some(observed_modified_at.into());
        self
    }

    pub fn with_fingerprint(mut self, fingerprint: Fingerprint) -> Self {
        self.fingerprint = Some(fingerprint);
        self
    }

    pub fn with_verified_at(mut self, verified_at: impl Into<String>) -> Self {
        self.verified_at = Some(verified_at.into());
        self
    }

    pub fn with_discovered_at(mut self, discovered_at: impl Into<String>) -> Self {
        self.discovered_at = Some(discovered_at.into());
        self
    }

    pub fn with_updated_at(mut self, updated_at: impl Into<String>) -> Self {
        self.updated_at = Some(updated_at.into());
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    pub fn id(&self) -> &ModelId {
        &self.id
    }

    pub fn model_series(&self) -> &ModelSeries {
        &self.model_series
    }

    pub fn variant(&self) -> &ModelVariant {
        &self.variant
    }

    pub fn roles(&self) -> &[ModelRole] {
        &self.roles
    }

    pub fn source(&self) -> &ModelSource {
        &self.source
    }

    pub fn source_status(&self) -> ModelSourceStatus {
        self.source_status
    }

    pub fn format(&self) -> ModelFormat {
        self.format
    }

    pub fn size_bytes(&self) -> Option<u64> {
        self.size_bytes
    }

    pub fn observed_size_bytes(&self) -> Option<u64> {
        self.observed_size_bytes
    }

    pub fn fingerprint(&self) -> Option<&Fingerprint> {
        self.fingerprint.as_ref()
    }

    pub fn is_runnable_candidate(&self) -> bool {
        self.model_series.as_str() != "unknown"
            && self.variant.as_str() != "unknown"
            && !self.roles.is_empty()
            && matches!(
                self.source_status,
                ModelSourceStatus::Available | ModelSourceStatus::Unverified
            )
    }
}
