use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use reimagine_core::diagnostic::{DiagnosticTarget, DiagnosticTargetDomain};
use reimagine_core::event::{
    DomainEvent, DomainEventId, DomainEventKind, DomainEventSource, OperationReport, Timestamp,
};

use crate::manifest::resolve_source_path;
use crate::{
    Fingerprint, ModelDescriptor, ModelManagerError, ModelManagerResult, ModelManifest,
    ModelSourceStatus,
};

use super::sha256::hash_file;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FingerprintRefresh {
    descriptor: ModelDescriptor,
    report: OperationReport,
}

impl FingerprintRefresh {
    fn new(descriptor: ModelDescriptor, report: OperationReport) -> Self {
        Self { descriptor, report }
    }

    pub fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    pub fn into_descriptor(self) -> ModelDescriptor {
        self.descriptor
    }

    pub fn report(&self) -> &OperationReport {
        &self.report
    }
}

pub struct ModelFingerprintVerifier<'a> {
    manifest: &'a ModelManifest,
    models_dir: PathBuf,
}

impl<'a> ModelFingerprintVerifier<'a> {
    pub fn new(manifest: &'a ModelManifest, models_dir: impl Into<PathBuf>) -> Self {
        Self {
            manifest,
            models_dir: models_dir.into(),
        }
    }

    pub async fn refresh_descriptor(
        &self,
        descriptor: &ModelDescriptor,
    ) -> ModelManagerResult<FingerprintRefresh> {
        let source_path = resolve_source_path(self.manifest, descriptor.source(), &self.models_dir)
            .ok_or_else(|| ModelManagerError::ReadFailed {
                path: descriptor.source().path().to_owned(),
                message: "model source root could not be resolved".to_owned(),
            })?;
        let metadata = tokio::fs::metadata(&source_path).await.map_err(|error| {
            ModelManagerError::ReadFailed {
                path: source_path.display().to_string(),
                message: error.to_string(),
            }
        })?;
        let observed_size_bytes = metadata.len();
        let observed_modified_at = modified_at_string(&metadata);
        let fingerprint = Fingerprint::sha256(hash_file(&source_path).await.map_err(|error| {
            ModelManagerError::ReadFailed {
                path: source_path.display().to_string(),
                message: error.to_string(),
            }
        })?);
        let refreshed_at = now_timestamp();

        let mut refreshed = descriptor.clone();
        refreshed.set_size_bytes(Some(observed_size_bytes));
        refreshed.set_observed_size_bytes(Some(observed_size_bytes));
        refreshed.set_observed_modified_at(observed_modified_at);
        refreshed.set_fingerprint(Some(fingerprint));
        refreshed.set_verified_at(Some(refreshed_at.clone()));
        refreshed.set_source_status(ModelSourceStatus::Available);
        refreshed.set_updated_at(Some(refreshed_at.clone()));

        let event = DomainEvent::new(
            DomainEventId::new(format!(
                "model-manager:model.verified:{}",
                refreshed.id().as_str()
            )),
            DomainEventSource::new("model-manager"),
            DomainEventKind::new("model.verified"),
            Timestamp::new(refreshed_at),
        )
        .with_subject(
            DiagnosticTarget::new(DiagnosticTargetDomain::new("model-manager"))
                .with_id(refreshed.id().as_str().to_owned())
                .with_path(source_path.display().to_string()),
        );

        Ok(FingerprintRefresh::new(
            refreshed,
            OperationReport::with_event(event),
        ))
    }
}

fn modified_at_string(metadata: &std::fs::Metadata) -> Option<String> {
    let duration = metadata.modified().ok()?.duration_since(UNIX_EPOCH).ok()?;
    Some(duration.as_secs().to_string())
}

fn now_timestamp() -> String {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    duration.as_secs().to_string()
}
