use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::event::{
    DomainEvent, DomainEventId, DomainEventKind, DomainEventSource, OperationReport, Timestamp,
};
use reimagine_core::model::{DiagnosticId, ModelRole};

use crate::{
    ClassificationCandidate, Classifier, IdPolicy, ModelDescriptor, ModelManifest, ModelRootId,
    ModelSource, ModelSourceStatus,
};

use super::ScanObservation;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestUpdate {
    manifest: ModelManifest,
    report: OperationReport,
}

impl ManifestUpdate {
    fn new(manifest: ModelManifest, report: OperationReport) -> Self {
        Self { manifest, report }
    }

    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    pub fn into_manifest(self) -> ModelManifest {
        self.manifest
    }

    pub fn report(&self) -> &OperationReport {
        &self.report
    }
}

pub struct ManifestUpdatePolicy<'a> {
    classifier: &'a Classifier<'a>,
}

impl<'a> ManifestUpdatePolicy<'a> {
    pub fn new(classifier: &'a Classifier<'a>) -> Self {
        Self { classifier }
    }

    pub fn apply_observations(
        &self,
        manifest: ModelManifest,
        observations: &[ScanObservation],
    ) -> ManifestUpdate {
        self.apply_observations_in_scope(manifest, None, observations)
    }

    pub fn apply_root_observations(
        &self,
        manifest: ModelManifest,
        root_id: ModelRootId,
        observations: &[ScanObservation],
    ) -> ManifestUpdate {
        self.apply_observations_in_scope(manifest, Some(&root_id), observations)
    }

    fn apply_observations_in_scope(
        &self,
        mut manifest: ModelManifest,
        scanned_root: Option<&ModelRootId>,
        observations: &[ScanObservation],
    ) -> ManifestUpdate {
        let mut report = OperationReport::new();
        let observed_sources = observations
            .iter()
            .map(|observation| observation.source().clone())
            .collect::<Vec<_>>();

        for observation in observations {
            if let Some(existing) = manifest
                .models_mut()
                .iter_mut()
                .find(|model| model.source() == observation.source())
            {
                update_existing(existing, observation, &mut report);
            } else {
                add_new_model(&mut manifest, observation, self.classifier, &mut report);
            }
        }

        for descriptor in manifest.models_mut() {
            if !descriptor_in_scope(descriptor, scanned_root) {
                continue;
            }

            if !observed_sources
                .iter()
                .any(|source| source == descriptor.source())
                && descriptor.source_status() != ModelSourceStatus::Missing
            {
                descriptor.set_source_status(ModelSourceStatus::Missing);
                report.push_event(model_event(
                    "model.marked_missing",
                    descriptor.id().as_str(),
                    descriptor.source().path(),
                ));
                report.push_diagnostic(model_diagnostic(
                    "scan_missing",
                    descriptor.id().as_str(),
                    descriptor.source().path(),
                    "MODEL_MANAGER/SCAN_SOURCE_MISSING",
                    "model source was not observed during scan",
                    DiagnosticSeverity::Warning,
                ));
            }
        }

        ManifestUpdate::new(manifest, report)
    }
}

fn descriptor_in_scope(descriptor: &ModelDescriptor, scanned_root: Option<&ModelRootId>) -> bool {
    match scanned_root {
        None => true,
        Some(scanned_root) => match descriptor.source() {
            ModelSource::LocalFileRelative { root_id, .. } => root_id == scanned_root,
            ModelSource::LocalFileAbsolute { .. } => false,
        },
    }
}

fn update_existing(
    descriptor: &mut ModelDescriptor,
    observation: &ScanObservation,
    report: &mut OperationReport,
) {
    let changed = descriptor
        .observed_size_bytes()
        .is_some_and(|size| size != observation.size_bytes())
        || descriptor
            .observed_modified_at()
            .zip(observation.modified_at())
            .is_some_and(|(left, right)| left != right);

    descriptor.set_observed_size_bytes(Some(observation.size_bytes()));
    descriptor.set_observed_modified_at(observation.modified_at().map(str::to_owned));

    if changed && descriptor.source_status() != ModelSourceStatus::Stale {
        descriptor.set_source_status(ModelSourceStatus::Stale);
        report.push_event(model_event(
            "model.marked_stale",
            descriptor.id().as_str(),
            descriptor.source().path(),
        ));
        report.push_diagnostic(model_diagnostic(
            "scan_stale",
            descriptor.id().as_str(),
            descriptor.source().path(),
            "MODEL_MANAGER/SCAN_SOURCE_STALE",
            "model source metadata changed during scan",
            DiagnosticSeverity::Warning,
        ));
    }
}

fn add_new_model(
    manifest: &mut ModelManifest,
    observation: &ScanObservation,
    classifier: &Classifier<'_>,
    report: &mut OperationReport,
) {
    let candidate = ClassificationCandidate::new(
        Some(observation.root_id().clone()),
        observation.relative_path(),
        observation.filename(),
        observation.extension(),
    )
    .with_observed_format(observation.format());
    let classification = classifier.classify(&candidate);
    let role = classification
        .roles()
        .first()
        .copied()
        .unwrap_or(ModelRole::CheckpointBundle);
    let id_result = IdPolicy::new(manifest.models()).generate_auto_id_with_resolution(
        classification.model_series(),
        classification.variant(),
        role,
        observation.source(),
        None,
    );
    report.extend(id_result.report().clone());

    let descriptor = ModelDescriptor::new(
        id_result.id().clone(),
        classification.model_series().clone(),
        classification.variant().clone(),
        classification.roles().to_vec(),
        observation.source().clone(),
        classification.format().unwrap_or(observation.format()),
    )
    .with_source_status(ModelSourceStatus::Unverified)
    .with_observed_size_bytes(observation.size_bytes());
    let descriptor = if let Some(modified_at) = observation.modified_at() {
        descriptor.with_observed_modified_at(modified_at)
    } else {
        descriptor
    };
    let model_id = descriptor.id().as_str().to_owned();
    let model_path = descriptor.source().path().to_owned();

    manifest.models_mut().push(descriptor);
    report.push_event(model_event("model.added", &model_id, &model_path));
    report.push_diagnostic(model_diagnostic(
        "scan_added",
        &model_id,
        &model_path,
        "MODEL_MANAGER/SCAN_MODEL_ADDED",
        "model source was added to the manifest",
        DiagnosticSeverity::Info,
    ));
}

fn model_event(kind: &str, model_id: &str, path: &str) -> DomainEvent {
    DomainEvent::new(
        DomainEventId::new(format!("model-manager:{kind}:{model_id}")),
        DomainEventSource::new("model-manager"),
        DomainEventKind::new(kind),
        Timestamp::new("scan"),
    )
    .with_subject(
        DiagnosticTarget::new(DiagnosticTargetDomain::new("model-manager"))
            .with_id(model_id.to_owned())
            .with_path(path.to_owned()),
    )
}

fn model_diagnostic(
    suffix: &str,
    model_id: &str,
    path: &str,
    code: &str,
    message: &str,
    severity: DiagnosticSeverity,
) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new(format!("model_manager:{suffix}:{model_id}")),
        DiagnosticCode::new(code),
        severity,
        DiagnosticSourceName::new("model-manager"),
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new("model-manager"))
            .with_id(model_id.to_owned())
            .with_path(path.to_owned()),
    )
}
