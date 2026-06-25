use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::time::UNIX_EPOCH;

use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::event::OperationReport;
use reimagine_core::model::{
    DiagnosticId, ModelId, ModelRef, ModelRole, ModelSeries, ModelVariant,
};

use crate::manifest::resolve_source_path;
use crate::{
    ModelComponentSource, ModelDescriptor, ModelFormat, ModelManifest, ModelSource,
    ModelSourceStatus,
};

use super::readiness::ModelReadinessResolver;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModelInfo {
    id: ModelId,
    model_series: ModelSeries,
    variant: ModelVariant,
    roles: Vec<ModelRole>,
    format: ModelFormat,
    source_status: ModelSourceStatus,
    source_available: bool,
}

impl ResolvedModelInfo {
    fn from_descriptor(descriptor: &ModelDescriptor, source_available: bool) -> Self {
        Self {
            id: descriptor.id().clone(),
            model_series: descriptor.model_series().clone(),
            variant: descriptor.variant().clone(),
            roles: descriptor.roles().to_vec(),
            format: descriptor.format(),
            source_status: descriptor.source_status(),
            source_available,
        }
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

    pub fn format(&self) -> ModelFormat {
        self.format
    }

    pub fn source_status(&self) -> ModelSourceStatus {
        self.source_status
    }

    pub fn source_available(&self) -> bool {
        self.source_available
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelResolution<T> {
    value: Option<T>,
    report: OperationReport,
}

impl<T> ModelResolution<T> {
    fn resolved(value: T, report: OperationReport) -> Self {
        Self {
            value: Some(value),
            report,
        }
    }

    fn blocked(report: OperationReport) -> Self {
        Self {
            value: None,
            report,
        }
    }

    pub fn is_resolved(&self) -> bool {
        self.value.is_some()
    }

    pub fn value(&self) -> Option<&T> {
        self.value.as_ref()
    }

    pub fn into_value(self) -> Option<T> {
        self.value
    }

    pub fn report(&self) -> &OperationReport {
        &self.report
    }
}

#[allow(async_fn_in_trait)]
pub trait ModelDescriptorResolver {
    async fn resolve_descriptor(&self, model_ref: &ModelRef) -> ModelResolution<ModelDescriptor>;
}

/// One role-keyed component resolved to an absolute path with per-component metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedComponent {
    role: ModelRole,
    format: ModelFormat,
    metadata: BTreeMap<String, String>,
    path: PathBuf,
    exists: bool,
}

impl ResolvedComponent {
    fn from_component(component: &ModelComponentSource, path: PathBuf, exists: bool) -> Self {
        Self {
            role: component.role(),
            format: component.format(),
            metadata: component.metadata().clone(),
            path,
            exists,
        }
    }

    pub fn role(&self) -> ModelRole {
        self.role
    }

    pub fn format(&self) -> ModelFormat {
        self.format
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    pub fn is_missing(&self) -> bool {
        !self.exists
    }
}

/// A resolved descriptor view that exposes the role-keyed components
/// alongside the descriptor itself. Returned from
/// [`ManifestModelResolver::resolve_descriptor_with_components`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedDescriptorView {
    descriptor: ModelDescriptor,
    components: Vec<ResolvedComponent>,
}

impl ResolvedDescriptorView {
    pub fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    pub fn components(&self) -> &[ResolvedComponent] {
        &self.components
    }
}

pub struct ManifestModelResolver<'a> {
    manifest: &'a ModelManifest,
    models_dir: PathBuf,
}

impl<'a> ManifestModelResolver<'a> {
    pub fn new(manifest: &'a ModelManifest, models_dir: impl Into<PathBuf>) -> Self {
        Self {
            manifest,
            models_dir: models_dir.into(),
        }
    }

    /// Resolve a [`ModelRef`] together with the role-keyed component
    /// sources declared on the matching descriptor.
    ///
    /// Behaves like [`ModelDescriptorResolver::resolve_descriptor`] for
    /// the primary descriptor identity (series/variant/role/primary
    /// source availability) and additionally resolves every
    /// [`crate::ModelComponentSource`] entry to an absolute path,
    /// reporting per-component diagnostics (missing files, duplicate
    /// role+component pairs) on the returned
    /// [`ModelResolution::report`].
    pub async fn resolve_descriptor_with_components(
        &self,
        model_ref: &ModelRef,
    ) -> ModelResolution<ResolvedDescriptorView> {
        let base = self.resolve_internal(model_ref).await;
        let report = base.report().clone();
        match base.into_value() {
            Some(descriptor) => {
                let (components, component_report) = self.resolve_components(&descriptor).await;
                let mut merged = report;
                merged.extend(component_report);
                ModelResolution::resolved(
                    ResolvedDescriptorView {
                        descriptor,
                        components,
                    },
                    merged,
                )
            }
            None => ModelResolution::blocked(report),
        }
    }

    async fn resolve_components(
        &self,
        descriptor: &ModelDescriptor,
    ) -> (Vec<ResolvedComponent>, OperationReport) {
        if descriptor.components().is_empty() {
            return (Vec::new(), OperationReport::new());
        }

        let mut report = OperationReport::new();
        let mut components = Vec::with_capacity(descriptor.components().len());
        let mut seen_keys: HashSet<(ModelRole, String)> = HashSet::new();

        for component in descriptor.components() {
            let model_id = descriptor.id().as_str();
            let component_label = component.label();

            if !component.format().is_supported() {
                report.push_diagnostic(model_diagnostic(
                    "component_format_unsupported",
                    model_id,
                    Some(component.source().path().to_owned()),
                    "MODEL_MANAGER/COMPONENT_FORMAT_UNSUPPORTED",
                    "component source format is unsupported",
                    DiagnosticSeverity::Error,
                ));
            }

            if let ModelSource::LocalFileAbsolute { path } = component.source()
                && (path.trim().is_empty() || !std::path::Path::new(path).is_absolute())
            {
                report.push_diagnostic(model_diagnostic(
                    "component_source_invalid",
                    model_id,
                    Some(path.clone()),
                    "MODEL_MANAGER/SOURCE_PATH_INVALID",
                    "component absolute source path is invalid",
                    DiagnosticSeverity::Error,
                ));
                continue;
            }

            let resolved_path =
                resolve_source_path(self.manifest, component.source(), &self.models_dir);
            let (path, path_known) = match resolved_path.as_ref() {
                Some(path) => (path.clone(), true),
                None => {
                    report.push_diagnostic(model_diagnostic(
                        "component_source_invalid",
                        model_id,
                        Some(component.source().path().to_owned()),
                        "MODEL_MANAGER/SOURCE_PATH_INVALID",
                        "component source path is invalid",
                        DiagnosticSeverity::Error,
                    ));
                    (PathBuf::from(component.source().path()), false)
                }
            };

            let exists = if path_known {
                tokio::fs::try_exists(&path).await.unwrap_or(false)
            } else {
                false
            };

            if !exists {
                report.push_diagnostic(model_diagnostic(
                    "component_source_missing",
                    model_id,
                    Some(path.display().to_string()),
                    "MODEL_MANAGER/COMPONENT_SOURCE_MISSING",
                    &format!(
                        "model component `{}` source file is missing",
                        component_label
                    ),
                    DiagnosticSeverity::Error,
                ));
            }

            if !seen_keys.insert((component.role(), component_label.clone())) {
                report.push_diagnostic(model_diagnostic(
                    "component_duplicate",
                    model_id,
                    Some(path.display().to_string()),
                    "MODEL_MANAGER/COMPONENT_DUPLICATE",
                    &format!(
                        "duplicate model component entry for role+component pair `{component_label}`"
                    ),
                    DiagnosticSeverity::Error,
                ));
            }

            components.push(ResolvedComponent::from_component(component, path, exists));
        }

        (components, report)
    }

    async fn resolve_internal(&self, model_ref: &ModelRef) -> ModelResolution<ModelDescriptor> {
        let Some(descriptor) = self
            .manifest
            .models()
            .iter()
            .find(|descriptor| descriptor.id() == model_ref.id())
        else {
            return ModelResolution::blocked(OperationReport::with_diagnostic(model_diagnostic(
                "not_found",
                model_ref.id().as_str(),
                None,
                "MODEL_MANAGER/MODEL_REF_NOT_FOUND",
                "requested model id does not exist in the manifest",
                DiagnosticSeverity::Error,
            )));
        };

        if descriptor.model_series() != model_ref.model_series() {
            return ModelResolution::blocked(OperationReport::with_diagnostic(model_diagnostic(
                "series_mismatch",
                descriptor.id().as_str(),
                Some(descriptor.source().path().to_owned()),
                "MODEL_MANAGER/MODEL_SERIES_MISMATCH",
                "requested model series does not match the manifest descriptor",
                DiagnosticSeverity::Error,
            )));
        }

        if descriptor.variant() != model_ref.variant() {
            return ModelResolution::blocked(OperationReport::with_diagnostic(model_diagnostic(
                "variant_mismatch",
                descriptor.id().as_str(),
                Some(descriptor.source().path().to_owned()),
                "MODEL_MANAGER/MODEL_VARIANT_MISMATCH",
                "requested model variant does not match the manifest descriptor",
                DiagnosticSeverity::Error,
            )));
        }

        if !descriptor.roles().contains(&model_ref.role()) {
            return ModelResolution::blocked(OperationReport::with_diagnostic(model_diagnostic(
                "role_missing",
                descriptor.id().as_str(),
                Some(descriptor.source().path().to_owned()),
                "MODEL_MANAGER/MODEL_ROLE_MISSING",
                "requested model role is not provided by the manifest descriptor",
                DiagnosticSeverity::Error,
            )));
        }

        let source_path = resolve_source_path(self.manifest, descriptor.source(), &self.models_dir);
        let source_available = match &source_path {
            Some(path) => tokio::fs::try_exists(path).await.unwrap_or(false),
            None => false,
        };

        if !source_available {
            return ModelResolution::blocked(OperationReport::with_diagnostic(model_diagnostic(
                "source_missing",
                descriptor.id().as_str(),
                source_path
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .or_else(|| Some(descriptor.source().path().to_owned())),
                "MODEL_MANAGER/MODEL_SOURCE_MISSING",
                "model source is missing",
                DiagnosticSeverity::Error,
            )));
        }

        if descriptor.source_status() == ModelSourceStatus::Missing {
            return ModelResolution::blocked(OperationReport::with_diagnostic(model_diagnostic(
                "status_missing",
                descriptor.id().as_str(),
                Some(descriptor.source().path().to_owned()),
                "MODEL_MANAGER/MODEL_SOURCE_MISSING",
                "model source is marked missing",
                DiagnosticSeverity::Error,
            )));
        }

        if descriptor.source_status() == ModelSourceStatus::Stale {
            return ModelResolution::blocked(OperationReport::with_diagnostic(model_diagnostic(
                "stale",
                descriptor.id().as_str(),
                Some(descriptor.source().path().to_owned()),
                "MODEL_MANAGER/MODEL_SOURCE_STALE",
                "model source metadata changed and requires explicit refresh",
                DiagnosticSeverity::Error,
            )));
        }

        let fingerprint_missing = descriptor.fingerprint().is_none();
        let recorded_size_mismatch = descriptor
            .size_bytes()
            .zip(descriptor.observed_size_bytes())
            .is_some_and(|(left, right)| left != right);
        let observed_metadata_mismatch = match source_path.as_ref() {
            Some(path) => match tokio::fs::metadata(path).await {
                Ok(metadata) => {
                    let size_mismatch = descriptor
                        .observed_size_bytes()
                        .is_some_and(|size| size != metadata.len());
                    let modified_mismatch = descriptor
                        .observed_modified_at()
                        .zip(modified_at_string(&metadata).as_deref())
                        .is_some_and(|(left, right)| left != right);
                    size_mismatch || modified_mismatch
                }
                Err(_) => true,
            },
            None => true,
        };

        if !fingerprint_missing && (recorded_size_mismatch || observed_metadata_mismatch) {
            return ModelResolution::blocked(OperationReport::with_diagnostic(model_diagnostic(
                "fingerprint_mismatch",
                descriptor.id().as_str(),
                Some(descriptor.source().path().to_owned()),
                "MODEL_MANAGER/MODEL_FINGERPRINT_MISMATCH",
                "recorded fingerprint can no longer be trusted for the current file",
                DiagnosticSeverity::Error,
            )));
        }

        let mut report = OperationReport::new();
        if fingerprint_missing {
            report.push_diagnostic(model_diagnostic(
                "fingerprint_missing",
                descriptor.id().as_str(),
                Some(descriptor.source().path().to_owned()),
                "MODEL_MANAGER/MODEL_FINGERPRINT_MISSING",
                "model source exists but has not been explicitly fingerprinted yet",
                DiagnosticSeverity::Warning,
            ));
        }

        ModelResolution::resolved(descriptor.clone(), report)
    }
}

impl ModelReadinessResolver for ManifestModelResolver<'_> {
    async fn resolve_readiness(&self, model_ref: &ModelRef) -> ModelResolution<ResolvedModelInfo> {
        let resolution = self.resolve_internal(model_ref).await;
        let report = resolution.report().clone();
        match resolution.into_value() {
            Some(descriptor) => ModelResolution::resolved(
                ResolvedModelInfo::from_descriptor(&descriptor, true),
                report,
            ),
            None => ModelResolution::blocked(report),
        }
    }
}

impl ModelDescriptorResolver for ManifestModelResolver<'_> {
    async fn resolve_descriptor(&self, model_ref: &ModelRef) -> ModelResolution<ModelDescriptor> {
        self.resolve_internal(model_ref).await
    }
}

fn modified_at_string(metadata: &std::fs::Metadata) -> Option<String> {
    let duration = metadata.modified().ok()?.duration_since(UNIX_EPOCH).ok()?;
    Some(duration.as_secs().to_string())
}

// TODO: consolidate duplicated `model_diagnostic` helpers
// (`manifest::validation::model_diagnostic` and
// `resolve::descriptor::model_diagnostic` both build a model-manager
// diagnostic with similar field shapes; the resolver variant takes
// a severity, the validator variant hard-codes `Error`). Move to a
// single shared helper in a small private `diagnostic` module.
fn model_diagnostic(
    suffix: &str,
    model_id: &str,
    path: Option<String>,
    code: &str,
    message: &str,
    severity: DiagnosticSeverity,
) -> Diagnostic {
    let target = DiagnosticTarget::new(DiagnosticTargetDomain::new("model-manager"))
        .with_id(model_id.to_owned());
    let target = if let Some(path) = path {
        target.with_path(path)
    } else {
        target
    };

    Diagnostic::new(
        DiagnosticId::new(format!("model_manager:resolve:{suffix}:{model_id}")),
        DiagnosticCode::new(code),
        severity,
        DiagnosticSourceName::new("model-manager"),
        message,
        target,
    )
}
