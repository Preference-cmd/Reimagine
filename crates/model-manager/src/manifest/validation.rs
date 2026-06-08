use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};

use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::event::OperationReport;
use reimagine_core::model::DiagnosticId;

use super::{
    Fingerprint, MODEL_MANIFEST_SCHEMA_VERSION, ModelDescriptor, ModelManifest, ModelRoot,
    ModelRootId, ModelRootKind, ModelSource, ModelSourceStatus,
};

pub type ManifestValidationReport = OperationReport;

pub async fn validate_manifest(
    manifest: &ModelManifest,
    models_dir: impl Into<PathBuf>,
) -> ManifestValidationReport {
    let models_dir = models_dir.into();
    let mut report = ManifestValidationReport::new();

    if manifest.schema_version() != MODEL_MANIFEST_SCHEMA_VERSION {
        report.push_diagnostic(model_diagnostic(
            "schema_version",
            None,
            Some("models/manifest.json".to_owned()),
            "MODEL_MANAGER/SCHEMA_VERSION_UNSUPPORTED",
            "schema version is unsupported",
        ));
    }

    let mut seen_ids = HashSet::new();
    for root in manifest.model_roots() {
        if !is_valid_root_path(root.path(), root.kind()) {
            report.push_diagnostic(model_diagnostic(
                "root_invalid",
                Some(root.id().as_str().to_owned()),
                Some(root.path().to_owned()),
                "MODEL_MANAGER/MODEL_ROOT_INVALID",
                "model root path is invalid",
            ));
        }
    }

    for descriptor in manifest.models() {
        let model_id = descriptor.id().as_str().to_owned();
        if !seen_ids.insert(model_id.clone()) {
            report.push_diagnostic(model_diagnostic(
                "duplicate_id",
                Some(model_id.clone()),
                Some(descriptor.source().path().to_owned()),
                "MODEL_MANAGER/MODEL_ID_DUPLICATE",
                "model id is duplicated in the manifest",
            ));
        }

        if descriptor.model_series().as_str().trim().is_empty()
            || descriptor.variant().as_str().trim().is_empty()
        {
            report.push_diagnostic(model_diagnostic(
                "descriptor_unknown",
                Some(model_id.clone()),
                Some(descriptor.source().path().to_owned()),
                "MODEL_MANAGER/MODEL_DESCRIPTOR_UNKNOWN",
                "model descriptor has unknown series or variant",
            ));
        }

        let series_unknown = descriptor.model_series().as_str() == "unknown";
        let variant_unknown = descriptor.variant().as_str() == "unknown";
        let fully_unknown = series_unknown
            && variant_unknown
            && descriptor.roles().is_empty()
            && matches!(descriptor.format(), super::ModelFormat::Unknown);

        if !fully_unknown && (series_unknown || variant_unknown) {
            report.push_diagnostic(model_diagnostic(
                "descriptor_unknown",
                Some(model_id.clone()),
                Some(descriptor.source().path().to_owned()),
                "MODEL_MANAGER/MODEL_DESCRIPTOR_UNKNOWN",
                "model descriptor has unknown series or variant",
            ));
        }

        if !fully_unknown
            && !series_unknown
            && !variant_unknown
            && !is_supported_series_variant(
                descriptor.model_series().as_str(),
                descriptor.variant().as_str(),
            )
        {
            report.push_diagnostic(model_diagnostic(
                "descriptor_unknown",
                Some(model_id.clone()),
                Some(descriptor.source().path().to_owned()),
                "MODEL_MANAGER/MODEL_DESCRIPTOR_UNKNOWN",
                "model descriptor has unsupported series or variant",
            ));
        }

        if descriptor.roles().is_empty() && !series_unknown && !variant_unknown {
            report.push_diagnostic(model_diagnostic(
                "roles_missing",
                Some(model_id.clone()),
                Some(descriptor.source().path().to_owned()),
                "MODEL_MANAGER/MODEL_ROLE_MISSING",
                "model descriptor must provide at least one role",
            ));
        }

        if !descriptor.format().is_supported() && !fully_unknown {
            report.push_diagnostic(model_diagnostic(
                "format_unsupported",
                Some(model_id.clone()),
                Some(descriptor.source().path().to_owned()),
                "MODEL_MANAGER/MODEL_FORMAT_UNSUPPORTED",
                "model format is unsupported",
            ));
        }

        validate_source_shape(&mut report, descriptor, manifest, &models_dir).await;
        validate_source_status(&mut report, descriptor, manifest, &models_dir).await;
        validate_size_and_fingerprint(&mut report, descriptor);
    }

    report
}

async fn validate_source_shape(
    report: &mut ManifestValidationReport,
    descriptor: &ModelDescriptor,
    manifest: &ModelManifest,
    models_dir: &Path,
) {
    let model_id = descriptor.id().as_str().to_owned();

    match descriptor.source() {
        ModelSource::LocalFileRelative { root_id, path } => {
            if !is_valid_relative_path(path) {
                report.push_diagnostic(model_diagnostic(
                    "source_invalid",
                    Some(model_id),
                    Some(path.clone()),
                    "MODEL_MANAGER/SOURCE_PATH_INVALID",
                    "relative model source path is invalid",
                ));
            } else if !root_exists(manifest, root_id) {
                report.push_diagnostic(model_diagnostic(
                    "source_root_missing",
                    Some(descriptor.id().as_str().to_owned()),
                    Some(path.clone()),
                    "MODEL_MANAGER/SOURCE_ROOT_MISSING",
                    "referenced model root does not exist",
                ));
            } else if let Some(root_path) =
                resolve_relative_root_path(manifest, root_id, models_dir)
                && !tokio::fs::try_exists(&root_path).await.unwrap_or(false)
            {
                report.push_diagnostic(model_diagnostic(
                    "model_root_missing",
                    Some(descriptor.id().as_str().to_owned()),
                    Some(root_path.display().to_string()),
                    "MODEL_MANAGER/MODEL_ROOT_MISSING",
                    "declared model root directory does not exist",
                ));
            }
        }
        ModelSource::LocalFileAbsolute { path } => {
            if path.trim().is_empty() || !Path::new(path).is_absolute() {
                report.push_diagnostic(model_diagnostic(
                    "source_invalid",
                    Some(model_id),
                    Some(path.clone()),
                    "MODEL_MANAGER/SOURCE_PATH_INVALID",
                    "absolute model source path is invalid",
                ));
            }
        }
    }
}

async fn validate_source_status(
    report: &mut ManifestValidationReport,
    descriptor: &ModelDescriptor,
    manifest: &ModelManifest,
    models_dir: &Path,
) {
    if let ModelSource::LocalFileRelative { root_id, .. } = descriptor.source()
        && relative_root_missing(manifest, root_id, models_dir).await
    {
        return;
    }

    let Some(source_path) = resolve_source_path(manifest, descriptor.source(), models_dir) else {
        return;
    };

    let exists = tokio::fs::try_exists(&source_path).await.unwrap_or(false);
    let status = descriptor.source_status();

    if !exists {
        report.push_diagnostic(model_diagnostic(
            "source_file_missing",
            Some(descriptor.id().as_str().to_owned()),
            Some(source_path.display().to_string()),
            "MODEL_MANAGER/SOURCE_FILE_MISSING",
            "model source file does not exist",
        ));
    }

    let inconsistent = match status {
        ModelSourceStatus::Available => !exists,
        ModelSourceStatus::Missing => exists,
        ModelSourceStatus::Stale => !exists,
        ModelSourceStatus::Unverified => false,
    };

    if inconsistent {
        report.push_diagnostic(model_diagnostic(
            "source_status_inconsistent",
            Some(descriptor.id().as_str().to_owned()),
            Some(source_path.display().to_string()),
            "MODEL_MANAGER/SOURCE_STATUS_INCONSISTENT",
            "source status does not match the source file state",
        ));
    }
}

fn validate_size_and_fingerprint(
    report: &mut ManifestValidationReport,
    descriptor: &ModelDescriptor,
) {
    if let (Some(expected), Some(observed)) =
        (descriptor.size_bytes(), descriptor.observed_size_bytes())
        && expected != observed
    {
        report.push_diagnostic(model_diagnostic(
            "size_mismatch",
            Some(descriptor.id().as_str().to_owned()),
            Some(descriptor.source().path().to_owned()),
            "MODEL_MANAGER/SIZE_MISMATCH",
            "recorded size does not match observed size",
        ));
    }

    if let Some(fingerprint) = descriptor.fingerprint()
        && !is_valid_fingerprint(fingerprint)
    {
        report.push_diagnostic(model_diagnostic(
            "fingerprint_invalid",
            Some(descriptor.id().as_str().to_owned()),
            Some(descriptor.source().path().to_owned()),
            "MODEL_MANAGER/FINGERPRINT_INVALID",
            "recorded fingerprint is invalid",
        ));
    }
}

fn find_root<'a>(manifest: &'a ModelManifest, root_id: &ModelRootId) -> Option<&'a ModelRoot> {
    manifest
        .model_roots()
        .iter()
        .find(|root| root.id() == root_id)
}

fn root_exists(manifest: &ModelManifest, root_id: &ModelRootId) -> bool {
    root_id.as_str() == "base" || find_root(manifest, root_id).is_some()
}

async fn relative_root_missing(
    manifest: &ModelManifest,
    root_id: &ModelRootId,
    models_dir: &Path,
) -> bool {
    let Some(root_path) = resolve_relative_root_path(manifest, root_id, models_dir) else {
        return false;
    };

    !tokio::fs::try_exists(root_path).await.unwrap_or(false)
}

fn resolve_source_path(
    manifest: &ModelManifest,
    source: &ModelSource,
    models_dir: &Path,
) -> Option<PathBuf> {
    match source {
        ModelSource::LocalFileRelative { root_id, path } => {
            if root_id.as_str() == "base" {
                Some(models_dir.join(path))
            } else {
                let root = find_root(manifest, root_id)?;
                Some(resolve_root_path(root, models_dir).join(path))
            }
        }
        ModelSource::LocalFileAbsolute { path } => Some(PathBuf::from(path)),
    }
}

fn resolve_relative_root_path(
    manifest: &ModelManifest,
    root_id: &ModelRootId,
    models_dir: &Path,
) -> Option<PathBuf> {
    if root_id.as_str() == "base" {
        Some(models_dir.to_path_buf())
    } else {
        let root = find_root(manifest, root_id)?;
        Some(resolve_root_path(root, models_dir))
    }
}

fn resolve_root_path(root: &ModelRoot, models_dir: &Path) -> PathBuf {
    let root_path = Path::new(root.path());
    if root_path.is_absolute() {
        root_path.to_path_buf()
    } else {
        models_dir.join(root_path)
    }
}

fn is_valid_root_path(path: &str, kind: ModelRootKind) -> bool {
    if path.trim().is_empty() {
        return false;
    }

    let candidate = Path::new(path);
    if candidate.is_absolute() {
        return true;
    }

    if matches!(kind, ModelRootKind::BasePathModels) && path == "." {
        return true;
    }

    !contains_parent_dir(candidate)
}

fn is_valid_relative_path(path: &str) -> bool {
    if path.trim().is_empty() {
        return false;
    }

    let candidate = Path::new(path);
    !candidate.is_absolute() && !contains_parent_dir(candidate)
}

fn contains_parent_dir(path: &Path) -> bool {
    path.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::Prefix(_) | Component::RootDir
        )
    })
}

fn is_valid_fingerprint(fingerprint: &Fingerprint) -> bool {
    !fingerprint.kind().trim().is_empty()
        && !fingerprint.value().trim().is_empty()
        && matches!(fingerprint.kind(), "sha256")
}

fn is_supported_series_variant(series: &str, variant: &str) -> bool {
    matches!((series, variant), ("stable_diffusion", "sdxl" | "sd15"))
}

fn model_diagnostic(
    suffix: &str,
    id: Option<String>,
    path: Option<String>,
    code: &str,
    message: &str,
) -> Diagnostic {
    let target = DiagnosticTarget::new(DiagnosticTargetDomain::new("model-manager"))
        .with_id(id.unwrap_or_else(|| "manifest".to_owned()));
    let target = if let Some(path) = path {
        target.with_path(path)
    } else {
        target
    };

    Diagnostic::new(
        DiagnosticId::new(format!("model_manager:{suffix}")),
        DiagnosticCode::new(code),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("model-manager"),
        message,
        target,
    )
}
