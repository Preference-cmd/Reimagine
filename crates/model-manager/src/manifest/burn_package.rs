use std::collections::{BTreeMap, HashSet};
use std::path::{Component, Path, PathBuf};

use reimagine_core::model::{ModelId, ModelRole, ModelSeries, ModelVariant};
use serde::Deserialize;

use crate::{ModelManagerError, ModelManagerResult};

use super::{
    ModelComponentSource, ModelDescriptor, ModelFormat, ModelManifest, ModelRootId, ModelSource,
    ModelSourceStatus,
};

const PACKAGE_SCHEMA_VERSION: u32 = 1;
const PACKAGE_LAYOUT: &str = "burn_native_component_package";
const TARGET_BACKEND: &str = "burn";
const TARGET_CONTRACT: &str = "burn.component";
const TARGET_CONTRACT_VERSION: u32 = 1;
const MODEL_SERIES_STABLE_DIFFUSION: &str = "stable_diffusion";
const MODEL_VARIANT_SDXL: &str = "sdxl";
const REPORT_FILE_NAME: &str = "conversion-report.json";

pub async fn import_burn_package_descriptor(
    report_path: impl AsRef<Path>,
    models_dir: impl AsRef<Path>,
) -> ModelManagerResult<ModelDescriptor> {
    let report_path = report_path.as_ref();
    let models_dir = models_dir.as_ref();
    let bytes =
        tokio::fs::read(report_path)
            .await
            .map_err(|error| ModelManagerError::ReadFailed {
                path: display_path(report_path),
                message: error.to_string(),
            })?;
    let report = serde_json::from_slice::<BurnConversionReport>(&bytes).map_err(|error| {
        ModelManagerError::ManifestInvalid {
            path: display_path(report_path),
            message: error.to_string(),
        }
    })?;

    descriptor_from_burn_package_report(&report, report_path, models_dir).await
}

pub async fn upsert_burn_package_descriptor(
    manifest: &mut ModelManifest,
    report_path: impl AsRef<Path>,
    models_dir: impl AsRef<Path>,
) -> ModelManagerResult<ModelDescriptor> {
    let descriptor = import_burn_package_descriptor(report_path, models_dir).await?;
    if let Some(existing) = manifest
        .models()
        .iter()
        .find(|existing| existing.id() == descriptor.id())
        && !same_burn_package_descriptor(existing, &descriptor)
    {
        return Err(ModelManagerError::ManifestInvalid {
            path: descriptor
                .metadata()
                .get("package_report")
                .cloned()
                .unwrap_or_else(|| descriptor.source().path().to_owned()),
            message: format!(
                "descriptor id collision for Burn package import `{}`",
                descriptor.id()
            ),
        });
    }

    manifest.upsert_model(descriptor.clone());
    Ok(descriptor)
}

async fn descriptor_from_burn_package_report(
    report: &BurnConversionReport,
    report_path: &Path,
    models_dir: &Path,
) -> ModelManagerResult<ModelDescriptor> {
    let package = report
        .package
        .as_ref()
        .ok_or_else(|| invalid_report(report_path, "missing `package` section"))?;
    validate_report_shape(report, package, report_path)?;

    let package_root = package_root(report_path, package, models_dir)?;
    let package_root_relative = relative_to_models_dir(&package_root, models_dir, report_path)?;
    let report_relative = relative_to_models_dir(report_path, models_dir, report_path)?;

    let components =
        package_components(package, &package_root, &package_root_relative, report_path).await?;
    let descriptor_id = ModelId::new(format!("{}-burn", package.source.source_model_id));
    let primary_path = join_manifest_path(&package_root_relative, "diffusion/model.safetensors");

    Ok(ModelDescriptor::new(
        descriptor_id,
        ModelSeries::new(MODEL_SERIES_STABLE_DIFFUSION),
        ModelVariant::new(MODEL_VARIANT_SDXL),
        vec![
            ModelRole::CheckpointBundle,
            ModelRole::DiffusionModel,
            ModelRole::TextEncoder,
            ModelRole::Vae,
        ],
        ModelSource::relative(ModelRootId::new("base"), primary_path),
        ModelFormat::Safetensors,
    )
    .with_source_status(ModelSourceStatus::Available)
    .with_metadata("backend", TARGET_BACKEND)
    .with_metadata("converted_layout", PACKAGE_LAYOUT)
    .with_metadata("source_model_id", &package.source.source_model_id)
    .with_metadata("source_fingerprint", &package.source.source_fingerprint)
    .with_metadata("package_root", package_root_relative)
    .with_metadata("package_report", report_relative)
    .with_components(components))
}

fn validate_report_shape(
    report: &BurnConversionReport,
    package: &BurnPackageReport,
    report_path: &Path,
) -> ModelManagerResult<()> {
    if report.target_contract_version != TARGET_CONTRACT_VERSION {
        return Err(invalid_report(
            report_path,
            format!(
                "unsupported target contract version `{}`",
                report.target_contract_version
            ),
        ));
    }

    if package.schema_version != PACKAGE_SCHEMA_VERSION
        || package.layout != PACKAGE_LAYOUT
        || package.target.backend != TARGET_BACKEND
        || package.target.contract != TARGET_CONTRACT
        || package.target.contract_version != TARGET_CONTRACT_VERSION
        || package.target.model_series != MODEL_SERIES_STABLE_DIFFUSION
        || package.target.variant != MODEL_VARIANT_SDXL
    {
        return Err(invalid_report(
            report_path,
            "package metadata is not a supported Burn SDXL component package",
        ));
    }

    if package.source.source_model_id.trim().is_empty() {
        return Err(invalid_report(
            report_path,
            "package source_model_id must not be empty",
        ));
    }
    if package.source.source_fingerprint.trim().is_empty() {
        return Err(invalid_report(
            report_path,
            "package source_fingerprint must not be empty",
        ));
    }

    validate_component_set(package, report_path)
}

fn validate_component_set(
    package: &BurnPackageReport,
    report_path: &Path,
) -> ModelManagerResult<()> {
    let mut seen = HashSet::new();
    for expected in ExpectedBurnComponent::all() {
        let matching = package
            .components
            .iter()
            .filter(|component| component.component_role == expected.component_role)
            .collect::<Vec<_>>();
        if matching.len() != 1 {
            return Err(invalid_report(
                report_path,
                format!(
                    "expected exactly one `{}` component, found {}",
                    expected.component_role,
                    matching.len()
                ),
            ));
        }

        let component = matching[0];
        if component.model_role != expected.model_role
            || component.relative_path != expected.relative_path
            || component.format != "safetensors"
        {
            return Err(invalid_report(
                report_path,
                format!(
                    "component `{}` metadata is stale or incompatible",
                    expected.component
                ),
            ));
        }
        validate_component_metadata(component, expected, report_path)?;
        if !seen.insert((&component.component_role, &component.relative_path)) {
            return Err(invalid_report(
                report_path,
                format!("duplicate component `{}`", expected.component),
            ));
        }
    }

    if package.components.len() != ExpectedBurnComponent::all().len() {
        return Err(invalid_report(
            report_path,
            format!(
                "expected {} package components, found {}",
                ExpectedBurnComponent::all().len(),
                package.components.len()
            ),
        ));
    }

    Ok(())
}

fn validate_component_metadata(
    component: &BurnPackageComponentReport,
    expected: &ExpectedBurnComponent,
    report_path: &Path,
) -> ModelManagerResult<()> {
    let expected_metadata = [
        ("component", expected.component),
        ("backend", TARGET_BACKEND),
        ("converted_layout", PACKAGE_LAYOUT),
        ("contract", TARGET_CONTRACT),
        ("contract_version", "1"),
    ];

    for (key, expected_value) in expected_metadata {
        if component.metadata.get(key).map(String::as_str) != Some(expected_value) {
            return Err(invalid_report(
                report_path,
                format!(
                    "component `{}` metadata key `{key}` is stale or incompatible",
                    expected.component
                ),
            ));
        }
    }

    Ok(())
}

async fn package_components(
    package: &BurnPackageReport,
    package_root: &Path,
    package_root_relative: &str,
    report_path: &Path,
) -> ModelManagerResult<Vec<ModelComponentSource>> {
    let mut components = Vec::new();
    for expected in ExpectedBurnComponent::all() {
        let report_component = package
            .components
            .iter()
            .find(|component| component.component_role == expected.component_role)
            .expect("component set already validated");
        let relative_path = safe_package_relative_path(&report_component.relative_path)
            .map_err(|message| invalid_report(report_path, message))?;
        let component_path = package_root.join(relative_path);
        if !tokio::fs::try_exists(&component_path)
            .await
            .unwrap_or(false)
        {
            return Err(invalid_report(
                report_path,
                format!(
                    "package component `{}` file is missing at `{}`",
                    expected.component,
                    component_path.display()
                ),
            ));
        }

        let mut metadata = report_component.metadata.clone();
        metadata.insert("component".to_owned(), expected.component.to_owned());
        metadata.insert("backend".to_owned(), TARGET_BACKEND.to_owned());
        metadata.insert("converted_layout".to_owned(), PACKAGE_LAYOUT.to_owned());
        metadata.insert("contract".to_owned(), TARGET_CONTRACT.to_owned());
        metadata.insert(
            "contract_version".to_owned(),
            TARGET_CONTRACT_VERSION.to_string(),
        );
        metadata.insert(
            "source_model_id".to_owned(),
            package.source.source_model_id.clone(),
        );
        metadata.insert(
            "source_fingerprint".to_owned(),
            package.source.source_fingerprint.clone(),
        );

        components.push(component_source(
            expected.model_role_enum,
            join_manifest_path(package_root_relative, &report_component.relative_path),
            metadata,
        ));
    }

    Ok(components)
}

fn component_source(
    role: ModelRole,
    path: String,
    metadata: BTreeMap<String, String>,
) -> ModelComponentSource {
    metadata.into_iter().fold(
        ModelComponentSource::new(
            role,
            ModelSource::relative(ModelRootId::new("base"), path),
            ModelFormat::Safetensors,
        ),
        |source, (key, value)| source.with_metadata(key, value),
    )
}

fn same_burn_package_descriptor(existing: &ModelDescriptor, imported: &ModelDescriptor) -> bool {
    if existing.metadata().get("backend").map(String::as_str) != Some(TARGET_BACKEND) {
        return false;
    }

    let metadata_keys = [
        "backend",
        "converted_layout",
        "source_model_id",
        "source_fingerprint",
        "package_root",
        "package_report",
    ];
    metadata_keys
        .iter()
        .all(|key| existing.metadata().get(*key) == imported.metadata().get(*key))
        && existing.source() == imported.source()
        && same_component_sources(existing.components(), imported.components())
}

fn same_component_sources(
    existing: &[ModelComponentSource],
    imported: &[ModelComponentSource],
) -> bool {
    existing.len() == imported.len()
        && imported.iter().all(|component| {
            existing.iter().any(|existing| {
                existing.role() == component.role()
                    && existing.source() == component.source()
                    && existing.format() == component.format()
                    && existing.metadata().get("component") == component.metadata().get("component")
                    && existing.metadata().get("backend") == component.metadata().get("backend")
                    && existing.metadata().get("converted_layout")
                        == component.metadata().get("converted_layout")
                    && existing.metadata().get("contract") == component.metadata().get("contract")
                    && existing.metadata().get("contract_version")
                        == component.metadata().get("contract_version")
                    && existing.metadata().get("source_model_id")
                        == component.metadata().get("source_model_id")
                    && existing.metadata().get("source_fingerprint")
                        == component.metadata().get("source_fingerprint")
            })
        })
}

fn package_root(
    report_path: &Path,
    package: &BurnPackageReport,
    models_dir: &Path,
) -> ModelManagerResult<PathBuf> {
    if report_path.file_name().and_then(|name| name.to_str()) != Some(REPORT_FILE_NAME) {
        return Err(invalid_report(
            report_path,
            format!("Burn package report must be named `{REPORT_FILE_NAME}`"),
        ));
    }

    let report_parent = report_path.parent().ok_or_else(|| {
        invalid_report(
            report_path,
            "Burn package report must have a package directory",
        )
    })?;
    let root = safe_package_relative_path(&package.package_root)
        .map_err(|message| invalid_report(report_path, message))?;
    let package_root = normalize_package_path(report_parent.join(root));

    if !path_is_within(&package_root, models_dir) {
        return Err(invalid_report(
            report_path,
            "Burn package root must live under the models directory",
        ));
    }

    Ok(package_root)
}

fn relative_to_models_dir(
    path: &Path,
    models_dir: &Path,
    report_path: &Path,
) -> ModelManagerResult<String> {
    path.strip_prefix(models_dir)
        .map(to_manifest_path)
        .map_err(|_| {
            invalid_report(
                report_path,
                format!(
                    "path `{}` is not under models directory `{}`",
                    path.display(),
                    models_dir.display()
                ),
            )
        })
}

fn safe_package_relative_path(path: &str) -> Result<&Path, String> {
    if path.trim().is_empty() {
        return Err("package relative path must not be empty".to_owned());
    }
    let path = Path::new(path);
    if path.is_absolute() {
        return Err("package relative path must not be absolute".to_owned());
    }
    if path.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::Prefix(_) | Component::RootDir
        )
    }) {
        return Err("package relative path must stay inside the package".to_owned());
    }
    Ok(path)
}

fn path_is_within(path: &Path, parent: &Path) -> bool {
    path.strip_prefix(parent).is_ok()
}

fn normalize_package_path(path: PathBuf) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            _ => normalized.push(component.as_os_str()),
        }
    }
    normalized
}

fn join_manifest_path(root: &str, path: &str) -> String {
    format!("{root}/{path}")
}

fn to_manifest_path(path: impl AsRef<Path>) -> String {
    path.as_ref().to_string_lossy().replace('\\', "/")
}

fn display_path(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn invalid_report(path: &Path, message: impl Into<String>) -> ModelManagerError {
    ModelManagerError::ManifestInvalid {
        path: display_path(path),
        message: message.into(),
    }
}

#[derive(Debug, Deserialize)]
struct BurnConversionReport {
    target_contract_version: u32,
    package: Option<BurnPackageReport>,
}

#[derive(Debug, Deserialize)]
struct BurnPackageReport {
    schema_version: u32,
    layout: String,
    package_root: String,
    source: BurnPackageSourceReport,
    target: BurnPackageTargetReport,
    components: Vec<BurnPackageComponentReport>,
}

#[derive(Debug, Deserialize)]
struct BurnPackageSourceReport {
    source_model_id: String,
    source_fingerprint: String,
}

#[derive(Debug, Deserialize)]
struct BurnPackageTargetReport {
    backend: String,
    contract: String,
    contract_version: u32,
    model_series: String,
    variant: String,
}

#[derive(Debug, Deserialize)]
struct BurnPackageComponentReport {
    component_role: String,
    model_role: String,
    relative_path: String,
    format: String,
    #[serde(default)]
    metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy)]
struct ExpectedBurnComponent {
    component_role: &'static str,
    component: &'static str,
    model_role: &'static str,
    model_role_enum: ModelRole,
    relative_path: &'static str,
}

impl ExpectedBurnComponent {
    fn all() -> &'static [Self; 4] {
        &[
            Self {
                component_role: "diffusion",
                component: "diffusion",
                model_role: "DiffusionModel",
                model_role_enum: ModelRole::DiffusionModel,
                relative_path: "diffusion/model.safetensors",
            },
            Self {
                component_role: "vae",
                component: "vae",
                model_role: "Vae",
                model_role_enum: ModelRole::Vae,
                relative_path: "vae/model.safetensors",
            },
            Self {
                component_role: "text_encoder",
                component: "text_encoder",
                model_role: "TextEncoder",
                model_role_enum: ModelRole::TextEncoder,
                relative_path: "text_encoder/model.safetensors",
            },
            Self {
                component_role: "text_encoder_2",
                component: "text_encoder_2",
                model_role: "TextEncoder",
                model_role_enum: ModelRole::TextEncoder,
                relative_path: "text_encoder_2/model.safetensors",
            },
        ]
    }
}
