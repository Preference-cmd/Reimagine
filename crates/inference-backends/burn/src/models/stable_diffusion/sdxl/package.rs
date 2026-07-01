use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::UNIX_EPOCH;

use super::component::BurnSdxlComponentRole;
use super::contract::BURN_SDXL_COMPONENT_CONTRACT_VERSION;
use super::conversion::{
    BURN_SDXL_CONVERSION_REPORT_FILE, BurnSdxlConversionError, BurnSdxlConversionReport,
    BurnSdxlPackageComponentReport, BurnSdxlPackageReport, BurnSdxlPackageSourceFileReport,
    BurnSdxlPackageSourceReport, BurnSdxlPackageTargetReport,
};
use super::source_layout::{BurnSdxlSourceSet, DIFFUSERS_STYLE_SPLIT_SAFETENSORS};
use super::source_mapping::map_diffusers_style_split_source;
use super::validation::validate_component_inventory;
use super::writer::{inspect_component_safetensors, write_conversion_report};

const PACKAGE_SCHEMA_VERSION: u32 = 1;
const PACKAGE_LAYOUT: &str = "burn_native_component_package";
const CONVERTER_VERSION: &str = "burn-sdxl-package-04c-v1";
const FINGERPRINT_KIND_SUPPLIED: &str = "supplied";
const FINGERPRINT_KIND_STAT: &str = "stat-v1";
const PORTABLE_PACKAGE_ROOT: &str = ".";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BurnSdxlPackageRequest {
    pub(crate) source_set: BurnSdxlSourceSet,
    pub(crate) source_model_id: String,
    pub(crate) source_fingerprint: Option<String>,
    pub(crate) converted_models_root: PathBuf,
    pub(crate) overwrite: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BurnSdxlPackageResult {
    pub(crate) package_root: PathBuf,
    pub(crate) report_path: PathBuf,
    pub(crate) report: BurnSdxlConversionReport,
    pub(crate) reused_existing: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PackageSourceIdentity {
    source_model_id: String,
    source_fingerprint: String,
    fingerprint_kind: String,
    source_files: Vec<BurnSdxlPackageSourceFileReport>,
}

pub(crate) fn package_diffusers_style_split_source(
    request: &BurnSdxlPackageRequest,
) -> Result<BurnSdxlPackageResult, BurnSdxlConversionError> {
    let source_identity = package_source_identity(request)?;
    let package_root = request
        .converted_models_root
        .join("burn")
        .join(&source_identity.source_model_id)
        .join(&source_identity.source_fingerprint);
    let report_path = package_root.join(BURN_SDXL_CONVERSION_REPORT_FILE);

    if package_root.exists() && !request.overwrite {
        let report = read_package_report(&report_path)?;
        validate_existing_package(&report, &package_root, &source_identity)?;
        return Ok(BurnSdxlPackageResult {
            package_root,
            report_path,
            report,
            reused_existing: true,
        });
    }

    let parent = package_root.parent().expect("package root has parent");
    fs::create_dir_all(parent).map_err(|source| BurnSdxlConversionError::Io {
        path: parent.to_path_buf(),
        source,
    })?;
    let temp_root = sibling_temp_root(&package_root);
    remove_dir_if_exists(&temp_root)?;

    let package_result = write_fresh_package(request, &source_identity, &temp_root);
    match package_result {
        Ok(report) => {
            publish_staged_package(&temp_root, &package_root, request.overwrite)?;
            Ok(BurnSdxlPackageResult {
                package_root,
                report_path,
                report,
                reused_existing: false,
            })
        }
        Err(err) => {
            let _ = remove_dir_if_exists(&temp_root);
            Err(err)
        }
    }
}

fn write_fresh_package(
    request: &BurnSdxlPackageRequest,
    source_identity: &PackageSourceIdentity,
    temp_root: &Path,
) -> Result<BurnSdxlConversionReport, BurnSdxlConversionError> {
    let mut report = map_diffusers_style_split_source(&request.source_set, temp_root)?;
    validate_package_components(temp_root)?;
    report.package = Some(package_report(source_identity, &report.output_components));
    write_conversion_report(&report, temp_root.join(BURN_SDXL_CONVERSION_REPORT_FILE))?;
    Ok(report)
}

fn read_package_report(path: &Path) -> Result<BurnSdxlConversionReport, BurnSdxlConversionError> {
    let json = fs::read_to_string(path).map_err(|source| BurnSdxlConversionError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&json).map_err(|source| BurnSdxlConversionError::Json {
        path: path.to_path_buf(),
        source,
    })
}

fn validate_existing_package(
    report: &BurnSdxlConversionReport,
    package_root: &Path,
    expected_source: &PackageSourceIdentity,
) -> Result<(), BurnSdxlConversionError> {
    let package = report.package.as_ref().ok_or_else(stale_package_error)?;
    if report.source_layout != DIFFUSERS_STYLE_SPLIT_SAFETENSORS
        || report.target_contract_version != BURN_SDXL_COMPONENT_CONTRACT_VERSION
        || package.schema_version != PACKAGE_SCHEMA_VERSION
        || package.layout != PACKAGE_LAYOUT
        || package.converter_version != CONVERTER_VERSION
        || package.package_root != PORTABLE_PACKAGE_ROOT
        || package.source.source_model_id != expected_source.source_model_id
        || package.source.source_layout != DIFFUSERS_STYLE_SPLIT_SAFETENSORS
        || package.source.source_fingerprint != expected_source.source_fingerprint
        || package.source.fingerprint_kind != expected_source.fingerprint_kind
        || package.source.source_files != expected_source.source_files
        || package.target.backend != "burn"
        || package.target.contract != "burn.component"
        || package.target.contract_version != BURN_SDXL_COMPONENT_CONTRACT_VERSION
        || package.target.model_series != "stable_diffusion"
        || package.target.variant != "sdxl"
    {
        return Err(stale_package_error());
    }

    validate_report_component_inventory(report)?;
    validate_package_components(package_root)
}

fn validate_package_components(package_root: &Path) -> Result<(), BurnSdxlConversionError> {
    for role in BurnSdxlComponentRole::all() {
        let path = package_root.join(role.as_str()).join("model.safetensors");
        let inspected = inspect_component_safetensors(&path)?;
        validate_component_inventory(&inspected.metadata, &inspected.inventory)
            .map_err(|source| BurnSdxlConversionError::Validation { role, source })?;
    }
    Ok(())
}

fn package_report(
    source_identity: &PackageSourceIdentity,
    output_components: &[super::conversion::BurnSdxlOutputComponentReport],
) -> BurnSdxlPackageReport {
    BurnSdxlPackageReport {
        schema_version: PACKAGE_SCHEMA_VERSION,
        layout: PACKAGE_LAYOUT.to_owned(),
        converter_version: CONVERTER_VERSION.to_owned(),
        package_root: PORTABLE_PACKAGE_ROOT.to_owned(),
        created_at: now_unix_seconds(),
        source: BurnSdxlPackageSourceReport {
            source_model_id: source_identity.source_model_id.clone(),
            source_layout: DIFFUSERS_STYLE_SPLIT_SAFETENSORS.to_owned(),
            source_fingerprint: source_identity.source_fingerprint.clone(),
            fingerprint_kind: source_identity.fingerprint_kind.clone(),
            source_files: source_identity.source_files.clone(),
        },
        target: BurnSdxlPackageTargetReport {
            backend: "burn".to_owned(),
            contract: "burn.component".to_owned(),
            contract_version: BURN_SDXL_COMPONENT_CONTRACT_VERSION,
            model_series: "stable_diffusion".to_owned(),
            variant: "sdxl".to_owned(),
        },
        components: output_components
            .iter()
            .map(|component| package_component_report(component.role, component.path.clone()))
            .collect(),
    }
}

fn package_component_report(
    role: BurnSdxlComponentRole,
    relative_path: String,
) -> BurnSdxlPackageComponentReport {
    BurnSdxlPackageComponentReport {
        component_role: role,
        model_role: model_role(role).to_owned(),
        relative_path,
        format: "safetensors".to_owned(),
        metadata: BTreeMap::from([
            ("component".to_owned(), role.as_str().to_owned()),
            ("backend".to_owned(), "burn".to_owned()),
            ("converted_layout".to_owned(), PACKAGE_LAYOUT.to_owned()),
            ("contract".to_owned(), "burn.component".to_owned()),
            (
                "contract_version".to_owned(),
                BURN_SDXL_COMPONENT_CONTRACT_VERSION.to_string(),
            ),
        ]),
    }
}

fn model_role(role: BurnSdxlComponentRole) -> &'static str {
    match role {
        BurnSdxlComponentRole::Diffusion => "DiffusionModel",
        BurnSdxlComponentRole::Vae => "Vae",
        BurnSdxlComponentRole::TextEncoder | BurnSdxlComponentRole::TextEncoder2 => "TextEncoder",
    }
}

fn package_source_identity(
    request: &BurnSdxlPackageRequest,
) -> Result<PackageSourceIdentity, BurnSdxlConversionError> {
    let source_model_id = source_model_id(request)?;
    let supplied_fingerprint = request
        .source_fingerprint
        .as_ref()
        .map(|value| validate_explicit_segment("source_fingerprint", value))
        .transpose()?;
    let source_files = source_file_reports(&request.source_set)?;
    let (source_fingerprint, fingerprint_kind) = match supplied_fingerprint {
        Some(value) => (value, FINGERPRINT_KIND_SUPPLIED.to_owned()),
        None => (
            stat_fingerprint(&source_files),
            FINGERPRINT_KIND_STAT.to_owned(),
        ),
    };

    Ok(PackageSourceIdentity {
        source_model_id,
        source_fingerprint,
        fingerprint_kind,
        source_files,
    })
}

fn source_model_id(request: &BurnSdxlPackageRequest) -> Result<String, BurnSdxlConversionError> {
    if !request.source_model_id.is_empty() {
        return validate_explicit_segment("source_model_id", &request.source_model_id);
    }

    let derived = request
        .source_set
        .root()
        .file_name()
        .and_then(|name| name.to_str())
        .map(slugify_segment)
        .filter(|value| !value.is_empty());

    match derived {
        Some(value) => validate_explicit_segment("source_model_id", &value),
        None => Err(unsafe_segment_error("source_model_id")),
    }
}

fn source_file_reports(
    source_set: &BurnSdxlSourceSet,
) -> Result<Vec<BurnSdxlPackageSourceFileReport>, BurnSdxlConversionError> {
    [
        ("unet/model.safetensors", source_set.diffusion_path()),
        ("vae/model.safetensors", source_set.vae_path()),
        (
            "text_encoder/model.safetensors",
            source_set.text_encoder_path(),
        ),
        (
            "text_encoder_2/model.safetensors",
            source_set.text_encoder_2_path(),
        ),
    ]
    .into_iter()
    .map(|(relative_path, path)| source_file_report(relative_path, &path))
    .collect()
}

fn source_file_report(
    relative_path: &str,
    path: &Path,
) -> Result<BurnSdxlPackageSourceFileReport, BurnSdxlConversionError> {
    let metadata = fs::metadata(path).map_err(|source| BurnSdxlConversionError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let modified_at = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs());

    Ok(BurnSdxlPackageSourceFileReport {
        relative_path: relative_path.to_owned(),
        size_bytes: metadata.len(),
        modified_at,
        fingerprint: None,
    })
}

fn stat_fingerprint(source_files: &[BurnSdxlPackageSourceFileReport]) -> String {
    let source_facts = source_files
        .iter()
        .map(|file| {
            format!(
                "{}:{}:{}",
                file.relative_path,
                file.size_bytes,
                file.modified_at.unwrap_or_default()
            )
        })
        .collect::<Vec<_>>()
        .join("|");
    let joined = format!("{DIFFUSERS_STYLE_SPLIT_SAFETENSORS}|{source_facts}");
    format!("stat-{:016x}", stable_hash(joined.as_bytes()))
}

fn stable_hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn now_unix_seconds() -> Option<u64> {
    std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_secs())
}

fn slugify_segment(value: &str) -> String {
    let mut safe = String::new();
    let mut last_was_dash = false;
    for ch in value.chars() {
        let lower = ch.to_ascii_lowercase();
        if lower.is_ascii_alphanumeric() || lower == '.' {
            safe.push(lower);
            last_was_dash = false;
        } else if !last_was_dash {
            safe.push('-');
            last_was_dash = true;
        }
    }
    let trimmed = safe.trim_matches(['-', '.']);
    if trimmed.is_empty() || trimmed.split('.').all(str::is_empty) {
        String::new()
    } else {
        trimmed.to_owned()
    }
}

fn validate_explicit_segment(field: &str, value: &str) -> Result<String, BurnSdxlConversionError> {
    if is_safe_path_segment(value) {
        Ok(value.to_owned())
    } else {
        Err(unsafe_segment_error(field))
    }
}

fn is_safe_path_segment(value: &str) -> bool {
    if value.is_empty()
        || value == "."
        || value == ".."
        || value.starts_with('.')
        || value.contains("..")
        || value.contains('/')
        || value.contains('\\')
    {
        return false;
    }

    value
        .chars()
        .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || matches!(ch, '-' | '_' | '.'))
}

fn unsafe_segment_error(field: &str) -> BurnSdxlConversionError {
    BurnSdxlConversionError::InvalidComponentSet {
        reason: format!("unsafe {field}"),
    }
}

fn validate_report_component_inventory(
    report: &BurnSdxlConversionReport,
) -> Result<(), BurnSdxlConversionError> {
    let package = report.package.as_ref().ok_or_else(stale_package_error)?;
    if package.components.len() != BurnSdxlComponentRole::all().len()
        || report.output_components.len() != BurnSdxlComponentRole::all().len()
    {
        return Err(stale_package_error());
    }

    for role in BurnSdxlComponentRole::all() {
        let expected_path = expected_component_relative_path(role);
        let component = package
            .components
            .iter()
            .find(|component| component.component_role == role)
            .ok_or_else(stale_package_error)?;
        let output = report
            .output_components
            .iter()
            .find(|component| component.role == role)
            .ok_or_else(stale_package_error)?;

        if component.model_role != model_role(role)
            || component.relative_path != expected_path
            || !is_plain_relative_path(&component.relative_path)
            || component.format != "safetensors"
            || component.metadata != package_component_report(role, expected_path.clone()).metadata
            || output.path != expected_path
            || !is_plain_relative_path(&output.path)
        {
            return Err(stale_package_error());
        }
    }

    Ok(())
}

fn expected_component_relative_path(role: BurnSdxlComponentRole) -> String {
    format!("{}/model.safetensors", role.as_str())
}

fn publish_staged_package(
    temp_root: &Path,
    package_root: &Path,
    overwrite: bool,
) -> Result<(), BurnSdxlConversionError> {
    if !package_root.exists() {
        fs::rename(temp_root, package_root).map_err(|source| BurnSdxlConversionError::Io {
            path: package_root.to_path_buf(),
            source,
        })?;
        return Ok(());
    }

    if !overwrite {
        return Err(stale_package_error());
    }

    let backup_root = sibling_backup_root(package_root);
    remove_dir_if_exists(&backup_root)?;

    fs::rename(package_root, &backup_root).map_err(|source| BurnSdxlConversionError::Io {
        path: package_root.to_path_buf(),
        source,
    })?;

    match fs::rename(temp_root, package_root) {
        Ok(()) => {
            remove_dir_if_exists(&backup_root)?;
            Ok(())
        }
        Err(source) => {
            let err = BurnSdxlConversionError::Io {
                path: package_root.to_path_buf(),
                source,
            };
            let _ = remove_dir_if_exists(package_root);
            let _ = fs::rename(&backup_root, package_root);
            Err(err)
        }
    }
}

fn sibling_temp_root(package_root: &Path) -> PathBuf {
    let name = package_root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("package");
    package_root.with_file_name(format!(".{name}.tmp"))
}

fn sibling_backup_root(package_root: &Path) -> PathBuf {
    let name = package_root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("package");
    package_root.with_file_name(format!(".{name}.backup"))
}

fn is_plain_relative_path(path: &str) -> bool {
    let path = Path::new(path);
    !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn remove_dir_if_exists(path: &Path) -> Result<(), BurnSdxlConversionError> {
    if path.exists() {
        fs::remove_dir_all(path).map_err(|source| BurnSdxlConversionError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    }
    Ok(())
}

fn stale_package_error() -> BurnSdxlConversionError {
    BurnSdxlConversionError::InvalidComponentSet {
        reason: "stale_or_incompatible_package".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::collections::HashMap;
    use std::fs;
    use std::path::{Path, PathBuf};

    use safetensors::tensor::{Dtype, View, serialize_to_file};

    use super::super::component::BurnSdxlComponentRole;
    use super::super::conversion::{BURN_SDXL_CONVERSION_REPORT_FILE, BurnSdxlConversionReport};
    use super::super::source_layout::BurnSdxlSourceSet;
    use super::super::validation::validate_component_inventory;
    use super::super::writer::inspect_component_safetensors;
    use super::{BurnSdxlPackageRequest, package_diffusers_style_split_source};

    #[derive(Debug, Clone)]
    struct TestTensorView {
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl TestTensorView {
        fn f32(shape: Vec<usize>) -> Self {
            let len = shape.iter().product::<usize>() * 4;
            Self {
                dtype: Dtype::F32,
                shape,
                data: vec![0; len],
            }
        }
    }

    impl View for TestTensorView {
        fn dtype(&self) -> Dtype {
            self.dtype
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }

        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    fn write_source_file(path: &Path, tensors: &[(&str, Vec<usize>)]) {
        fs::create_dir_all(path.parent().expect("source path has parent")).unwrap();
        let views = tensors
            .iter()
            .map(|(name, shape)| ((*name).to_owned(), TestTensorView::f32(shape.clone())))
            .collect::<Vec<_>>();
        serialize_to_file(views, Some(HashMap::new()), path).unwrap();
    }

    fn write_complete_split_source(root: &Path) {
        write_source_file(
            &root.join("unet/model.safetensors"),
            &[
                ("conv_in.weight", vec![1, 1, 1, 1]),
                ("time_embedding.linear_1.weight", vec![1, 1]),
            ],
        );
        write_source_file(
            &root.join("vae/model.safetensors"),
            &[
                ("encoder.conv_in.weight", vec![1, 1, 1, 1]),
                ("decoder.conv_out.weight", vec![1, 1, 1, 1]),
            ],
        );
        for role_dir in ["text_encoder", "text_encoder_2"] {
            write_source_file(
                &root.join(role_dir).join("model.safetensors"),
                &[
                    (
                        "transformer.text_model.embeddings.token_embedding.weight",
                        vec![1, 1],
                    ),
                    ("transformer.text_model.final_layer_norm.weight", vec![1]),
                ],
            );
        }
    }

    fn fixed_request(
        source_root: &Path,
        models_dir: &Path,
        source_fingerprint: &str,
        overwrite: bool,
    ) -> BurnSdxlPackageRequest {
        BurnSdxlPackageRequest {
            source_set: BurnSdxlSourceSet::diffusers_style_split_safetensors(
                source_root.to_path_buf(),
            ),
            source_model_id: "sdxl-base-1.0".to_owned(),
            source_fingerprint: Some(source_fingerprint.to_owned()),
            converted_models_root: models_dir.join("converted"),
            overwrite,
        }
    }

    fn expected_fixed_package_root(models_dir: &Path, source_fingerprint: &str) -> PathBuf {
        models_dir
            .join("converted/burn")
            .join("sdxl-base-1.0")
            .join(source_fingerprint)
    }

    #[test]
    fn packages_diffusers_split_source_into_generated_burn_layout() {
        let source = tempfile::tempdir().expect("source temp dir");
        let models = tempfile::tempdir().expect("models temp dir");
        write_complete_split_source(source.path());
        let request = fixed_request(source.path(), models.path(), "fixed-source", false);

        let result = package_diffusers_style_split_source(&request).expect("package source");

        let expected_root = expected_fixed_package_root(models.path(), "fixed-source");
        assert_eq!(result.package_root, expected_root);
        assert_eq!(
            result.report_path,
            expected_root.join(BURN_SDXL_CONVERSION_REPORT_FILE)
        );
        assert!(!result.reused_existing);
        assert!(result.report_path.is_file());

        for role in BurnSdxlComponentRole::all() {
            let component_path = result
                .package_root
                .join(role.as_str())
                .join("model.safetensors");
            assert!(component_path.is_file());
            let inspected =
                inspect_component_safetensors(&component_path).expect("inspect component");
            let validation =
                validate_component_inventory(&inspected.metadata, &inspected.inventory)
                    .expect("component validates");
            assert_eq!(validation.component_role, role);
        }

        let report_json = fs::read_to_string(&result.report_path).expect("report json");
        assert!(
            !report_json.contains("reused_existing"),
            "manifest must not record per-call reuse state"
        );
        let report_from_disk: BurnSdxlConversionReport =
            serde_json::from_str(&report_json).expect("parse report");
        assert_eq!(report_from_disk, result.report);
        assert_eq!(
            report_from_disk.source_layout,
            "diffusers_style_split_safetensors"
        );
        assert_eq!(report_from_disk.mapped_tensor_count, 8);

        let package = report_from_disk.package.expect("package report");
        assert_eq!(package.schema_version, 1);
        assert_eq!(package.layout, "burn_native_component_package");
        assert_eq!(package.converter_version, "burn-sdxl-package-04c-v1");
        assert!(package.created_at.is_some());
        assert_eq!(package.package_root, ".");
        assert_eq!(package.source.source_model_id, "sdxl-base-1.0");
        assert_eq!(package.source.source_fingerprint, "fixed-source");
        assert_eq!(package.source.source_files.len(), 4);
        assert_eq!(package.target.backend, "burn");
        assert_eq!(package.target.contract, "burn.component");
        assert_eq!(package.target.contract_version, 1);
        assert_eq!(package.target.model_series, "stable_diffusion");
        assert_eq!(package.target.variant, "sdxl");
        assert_eq!(package.components.len(), 4);
        assert!(package.components.iter().all(|component| {
            component.format == "safetensors"
                && component.metadata.get("backend").map(String::as_str) == Some("burn")
                && component
                    .metadata
                    .get("converted_layout")
                    .map(String::as_str)
                    == Some("burn_native_component_package")
        }));
    }

    #[test]
    fn derives_source_model_id_and_stat_fingerprint_when_not_supplied() {
        let source = tempfile::tempdir().expect("source temp dir");
        let source_root = source.path().join("My Source Model");
        let models = tempfile::tempdir().expect("models temp dir");
        write_complete_split_source(&source_root);
        let request = BurnSdxlPackageRequest {
            source_set: BurnSdxlSourceSet::diffusers_style_split_safetensors(source_root),
            source_model_id: String::new(),
            source_fingerprint: None,
            converted_models_root: models.path().join("converted"),
            overwrite: false,
        };

        let result = package_diffusers_style_split_source(&request).expect("package source");
        let package = result.report.package.expect("package report");

        assert_eq!(package.source.source_model_id, "my-source-model");
        assert_eq!(package.source.fingerprint_kind, "stat-v1");
        assert!(package.source.source_fingerprint.starts_with("stat-"));
        assert!(
            result
                .package_root
                .starts_with(models.path().join("converted/burn/my-source-model"))
        );
    }

    #[test]
    fn rejects_unsafe_explicit_source_model_id_before_writing() {
        let source = tempfile::tempdir().expect("source temp dir");
        let source_root = source.path().join("Real Model");
        let models = tempfile::tempdir().expect("models temp dir");
        write_complete_split_source(&source_root);
        let request = BurnSdxlPackageRequest {
            source_set: BurnSdxlSourceSet::diffusers_style_split_safetensors(source_root),
            source_model_id: "../..".to_owned(),
            source_fingerprint: Some("fixed-source".to_owned()),
            converted_models_root: models.path().join("converted"),
            overwrite: false,
        };

        let err = package_diffusers_style_split_source(&request)
            .expect_err("unsafe explicit source id should fail");

        assert!(err.to_string().contains("unsafe source_model_id"));
        assert!(
            !models.path().join("converted").exists(),
            "invalid request should fail before filesystem writes"
        );
    }

    #[test]
    fn rejects_unsafe_explicit_source_fingerprint_before_writing() {
        let source = tempfile::tempdir().expect("source temp dir");
        let models = tempfile::tempdir().expect("models temp dir");
        write_complete_split_source(source.path());
        let request = BurnSdxlPackageRequest {
            source_set: BurnSdxlSourceSet::diffusers_style_split_safetensors(
                source.path().to_path_buf(),
            ),
            source_model_id: "safe-model".to_owned(),
            source_fingerprint: Some("../fingerprint".to_owned()),
            converted_models_root: models.path().join("converted"),
            overwrite: false,
        };

        let err = package_diffusers_style_split_source(&request)
            .expect_err("unsafe explicit fingerprint should fail");

        assert!(err.to_string().contains("unsafe source_fingerprint"));
        assert!(
            !models.path().join("converted").exists(),
            "invalid request should fail before filesystem writes"
        );
    }

    #[test]
    fn rejects_existing_package_with_mismatched_converter_version() {
        let source = tempfile::tempdir().expect("source temp dir");
        let models = tempfile::tempdir().expect("models temp dir");
        write_complete_split_source(source.path());
        let request = fixed_request(source.path(), models.path(), "fixed-source", false);
        let first = package_diffusers_style_split_source(&request).expect("first package");
        let mut stale_report = first.report;
        stale_report
            .package
            .as_mut()
            .expect("package report")
            .converter_version = "older-converter".to_owned();
        fs::write(
            &first.report_path,
            serde_json::to_vec_pretty(&stale_report).expect("report json"),
        )
        .unwrap();

        let err = package_diffusers_style_split_source(&request)
            .expect_err("converter mismatch should be rejected");

        assert!(err.to_string().contains("stale_or_incompatible_package"));
    }

    #[test]
    fn rejects_existing_package_with_mismatched_component_manifest() {
        let source = tempfile::tempdir().expect("source temp dir");
        let models = tempfile::tempdir().expect("models temp dir");
        write_complete_split_source(source.path());
        let request = fixed_request(source.path(), models.path(), "fixed-source", false);
        let first = package_diffusers_style_split_source(&request).expect("first package");
        let mut stale_report = first.report;
        stale_report
            .package
            .as_mut()
            .expect("package report")
            .components
            .iter_mut()
            .find(|component| component.component_role == BurnSdxlComponentRole::Diffusion)
            .expect("diffusion component")
            .relative_path = "/tmp/model.safetensors".to_owned();
        fs::write(
            &first.report_path,
            serde_json::to_vec_pretty(&stale_report).expect("report json"),
        )
        .unwrap();

        let err = package_diffusers_style_split_source(&request)
            .expect_err("component manifest mismatch should be rejected");

        assert!(err.to_string().contains("stale_or_incompatible_package"));
    }

    #[test]
    fn reuses_compatible_existing_package_without_rewriting_files() {
        let source = tempfile::tempdir().expect("source temp dir");
        let models = tempfile::tempdir().expect("models temp dir");
        write_complete_split_source(source.path());
        let request = fixed_request(source.path(), models.path(), "fixed-source", false);
        let first = package_diffusers_style_split_source(&request).expect("first package");
        let sentinel = first.package_root.join("sentinel.txt");
        fs::write(&sentinel, "do not remove").unwrap();

        let second = package_diffusers_style_split_source(&request).expect("reuse package");

        assert_eq!(second.package_root, first.package_root);
        assert_eq!(second.report_path, first.report_path);
        assert!(second.reused_existing);
        assert!(
            sentinel.is_file(),
            "reuse must not replace package contents"
        );
    }

    #[test]
    fn rejects_stale_existing_package_unless_overwrite_is_requested() {
        let source = tempfile::tempdir().expect("source temp dir");
        let models = tempfile::tempdir().expect("models temp dir");
        write_complete_split_source(source.path());
        let request = fixed_request(source.path(), models.path(), "fixed-source", false);
        let first = package_diffusers_style_split_source(&request).expect("first package");
        write_source_file(
            &source.path().join("unet/model.safetensors"),
            &[
                ("conv_in.weight", vec![1, 1, 1, 2]),
                ("time_embedding.linear_1.weight", vec![1, 1]),
            ],
        );

        let err = package_diffusers_style_split_source(&request)
            .expect_err("stale package should be rejected");

        assert!(err.to_string().contains("stale_or_incompatible_package"));
        assert!(first.package_root.is_dir());

        let overwrite_request = fixed_request(source.path(), models.path(), "fixed-source", true);
        let replaced =
            package_diffusers_style_split_source(&overwrite_request).expect("overwrite package");
        assert_eq!(replaced.package_root, first.package_root);
        assert!(!replaced.reused_existing);
        assert_eq!(
            replaced.report.package.unwrap().source.source_files.len(),
            4
        );
    }

    #[test]
    fn failed_conversion_does_not_publish_partial_package_root() {
        let source = tempfile::tempdir().expect("source temp dir");
        let models = tempfile::tempdir().expect("models temp dir");
        write_complete_split_source(source.path());
        fs::remove_file(source.path().join("text_encoder_2/model.safetensors")).unwrap();
        let request = fixed_request(source.path(), models.path(), "fixed-source", false);
        let expected_root = expected_fixed_package_root(models.path(), "fixed-source");

        let err =
            package_diffusers_style_split_source(&request).expect_err("invalid source should fail");

        assert!(err.to_string().contains("text_encoder_2/model.safetensors"));
        assert!(
            !expected_root.exists(),
            "partial package must not be published"
        );
    }

    #[test]
    fn overwrite_failed_staging_preserves_existing_package() {
        let source = tempfile::tempdir().expect("source temp dir");
        let models = tempfile::tempdir().expect("models temp dir");
        write_complete_split_source(source.path());
        let request = fixed_request(source.path(), models.path(), "fixed-source", false);
        let first = package_diffusers_style_split_source(&request).expect("first package");
        let sentinel = first.package_root.join("sentinel.txt");
        fs::write(&sentinel, "keep old package").unwrap();

        fs::remove_file(source.path().join("text_encoder_2/model.safetensors")).unwrap();
        let overwrite_request = fixed_request(source.path(), models.path(), "fixed-source", true);

        let err = package_diffusers_style_split_source(&overwrite_request)
            .expect_err("failed staging should not replace existing package");

        assert!(err.to_string().contains("text_encoder_2/model.safetensors"));
        assert!(first.package_root.is_dir());
        assert_eq!(
            fs::read_to_string(&sentinel).expect("sentinel"),
            "keep old package"
        );
    }
}
