use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::Path;

use safetensors::tensor::{Dtype, SafeTensors, View, serialize_to_file};

use super::component::{BurnSdxlComponentRole, BurnTensorDType, BurnTensorInventoryEntry};
use super::conversion::{
    BURN_SDXL_CONVERSION_REPORT_FILE, BurnSdxlConversionError, BurnSdxlConversionReport,
    BurnSdxlSyntheticComponent, BurnTensorSource, InspectedBurnSdxlComponent,
    SyntheticSdxlConversionPlan, output_component_report, validation_error,
};
use super::validation::{BurnSdxlComponentValidationReport, validate_component_inventory};

type PreflightComponent<'a> = (
    &'a BurnSdxlSyntheticComponent,
    BurnSdxlComponentValidationReport,
);

#[derive(Debug, Clone)]
struct OwnedTensorView {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for OwnedTensorView {
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

pub fn write_synthetic_sdxl_components(
    plan: &SyntheticSdxlConversionPlan,
    output_dir: impl AsRef<Path>,
) -> Result<BurnSdxlConversionReport, BurnSdxlConversionError> {
    let output_dir = output_dir.as_ref();
    let preflight = preflight_plan(plan)?;

    fs::create_dir_all(output_dir).map_err(|source| BurnSdxlConversionError::Io {
        path: output_dir.to_path_buf(),
        source,
    })?;

    let mut report = BurnSdxlConversionReport::synthetic(&plan.source_identity);

    for (component, validation) in preflight {
        let relative_path = format!("{}/model.safetensors", component.role.as_str());
        let path = output_dir.join(&relative_path);
        let component_dir = path.parent().expect("component path has parent");
        fs::create_dir_all(component_dir).map_err(|source| BurnSdxlConversionError::Io {
            path: component_dir.to_path_buf(),
            source,
        })?;
        write_component_safetensors(component, &path)?;

        let inspected = inspect_component_safetensors(&path)?;
        validate_component_inventory(&inspected.metadata, &inspected.inventory)
            .map_err(|source| validation_error(component.role, source))?;

        report.mapped_tensor_count += component.tensors.len();
        report.output_components.push(output_component_report(
            component.role,
            relative_path,
            component.tensors.len(),
            &validation,
        ));
    }

    let report_path = output_dir.join(BURN_SDXL_CONVERSION_REPORT_FILE);
    write_conversion_report(&report, &report_path)?;

    Ok(report)
}

fn preflight_plan(
    plan: &SyntheticSdxlConversionPlan,
) -> Result<Vec<PreflightComponent<'_>>, BurnSdxlConversionError> {
    validate_component_roles(plan)?;

    plan.components
        .iter()
        .map(|component| {
            let metadata = component.metadata();
            let inventory = component.inventory();
            let validation = validate_component_inventory(&metadata, &inventory)
                .map_err(|source| validation_error(component.role, source))?;
            for tensor in &component.tensors {
                tensor_data(tensor)?;
            }
            Ok((component, validation))
        })
        .collect()
}

fn validate_component_roles(
    plan: &SyntheticSdxlConversionPlan,
) -> Result<(), BurnSdxlConversionError> {
    let mut seen = BTreeSet::new();
    let mut duplicates = Vec::new();

    for component in &plan.components {
        if !seen.insert(component.role.as_str()) {
            duplicates.push(component.role);
        }
    }

    if let Some(role) = duplicates.first() {
        return Err(BurnSdxlConversionError::InvalidComponentSet {
            reason: format!("duplicate Burn SDXL component role `{role}`"),
        });
    }

    let missing = BurnSdxlComponentRole::all()
        .into_iter()
        .filter(|role| !seen.contains(role.as_str()))
        .map(|role| role.as_str())
        .collect::<Vec<_>>();

    if !missing.is_empty() {
        return Err(BurnSdxlConversionError::InvalidComponentSet {
            reason: format!("missing Burn SDXL component roles: {}", missing.join(", ")),
        });
    }

    if plan.components.len() != BurnSdxlComponentRole::all().len() {
        return Err(BurnSdxlConversionError::InvalidComponentSet {
            reason: format!(
                "expected exactly {} Burn SDXL components, found {}",
                BurnSdxlComponentRole::all().len(),
                plan.components.len()
            ),
        });
    }

    Ok(())
}

pub fn write_conversion_report(
    report: &BurnSdxlConversionReport,
    path: impl AsRef<Path>,
) -> Result<(), BurnSdxlConversionError> {
    let path = path.as_ref();
    let bytes =
        serde_json::to_vec_pretty(report).map_err(|source| BurnSdxlConversionError::Json {
            path: path.to_path_buf(),
            source,
        })?;
    fs::write(path, bytes).map_err(|source| BurnSdxlConversionError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn write_component_safetensors(
    component: &BurnSdxlSyntheticComponent,
    path: &Path,
) -> Result<(), BurnSdxlConversionError> {
    let tensors = component
        .tensors
        .iter()
        .map(|tensor| tensor_data(tensor).map(|view| (tensor.key.clone(), view)))
        .collect::<Result<Vec<_>, _>>()?;
    let metadata = component
        .metadata()
        .into_iter()
        .collect::<HashMap<String, String>>();

    serialize_to_file(tensors, Some(metadata), path).map_err(|source| {
        BurnSdxlConversionError::SafetensorsWrite {
            path: path.to_path_buf(),
            source,
        }
    })
}

fn tensor_data(
    tensor: &super::conversion::BurnSyntheticTensor,
) -> Result<OwnedTensorView, BurnSdxlConversionError> {
    let dtype = safetensors_dtype(&tensor.dtype).ok_or_else(|| {
        BurnSdxlConversionError::InvalidTensorData {
            key: tensor.key.clone(),
            reason: format!("unsupported dtype `{}`", tensor.dtype.as_str()),
        }
    })?;
    let byte_len = tensor_byte_len(&tensor.shape, dtype).ok_or_else(|| {
        BurnSdxlConversionError::InvalidTensorData {
            key: tensor.key.clone(),
            reason: "tensor byte length overflowed".to_owned(),
        }
    })?;
    let data = match &tensor.source {
        BurnTensorSource::Zeros => vec![0; byte_len],
        BurnTensorSource::Data(data) => {
            if data.len() != byte_len {
                return Err(BurnSdxlConversionError::InvalidTensorData {
                    key: tensor.key.clone(),
                    reason: format!("expected {byte_len} bytes, found {}", data.len()),
                });
            }
            data.clone()
        }
    };

    Ok(OwnedTensorView {
        dtype,
        shape: tensor.shape.clone(),
        data,
    })
}

pub fn inspect_component_safetensors(
    path: impl AsRef<Path>,
) -> Result<InspectedBurnSdxlComponent, BurnSdxlConversionError> {
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|source| BurnSdxlConversionError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let (_, file_metadata) = SafeTensors::read_metadata(&bytes).map_err(|source| {
        BurnSdxlConversionError::SafetensorsReadBack {
            path: path.to_path_buf(),
            source,
        }
    })?;
    let metadata = file_metadata
        .metadata()
        .clone()
        .unwrap_or_default()
        .into_iter()
        .collect();
    let safetensors = SafeTensors::deserialize(&bytes).map_err(|source| {
        BurnSdxlConversionError::SafetensorsReadBack {
            path: path.to_path_buf(),
            source,
        }
    })?;
    let inventory = safetensors
        .names()
        .into_iter()
        .map(|name| {
            let tensor = safetensors.tensor(name).map_err(|source| {
                BurnSdxlConversionError::SafetensorsReadBack {
                    path: path.to_path_buf(),
                    source,
                }
            })?;
            Ok(BurnTensorInventoryEntry::new(
                name.to_owned(),
                tensor.shape().to_vec(),
                burn_dtype(tensor.dtype()),
            ))
        })
        .collect::<Result<Vec<_>, BurnSdxlConversionError>>()?;

    Ok(InspectedBurnSdxlComponent {
        metadata,
        inventory,
    })
}

fn safetensors_dtype(dtype: &BurnTensorDType) -> Option<Dtype> {
    match dtype {
        BurnTensorDType::F32 => Some(Dtype::F32),
        BurnTensorDType::F16 => Some(Dtype::F16),
        BurnTensorDType::Bf16 => Some(Dtype::BF16),
        BurnTensorDType::Unsupported(_) => None,
    }
}

fn burn_dtype(dtype: Dtype) -> BurnTensorDType {
    match dtype {
        Dtype::F32 => BurnTensorDType::F32,
        Dtype::F16 => BurnTensorDType::F16,
        Dtype::BF16 => BurnTensorDType::Bf16,
        other => BurnTensorDType::Unsupported(format!("{other:?}")),
    }
}

fn tensor_byte_len(shape: &[usize], dtype: Dtype) -> Option<usize> {
    let elements = shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))?;
    elements.checked_mul(dtype.bitsize())?.checked_div(8)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::super::component::{BurnSdxlComponentRole, BurnTensorDType};
    use super::super::contract::{BURN_SDXL_COMPONENT_CONTRACT_VERSION, BurnDTypePolicy};
    use super::super::conversion::{
        BurnSdxlConversionReport, BurnSdxlSyntheticComponent, BurnSyntheticTensor,
        BurnTensorSource, SyntheticSdxlConversionPlan,
    };
    use super::super::validation::validate_component_inventory;
    use super::{inspect_component_safetensors, write_synthetic_sdxl_components};

    fn required_component(role: BurnSdxlComponentRole) -> BurnSdxlSyntheticComponent {
        let tensors = role
            .contract()
            .expected_tensor_specs()
            .iter()
            .filter(|spec| spec.required)
            .map(|spec| {
                BurnSyntheticTensor::zeros(
                    spec.key,
                    vec![1; spec.shape.rank()],
                    BurnTensorDType::F32,
                )
            })
            .collect();

        BurnSdxlSyntheticComponent {
            role,
            dtype_policy: BurnDTypePolicy::Fp32,
            tensors,
        }
    }

    /// Build a synthetic text-encoder component with the full
    /// executable spec set. This generates all transformer-block
    /// tensors that the burn/08b contract requires.
    fn full_text_encoder(role: BurnSdxlComponentRole) -> BurnSdxlSyntheticComponent {
        let specs = role.contract().all_expected_tensor_specs();
        let tensors = specs
            .into_iter()
            .filter(|spec| spec.required)
            .map(|spec| {
                BurnSyntheticTensor::zeros(
                    &spec.key,
                    vec![1; spec.shape.rank()],
                    BurnTensorDType::F32,
                )
            })
            .collect();

        BurnSdxlSyntheticComponent {
            role,
            dtype_policy: BurnDTypePolicy::Fp32,
            tensors,
        }
    }

    fn all_required_components() -> Vec<BurnSdxlSyntheticComponent> {
        BurnSdxlComponentRole::all()
            .into_iter()
            .map(|role| match role {
                BurnSdxlComponentRole::TextEncoder | BurnSdxlComponentRole::TextEncoder2 => {
                    full_text_encoder(role)
                }
                _ => required_component(role),
            })
            .collect()
    }

    fn all_component_plan(source_identity: &str) -> SyntheticSdxlConversionPlan {
        SyntheticSdxlConversionPlan {
            source_identity: source_identity.to_owned(),
            components: all_required_components(),
        }
    }

    #[test]
    fn writes_all_synthetic_component_safetensors_and_validates_read_back_inventory() {
        let temp = tempfile::tempdir().expect("temp dir");
        let plan = all_component_plan("unit-test-sdxl-fixture");

        let report = write_synthetic_sdxl_components(&plan, temp.path()).expect("write components");

        assert_eq!(report.source_identity, "unit-test-sdxl-fixture");
        assert_eq!(report.source_layout, "synthetic_burn_native");
        assert_eq!(
            report.target_contract_version,
            BURN_SDXL_COMPONENT_CONTRACT_VERSION
        );
        assert_eq!(report.output_components.len(), 4);
        assert_eq!(
            report.mapped_tensor_count,
            plan.components
                .iter()
                .map(|c| c.tensors.len())
                .sum::<usize>()
        );
        assert!(report.ignored_tensor_families.is_empty());
        assert!(report.diagnostics.is_empty());

        let report_json = fs::read_to_string(temp.path().join("conversion-report.json"))
            .expect("conversion report file");
        let report_from_disk: BurnSdxlConversionReport =
            serde_json::from_str(&report_json).expect("parse report file");
        assert_eq!(report_from_disk, report);

        for role in BurnSdxlComponentRole::all() {
            let expected_path = format!("{}/model.safetensors", role.as_str());
            let output = report
                .output_components
                .iter()
                .find(|component| component.role == role)
                .expect("role output report");
            let expected_tensor_count: usize = match role {
                BurnSdxlComponentRole::TextEncoder => 148,
                BurnSdxlComponentRole::TextEncoder2 => 389,
                BurnSdxlComponentRole::Vae => 20,
                _ => 2,
            };
            assert_eq!(
                output.tensor_count, expected_tensor_count,
                "tensor count mismatch for role {role}"
            );
            assert_eq!(output.path, expected_path);
            assert!(temp.path().join(&output.path).is_file());

            let inspected = inspect_component_safetensors(temp.path().join(&output.path))
                .expect("inspect output");
            let validation =
                validate_component_inventory(&inspected.metadata, &inspected.inventory)
                    .expect("read-back inventory validates");

            assert_eq!(validation.component_role, role);
            // validate_component_inventory checks the static contract
            // specs. Text encoders still use representative specs here;
            // VAE uses the 15f decoder key-space.
            let expected_matched_required_tensors = match role {
                BurnSdxlComponentRole::Vae => 20,
                _ => 2,
            };
            assert_eq!(
                validation.matched_required_tensors.len(),
                expected_matched_required_tensors,
                "matched tensor count mismatch for role {role}"
            );
            assert_eq!(
                inspected
                    .metadata
                    .get("reimagine.dtype_policy")
                    .map(String::as_str),
                Some("fp32")
            );
            assert!(
                inspected
                    .inventory
                    .iter()
                    .all(|entry| matches!(entry.dtype, BurnTensorDType::F32))
            );
        }
    }

    #[test]
    fn rejects_synthetic_component_missing_required_tensor_before_writing() {
        let temp = tempfile::tempdir().expect("temp dir");
        let mut plan = all_component_plan("missing-required");
        plan.components
            .iter_mut()
            .find(|component| component.role == BurnSdxlComponentRole::Vae)
            .expect("vae component")
            .tensors
            .pop();

        let err = write_synthetic_sdxl_components(&plan, temp.path())
            .expect_err("missing required tensor should fail");

        assert!(
            err.to_string()
                .contains("missing required Burn SDXL tensors")
        );
        assert!(
            !temp.path().join("vae/model.safetensors").exists(),
            "invalid component should not be written"
        );
    }

    #[test]
    fn rejects_unsupported_synthetic_tensor_dtype() {
        let temp = tempfile::tempdir().expect("temp dir");
        let mut plan = all_component_plan("unsupported-dtype");
        plan.components
            .iter_mut()
            .find(|component| component.role == BurnSdxlComponentRole::TextEncoder)
            .expect("text encoder component")
            .tensors[0]
            .dtype = BurnTensorDType::Unsupported("q8".to_owned());

        let err = write_synthetic_sdxl_components(&plan, temp.path())
            .expect_err("unsupported dtype should fail");

        assert!(err.to_string().contains("unsupported dtype `q8`"));
    }

    #[test]
    fn serializes_conversion_report_as_json() {
        let report = BurnSdxlConversionReport {
            source_identity: "fixture".to_owned(),
            source_layout: "synthetic_burn_native".to_owned(),
            target_contract_version: BURN_SDXL_COMPONENT_CONTRACT_VERSION,
            output_components: Vec::new(),
            mapped_tensor_count: 0,
            ignored_tensor_families: vec!["source.unused".to_owned()],
            diagnostics: vec!["synthetic fixture".to_owned()],
            package: None,
        };

        let json = serde_json::to_string_pretty(&report).expect("serialize report");

        assert!(json.contains("\"source_layout\": \"synthetic_burn_native\""));
        assert!(json.contains("\"target_contract_version\": 1"));
        assert!(json.contains("\"ignored_tensor_families\""));
    }

    #[test]
    fn rejects_missing_component_role_before_writing() {
        let temp = tempfile::tempdir().expect("temp dir");
        let mut plan = all_component_plan("missing-role");
        plan.components
            .retain(|component| component.role != BurnSdxlComponentRole::TextEncoder2);

        let err = write_synthetic_sdxl_components(&plan, temp.path())
            .expect_err("missing role should fail");

        assert!(
            err.to_string()
                .contains("missing Burn SDXL component roles")
        );
        assert_eq!(
            fs::read_dir(temp.path()).expect("read temp dir").count(),
            0,
            "missing role should not leave artifacts"
        );
    }

    #[test]
    fn rejects_duplicate_component_role_before_writing() {
        let temp = tempfile::tempdir().expect("temp dir");
        let mut plan = all_component_plan("duplicate-role");
        plan.components[0] = required_component(BurnSdxlComponentRole::Vae);

        let err = write_synthetic_sdxl_components(&plan, temp.path())
            .expect_err("duplicate role should fail");

        assert!(
            err.to_string()
                .contains("duplicate Burn SDXL component role `vae`")
        );
        assert_eq!(
            fs::read_dir(temp.path()).expect("read temp dir").count(),
            0,
            "duplicate role should not leave artifacts"
        );
    }

    #[test]
    fn preflight_rejects_later_invalid_component_without_partial_artifacts() {
        let temp = tempfile::tempdir().expect("temp dir");
        let mut plan = all_component_plan("later-invalid");
        plan.components
            .iter_mut()
            .find(|component| component.role == BurnSdxlComponentRole::TextEncoder2)
            .expect("text encoder 2 component")
            .tensors[0]
            .source = BurnTensorSource::Data(vec![0]);

        let err = write_synthetic_sdxl_components(&plan, temp.path())
            .expect_err("later invalid component should fail preflight");

        assert!(err.to_string().contains("expected 4 bytes, found 1"));
        assert_eq!(
            fs::read_dir(temp.path()).expect("read temp dir").count(),
            0,
            "failed preflight should not leave component directories, files, or report"
        );
    }

    #[test]
    fn writes_valid_explicit_synthetic_data_source() {
        let temp = tempfile::tempdir().expect("temp dir");
        let mut plan = all_component_plan("explicit-data-valid");
        let diffusion = plan
            .components
            .iter_mut()
            .find(|component| component.role == BurnSdxlComponentRole::Diffusion)
            .expect("diffusion component");
        diffusion.tensors[0].source = BurnTensorSource::Data(vec![1, 0, 0, 0]);
        let explicit_key = diffusion.tensors[0].key.clone();
        let explicit_shape = diffusion.tensors[0].shape.clone();

        let report =
            write_synthetic_sdxl_components(&plan, temp.path()).expect("write explicit data plan");
        assert_eq!(report.output_components.len(), 4);

        let inspected =
            inspect_component_safetensors(temp.path().join("diffusion/model.safetensors"))
                .expect("inspect diffusion output");
        let inventory = inspected
            .inventory
            .iter()
            .find(|entry| entry.key == explicit_key)
            .expect("explicit tensor inventory");
        assert_eq!(inventory.shape, explicit_shape);
        assert_eq!(inventory.dtype, BurnTensorDType::F32);
    }

    #[test]
    fn rejects_explicit_synthetic_data_byte_length_mismatch() {
        let temp = tempfile::tempdir().expect("temp dir");
        let mut plan = all_component_plan("explicit-data-mismatch");
        plan.components
            .iter_mut()
            .find(|component| component.role == BurnSdxlComponentRole::TextEncoder)
            .expect("text encoder component")
            .tensors[0]
            .source = BurnTensorSource::Data(vec![0, 0]);

        let err = write_synthetic_sdxl_components(&plan, temp.path())
            .expect_err("explicit data byte length mismatch should fail");

        assert!(err.to_string().contains("expected 4 bytes, found 2"));
        assert_eq!(
            fs::read_dir(temp.path()).expect("read temp dir").count(),
            0,
            "byte length mismatch should not leave artifacts"
        );
    }
}
