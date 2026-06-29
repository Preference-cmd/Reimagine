use std::collections::BTreeMap;

use reimagine_inference_burn::models::stable_diffusion::sdxl::{
    BurnSdxlComponentRole, BurnSdxlContractError, BurnTensorDType, BurnTensorInventoryEntry,
    metadata_keys, validate_component_inventory,
};

fn metadata(role: BurnSdxlComponentRole) -> BTreeMap<String, String> {
    BTreeMap::from([
        (
            metadata_keys::CONTRACT.to_owned(),
            "burn.component".to_owned(),
        ),
        (metadata_keys::CONTRACT_VERSION.to_owned(), "1".to_owned()),
        (metadata_keys::BACKEND.to_owned(), "burn".to_owned()),
        (
            metadata_keys::MODEL_SERIES.to_owned(),
            "stable_diffusion".to_owned(),
        ),
        (metadata_keys::VARIANT.to_owned(), "sdxl".to_owned()),
        (
            metadata_keys::COMPONENT_ROLE.to_owned(),
            role.as_str().to_owned(),
        ),
        (
            metadata_keys::TENSOR_LAYOUT.to_owned(),
            "burn-module-snapshot".to_owned(),
        ),
        (metadata_keys::DTYPE_POLICY.to_owned(), "mixed".to_owned()),
    ])
}

fn inventory_for(role: BurnSdxlComponentRole) -> Vec<BurnTensorInventoryEntry> {
    role.contract()
        .expected_tensor_specs()
        .iter()
        .filter(|spec| spec.required)
        .map(|spec| {
            BurnTensorInventoryEntry::new(
                spec.key,
                vec![1; spec.shape.rank()],
                BurnTensorDType::F32,
            )
        })
        .collect()
}

#[test]
fn validates_each_supported_sdxl_component_role() {
    for role in BurnSdxlComponentRole::all() {
        let report = validate_component_inventory(&metadata(role), &inventory_for(role))
            .expect("valid component inventory");

        assert_eq!(report.component_role, role);
        assert!(!report.matched_required_tensors.is_empty());
        assert!(report.missing_required_tensors.is_empty());
        assert!(report.unused_tensors.is_empty());
    }
}

#[test]
fn rejects_missing_contract_metadata() {
    let mut metadata = metadata(BurnSdxlComponentRole::Diffusion);
    metadata.remove(metadata_keys::CONTRACT);

    let err =
        validate_component_inventory(&metadata, &inventory_for(BurnSdxlComponentRole::Diffusion))
            .expect_err("missing contract metadata should fail");

    assert_eq!(
        err,
        BurnSdxlContractError::MissingMetadata {
            key: metadata_keys::CONTRACT.to_owned()
        }
    );
}

#[test]
fn rejects_unknown_component_role() {
    let mut metadata = metadata(BurnSdxlComponentRole::Diffusion);
    metadata.insert(
        metadata_keys::COMPONENT_ROLE.to_owned(),
        "tokenizer".to_owned(),
    );

    let err =
        validate_component_inventory(&metadata, &inventory_for(BurnSdxlComponentRole::Diffusion))
            .expect_err("unknown component role should fail");

    assert!(matches!(
        err,
        BurnSdxlContractError::InvalidMetadata { key, .. }
            if key == metadata_keys::COMPONENT_ROLE
    ));
}

#[test]
fn rejects_unsupported_contract_version() {
    let mut metadata = metadata(BurnSdxlComponentRole::Vae);
    metadata.insert(metadata_keys::CONTRACT_VERSION.to_owned(), "2".to_owned());

    let err = validate_component_inventory(&metadata, &inventory_for(BurnSdxlComponentRole::Vae))
        .expect_err("unsupported version should fail");

    assert_eq!(
        err,
        BurnSdxlContractError::UnsupportedContractVersion {
            found: "2".to_owned()
        }
    );
}

#[test]
fn rejects_wrong_tensor_layout() {
    let role = BurnSdxlComponentRole::Vae;
    let mut metadata = metadata(role);
    metadata.insert(
        metadata_keys::TENSOR_LAYOUT.to_owned(),
        "pytorch-state-dict".to_owned(),
    );

    let err = validate_component_inventory(&metadata, &inventory_for(role))
        .expect_err("wrong tensor layout should fail");

    assert!(matches!(
        err,
        BurnSdxlContractError::InvalidMetadata { key, expected, found }
            if key == metadata_keys::TENSOR_LAYOUT
                && expected == "burn-module-snapshot"
                && found == "pytorch-state-dict"
    ));
}

#[test]
fn rejects_missing_required_tensors() {
    let err = validate_component_inventory(&metadata(BurnSdxlComponentRole::TextEncoder), &[])
        .expect_err("missing required tensors should fail");

    assert!(matches!(
        err,
        BurnSdxlContractError::MissingRequiredTensors { keys } if !keys.is_empty()
    ));
}

#[test]
fn rejects_unsupported_tensor_dtype() {
    let role = BurnSdxlComponentRole::TextEncoder2;
    let mut inventory = inventory_for(role);
    inventory[0].dtype = BurnTensorDType::Unsupported("q8".to_owned());

    let err = validate_component_inventory(&metadata(role), &inventory)
        .expect_err("unsupported dtype should fail");

    assert!(matches!(
        err,
        BurnSdxlContractError::UnsupportedTensorDType { key, .. } if key == inventory[0].key
    ));
}

#[test]
fn reports_extra_tensors_without_failing() {
    let role = BurnSdxlComponentRole::Diffusion;
    let mut inventory = inventory_for(role);
    inventory.push(BurnTensorInventoryEntry::new(
        "unexpected.extra.weight",
        vec![1, 1],
        BurnTensorDType::F32,
    ));

    let report = validate_component_inventory(&metadata(role), &inventory)
        .expect("extra tensors should be reported, not rejected");

    assert_eq!(report.unused_tensors, vec!["unexpected.extra.weight"]);
    assert_eq!(report.warnings.len(), 1);
}
