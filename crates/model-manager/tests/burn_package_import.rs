use std::path::{Path, PathBuf};

use reimagine_core::model::ModelRole;
use reimagine_model_manager::{
    ModelFormat, ModelManifest, ModelRoot, ModelSource, import_burn_package_descriptor,
    upsert_burn_package_descriptor, validate_manifest,
};

#[tokio::test]
async fn burn_package_report_projects_to_split_sdxl_descriptor() {
    let base = test_base("projects");
    let models_dir = base.join("models");
    let report_path = write_burn_package(&models_dir, "sdxl-base-1.0", "sha256-abc123").await;

    let descriptor = import_burn_package_descriptor(&report_path, &models_dir)
        .await
        .expect("burn package report should import");

    assert_eq!(descriptor.id().as_str(), "sdxl-base-1.0-burn");
    assert_eq!(descriptor.model_series().as_str(), "stable_diffusion");
    assert_eq!(descriptor.variant().as_str(), "sdxl");
    assert_eq!(
        descriptor.roles(),
        &[
            ModelRole::CheckpointBundle,
            ModelRole::DiffusionModel,
            ModelRole::TextEncoder,
            ModelRole::Vae,
        ]
    );
    assert_eq!(descriptor.format(), ModelFormat::Safetensors);
    assert!(matches!(
        descriptor.source(),
        ModelSource::LocalFileRelative { root_id, path }
            if root_id.as_str() == "base"
                && path == "converted/burn/sdxl-base-1.0/sha256-abc123/diffusion/model.safetensors"
    ));
    assert_eq!(
        descriptor.metadata().get("backend").map(String::as_str),
        Some("burn")
    );
    assert_eq!(
        descriptor
            .metadata()
            .get("converted_layout")
            .map(String::as_str),
        Some("burn_native_component_package")
    );
    assert_eq!(
        descriptor
            .metadata()
            .get("source_model_id")
            .map(String::as_str),
        Some("sdxl-base-1.0")
    );
    assert_eq!(
        descriptor
            .metadata()
            .get("source_fingerprint")
            .map(String::as_str),
        Some("sha256-abc123")
    );
    assert_eq!(
        descriptor
            .metadata()
            .get("package_root")
            .map(String::as_str),
        Some("converted/burn/sdxl-base-1.0/sha256-abc123")
    );
    assert_eq!(
        descriptor
            .metadata()
            .get("package_report")
            .map(String::as_str),
        Some("converted/burn/sdxl-base-1.0/sha256-abc123/conversion-report.json")
    );

    let components = descriptor.components();
    assert_eq!(components.len(), 4);
    assert_component(
        components,
        ModelRole::DiffusionModel,
        "diffusion",
        "converted/burn/sdxl-base-1.0/sha256-abc123/diffusion/model.safetensors",
    );
    assert_component(
        components,
        ModelRole::Vae,
        "vae",
        "converted/burn/sdxl-base-1.0/sha256-abc123/vae/model.safetensors",
    );
    assert_component(
        components,
        ModelRole::TextEncoder,
        "text_encoder",
        "converted/burn/sdxl-base-1.0/sha256-abc123/text_encoder/model.safetensors",
    );
    assert_component(
        components,
        ModelRole::TextEncoder,
        "text_encoder_2",
        "converted/burn/sdxl-base-1.0/sha256-abc123/text_encoder_2/model.safetensors",
    );

    let manifest = reimagine_model_manager::ModelManifest::new()
        .with_root(reimagine_model_manager::ModelRoot::base_models())
        .with_model(descriptor);
    let validation = validate_manifest(&manifest, &models_dir).await;
    assert!(
        validation.diagnostics().is_empty(),
        "imported descriptor should validate cleanly: {:?}",
        validation.diagnostics()
    );

    cleanup(base).await;
}

#[tokio::test]
async fn burn_package_import_reports_missing_component_before_descriptor_creation() {
    let base = test_base("missing-component");
    let models_dir = base.join("models");
    let report_path = write_burn_package(&models_dir, "sdxl-base-1.0", "sha256-abc123").await;
    tokio::fs::remove_file(
        models_dir.join("converted/burn/sdxl-base-1.0/sha256-abc123/vae/model.safetensors"),
    )
    .await
    .unwrap();

    let error = import_burn_package_descriptor(&report_path, &models_dir)
        .await
        .expect_err("missing component should reject import");

    assert!(
        error.to_string().contains("vae/model.safetensors"),
        "expected missing component path in error, got: {error}"
    );

    cleanup(base).await;
}

#[tokio::test]
async fn burn_package_import_reports_stale_component_metadata() {
    let base = test_base("stale-component-metadata");
    let models_dir = base.join("models");
    let report_path = write_burn_package(&models_dir, "sdxl-base-1.0", "sha256-abc123").await;
    let mut report = read_report_json(&report_path).await;
    report["package"]["components"][0]["metadata"]["contract"] = "burn.other".into();
    tokio::fs::write(&report_path, serde_json::to_vec_pretty(&report).unwrap())
        .await
        .unwrap();

    let error = import_burn_package_descriptor(&report_path, &models_dir)
        .await
        .expect_err("stale component metadata should reject import");

    assert!(
        error.to_string().contains("component `diffusion` metadata"),
        "expected component metadata diagnostic, got: {error}"
    );

    cleanup(base).await;
}

#[tokio::test]
async fn burn_package_upsert_is_idempotent_for_same_package_identity() {
    let base = test_base("idempotent");
    let models_dir = base.join("models");
    let report_path = write_burn_package(&models_dir, "sdxl-base-1.0", "sha256-abc123").await;
    let mut manifest = ModelManifest::new().with_root(ModelRoot::base_models());

    let first = upsert_burn_package_descriptor(&mut manifest, &report_path, &models_dir)
        .await
        .expect("first import should insert descriptor");
    let second = upsert_burn_package_descriptor(&mut manifest, &report_path, &models_dir)
        .await
        .expect("second import of same package should update deterministically");

    assert_eq!(manifest.models().len(), 1);
    assert_eq!(first.id(), second.id());
    assert_eq!(manifest.models()[0], second);

    cleanup(base).await;
}

#[tokio::test]
async fn burn_package_upsert_rejects_same_descriptor_id_with_different_identity() {
    let base = test_base("collision");
    let models_dir = base.join("models");
    let first_report = write_burn_package(&models_dir, "sdxl-base-1.0", "sha256-abc123").await;
    let second_report = write_burn_package(&models_dir, "sdxl-base-1.0", "sha256-def456").await;
    let mut manifest = ModelManifest::new().with_root(ModelRoot::base_models());
    let original = upsert_burn_package_descriptor(&mut manifest, &first_report, &models_dir)
        .await
        .expect("first import should insert descriptor");

    let error = upsert_burn_package_descriptor(&mut manifest, &second_report, &models_dir)
        .await
        .expect_err("different package identity should collide");

    assert!(
        error.to_string().contains("descriptor id collision"),
        "expected collision diagnostic, got: {error}"
    );
    assert_eq!(manifest.models(), &[original]);

    cleanup(base).await;
}

fn assert_component(
    components: &[reimagine_model_manager::ModelComponentSource],
    role: ModelRole,
    component: &str,
    path: &str,
) {
    let component_source = components
        .iter()
        .find(|source| {
            source.role() == role
                && source.metadata().get("component").map(String::as_str) == Some(component)
        })
        .expect("component source should exist");

    assert_eq!(component_source.format(), ModelFormat::Safetensors);
    assert!(matches!(
        component_source.source(),
        ModelSource::LocalFileRelative { root_id, path: actual }
            if root_id.as_str() == "base" && actual == path
    ));
    assert_eq!(
        component_source
            .metadata()
            .get("backend")
            .map(String::as_str),
        Some("burn")
    );
    assert_eq!(
        component_source
            .metadata()
            .get("converted_layout")
            .map(String::as_str),
        Some("burn_native_component_package")
    );
    assert_eq!(
        component_source
            .metadata()
            .get("contract")
            .map(String::as_str),
        Some("burn.component")
    );
    assert_eq!(
        component_source
            .metadata()
            .get("contract_version")
            .map(String::as_str),
        Some("1")
    );
    assert_eq!(
        component_source
            .metadata()
            .get("source_model_id")
            .map(String::as_str),
        Some("sdxl-base-1.0")
    );
    assert_eq!(
        component_source
            .metadata()
            .get("source_fingerprint")
            .map(String::as_str),
        Some("sha256-abc123")
    );
}

async fn write_burn_package(
    models_dir: &Path,
    source_model_id: &str,
    source_fingerprint: &str,
) -> PathBuf {
    let package_root = models_dir
        .join("converted/burn")
        .join(source_model_id)
        .join(source_fingerprint);
    for component in ["diffusion", "vae", "text_encoder", "text_encoder_2"] {
        let path = package_root.join(component).join("model.safetensors");
        tokio::fs::create_dir_all(path.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(path, b"component").await.unwrap();
    }

    let report_path = package_root.join("conversion-report.json");
    tokio::fs::write(
        &report_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "source_identity": source_model_id,
            "source_layout": "diffusers_style_split_safetensors",
            "target_contract_version": 1,
            "output_components": [],
            "mapped_tensor_count": 0,
            "ignored_tensor_families": [],
            "diagnostics": [],
            "package": {
                "schema_version": 1,
                "layout": "burn_native_component_package",
                "converter_version": "burn-sdxl-package-04c-v1",
                "package_root": ".",
                "created_at": 1,
                "source": {
                    "source_model_id": source_model_id,
                    "source_layout": "diffusers_style_split_safetensors",
                    "source_fingerprint": source_fingerprint,
                    "fingerprint_kind": "supplied",
                    "source_files": []
                },
                "target": {
                    "backend": "burn",
                    "contract": "burn.component",
                    "contract_version": 1,
                    "model_series": "stable_diffusion",
                    "variant": "sdxl"
                },
                "components": [
                    {
                        "component_role": "diffusion",
                        "model_role": "DiffusionModel",
                        "relative_path": "diffusion/model.safetensors",
                        "format": "safetensors",
                        "metadata": {
                            "component": "diffusion",
                            "backend": "burn",
                            "converted_layout": "burn_native_component_package",
                            "contract": "burn.component",
                            "contract_version": "1"
                        }
                    },
                    {
                        "component_role": "vae",
                        "model_role": "Vae",
                        "relative_path": "vae/model.safetensors",
                        "format": "safetensors",
                        "metadata": {
                            "component": "vae",
                            "backend": "burn",
                            "converted_layout": "burn_native_component_package",
                            "contract": "burn.component",
                            "contract_version": "1"
                        }
                    },
                    {
                        "component_role": "text_encoder",
                        "model_role": "TextEncoder",
                        "relative_path": "text_encoder/model.safetensors",
                        "format": "safetensors",
                        "metadata": {
                            "component": "text_encoder",
                            "backend": "burn",
                            "converted_layout": "burn_native_component_package",
                            "contract": "burn.component",
                            "contract_version": "1"
                        }
                    },
                    {
                        "component_role": "text_encoder_2",
                        "model_role": "TextEncoder",
                        "relative_path": "text_encoder_2/model.safetensors",
                        "format": "safetensors",
                        "metadata": {
                            "component": "text_encoder_2",
                            "backend": "burn",
                            "converted_layout": "burn_native_component_package",
                            "contract": "burn.component",
                            "contract_version": "1"
                        }
                    }
                ]
            }
        }))
        .unwrap(),
    )
    .await
    .unwrap();

    report_path
}

async fn read_report_json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&tokio::fs::read(path).await.unwrap()).unwrap()
}

fn test_base(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "reimagine-burn-package-import-{name}-{}",
        std::process::id()
    ))
}

async fn cleanup(path: PathBuf) {
    let _ = tokio::fs::remove_dir_all(path).await;
}
