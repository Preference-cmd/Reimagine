use std::path::PathBuf;

use reimagine_config::AppPaths;
use reimagine_core::model::{ModelId, ModelRole, ModelSeries, ModelVariant};
use reimagine_model_manager::{
    Fingerprint, ManifestValidationReport, ModelDescriptor, ModelFormat, ModelManifest, ModelRoot,
    ModelRootId, ModelRootKind, ModelSource, ModelSourceStatus, validate_manifest,
};

#[tokio::test]
async fn unsupported_schema_version_produces_diagnostic() {
    let base = test_base("unsupported-schema");
    let path = base.join("models/manifest.json");
    tokio::fs::create_dir_all(path.parent().unwrap())
        .await
        .unwrap();
    tokio::fs::write(
        &path,
        serde_json::json!({
            "schema_version": "reimagine.model_manifest.v9",
            "model_roots": [],
            "models": [],
        })
        .to_string(),
    )
    .await
    .unwrap();

    let store = reimagine_model_manager::ModelManifestStore::new(AppPaths::new(base.clone()));
    let (manifest, report) = store.load().await.unwrap();

    assert_eq!(manifest.schema_version(), "reimagine.model_manifest.v9");
    assert_eq!(report.diagnostics().len(), 1);
    assert_eq!(
        report.diagnostics()[0].code().as_str(),
        "MODEL_MANAGER/SCHEMA_VERSION_UNSUPPORTED"
    );

    cleanup(base).await;
}

#[tokio::test]
async fn duplicate_ids_and_invalid_unknown_descriptor_are_reported() {
    let base = test_base("duplicate-and-unknown");
    tokio::fs::create_dir_all(&base).await.unwrap();
    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(
            descriptor("dup")
                .with_source_status(ModelSourceStatus::Available)
                .with_size_bytes(10)
                .with_observed_size_bytes(10),
        )
        .with_model(
            descriptor("dup")
                .with_source_status(ModelSourceStatus::Available)
                .with_size_bytes(10)
                .with_observed_size_bytes(10),
        )
        .with_model(
            ModelDescriptor::new(
                ModelId::new("unknown-runnable"),
                ModelSeries::new("unknown"),
                ModelVariant::new("unknown"),
                vec![ModelRole::DiffusionModel],
                ModelSource::absolute(base.join("unknown.safetensors").display().to_string()),
                ModelFormat::Safetensors,
            )
            .with_source_status(ModelSourceStatus::Available),
        );

    tokio::fs::write(base.join("unknown.safetensors"), b"weights")
        .await
        .unwrap();

    let report = validate_manifest(&manifest, base.join("models")).await;

    assert_codes(
        &report,
        &[
            "MODEL_MANAGER/MODEL_ID_DUPLICATE",
            "MODEL_MANAGER/MODEL_DESCRIPTOR_UNKNOWN",
        ],
    );

    cleanup(base).await;
}

#[tokio::test]
async fn fully_unknown_descriptor_is_allowed_without_diagnostic() {
    let base = test_base("unknown-allowed");
    tokio::fs::create_dir_all(base.join("models"))
        .await
        .unwrap();
    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(
            ModelDescriptor::new(
                ModelId::new("unknown-local-file"),
                ModelSeries::new("unknown"),
                ModelVariant::new("unknown"),
                Vec::new(),
                ModelSource::absolute(base.join("mystery.bin").display().to_string()),
                ModelFormat::Unknown,
            )
            .with_source_status(ModelSourceStatus::Unverified),
        );

    tokio::fs::write(base.join("mystery.bin"), b"weights")
        .await
        .unwrap();

    let report = validate_manifest(&manifest, base.join("models")).await;

    assert!(
        report.diagnostics().is_empty(),
        "expected no diagnostics, got {:?}",
        report
            .diagnostics()
            .iter()
            .map(|diagnostic| diagnostic.code().as_str())
            .collect::<Vec<_>>()
    );

    cleanup(base).await;
}

#[tokio::test]
async fn source_status_consistency_and_root_existence_are_reported() {
    let base = test_base("source-status");
    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_root(ModelRoot::new(
            ModelRootId::new("external"),
            "../outside",
            ModelRootKind::UserSelected,
        ))
        .with_model(
            ModelDescriptor::new(
                ModelId::new("missing-file"),
                ModelSeries::new("stable_diffusion"),
                ModelVariant::new("sd15"),
                vec![ModelRole::CheckpointBundle],
                ModelSource::relative(
                    ModelRootId::new("missing-root"),
                    "checkpoints/missing.safetensors",
                ),
                ModelFormat::Safetensors,
            )
            .with_source_status(ModelSourceStatus::Available),
        )
        .with_model(
            ModelDescriptor::new(
                ModelId::new("size-mismatch"),
                ModelSeries::new("stable_diffusion"),
                ModelVariant::new("sdxl"),
                vec![ModelRole::DiffusionModel],
                ModelSource::absolute(base.join("size-mismatch.safetensors").display().to_string()),
                ModelFormat::Safetensors,
            )
            .with_source_status(ModelSourceStatus::Missing)
            .with_size_bytes(100)
            .with_observed_size_bytes(80)
            .with_fingerprint(Fingerprint::sha256("abc123")),
        );

    tokio::fs::create_dir_all(base.join("models"))
        .await
        .unwrap();
    tokio::fs::write(base.join("size-mismatch.safetensors"), b"weights")
        .await
        .unwrap();

    let report = validate_manifest(&manifest, base.join("models")).await;

    assert_codes(
        &report,
        &[
            "MODEL_MANAGER/MODEL_ROOT_INVALID",
            "MODEL_MANAGER/SOURCE_ROOT_MISSING",
            "MODEL_MANAGER/SOURCE_STATUS_INCONSISTENT",
            "MODEL_MANAGER/SIZE_MISMATCH",
        ],
    );

    cleanup(base).await;
}

#[tokio::test]
async fn declared_relative_root_missing_on_disk_is_reported() {
    let base = test_base("declared-root-missing");
    tokio::fs::create_dir_all(base.join("models"))
        .await
        .unwrap();

    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_root(ModelRoot::new(
            ModelRootId::new("external"),
            "missing-root",
            ModelRootKind::UserSelected,
        ))
        .with_model(
            ModelDescriptor::new(
                ModelId::new("declared-root-missing"),
                ModelSeries::new("stable_diffusion"),
                ModelVariant::new("sdxl"),
                vec![ModelRole::DiffusionModel],
                ModelSource::relative(ModelRootId::new("external"), "file.safetensors"),
                ModelFormat::Safetensors,
            )
            .with_source_status(ModelSourceStatus::Missing),
        );

    let report = validate_manifest(&manifest, base.join("models")).await;

    assert_codes(&report, &["MODEL_MANAGER/MODEL_ROOT_MISSING"]);
    assert_lacks_code(&report, "MODEL_MANAGER/SOURCE_ROOT_MISSING");
    assert_lacks_code(&report, "MODEL_MANAGER/SOURCE_FILE_MISSING");

    cleanup(base).await;
}

fn descriptor(id: &str) -> ModelDescriptor {
    ModelDescriptor::new(
        ModelId::new(id),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        vec![ModelRole::DiffusionModel],
        ModelSource::relative(ModelRootId::new("base"), "checkpoints/demo.safetensors"),
        ModelFormat::Safetensors,
    )
}

fn assert_codes(report: &ManifestValidationReport, expected: &[&str]) {
    let codes: Vec<_> = report
        .diagnostics()
        .iter()
        .map(|diagnostic| diagnostic.code().as_str().to_owned())
        .collect();

    for code in expected {
        assert!(
            codes.iter().any(|actual| actual == code),
            "expected diagnostic code {code} in {:?}",
            codes
        );
    }
}

fn assert_lacks_code(report: &ManifestValidationReport, unexpected: &str) {
    let codes: Vec<_> = report
        .diagnostics()
        .iter()
        .map(|diagnostic| diagnostic.code().as_str().to_owned())
        .collect();

    assert!(
        !codes.iter().any(|actual| actual == unexpected),
        "did not expect diagnostic code {unexpected} in {:?}",
        codes
    );
}

fn test_base(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "reimagine-model-manager-validation-{name}-{}",
        std::process::id()
    ))
}

async fn cleanup(path: PathBuf) {
    let _ = tokio::fs::remove_dir_all(path).await;
}
