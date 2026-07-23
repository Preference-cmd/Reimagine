use std::path::PathBuf;

use reimagine_config::AppPaths;
use reimagine_core::model::{ModelId, ModelRole, ModelSeries, ModelVariant};
use reimagine_model_manager::{
    Fingerprint, ManifestValidationReport, ModelComponentSource, ModelDescriptor, ModelFormat,
    ModelManifest, ModelRoot, ModelRootId, ModelRootKind, ModelSeriesConfig, ModelSeriesRule,
    ModelSource, ModelSourceStatus, validate_manifest, validate_manifest_with_series_config,
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
async fn unsupported_series_and_variant_are_reported() {
    let base = test_base("unsupported-series-variant");
    tokio::fs::create_dir_all(&base).await.unwrap();
    let source_path = base.join("unsupported.safetensors");
    tokio::fs::write(&source_path, b"weights").await.unwrap();
    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(
            ModelDescriptor::new(
                ModelId::new("unsupported-series"),
                ModelSeries::new("totally_not_a_series"),
                ModelVariant::new("totally_not_a_variant"),
                vec![ModelRole::DiffusionModel],
                ModelSource::absolute(source_path.display().to_string()),
                ModelFormat::Safetensors,
            )
            .with_source_status(ModelSourceStatus::Available),
        );

    let report = validate_manifest(&manifest, base.join("models")).await;

    assert_codes(&report, &["MODEL_MANAGER/MODEL_DESCRIPTOR_UNKNOWN"]);

    cleanup(base).await;
}

#[tokio::test]
async fn series_config_is_the_supported_series_variant_source() {
    let base = test_base("series-config-source");
    tokio::fs::create_dir_all(&base).await.unwrap();
    let source_path = base.join("custom.safetensors");
    tokio::fs::write(&source_path, b"weights").await.unwrap();
    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(
            ModelDescriptor::new(
                ModelId::new("custom-variant"),
                ModelSeries::new("stable_diffusion"),
                ModelVariant::new("sd3"),
                vec![ModelRole::DiffusionModel],
                ModelSource::absolute(source_path.display().to_string()),
                ModelFormat::Safetensors,
            )
            .with_source_status(ModelSourceStatus::Available),
        );
    let series_config = ModelSeriesConfig::default().with_rule(ModelSeriesRule::new(
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sd3"),
    ));

    let report =
        validate_manifest_with_series_config(&manifest, base.join("models"), &series_config).await;

    assert_lacks_code(&report, "MODEL_MANAGER/MODEL_DESCRIPTOR_UNKNOWN");

    cleanup(base).await;
}

#[tokio::test]
async fn empty_series_and_variant_emit_one_descriptor_diagnostic() {
    let base = test_base("empty-series-variant");
    tokio::fs::create_dir_all(&base).await.unwrap();
    let source_path = base.join("empty.safetensors");
    tokio::fs::write(&source_path, b"weights").await.unwrap();
    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(
            ModelDescriptor::new(
                ModelId::new("empty-series"),
                ModelSeries::new(""),
                ModelVariant::new(""),
                vec![ModelRole::DiffusionModel],
                ModelSource::absolute(source_path.display().to_string()),
                ModelFormat::Safetensors,
            )
            .with_source_status(ModelSourceStatus::Available),
        );

    let report = validate_manifest(&manifest, base.join("models")).await;

    assert_code_count(&report, "MODEL_MANAGER/MODEL_DESCRIPTOR_UNKNOWN", 1);

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

fn split_sdxl_descriptor(source_status: ModelSourceStatus) -> ModelDescriptor {
    ModelDescriptor::new(
        ModelId::new("sdxl-base-1.0"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        vec![
            ModelRole::CheckpointBundle,
            ModelRole::DiffusionModel,
            ModelRole::TextEncoder,
            ModelRole::Vae,
        ],
        ModelSource::relative(
            ModelRootId::new("base"),
            "sdxl-base-1.0/manifest.safetensors",
        ),
        ModelFormat::Safetensors,
    )
    .with_source_status(source_status)
    .with_component(
        ModelComponentSource::new(
            ModelRole::DiffusionModel,
            ModelSource::relative(
                ModelRootId::new("base"),
                "sdxl-base-1.0/unet/model.safetensors",
            ),
            ModelFormat::Safetensors,
        )
        .with_metadata("component", "unet"),
    )
    .with_component(
        ModelComponentSource::new(
            ModelRole::TextEncoder,
            ModelSource::relative(
                ModelRootId::new("base"),
                "sdxl-base-1.0/text_encoder/model.safetensors",
            ),
            ModelFormat::Safetensors,
        )
        .with_metadata("component", "clip_l"),
    )
    .with_component(
        ModelComponentSource::new(
            ModelRole::TextEncoder,
            ModelSource::relative(
                ModelRootId::new("base"),
                "sdxl-base-1.0/text_encoder_2/model.safetensors",
            ),
            ModelFormat::Safetensors,
        )
        .with_metadata("component", "clip_g"),
    )
    .with_component(
        ModelComponentSource::new(
            ModelRole::Vae,
            ModelSource::relative(
                ModelRootId::new("base"),
                "sdxl-base-1.0/vae/model.safetensors",
            ),
            ModelFormat::Safetensors,
        )
        .with_metadata("component", "vae"),
    )
}

#[tokio::test]
async fn missing_component_source_file_emits_component_source_missing_diagnostic() {
    let base = test_base("split-component-missing");
    tokio::fs::create_dir_all(base.join("models"))
        .await
        .unwrap();

    let unet_path = base
        .join("models")
        .join("sdxl-base-1.0")
        .join("unet")
        .join("model.safetensors");
    let clip_l_path = base
        .join("models")
        .join("sdxl-base-1.0")
        .join("text_encoder")
        .join("model.safetensors");
    let vae_path = base
        .join("models")
        .join("sdxl-base-1.0")
        .join("vae")
        .join("model.safetensors");

    tokio::fs::create_dir_all(unet_path.parent().unwrap())
        .await
        .unwrap();
    tokio::fs::create_dir_all(clip_l_path.parent().unwrap())
        .await
        .unwrap();
    tokio::fs::create_dir_all(vae_path.parent().unwrap())
        .await
        .unwrap();
    tokio::fs::write(&unet_path, b"unet-weights").await.unwrap();
    tokio::fs::write(&clip_l_path, b"clip_l-weights")
        .await
        .unwrap();
    tokio::fs::write(&vae_path, b"vae-weights").await.unwrap();
    // Intentionally do NOT create clip_g_path to test missing-component diagnostics.

    let descriptor = split_sdxl_descriptor(ModelSourceStatus::Available);

    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(descriptor);

    let report = validate_manifest(&manifest, base.join("models")).await;

    assert_codes(
        &report,
        &[
            "MODEL_MANAGER/COMPONENT_SOURCE_MISSING",
            "MODEL_MANAGER/SOURCE_FILE_MISSING",
        ],
    );

    cleanup(base).await;
}

#[tokio::test]
async fn component_with_invalid_absolute_source_path_emits_source_path_invalid() {
    let base = test_base("split-component-absolute-invalid");
    tokio::fs::create_dir_all(base.join("models"))
        .await
        .unwrap();

    let descriptor = ModelDescriptor::new(
        ModelId::new("sdxl-base-1.0"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        vec![
            ModelRole::DiffusionModel,
            ModelRole::TextEncoder,
            ModelRole::Vae,
        ],
        ModelSource::relative(
            ModelRootId::new("base"),
            "sdxl-base-1.0/manifest.safetensors",
        ),
        ModelFormat::Safetensors,
    )
    .with_source_status(ModelSourceStatus::Available)
    .with_component(
        ModelComponentSource::new(
            ModelRole::DiffusionModel,
            ModelSource::absolute("relative/path/model.safetensors".to_owned()),
            ModelFormat::Safetensors,
        )
        .with_metadata("component", "unet"),
    )
    .with_component(
        ModelComponentSource::new(
            ModelRole::TextEncoder,
            ModelSource::absolute(String::new()),
            ModelFormat::Safetensors,
        )
        .with_metadata("component", "clip_l"),
    );

    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(descriptor);

    let report = validate_manifest(&manifest, base.join("models")).await;

    let count = report
        .diagnostics()
        .iter()
        .filter(|d| d.code().as_str() == "MODEL_MANAGER/SOURCE_PATH_INVALID")
        .count();
    assert_eq!(
        count, 2,
        "expected one SOURCE_PATH_INVALID diagnostic per offending component, got {count}"
    );

    cleanup(base).await;
}

#[tokio::test]
async fn component_with_unsupported_format_emits_component_format_unsupported() {
    let base = test_base("split-component-format-unsupported");
    tokio::fs::create_dir_all(base.join("models"))
        .await
        .unwrap();

    let descriptor = ModelDescriptor::new(
        ModelId::new("sdxl-base-1.0"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        vec![
            ModelRole::DiffusionModel,
            ModelRole::TextEncoder,
            ModelRole::Vae,
        ],
        ModelSource::relative(
            ModelRootId::new("base"),
            "sdxl-base-1.0/manifest.safetensors",
        ),
        ModelFormat::Safetensors,
    )
    .with_source_status(ModelSourceStatus::Available)
    .with_component(
        ModelComponentSource::new(
            ModelRole::DiffusionModel,
            ModelSource::relative(
                ModelRootId::new("base"),
                "sdxl-base-1.0/unet/model.safetensors",
            ),
            ModelFormat::Unknown,
        )
        .with_metadata("component", "unet"),
    );

    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(descriptor);

    let report = validate_manifest(&manifest, base.join("models")).await;

    assert_codes(&report, &["MODEL_MANAGER/COMPONENT_FORMAT_UNSUPPORTED"]);

    cleanup(base).await;
}

#[tokio::test]
async fn duplicate_component_role_metadata_pair_emits_component_duplicate_diagnostic() {
    let base = test_base("split-component-duplicate");
    tokio::fs::create_dir_all(base.join("models"))
        .await
        .unwrap();

    let descriptor = ModelDescriptor::new(
        ModelId::new("sdxl-base-1.0"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        vec![
            ModelRole::CheckpointBundle,
            ModelRole::DiffusionModel,
            ModelRole::TextEncoder,
        ],
        ModelSource::relative(
            ModelRootId::new("base"),
            "sdxl-base-1.0/manifest.safetensors",
        ),
        ModelFormat::Safetensors,
    )
    .with_source_status(ModelSourceStatus::Unverified)
    .with_component(
        ModelComponentSource::new(
            ModelRole::TextEncoder,
            ModelSource::relative(
                ModelRootId::new("base"),
                "sdxl-base-1.0/text_encoder/model.safetensors",
            ),
            ModelFormat::Safetensors,
        )
        .with_metadata("component", "clip_l"),
    )
    .with_component(
        ModelComponentSource::new(
            ModelRole::TextEncoder,
            ModelSource::relative(
                ModelRootId::new("base"),
                "sdxl-base-1.0/text_encoder_dup/model.safetensors",
            ),
            ModelFormat::Safetensors,
        )
        .with_metadata("component", "clip_l"),
    );

    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(descriptor);

    let report = validate_manifest(&manifest, base.join("models")).await;

    assert_codes(&report, &["MODEL_MANAGER/COMPONENT_DUPLICATE"]);

    cleanup(base).await;
}

#[tokio::test]
async fn happy_path_split_descriptor_has_no_component_diagnostics() {
    let base = test_base("split-component-happy");
    tokio::fs::create_dir_all(base.join("models"))
        .await
        .unwrap();

    for relative in [
        "sdxl-base-1.0/unet/model.safetensors",
        "sdxl-base-1.0/text_encoder/model.safetensors",
        "sdxl-base-1.0/text_encoder_2/model.safetensors",
        "sdxl-base-1.0/vae/model.safetensors",
    ] {
        let p = base.join("models").join(relative);
        tokio::fs::create_dir_all(p.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&p, b"weights").await.unwrap();
    }

    let descriptor = split_sdxl_descriptor(ModelSourceStatus::Unverified);

    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(descriptor);

    let report = validate_manifest(&manifest, base.join("models")).await;

    assert_lacks_code(&report, "MODEL_MANAGER/COMPONENT_DUPLICATE");
    assert_lacks_code(&report, "MODEL_MANAGER/COMPONENT_SOURCE_MISSING");

    cleanup(base).await;
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

fn assert_code_count(report: &ManifestValidationReport, expected: &str, count: usize) {
    let actual_count = report
        .diagnostics()
        .iter()
        .filter(|diagnostic| diagnostic.code().as_str() == expected)
        .count();

    assert_eq!(
        actual_count, count,
        "expected diagnostic code {expected} to appear {count} time(s)"
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
