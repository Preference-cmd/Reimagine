use std::path::{Path, PathBuf};

use reimagine_core::model::{ModelId, ModelRole, ModelSeries, ModelVariant};
use reimagine_model_manager::{
    Classifier, ManifestUpdatePolicy, ModelDescriptor, ModelFormat, ModelManifest, ModelRoot,
    ModelRootId, ModelRootKind, ModelScanner, ModelSeriesConfig, ModelSeriesRule, ModelSource,
    ModelSourceStatus, ScanConfig,
};

#[tokio::test]
async fn scanner_default_ignores_hidden_and_unsupported_files() {
    let base = test_base("default-ignore");
    let models_dir = base.join("models");
    write_file(
        &models_dir.join("checkpoints/visible.safetensors"),
        b"visible",
    )
    .await;
    write_file(&models_dir.join(".hidden/secret.safetensors"), b"hidden").await;
    write_file(&models_dir.join("target/cache.safetensors"), b"target").await;
    write_file(
        &models_dir.join("node_modules/pkg/model.safetensors"),
        b"node",
    )
    .await;
    write_file(&models_dir.join("checkpoints/readme.txt"), b"text").await;

    let scanner = ModelScanner::new(ScanConfig::default());
    let observations = scanner
        .scan_root(&ModelRoot::base_models(), &models_dir)
        .await
        .unwrap();

    assert_eq!(observations.len(), 1);
    assert_eq!(observations[0].root_id().as_str(), "base");
    assert_eq!(
        observations[0].relative_path(),
        "checkpoints/visible.safetensors"
    );
    assert_eq!(observations[0].filename(), "visible");
    assert_eq!(observations[0].extension(), "safetensors");
    assert_eq!(observations[0].size_bytes(), 7);
    assert!(observations[0].modified_at().is_some());

    cleanup(base).await;
}

#[tokio::test]
async fn scanner_custom_config_controls_recursion_patterns_and_extensions() {
    let base = test_base("custom-config");
    let models_dir = base.join("models");
    write_file(&models_dir.join("top.gguf"), b"gguf").await;
    write_file(&models_dir.join("nested/deep.gguf"), b"deep").await;
    write_file(&models_dir.join("nested/skip.safetensors"), b"safe").await;

    let config = ScanConfig::default()
        .with_recursive(true)
        .with_supported_extension("gguf")
        .with_include_pattern("**/*.gguf")
        .with_exclude_pattern("nested/**");
    let scanner = ModelScanner::new(config);
    let observations = scanner
        .scan_root(
            &ModelRoot::new(
                ModelRootId::new("external"),
                ".",
                ModelRootKind::UserSelected,
            ),
            &models_dir,
        )
        .await
        .unwrap();

    assert_eq!(observations.len(), 1);
    assert_eq!(observations[0].root_id().as_str(), "external");
    assert_eq!(observations[0].relative_path(), "top.gguf");
    assert_eq!(observations[0].extension(), "gguf");

    cleanup(base).await;
}

#[tokio::test]
async fn manifest_update_policy_auto_adds_new_supported_files() {
    let base = test_base("auto-add");
    let models_dir = base.join("models");
    write_file(
        &models_dir.join("checkpoints/sdxl_base.safetensors"),
        b"weights",
    )
    .await;

    let observations = ModelScanner::new(ScanConfig::default())
        .scan_root(&ModelRoot::base_models(), &models_dir)
        .await
        .unwrap();
    let manifest = ModelManifest::new().with_root(ModelRoot::base_models());
    let classifier_config = sdxl_config();
    let classifier = Classifier::new(&classifier_config);
    let policy = ManifestUpdatePolicy::new(&classifier);

    let update = policy.apply_observations(manifest, &observations);

    assert_eq!(update.manifest().models().len(), 1);
    let model = &update.manifest().models()[0];
    assert_eq!(model.model_series().as_str(), "stable_diffusion");
    assert_eq!(model.variant().as_str(), "sdxl");
    assert_eq!(model.roles().len(), 4);
    assert_eq!(model.source_status(), ModelSourceStatus::Unverified);
    assert_eq!(model.observed_size_bytes(), Some(7));
    assert!(model.fingerprint().is_none());
    assert!(model.id().as_str().contains("stable_diffusion-sdxl"));
    assert_eq!(update.report().events().len(), 1);
    assert_eq!(update.report().events()[0].kind().as_str(), "model.added");

    cleanup(base).await;
}

#[test]
fn manifest_update_policy_marks_missing_without_deleting_entry() {
    let existing = existing_descriptor("demo", "checkpoints/missing.safetensors")
        .with_source_status(ModelSourceStatus::Available)
        .with_observed_size_bytes(7)
        .with_observed_modified_at("1");
    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(existing);
    let classifier_config = sdxl_config();
    let classifier = Classifier::new(&classifier_config);
    let policy = ManifestUpdatePolicy::new(&classifier);

    let update = policy.apply_observations(manifest, &[]);

    assert_eq!(update.manifest().models().len(), 1);
    assert_eq!(
        update.manifest().models()[0].source_status(),
        ModelSourceStatus::Missing
    );
    assert_eq!(update.report().events().len(), 1);
    assert_eq!(
        update.report().events()[0].kind().as_str(),
        "model.marked_missing"
    );
}

#[test]
fn manifest_update_policy_marks_unavailable_root_entries_missing() {
    let existing = existing_descriptor("demo", "checkpoints/on-drive.safetensors")
        .with_source_status(ModelSourceStatus::Available)
        .with_observed_size_bytes(7);
    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(existing);
    let classifier_config = sdxl_config();
    let classifier = Classifier::new(&classifier_config);
    let policy = ManifestUpdatePolicy::new(&classifier);

    let update = policy.apply_observations(manifest, &[]);

    assert_eq!(update.manifest().models().len(), 1);
    assert_eq!(
        update.manifest().models()[0].source_status(),
        ModelSourceStatus::Missing
    );
    assert!(
        update
            .report()
            .events()
            .iter()
            .any(|event| { event.kind().as_str() == "model.marked_missing" })
    );
}

#[tokio::test]
async fn manifest_update_policy_only_marks_missing_for_scanned_roots() {
    let base = test_base("scanned-root-scope");
    let models_dir = base.join("models");
    write_file(&models_dir.join("checkpoints/base.safetensors"), b"base").await;
    let observations = ModelScanner::new(ScanConfig::default())
        .scan_root(&ModelRoot::base_models(), &models_dir)
        .await
        .unwrap();
    let base_entry = existing_descriptor("base", "checkpoints/base.safetensors")
        .with_source_status(ModelSourceStatus::Available)
        .with_observed_size_bytes(4);
    let external_entry = ModelDescriptor::new(
        ModelId::new("external"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        vec![ModelRole::CheckpointBundle],
        ModelSource::relative(
            ModelRootId::new("external"),
            "checkpoints/external.safetensors",
        ),
        ModelFormat::Safetensors,
    )
    .with_source_status(ModelSourceStatus::Available)
    .with_observed_size_bytes(8);
    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_root(ModelRoot::new(
            ModelRootId::new("external"),
            "../external-models",
            ModelRootKind::UserSelected,
        ))
        .with_model(base_entry)
        .with_model(external_entry);
    let classifier_config = sdxl_config();
    let classifier = Classifier::new(&classifier_config);
    let policy = ManifestUpdatePolicy::new(&classifier);

    let update = policy.apply_root_observations(manifest, ModelRootId::new("base"), &observations);

    let external = update
        .manifest()
        .models()
        .iter()
        .find(|model| model.id().as_str() == "external")
        .unwrap();
    assert_eq!(external.source_status(), ModelSourceStatus::Available);
    assert_eq!(update.report().events().len(), 0);

    cleanup(base).await;
}

#[tokio::test]
async fn manifest_update_policy_marks_changed_known_files_stale() {
    let base = test_base("mark-stale");
    let models_dir = base.join("models");
    write_file(
        &models_dir.join("checkpoints/demo.safetensors"),
        b"new-size",
    )
    .await;
    let observations = ModelScanner::new(ScanConfig::default())
        .scan_root(&ModelRoot::base_models(), &models_dir)
        .await
        .unwrap();
    let existing = existing_descriptor("demo", "checkpoints/demo.safetensors")
        .with_source_status(ModelSourceStatus::Available)
        .with_size_bytes(7)
        .with_observed_size_bytes(7)
        .with_observed_modified_at("1");
    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(existing);
    let classifier_config = sdxl_config();
    let classifier = Classifier::new(&classifier_config);
    let policy = ManifestUpdatePolicy::new(&classifier);

    let update = policy.apply_observations(manifest, &observations);

    assert_eq!(update.manifest().models().len(), 1);
    let model = &update.manifest().models()[0];
    assert_eq!(model.source_status(), ModelSourceStatus::Stale);
    assert_eq!(model.observed_size_bytes(), Some(8));
    assert_eq!(update.report().events().len(), 1);
    assert_eq!(
        update.report().events()[0].kind().as_str(),
        "model.marked_stale"
    );

    cleanup(base).await;
}

fn sdxl_config() -> ModelSeriesConfig {
    ModelSeriesConfig::new().with_rule(
        ModelSeriesRule::new(
            ModelSeries::new("stable_diffusion"),
            ModelVariant::new("sdxl"),
        )
        .with_extension("safetensors")
        .with_role(ModelRole::CheckpointBundle)
        .with_role(ModelRole::DiffusionModel)
        .with_role(ModelRole::TextEncoder)
        .with_role(ModelRole::Vae)
        .with_format(ModelFormat::Safetensors),
    )
}

fn existing_descriptor(id: &str, path: &str) -> ModelDescriptor {
    ModelDescriptor::new(
        ModelId::new(id),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        vec![ModelRole::CheckpointBundle],
        ModelSource::relative(ModelRootId::new("base"), path),
        ModelFormat::Safetensors,
    )
}

fn test_base(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "reimagine-model-manager-scanner-{name}-{}",
        std::process::id()
    ))
}

async fn write_file(path: &Path, bytes: &[u8]) {
    tokio::fs::create_dir_all(path.parent().unwrap())
        .await
        .unwrap();
    tokio::fs::write(path, bytes).await.unwrap();
}

async fn cleanup(path: PathBuf) {
    let _ = tokio::fs::remove_dir_all(path).await;
}
