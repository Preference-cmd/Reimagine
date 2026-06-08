use std::path::PathBuf;

use reimagine_config::AppPaths;
use reimagine_core::model::{ModelId, ModelRole, ModelSeries, ModelVariant};
use reimagine_model_manager::{
    Fingerprint, ModelDescriptor, ModelFormat, ModelManifest, ModelManifestStore, ModelRoot,
    ModelRootId, ModelSource, ModelSourceStatus, load_model_manifest,
};

#[tokio::test]
async fn missing_manifest_loads_empty_v1_with_default_base_root() {
    let base = test_base("missing-manifest");
    let store = ModelManifestStore::new(AppPaths::new(base.clone()));

    let (manifest, report) = store.load().await.unwrap();

    assert_eq!(manifest.schema_version(), "reimagine.model_manifest.v1");
    assert_eq!(manifest.model_roots(), &[ModelRoot::base_models()]);
    assert!(manifest.models().is_empty());
    assert!(report.diagnostics().is_empty());

    cleanup(base).await;
}

#[tokio::test]
async fn save_and_reload_roundtrip_preserves_manifest() {
    let base = test_base("roundtrip");
    let store = ModelManifestStore::new(AppPaths::new(base.clone()));
    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(sample_descriptor().with_source_status(ModelSourceStatus::Available));
    let source_path = base.join("models/checkpoints/demo.safetensors");

    tokio::fs::create_dir_all(source_path.parent().unwrap())
        .await
        .unwrap();
    tokio::fs::write(&source_path, b"weights").await.unwrap();

    let save_report = store.save(&manifest).await.unwrap();
    assert!(save_report.diagnostics().is_empty());

    let (loaded, load_report) = store.load().await.unwrap();

    assert_eq!(loaded, manifest);
    assert!(load_report.diagnostics().is_empty());
    assert_eq!(store.path(), &base.join("models/manifest.json"));

    cleanup(base).await;
}

#[tokio::test]
async fn invalid_json_returns_error_with_diagnostic() {
    let base = test_base("invalid-json");
    let store = ModelManifestStore::new(AppPaths::new(base.clone()));
    tokio::fs::create_dir_all(base.join("models"))
        .await
        .unwrap();
    tokio::fs::write(store.path(), b"{bad-json").await.unwrap();

    let error = store.load().await.unwrap_err();
    let diagnostic = error.to_diagnostic(None);

    assert_eq!(diagnostic.code().as_str(), "MODEL_MANAGER/MANIFEST_INVALID");
    assert_eq!(diagnostic.primary().path(), Some("models/manifest.json"));

    cleanup(base).await;
}

#[tokio::test]
async fn remove_model_updates_manifest_only() {
    let base = test_base("remove-model");
    let store = ModelManifestStore::new(AppPaths::new(base.clone()));
    let descriptor = sample_descriptor()
        .with_source_status(ModelSourceStatus::Available)
        .with_fingerprint(Fingerprint::sha256("abc123"))
        .with_size_bytes(42)
        .with_observed_size_bytes(42);
    let relative_path = base.join("models").join("checkpoints/demo.safetensors");

    tokio::fs::create_dir_all(relative_path.parent().unwrap())
        .await
        .unwrap();
    tokio::fs::write(&relative_path, b"weights").await.unwrap();

    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(descriptor.clone());
    store.save(&manifest).await.unwrap();

    let (updated, report) = store.remove_model(descriptor.id()).await.unwrap();

    assert!(updated.models().is_empty());
    assert!(report.diagnostics().is_empty());
    assert!(tokio::fs::try_exists(&relative_path).await.unwrap());

    cleanup(base).await;
}

#[tokio::test]
async fn top_level_load_helper_matches_store_behavior() {
    let base = test_base("helper-load");

    let (manifest, report) = load_model_manifest(AppPaths::new(base.clone()))
        .await
        .unwrap();

    assert_eq!(manifest.model_roots(), &[ModelRoot::base_models()]);
    assert!(report.diagnostics().is_empty());

    cleanup(base).await;
}

fn sample_descriptor() -> ModelDescriptor {
    ModelDescriptor::new(
        ModelId::new("demo-model"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        vec![ModelRole::CheckpointBundle, ModelRole::DiffusionModel],
        ModelSource::relative(ModelRootId::new("base"), "checkpoints/demo.safetensors"),
        ModelFormat::Safetensors,
    )
}

fn test_base(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "reimagine-model-manager-{name}-{}",
        std::process::id()
    ))
}

async fn cleanup(path: PathBuf) {
    let _ = tokio::fs::remove_dir_all(path).await;
}
