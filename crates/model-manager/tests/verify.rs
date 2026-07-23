use std::path::PathBuf;

use reimagine_core::model::{ModelId, ModelRole, ModelSeries, ModelVariant};
use reimagine_model_manager::{
    ModelDescriptor, ModelFingerprintVerifier, ModelFormat, ModelManifest, ModelRoot, ModelRootId,
    ModelSource, ModelSourceStatus,
};

const WEIGHTS_SHA256: &str = "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c";

#[tokio::test]
async fn refresh_clears_stale_and_marks_descriptor_available() {
    let base = test_base("refresh-clears-stale");
    write_weights(&base, "checkpoints/demo-model.safetensors", b"weights").await;
    let manifest = ModelManifest::new().with_root(ModelRoot::base_models());
    let descriptor = sample_descriptor()
        .with_source_status(ModelSourceStatus::Stale)
        .with_size_bytes(6)
        .with_observed_size_bytes(6)
        .with_observed_modified_at("1");
    let verifier = ModelFingerprintVerifier::new(&manifest, base.join("models"));

    let refreshed = verifier.refresh_descriptor(&descriptor).await.unwrap();

    assert_eq!(
        refreshed.descriptor().source_status(),
        ModelSourceStatus::Available
    );
    assert_eq!(
        refreshed.descriptor().fingerprint().unwrap().value(),
        WEIGHTS_SHA256
    );
    assert_eq!(refreshed.descriptor().observed_size_bytes(), Some(7));
    assert!(refreshed.descriptor().observed_modified_at().is_some());
    assert!(refreshed.descriptor().verified_at().is_some());
    assert!(refreshed.descriptor().updated_at().is_some());
    assert_eq!(refreshed.report().events().len(), 1);
    assert_eq!(
        refreshed.report().events()[0].kind().as_str(),
        "model.verified"
    );

    cleanup(base).await;
}

#[tokio::test]
async fn first_add_refresh_computes_sha256_and_updates_descriptor_metadata() {
    let base = test_base("refresh-first-add");
    write_weights(&base, "checkpoints/demo-model.safetensors", b"weights").await;
    let manifest = ModelManifest::new().with_root(ModelRoot::base_models());
    let verifier = ModelFingerprintVerifier::new(&manifest, base.join("models"));

    let refreshed = verifier
        .refresh_descriptor(&sample_descriptor())
        .await
        .unwrap();

    assert_eq!(
        refreshed.descriptor().fingerprint().unwrap().kind(),
        "sha256"
    );
    assert_eq!(
        refreshed.descriptor().fingerprint().unwrap().value(),
        WEIGHTS_SHA256
    );
    assert_eq!(refreshed.descriptor().size_bytes(), Some(7));
    assert_eq!(refreshed.descriptor().observed_size_bytes(), Some(7));
    assert!(refreshed.descriptor().observed_modified_at().is_some());
    assert!(refreshed.descriptor().verified_at().is_some());
    assert!(refreshed.descriptor().updated_at().is_some());
    assert_eq!(
        refreshed.descriptor().source_status(),
        ModelSourceStatus::Available
    );
    assert!(refreshed.report().diagnostics().is_empty());
    assert_eq!(refreshed.report().events().len(), 1);
    assert_eq!(
        refreshed.report().events()[0].kind().as_str(),
        "model.verified"
    );

    cleanup(base).await;
}

#[tokio::test]
async fn refresh_fails_when_relative_root_is_not_in_manifest() {
    let base = test_base("missing-root");
    let manifest = ModelManifest::new().with_root(ModelRoot::base_models());
    let descriptor = ModelDescriptor::new(
        ModelId::new("external-model"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        vec![ModelRole::CheckpointBundle],
        ModelSource::relative(ModelRootId::new("external"), "demo.safetensors"),
        ModelFormat::Safetensors,
    );
    let verifier = ModelFingerprintVerifier::new(&manifest, base.join("models"));

    let error = verifier.refresh_descriptor(&descriptor).await.unwrap_err();

    assert!(
        error
            .to_string()
            .contains("model source root could not be resolved")
    );

    cleanup(base).await;
}

fn sample_descriptor() -> ModelDescriptor {
    ModelDescriptor::new(
        ModelId::new("demo-model"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        vec![ModelRole::CheckpointBundle],
        ModelSource::relative(
            ModelRootId::new("base"),
            "checkpoints/demo-model.safetensors",
        ),
        ModelFormat::Safetensors,
    )
}

fn test_base(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "reimagine-model-manager-verify-{name}-{}",
        std::process::id()
    ))
}

async fn write_weights(base: &std::path::Path, relative_path: &str, bytes: &[u8]) {
    let path = base.join("models").join(relative_path);
    tokio::fs::create_dir_all(path.parent().unwrap())
        .await
        .unwrap();
    tokio::fs::write(path, bytes).await.unwrap();
}

async fn cleanup(path: PathBuf) {
    let _ = tokio::fs::remove_dir_all(path).await;
}
