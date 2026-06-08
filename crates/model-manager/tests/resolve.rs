use std::path::PathBuf;

use reimagine_core::diagnostic::DiagnosticSeverity;
use reimagine_core::model::{ModelId, ModelRef, ModelRole, ModelSeries, ModelVariant};
use reimagine_model_manager::{
    Fingerprint, ManifestModelResolver, ModelDescriptor, ModelDescriptorResolver, ModelFormat,
    ModelManifest, ModelReadinessResolver, ModelRoot, ModelRootId, ModelSource, ModelSourceStatus,
};

#[tokio::test]
async fn successful_resolve_returns_readiness_info_and_descriptor() {
    let base = test_base("resolve-success");
    let manifest = manifest_with_model(
        sample_descriptor("demo-model")
            .with_source_status(ModelSourceStatus::Available)
            .with_size_bytes(7)
            .with_observed_size_bytes(7)
            .with_fingerprint(Fingerprint::sha256("abc123")),
    );
    write_weights(&base, "checkpoints/demo-model.safetensors", b"weights").await;
    let resolver = ManifestModelResolver::new(&manifest, base.join("models"));
    let model_ref = sample_model_ref("demo-model");

    let readiness = resolver.resolve_readiness(&model_ref).await;
    let descriptor = resolver.resolve_descriptor(&model_ref).await;

    assert!(readiness.is_resolved());
    assert!(descriptor.is_resolved());
    assert!(readiness.report().diagnostics().is_empty());
    assert!(descriptor.report().diagnostics().is_empty());

    let info = readiness.value().unwrap();
    assert_eq!(info.id().as_str(), "demo-model");
    assert_eq!(info.model_series().as_str(), "stable_diffusion");
    assert_eq!(info.variant().as_str(), "sdxl");
    assert_eq!(info.roles(), &[ModelRole::CheckpointBundle]);
    assert_eq!(info.format(), ModelFormat::Safetensors);
    assert_eq!(info.source_status(), ModelSourceStatus::Available);
    assert!(info.source_available());

    assert_eq!(descriptor.value().unwrap().id().as_str(), "demo-model");

    cleanup(base).await;
}

#[tokio::test]
async fn resolve_reports_not_found_when_id_is_missing() {
    let base = test_base("resolve-not-found");
    let manifest = ModelManifest::new().with_root(ModelRoot::base_models());
    let resolver = ManifestModelResolver::new(&manifest, base.join("models"));

    let resolution = resolver
        .resolve_readiness(&sample_model_ref("missing"))
        .await;

    assert!(!resolution.is_resolved());
    assert_eq!(resolution.report().diagnostics().len(), 1);
    assert_eq!(
        resolution.report().diagnostics()[0].code().as_str(),
        "MODEL_MANAGER/MODEL_REF_NOT_FOUND"
    );
    assert_eq!(
        resolution.report().diagnostics()[0].severity(),
        DiagnosticSeverity::Error
    );
}

#[tokio::test]
async fn resolve_blocks_when_model_series_mismatches() {
    let base = test_base("resolve-series-mismatch");
    let manifest = manifest_with_model(
        sample_descriptor("demo-model").with_source_status(ModelSourceStatus::Available),
    );
    write_weights(&base, "checkpoints/demo-model.safetensors", b"weights").await;
    let resolver = ManifestModelResolver::new(&manifest, base.join("models"));
    let model_ref = ModelRef::new(
        ModelId::new("demo-model"),
        ModelSeries::new("flux"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
    );

    let resolution = resolver.resolve_readiness(&model_ref).await;

    assert!(!resolution.is_resolved());
    assert_eq!(
        resolution.report().diagnostics()[0].code().as_str(),
        "MODEL_MANAGER/MODEL_SERIES_MISMATCH"
    );

    cleanup(base).await;
}

#[tokio::test]
async fn resolve_blocks_when_variant_mismatches() {
    let base = test_base("resolve-variant-mismatch");
    let manifest = manifest_with_model(
        sample_descriptor("demo-model").with_source_status(ModelSourceStatus::Available),
    );
    write_weights(&base, "checkpoints/demo-model.safetensors", b"weights").await;
    let resolver = ManifestModelResolver::new(&manifest, base.join("models"));
    let model_ref = ModelRef::new(
        ModelId::new("demo-model"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sd15"),
        ModelRole::CheckpointBundle,
    );

    let resolution = resolver.resolve_readiness(&model_ref).await;

    assert!(!resolution.is_resolved());
    assert_eq!(
        resolution.report().diagnostics()[0].code().as_str(),
        "MODEL_MANAGER/MODEL_VARIANT_MISMATCH"
    );

    cleanup(base).await;
}

#[tokio::test]
async fn resolve_blocks_when_requested_role_is_missing() {
    let base = test_base("resolve-role-missing");
    let manifest = manifest_with_model(
        sample_descriptor("demo-model").with_source_status(ModelSourceStatus::Available),
    );
    write_weights(&base, "checkpoints/demo-model.safetensors", b"weights").await;
    let resolver = ManifestModelResolver::new(&manifest, base.join("models"));
    let model_ref = ModelRef::new(
        ModelId::new("demo-model"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::TextEncoder,
    );

    let resolution = resolver.resolve_readiness(&model_ref).await;

    assert!(!resolution.is_resolved());
    assert_eq!(
        resolution.report().diagnostics()[0].code().as_str(),
        "MODEL_MANAGER/MODEL_ROLE_MISSING"
    );

    cleanup(base).await;
}

#[tokio::test]
async fn resolve_blocks_when_source_file_is_missing() {
    let base = test_base("resolve-source-missing");
    let manifest = manifest_with_model(
        sample_descriptor("demo-model").with_source_status(ModelSourceStatus::Available),
    );
    let resolver = ManifestModelResolver::new(&manifest, base.join("models"));

    let resolution = resolver
        .resolve_readiness(&sample_model_ref("demo-model"))
        .await;

    assert!(!resolution.is_resolved());
    assert_eq!(
        resolution.report().diagnostics()[0].code().as_str(),
        "MODEL_MANAGER/MODEL_SOURCE_MISSING"
    );
}

#[tokio::test]
async fn resolve_blocks_when_model_is_stale() {
    let base = test_base("resolve-stale");
    let manifest = manifest_with_model(
        sample_descriptor("demo-model")
            .with_source_status(ModelSourceStatus::Stale)
            .with_size_bytes(6)
            .with_observed_size_bytes(7)
            .with_fingerprint(Fingerprint::sha256("abc123")),
    );
    write_weights(&base, "checkpoints/demo-model.safetensors", b"weights").await;
    let resolver = ManifestModelResolver::new(&manifest, base.join("models"));

    let resolution = resolver
        .resolve_readiness(&sample_model_ref("demo-model"))
        .await;

    assert!(!resolution.is_resolved());
    assert_eq!(
        resolution.report().diagnostics()[0].code().as_str(),
        "MODEL_MANAGER/MODEL_SOURCE_STALE"
    );

    cleanup(base).await;
}

#[tokio::test]
async fn resolve_allows_unverified_models_with_warning_when_fingerprint_is_missing() {
    let base = test_base("resolve-unverified-warning");
    let manifest = manifest_with_model(
        sample_descriptor("demo-model").with_source_status(ModelSourceStatus::Unverified),
    );
    write_weights(&base, "checkpoints/demo-model.safetensors", b"weights").await;
    let resolver = ManifestModelResolver::new(&manifest, base.join("models"));

    let resolution = resolver
        .resolve_readiness(&sample_model_ref("demo-model"))
        .await;

    assert!(resolution.is_resolved());
    assert_eq!(resolution.report().diagnostics().len(), 1);
    assert_eq!(
        resolution.report().diagnostics()[0].code().as_str(),
        "MODEL_MANAGER/MODEL_FINGERPRINT_MISSING"
    );
    assert_eq!(
        resolution.report().diagnostics()[0].severity(),
        DiagnosticSeverity::Warning
    );

    cleanup(base).await;
}

#[tokio::test]
async fn resolve_blocks_when_recorded_fingerprint_can_no_longer_match_observed_metadata() {
    let base = test_base("resolve-fingerprint-mismatch");
    let manifest = manifest_with_model(
        sample_descriptor("demo-model")
            .with_source_status(ModelSourceStatus::Available)
            .with_size_bytes(6)
            .with_observed_size_bytes(7)
            .with_fingerprint(Fingerprint::sha256("abc123")),
    );
    write_weights(&base, "checkpoints/demo-model.safetensors", b"weights").await;
    let resolver = ManifestModelResolver::new(&manifest, base.join("models"));

    let resolution = resolver
        .resolve_readiness(&sample_model_ref("demo-model"))
        .await;

    assert!(!resolution.is_resolved());
    assert_eq!(
        resolution.report().diagnostics()[0].code().as_str(),
        "MODEL_MANAGER/MODEL_FINGERPRINT_MISMATCH"
    );

    cleanup(base).await;
}

fn manifest_with_model(model: ModelDescriptor) -> ModelManifest {
    ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(model)
}

fn sample_descriptor(id: &str) -> ModelDescriptor {
    ModelDescriptor::new(
        ModelId::new(id),
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

fn sample_model_ref(id: &str) -> ModelRef {
    ModelRef::new(
        ModelId::new(id),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
    )
}

fn test_base(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "reimagine-model-manager-resolve-{name}-{}",
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
