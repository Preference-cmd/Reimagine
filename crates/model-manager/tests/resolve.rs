use std::path::PathBuf;

use reimagine_core::diagnostic::DiagnosticSeverity;
use reimagine_core::model::{ModelId, ModelRef, ModelRole, ModelSeries, ModelVariant};
use reimagine_model_manager::{
    Fingerprint, ManifestModelResolver, ModelComponentSource, ModelDescriptor,
    ModelDescriptorResolver, ModelFormat, ModelManifest, ModelReadinessResolver, ModelRoot,
    ModelRootId, ModelSource, ModelSourceStatus,
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

fn split_sdxl_descriptor(id: &str) -> ModelDescriptor {
    ModelDescriptor::new(
        ModelId::new(id),
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
    .with_source_status(ModelSourceStatus::Available)
    .with_size_bytes(7)
    .with_observed_size_bytes(7)
    .with_fingerprint(Fingerprint::sha256("abc123"))
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

async fn write_split_sdxl_files(base: &std::path::Path) {
    for relative in [
        "sdxl-base-1.0/manifest.safetensors",
        "sdxl-base-1.0/unet/model.safetensors",
        "sdxl-base-1.0/text_encoder/model.safetensors",
        "sdxl-base-1.0/text_encoder_2/model.safetensors",
        "sdxl-base-1.0/vae/model.safetensors",
    ] {
        write_weights(base, relative, b"weights").await;
    }
}

#[tokio::test]
async fn split_descriptor_resolves_components_with_absolute_paths() {
    let base = test_base("split-success");
    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(split_sdxl_descriptor("sdxl-base-1.0"));
    write_split_sdxl_files(&base).await;
    let resolver = ManifestModelResolver::new(&manifest, base.join("models"));
    let model_ref = ModelRef::new(
        ModelId::new("sdxl-base-1.0"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::TextEncoder,
    );

    let view = resolver
        .resolve_descriptor_with_components(&model_ref)
        .await;

    assert!(view.is_resolved());
    assert!(view.report().diagnostics().is_empty());

    let resolved = view.value().expect("resolved view");
    let components = resolved.components();
    assert_eq!(components.len(), 4);

    let unet = components
        .iter()
        .find(|c| c.role() == ModelRole::DiffusionModel)
        .expect("diffusion component");
    assert!(!unet.is_missing());
    assert!(
        unet.path()
            .ends_with("sdxl-base-1.0/unet/model.safetensors")
    );
    assert_eq!(
        unet.metadata().get("component").map(String::as_str),
        Some("unet")
    );

    let clip_l = components
        .iter()
        .filter(|c| c.role() == ModelRole::TextEncoder)
        .find(|c| c.metadata().get("component").map(String::as_str) == Some("clip_l"))
        .expect("clip_l component");
    assert!(!clip_l.is_missing());

    let vae = components
        .iter()
        .find(|c| c.role() == ModelRole::Vae)
        .expect("vae component");
    assert!(!vae.is_missing());

    cleanup(base).await;
}

#[tokio::test]
async fn split_descriptor_marks_missing_components_with_specific_codes() {
    let base = test_base("split-missing-clip-l");
    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(split_sdxl_descriptor("sdxl-base-1.0"));
    // Only write the unet, vae, and clip_g files. CLIP-L is missing.
    for relative in [
        "sdxl-base-1.0/manifest.safetensors",
        "sdxl-base-1.0/unet/model.safetensors",
        "sdxl-base-1.0/text_encoder_2/model.safetensors",
        "sdxl-base-1.0/vae/model.safetensors",
    ] {
        write_weights(&base, relative, b"weights").await;
    }
    let resolver = ManifestModelResolver::new(&manifest, base.join("models"));
    let model_ref = ModelRef::new(
        ModelId::new("sdxl-base-1.0"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::TextEncoder,
    );

    let view = resolver
        .resolve_descriptor_with_components(&model_ref)
        .await;

    let clip_l_missing = view
        .value()
        .expect("view resolved")
        .components()
        .iter()
        .find(|c| {
            c.role() == ModelRole::TextEncoder
                && c.metadata().get("component").map(String::as_str) == Some("clip_l")
        })
        .expect("clip_l component entry");
    assert!(clip_l_missing.is_missing());
    assert!(view.report().diagnostics().iter().any(|d| d.code().as_str()
        == "MODEL_MANAGER/COMPONENT_SOURCE_MISSING"
        && d.message().contains("clip_l")));

    cleanup(base).await;
}

#[tokio::test]
async fn split_descriptor_reports_duplicate_component_metadata() {
    let base = test_base("split-duplicate-metadata");
    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(
            split_sdxl_descriptor("sdxl-base-1.0").with_component(
                ModelComponentSource::new(
                    ModelRole::TextEncoder,
                    ModelSource::relative(
                        ModelRootId::new("base"),
                        "sdxl-base-1.0/text_encoder_dup/model.safetensors",
                    ),
                    ModelFormat::Safetensors,
                )
                .with_metadata("component", "clip_l"),
            ),
        );
    write_split_sdxl_files(&base).await;
    let resolver = ManifestModelResolver::new(&manifest, base.join("models"));
    let model_ref = ModelRef::new(
        ModelId::new("sdxl-base-1.0"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::TextEncoder,
    );

    let view = resolver
        .resolve_descriptor_with_components(&model_ref)
        .await;

    let components = view.value().expect("resolved view").components();
    assert_eq!(components.len(), 5);
    let clip_l_count = components
        .iter()
        .filter(|c| {
            c.role() == ModelRole::TextEncoder
                && c.metadata().get("component").map(String::as_str) == Some("clip_l")
        })
        .count();
    assert_eq!(clip_l_count, 2);

    assert!(
        view.report()
            .diagnostics()
            .iter()
            .any(|d| d.code().as_str() == "MODEL_MANAGER/COMPONENT_DUPLICATE")
    );

    cleanup(base).await;
}

async fn assert_single_missing_component_reports_specific_code(
    name: &str,
    written: &[&str],
    target: ModelRole,
    expected_component_metadata: &str,
    expected_label: &str,
) {
    let base = test_base(name);
    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(split_sdxl_descriptor("sdxl-base-1.0"));
    for relative in written {
        write_weights(&base, relative, b"weights").await;
    }
    let resolver = ManifestModelResolver::new(&manifest, base.join("models"));
    let model_ref = ModelRef::new(
        ModelId::new("sdxl-base-1.0"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        target,
    );

    let view = resolver
        .resolve_descriptor_with_components(&model_ref)
        .await;

    let missing = view
        .value()
        .expect("view resolved")
        .components()
        .iter()
        .find(|c| {
            c.role() == target
                && c.metadata().get("component").map(String::as_str)
                    == Some(expected_component_metadata)
        })
        .expect("missing component entry");
    assert!(missing.is_missing(), "{expected_label} should be missing");
    assert!(view.report().diagnostics().iter().any(|d| d.code().as_str()
        == "MODEL_MANAGER/COMPONENT_SOURCE_MISSING"
        && d.message().contains(expected_label)));

    cleanup(base).await;
}

#[tokio::test]
async fn missing_clip_g_component_is_reported_with_specific_code() {
    assert_single_missing_component_reports_specific_code(
        "split-missing-clip-g",
        &[
            "sdxl-base-1.0/manifest.safetensors",
            "sdxl-base-1.0/unet/model.safetensors",
            "sdxl-base-1.0/text_encoder/model.safetensors",
            "sdxl-base-1.0/vae/model.safetensors",
        ],
        ModelRole::TextEncoder,
        "clip_g",
        "clip_g",
    )
    .await;
}

#[tokio::test]
async fn missing_unet_component_is_reported_with_specific_code() {
    assert_single_missing_component_reports_specific_code(
        "split-missing-unet",
        &[
            "sdxl-base-1.0/manifest.safetensors",
            "sdxl-base-1.0/text_encoder/model.safetensors",
            "sdxl-base-1.0/text_encoder_2/model.safetensors",
            "sdxl-base-1.0/vae/model.safetensors",
        ],
        ModelRole::DiffusionModel,
        "unet",
        "unet",
    )
    .await;
}

#[tokio::test]
async fn missing_vae_component_is_reported_with_specific_code() {
    assert_single_missing_component_reports_specific_code(
        "split-missing-vae",
        &[
            "sdxl-base-1.0/manifest.safetensors",
            "sdxl-base-1.0/unet/model.safetensors",
            "sdxl-base-1.0/text_encoder/model.safetensors",
            "sdxl-base-1.0/text_encoder_2/model.safetensors",
        ],
        ModelRole::Vae,
        "vae",
        "vae",
    )
    .await;
}

#[tokio::test]
async fn resolver_emits_component_format_unsupported_for_unknown_component_format() {
    let base = test_base("split-resolver-format-unsupported");
    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(
            split_sdxl_descriptor("sdxl-base-1.0").with_component(
                ModelComponentSource::new(
                    ModelRole::DiffusionModel,
                    ModelSource::relative(
                        ModelRootId::new("base"),
                        "sdxl-base-1.0/unet/model.safetensors",
                    ),
                    ModelFormat::Unknown,
                )
                .with_metadata("component", "unet"),
            ),
        );
    write_split_sdxl_files(&base).await;
    let resolver = ManifestModelResolver::new(&manifest, base.join("models"));
    let model_ref = ModelRef::new(
        ModelId::new("sdxl-base-1.0"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::DiffusionModel,
    );

    let view = resolver
        .resolve_descriptor_with_components(&model_ref)
        .await;

    assert!(
        view.report()
            .diagnostics()
            .iter()
            .any(|d| d.code().as_str() == "MODEL_MANAGER/COMPONENT_FORMAT_UNSUPPORTED"),
        "expected COMPONENT_FORMAT_UNSUPPORTED diagnostic, got {:?}",
        view.report()
            .diagnostics()
            .iter()
            .map(|d| d.code().as_str())
            .collect::<Vec<_>>()
    );

    cleanup(base).await;
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
