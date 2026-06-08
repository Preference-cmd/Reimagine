use reimagine_core::model::{ModelId, ModelRole, ModelSeries, ModelVariant};
use reimagine_model_manager::{
    Fingerprint, ModelDescriptor, ModelFormat, ModelManifest, ModelRoot, ModelRootId,
    ModelRootKind, ModelSeriesConfig, ModelSeriesRule, ModelSource, ModelSourceStatus, ScanConfig,
};

#[test]
fn sdxl_manifest_example_roundtrips_through_documented_json_shape() {
    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(
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
                    "checkpoints/sdxl_base_1.0.safetensors",
                ),
                ModelFormat::Safetensors,
            )
            .with_source_status(ModelSourceStatus::Available)
            .with_size_bytes(6_938_078_336)
            .with_observed_size_bytes(6_938_078_336)
            .with_observed_modified_at("2026-06-07T00:00:00Z")
            .with_fingerprint(Fingerprint::sha256("abc123"))
            .with_verified_at("2026-06-07T00:00:00Z")
            .with_discovered_at("2026-06-07T00:00:00Z")
            .with_updated_at("2026-06-07T00:00:00Z"),
        );

    let json = serde_json::to_value(&manifest).unwrap();

    assert_eq!(json["schema_version"], "reimagine.model_manifest.v1");
    assert_eq!(json["model_roots"][0]["id"], "base");
    assert_eq!(json["model_roots"][0]["path"], ".");
    assert_eq!(json["model_roots"][0]["kind"], "base_path_models");
    assert_eq!(json["models"][0]["id"], "sdxl-base-1.0");
    assert_eq!(json["models"][0]["model_series"], "stable_diffusion");
    assert_eq!(json["models"][0]["variant"], "sdxl");
    assert_eq!(json["models"][0]["roles"][0], "CheckpointBundle");
    assert_eq!(json["models"][0]["source"]["type"], "local_file_relative");
    assert_eq!(json["models"][0]["source"]["root_id"], "base");
    assert_eq!(
        json["models"][0]["source"]["path"],
        "checkpoints/sdxl_base_1.0.safetensors"
    );
    assert_eq!(json["models"][0]["source_status"], "Available");
    assert_eq!(json["models"][0]["format"], "safetensors");
    assert_eq!(json["models"][0]["fingerprint"]["kind"], "sha256");

    let decoded: ModelManifest = serde_json::from_value(json).unwrap();
    assert_eq!(decoded, manifest);
    assert_eq!(decoded.models()[0].roles().len(), 4);
    assert!(decoded.models()[0].is_runnable_candidate());
}

#[test]
fn unknown_descriptor_is_representable_but_not_runnable() {
    let descriptor = ModelDescriptor::new(
        ModelId::new("unknown-local-file"),
        ModelSeries::new("unknown"),
        ModelVariant::new("unknown"),
        Vec::new(),
        ModelSource::absolute("/tmp/mystery.bin"),
        ModelFormat::Unknown,
    )
    .with_source_status(ModelSourceStatus::Unverified);

    assert_eq!(descriptor.model_series().as_str(), "unknown");
    assert_eq!(descriptor.variant().as_str(), "unknown");
    assert!(descriptor.roles().is_empty());
    assert!(!descriptor.is_runnable_candidate());
    assert!(matches!(
        descriptor.source(),
        ModelSource::LocalFileAbsolute { .. }
    ));
}

#[test]
fn model_series_config_scan_config_and_rule_shape_are_serializable() {
    let config = ModelSeriesConfig::new().with_rule(
        ModelSeriesRule::new(
            ModelSeries::new("stable_diffusion"),
            ModelVariant::new("sdxl"),
        )
        .with_root_id(ModelRootId::new("base"))
        .with_filename_pattern("*sdxl*")
        .with_extension("safetensors")
        .with_role(ModelRole::CheckpointBundle)
        .with_role(ModelRole::DiffusionModel)
        .with_format(ModelFormat::Safetensors),
    );
    let scan = ScanConfig::default()
        .with_recursive(true)
        .with_ignore_hidden(true);

    let config_json = serde_json::to_value(&config).unwrap();
    let scan_json = serde_json::to_value(&scan).unwrap();

    assert_eq!(config_json["schema_version"], "reimagine.model_series.v1");
    assert_eq!(config_json["rules"][0]["model_series"], "stable_diffusion");
    assert_eq!(config_json["rules"][0]["variant"], "sdxl");
    assert_eq!(config_json["rules"][0]["roles"][0], "CheckpointBundle");
    assert_eq!(config_json["rules"][0]["format"], "safetensors");
    assert_eq!(scan_json["recursive"], true);
    assert_eq!(scan_json["ignore_hidden"], true);

    let decoded_config: ModelSeriesConfig = serde_json::from_value(config_json).unwrap();
    let decoded_scan: ScanConfig = serde_json::from_value(scan_json).unwrap();

    assert_eq!(decoded_config.rules().len(), 1);
    assert!(decoded_scan.recursive());
    assert!(decoded_scan.ignore_hidden());
}

#[test]
fn source_status_and_root_kinds_cover_v1_manifest_cases() {
    let custom_root = ModelRoot::new(
        ModelRootId::new("external"),
        "/Volumes/models",
        ModelRootKind::UserSelected,
    );
    let statuses = [
        ModelSourceStatus::Available,
        ModelSourceStatus::Missing,
        ModelSourceStatus::Stale,
        ModelSourceStatus::Unverified,
    ];

    assert_eq!(
        ModelRoot::base_models().kind(),
        ModelRootKind::BasePathModels
    );
    assert_eq!(custom_root.path(), "/Volumes/models");
    assert_eq!(
        serde_json::to_string(&statuses).unwrap(),
        r#"["Available","Missing","Stale","Unverified"]"#
    );
}
