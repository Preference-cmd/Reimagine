use reimagine_config::{AppConfig, AppPaths, ConfigDocument, ConfigKey, ConfigValidationContext};
use reimagine_core::model::{ModelRole, ModelSeries, ModelVariant};

use reimagine_model_manager::{
    ClassificationCandidate, Classifier, MODEL_SERIES_SCHEMA_VERSION, ModelFormat, ModelRootId,
    ModelSeriesConfig, ModelSeriesRule,
};

fn sdxl_config() -> ModelSeriesConfig {
    ModelSeriesConfig::new()
        .with_rule(
            ModelSeriesRule::new(
                ModelSeries::new("stable_diffusion"),
                ModelVariant::new("sdxl"),
            )
            .with_extension("safetensors")
            .with_filename_pattern("*sdxl*")
            .with_role(ModelRole::CheckpointBundle)
            .with_role(ModelRole::DiffusionModel)
            .with_role(ModelRole::TextEncoder)
            .with_role(ModelRole::Vae)
            .with_format(ModelFormat::Safetensors),
        )
        .with_rule(
            ModelSeriesRule::new(
                ModelSeries::new("stable_diffusion"),
                ModelVariant::new("sd15"),
            )
            .with_extension("safetensors")
            .with_filename_pattern("*sd1*")
            .with_role(ModelRole::CheckpointBundle)
            .with_format(ModelFormat::Safetensors),
        )
}

// --- ConfigDocument tests ---

#[test]
fn series_config_key_is_model_series_json() {
    assert_eq!(ModelSeriesConfig::KEY, "model_series.json");
}

#[test]
fn series_config_schema_version_matches_constant() {
    assert_eq!(
        ModelSeriesConfig::SCHEMA_VERSION,
        MODEL_SERIES_SCHEMA_VERSION
    );
}

#[test]
fn series_config_validate_accepts_matching_version() {
    let config = ModelSeriesConfig::new();
    let key = ConfigKey::new("model_series.json").unwrap();
    let ctx = ConfigValidationContext::new(key, "/tmp/model_series.json");
    let diagnostics = config.validate(&ctx);
    assert!(diagnostics.is_empty());
}

#[test]
fn series_config_validate_rejects_unsupported_version() {
    let config = ModelSeriesConfig::new().with_schema_version("reimagine.model_series.v99");
    let key = ConfigKey::new("model_series.json").unwrap();
    let ctx = ConfigValidationContext::new(key, "/tmp/model_series.json");
    let diagnostics = config.validate(&ctx);
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(
        diagnostics[0].code().as_str(),
        "CONFIG/MODEL_SERIES_SCHEMA_VERSION_UNSUPPORTED"
    );
}

#[tokio::test]
async fn series_config_loads_default_builtin_rules_from_config_handle() {
    let base = test_base("series-config-default");
    let app = AppConfig::new(AppPaths::new(&base));
    app.paths().ensure_all().await.unwrap();
    let handle = app.config::<ModelSeriesConfig>().unwrap();

    let (config, report) = handle.load().await.unwrap();

    assert!(report.diagnostics().is_empty());
    assert_eq!(handle.key().as_str(), "model_series.json");
    assert!(config.supports_series_variant(
        &ModelSeries::new("stable_diffusion"),
        &ModelVariant::new("sdxl"),
    ));

    cleanup(base).await;
}

// --- Classifier tests ---

#[test]
fn extension_match_sets_model_series_variant_roles_format() {
    let config = sdxl_config();
    let classifier = Classifier::new(&config);
    let candidate = ClassificationCandidate::new(
        Some(ModelRootId::new("base")),
        "checkpoints/sdxl_base_1.0.safetensors",
        "sdxl_base_1.0",
        "safetensors",
    );
    let result = classifier.classify(&candidate);
    assert_eq!(result.model_series().as_str(), "stable_diffusion");
    assert_eq!(result.variant().as_str(), "sdxl");
    assert_eq!(result.roles().len(), 4);
    assert_eq!(result.format(), Some(ModelFormat::Safetensors));
}

#[test]
fn first_matching_rule_wins() {
    // Both rules could match via extension, but filename_pattern narrows it.
    // "sdxl_base_1.0.safetensors" matches *sdxl*, so the first rule wins.
    let config = sdxl_config();
    let classifier = Classifier::new(&config);
    let candidate = ClassificationCandidate::new(
        None,
        "sdxl_base_1.0.safetensors",
        "sdxl_base_1.0",
        "safetensors",
    );
    let result = classifier.classify(&candidate);
    assert_eq!(result.variant().as_str(), "sdxl");
}

#[test]
fn no_match_returns_unknown_series_variant_empty_roles() {
    let config = sdxl_config();
    let classifier = Classifier::new(&config);
    let candidate = ClassificationCandidate::new(None, "misc/file.gguf", "file", "gguf");
    let result = classifier.classify(&candidate);
    assert_eq!(result.model_series().as_str(), "unknown");
    assert_eq!(result.variant().as_str(), "unknown");
    assert!(result.roles().is_empty());
}

#[test]
fn no_match_returns_observed_format_when_available() {
    let config = sdxl_config();
    let classifier = Classifier::new(&config);
    let candidate = ClassificationCandidate::new(None, "misc/file.gguf", "file", "gguf")
        .with_observed_format(ModelFormat::Gguf);
    let result = classifier.classify(&candidate);
    assert_eq!(result.format(), Some(ModelFormat::Gguf));
}

#[test]
fn no_match_returns_unknown_format_when_no_observed() {
    let config = sdxl_config();
    let classifier = Classifier::new(&config);
    let candidate = ClassificationCandidate::new(None, "misc/file.gguf", "file", "gguf");
    let result = classifier.classify(&candidate);
    assert_eq!(result.format(), Some(ModelFormat::Unknown));
}

#[test]
fn extension_strips_leading_dot_and_lowercases() {
    let config = ModelSeriesConfig::new().with_rule(
        ModelSeriesRule::new(ModelSeries::new("test"), ModelVariant::new("v1"))
            .with_extension("safetensors")
            .with_role(ModelRole::Upscaler),
    );
    let classifier = Classifier::new(&config);
    // Candidate extension has leading dot and mixed case, so it should still match.
    let candidate = ClassificationCandidate::new(None, "test.SAFETensors", "test", ".SAFETensors");
    let result = classifier.classify(&candidate);
    assert_eq!(result.model_series().as_str(), "test");
    assert_eq!(result.variant().as_str(), "v1");
}

#[test]
fn rule_extension_strips_leading_dot_and_lowercases() {
    let config = ModelSeriesConfig::new().with_rule(
        ModelSeriesRule::new(ModelSeries::new("test"), ModelVariant::new("v1"))
            .with_extension(".SAFEtensors")
            .with_role(ModelRole::Upscaler),
    );
    let classifier = Classifier::new(&config);
    let candidate = ClassificationCandidate::new(None, "test.safetensors", "test", "safetensors");
    let result = classifier.classify(&candidate);
    assert_eq!(result.model_series().as_str(), "test");
    assert_eq!(result.variant().as_str(), "v1");
}

#[test]
fn filename_glob_matches_relative_path_segments() {
    let config = ModelSeriesConfig::new().with_rule(
        ModelSeriesRule::new(ModelSeries::new("test"), ModelVariant::new("v1"))
            .with_path_pattern("models/**/*.safetensors")
            .with_role(ModelRole::Lora),
    );
    let classifier = Classifier::new(&config);
    let candidate = ClassificationCandidate::new(
        None,
        "models/checkpoints/demo.safetensors",
        "demo",
        "safetensors",
    );
    let result = classifier.classify(&candidate);
    assert_eq!(result.model_series().as_str(), "test");
}

#[test]
fn root_id_filter_blocks_non_matching_root() {
    let config = ModelSeriesConfig::new().with_rule(
        ModelSeriesRule::new(ModelSeries::new("test"), ModelVariant::new("v1"))
            .with_root_id(ModelRootId::new("custom"))
            .with_role(ModelRole::ControlNet),
    );
    let classifier = Classifier::new(&config);

    // Wrong root_id should not match.
    let candidate = ClassificationCandidate::new(
        Some(ModelRootId::new("base")),
        "file.safetensors",
        "file",
        "safetensors",
    );
    let result = classifier.classify(&candidate);
    assert_eq!(result.model_series().as_str(), "unknown");

    // Correct root_id should match.
    let candidate = ClassificationCandidate::new(
        Some(ModelRootId::new("custom")),
        "file.safetensors",
        "file",
        "safetensors",
    );
    let result = classifier.classify(&candidate);
    assert_eq!(result.model_series().as_str(), "test");
}

#[test]
fn rule_all_present_fields_must_match() {
    // Rule has all four matchers set, so the candidate must satisfy all.
    let config = ModelSeriesConfig::new().with_rule(
        ModelSeriesRule::new(ModelSeries::new("test"), ModelVariant::new("v1"))
            .with_root_id(ModelRootId::new("base"))
            .with_extension("safetensors")
            .with_filename_pattern("*demo*")
            .with_path_pattern("checkpoints/*"),
    );
    let classifier = Classifier::new(&config);

    // Matches root, ext, and filename, but path_pattern fails.
    let candidate = ClassificationCandidate::new(
        Some(ModelRootId::new("base")),
        "loras/demo_v1.safetensors",
        "demo_v1",
        "safetensors",
    );
    let result = classifier.classify(&candidate);
    assert_eq!(result.model_series().as_str(), "unknown");
}

#[test]
fn empty_config_classifies_everything_as_unknown() {
    let config = ModelSeriesConfig::new();
    let classifier = Classifier::new(&config);
    let candidate =
        ClassificationCandidate::new(None, "anything.safetensors", "anything", "safetensors");
    let result = classifier.classify(&candidate);
    assert_eq!(result.model_series().as_str(), "unknown");
    assert_eq!(result.variant().as_str(), "unknown");
    assert!(result.roles().is_empty());
}

fn test_base(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "reimagine-model-manager-classification-{name}-{}",
        std::process::id()
    ))
}

async fn cleanup(path: std::path::PathBuf) {
    let _ = tokio::fs::remove_dir_all(path).await;
}
