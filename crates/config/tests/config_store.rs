use std::path::PathBuf;

use reimagine_config::{
    AppConfig, AppPaths, ConfigDocument, ConfigError, ConfigKey, ConfigResult,
    ConfigValidationContext,
};
use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::model::DiagnosticId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
struct ExampleConfig {
    schema_version: String,
    name: String,
}

impl ConfigDocument for ExampleConfig {
    const KEY: &'static str = "example/settings.json";
    const SCHEMA_VERSION: &'static str = "1";

    fn validate(&self, context: &ConfigValidationContext) -> Vec<Diagnostic> {
        if self.schema_version == Self::SCHEMA_VERSION {
            return Vec::new();
        }

        vec![Diagnostic::new(
            DiagnosticId::new(format!("config:{}:schema_version", context.key().as_str())),
            DiagnosticCode::new("CONFIG/SCHEMA_VERSION_UNSUPPORTED"),
            DiagnosticSeverity::Error,
            DiagnosticSourceName::new("test"),
            "schema version is unsupported",
            DiagnosticTarget::new(DiagnosticTargetDomain::new("config"))
                .with_id(context.key().as_str())
                .with_path(context.path().display().to_string()),
        )]
    }
}

#[tokio::test]
async fn app_paths_ensure_all_creates_workspace_layout() -> ConfigResult<()> {
    let base = test_base("layout");
    let paths = AppPaths::new(&base);

    paths.ensure_all().await?;

    assert!(base.join("models").is_dir());
    assert!(base.join("input").is_dir());
    assert!(base.join("output").is_dir());
    assert!(base.join("workflows").is_dir());
    assert!(base.join("config").is_dir());
    assert_eq!(paths.models_dir(), base.join("models"));
    assert_eq!(paths.config_dir(), base.join("config"));

    cleanup(base).await;
    Ok(())
}

#[tokio::test]
async fn config_handle_loads_default_for_missing_file_and_reports_validation() -> ConfigResult<()> {
    let base = test_base("missing_default");
    let app = AppConfig::new(AppPaths::new(&base));
    app.paths().ensure_all().await?;

    let handle = app.config::<ExampleConfig>()?;
    let (value, report) = handle.load().await?;

    assert_eq!(value, ExampleConfig::default());
    assert_eq!(report.key().as_str(), ExampleConfig::KEY);
    assert_eq!(report.path(), &base.join("config/example/settings.json"));
    assert_eq!(report.diagnostics().len(), 1);
    assert_eq!(
        report.diagnostics()[0].id().as_str(),
        "config:example/settings.json:schema_version"
    );

    cleanup(base).await;
    Ok(())
}

#[tokio::test]
async fn config_handle_saves_loads_and_updates_json_documents() -> ConfigResult<()> {
    let base = test_base("save_update");
    let app = AppConfig::new(AppPaths::new(&base));
    app.paths().ensure_all().await?;
    let handle = app.config::<ExampleConfig>()?;

    let saved = handle
        .save(&ExampleConfig {
            schema_version: "1".to_owned(),
            name: "first".to_owned(),
        })
        .await?;
    assert!(saved.diagnostics().is_empty());

    let (loaded, loaded_report) = handle.load().await?;
    assert_eq!(loaded.name, "first");
    assert!(loaded_report.diagnostics().is_empty());

    let (updated, updated_report) = handle
        .update(|value| {
            value.name = "second".to_owned();
        })
        .await?;
    assert_eq!(updated.name, "second");
    assert!(updated_report.diagnostics().is_empty());

    let disk = tokio::fs::read_to_string(handle.path()).await.unwrap();
    assert!(disk.contains("\"second\""));

    cleanup(base).await;
    Ok(())
}

#[tokio::test]
async fn config_key_rejects_absolute_empty_and_parent_escape_paths() {
    assert!(matches!(
        ConfigKey::new(""),
        Err(ConfigError::PathInvalid { .. })
    ));
    assert!(matches!(
        ConfigKey::new("../outside.json"),
        Err(ConfigError::PathInvalid { .. })
    ));
    assert!(matches!(
        ConfigKey::new("/absolute.json"),
        Err(ConfigError::PathInvalid { .. })
    ));
    assert_eq!(
        ConfigKey::new("agent/providers.json").unwrap().as_str(),
        "agent/providers.json"
    );
}

#[tokio::test]
async fn invalid_json_returns_config_error_with_diagnostic() -> ConfigResult<()> {
    let base = test_base("invalid_json");
    let app = AppConfig::new(AppPaths::new(&base));
    app.paths().ensure_all().await?;
    let handle = app.config::<ExampleConfig>()?;
    tokio::fs::create_dir_all(handle.path().parent().unwrap())
        .await
        .unwrap();
    tokio::fs::write(handle.path(), b"{not-json").await.unwrap();

    let error = handle.load().await.unwrap_err();
    let diagnostic = error.to_diagnostic(None);

    assert!(matches!(error, ConfigError::JsonInvalid { .. }));
    assert_eq!(diagnostic.code().as_str(), "CONFIG/JSON_INVALID");
    assert_eq!(diagnostic.primary().id(), Some(ExampleConfig::KEY));

    cleanup(base).await;
    Ok(())
}

fn test_base(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("reimagine-config-{name}-{}", std::process::id()))
}

async fn cleanup(path: PathBuf) {
    let _ = tokio::fs::remove_dir_all(path).await;
}
