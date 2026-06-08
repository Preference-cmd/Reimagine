use reimagine_config::{ConfigDocument, ConfigValidationContext};
use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::model::{DiagnosticId, ModelRole, ModelSeries, ModelVariant};
use serde::{Deserialize, Serialize};

use crate::manifest::{ModelFormat, ModelRootId};

pub const MODEL_SERIES_SCHEMA_VERSION: &str = "reimagine.model_series.v1";

/// User-editable rules used by later scanner/classifier slices.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSeriesConfig {
    schema_version: String,
    rules: Vec<ModelSeriesRule>,
}

impl ModelSeriesConfig {
    pub fn new() -> Self {
        Self {
            schema_version: MODEL_SERIES_SCHEMA_VERSION.to_owned(),
            rules: Vec::new(),
        }
    }

    pub fn v1_builtin() -> Self {
        Self::new()
            .with_rule(ModelSeriesRule::new(
                ModelSeries::new("stable_diffusion"),
                ModelVariant::new("sdxl"),
            ))
            .with_rule(ModelSeriesRule::new(
                ModelSeries::new("stable_diffusion"),
                ModelVariant::new("sd15"),
            ))
    }

    pub fn with_rule(mut self, rule: ModelSeriesRule) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn with_schema_version(mut self, version: impl Into<String>) -> Self {
        self.schema_version = version.into();
        self
    }

    pub fn schema_version(&self) -> &str {
        &self.schema_version
    }

    pub fn rules(&self) -> &[ModelSeriesRule] {
        &self.rules
    }

    pub fn supports_series_variant(&self, series: &ModelSeries, variant: &ModelVariant) -> bool {
        self.rules
            .iter()
            .any(|rule| rule.model_series() == series && rule.variant() == variant)
    }
}

impl Default for ModelSeriesConfig {
    fn default() -> Self {
        Self::v1_builtin()
    }
}

impl ConfigDocument for ModelSeriesConfig {
    const KEY: &'static str = "model_series.json";
    const SCHEMA_VERSION: &'static str = MODEL_SERIES_SCHEMA_VERSION;

    fn validate(&self, context: &ConfigValidationContext) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        if self.schema_version != MODEL_SERIES_SCHEMA_VERSION {
            diagnostics.push(Diagnostic::new(
                DiagnosticId::new(format!(
                    "config:{}:schema_version_unsupported",
                    context.key()
                )),
                DiagnosticCode::new("CONFIG/MODEL_SERIES_SCHEMA_VERSION_UNSUPPORTED"),
                DiagnosticSeverity::Error,
                DiagnosticSourceName::new("model-manager"),
                format!(
                    "model series config schema version `{}` is not supported; expected `{}`",
                    self.schema_version, MODEL_SERIES_SCHEMA_VERSION,
                ),
                DiagnosticTarget::new(DiagnosticTargetDomain::new("config"))
                    .with_id(context.key().to_string())
                    .with_path(context.path().display().to_string()),
            ));
        }

        diagnostics
    }
}

/// One declarative classification rule. Matching behavior is implemented in a
/// later slice; this slice only owns the serializable shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSeriesRule {
    #[serde(skip_serializing_if = "Option::is_none")]
    root_id: Option<ModelRootId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    path_pattern: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    filename_pattern: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    extension: Option<String>,
    model_series: ModelSeries,
    variant: ModelVariant,
    roles: Vec<ModelRole>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<ModelFormat>,
}

impl ModelSeriesRule {
    pub fn new(model_series: ModelSeries, variant: ModelVariant) -> Self {
        Self {
            root_id: None,
            path_pattern: None,
            filename_pattern: None,
            extension: None,
            model_series,
            variant,
            roles: Vec::new(),
            format: None,
        }
    }

    pub fn with_root_id(mut self, root_id: ModelRootId) -> Self {
        self.root_id = Some(root_id);
        self
    }

    pub fn with_path_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.path_pattern = Some(pattern.into());
        self
    }

    pub fn with_filename_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.filename_pattern = Some(pattern.into());
        self
    }

    pub fn with_extension(mut self, extension: impl Into<String>) -> Self {
        self.extension = Some(extension.into());
        self
    }

    pub fn with_role(mut self, role: ModelRole) -> Self {
        self.roles.push(role);
        self
    }

    pub fn with_format(mut self, format: ModelFormat) -> Self {
        self.format = Some(format);
        self
    }

    pub fn root_id(&self) -> Option<&ModelRootId> {
        self.root_id.as_ref()
    }

    pub fn path_pattern(&self) -> Option<&str> {
        self.path_pattern.as_deref()
    }

    pub fn filename_pattern(&self) -> Option<&str> {
        self.filename_pattern.as_deref()
    }

    pub fn extension(&self) -> Option<&str> {
        self.extension.as_deref()
    }

    pub fn model_series(&self) -> &ModelSeries {
        &self.model_series
    }

    pub fn variant(&self) -> &ModelVariant {
        &self.variant
    }

    pub fn roles(&self) -> &[ModelRole] {
        &self.roles
    }

    pub fn format(&self) -> Option<ModelFormat> {
        self.format
    }
}
