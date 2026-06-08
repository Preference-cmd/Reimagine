use reimagine_core::model::{ModelRole, ModelSeries, ModelVariant};

use crate::manifest::ModelFormat;

use super::candidate::ClassificationCandidate;
use super::series_config::ModelSeriesConfig;

/// Result of classifying a candidate against the series config rules.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    model_series: ModelSeries,
    variant: ModelVariant,
    roles: Vec<ModelRole>,
    format: Option<ModelFormat>,
}

impl ClassificationResult {
    fn unknown(observed_format: Option<ModelFormat>) -> Self {
        Self {
            model_series: ModelSeries::new("unknown"),
            variant: ModelVariant::new("unknown"),
            roles: Vec::new(),
            format: Some(observed_format.unwrap_or(ModelFormat::Unknown)),
        }
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

/// Applies classification rules to candidates. First matching rule wins.
pub struct Classifier<'a> {
    config: &'a ModelSeriesConfig,
}

impl<'a> Classifier<'a> {
    pub fn new(config: &'a ModelSeriesConfig) -> Self {
        Self { config }
    }

    pub fn classify(&self, candidate: &ClassificationCandidate) -> ClassificationResult {
        for rule in self.config.rules() {
            if rule_matches(rule, candidate) {
                return ClassificationResult {
                    model_series: rule.model_series().clone(),
                    variant: rule.variant().clone(),
                    roles: rule.roles().to_vec(),
                    format: rule.format(),
                };
            }
        }
        ClassificationResult::unknown(candidate.observed_format())
    }
}

fn rule_matches(
    rule: &super::series_config::ModelSeriesRule,
    candidate: &ClassificationCandidate,
) -> bool {
    if let Some(rule_root) = rule.root_id() {
        match candidate.root_id() {
            Some(candidate_root) if candidate_root == rule_root => {}
            _ => return false,
        }
    }

    if let Some(pattern) = rule.path_pattern() {
        if !glob_match::glob_match(pattern, candidate.path()) {
            return false;
        }
    }

    if let Some(pattern) = rule.filename_pattern() {
        if !glob_match::glob_match(pattern, candidate.filename()) {
            return false;
        }
    }

    if let Some(rule_ext) = rule.extension() {
        let candidate_ext = normalize_extension(candidate.extension());
        let rule_ext = normalize_extension(rule_ext);
        if candidate_ext != rule_ext {
            return false;
        }
    }

    true
}

fn normalize_extension(ext: &str) -> String {
    ext.trim_start_matches('.').to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_extension_strips_dot_and_lowercases() {
        assert_eq!(normalize_extension(".SAFETensors"), "safetensors");
        assert_eq!(normalize_extension("gguf"), "gguf");
        assert_eq!(normalize_extension(".Gguf"), "gguf");
    }

    #[test]
    fn empty_config_matches_nothing() {
        let config = ModelSeriesConfig::new();
        let classifier = Classifier::new(&config);
        let candidate =
            ClassificationCandidate::new(None, "file.safetensors", "file", "safetensors");
        let result = classifier.classify(&candidate);
        assert_eq!(result.model_series().as_str(), "unknown");
    }
}
