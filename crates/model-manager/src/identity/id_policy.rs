use std::fmt;

use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::event::OperationReport;
use reimagine_core::model::ModelId;
use reimagine_core::model::{ModelRole, ModelSeries, ModelVariant};
use sha2::{Digest, Sha256};

use crate::manifest::{Fingerprint, ModelDescriptor, ModelSource};

/// Outcome of auto-id generation when a collision was encountered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdResolution {
    /// No existing id matched - id is unique.
    NoConflict,
    /// Same fingerprint and source - treat as the same model.
    SameIdentity,
    /// Different fingerprint/source - id was suffixed to resolve collision.
    SuffixAppended,
}

impl fmt::Display for IdResolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoConflict => write!(f, "no_conflict"),
            Self::SameIdentity => write!(f, "same_identity"),
            Self::SuffixAppended => write!(f, "suffix_appended"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoIdResult {
    id: ModelId,
    resolution: IdResolution,
    report: OperationReport,
}

impl AutoIdResult {
    fn new(id: ModelId, resolution: IdResolution, report: OperationReport) -> Self {
        Self {
            id,
            resolution,
            report,
        }
    }

    pub fn id(&self) -> &ModelId {
        &self.id
    }

    pub fn resolution(&self) -> IdResolution {
        self.resolution
    }

    pub fn report(&self) -> &OperationReport {
        &self.report
    }
}

/// Policy for model id generation and conflict handling.
pub struct IdPolicy<'a> {
    existing: &'a [ModelDescriptor],
}

impl<'a> IdPolicy<'a> {
    pub fn new(existing: &'a [ModelDescriptor]) -> Self {
        Self { existing }
    }

    /// Validate a manually-chosen id against existing ids. Returns a report
    /// with a diagnostic if the id conflicts.
    pub fn validate_manual_id(&self, id: &str) -> OperationReport {
        let mut report = OperationReport::new();
        if self.existing.iter().any(|d| d.id().as_str() == id) {
            report.push_diagnostic(Diagnostic::new(
                reimagine_core::model::DiagnosticId::new(format!(
                    "model_manager:manual_id_conflict:{id}"
                )),
                DiagnosticCode::new("MODEL_MANAGER/MANUAL_ID_CONFLICT"),
                DiagnosticSeverity::Error,
                DiagnosticSourceName::new("model-manager"),
                format!("manual model id `{id}` conflicts with an existing entry"),
                DiagnosticTarget::new(DiagnosticTargetDomain::new("model-manager"))
                    .with_id(id.to_owned()),
            ));
        }
        report
    }

    /// Generate a deterministic auto-id from descriptor data without collision
    /// handling. Returns the base id string.
    pub fn generate_auto_id(
        &self,
        series: &ModelSeries,
        variant: &ModelVariant,
        role: ModelRole,
        source: &ModelSource,
    ) -> String {
        let stem = stem_from_path(source.path());
        let role_str = format!("{role:?}");
        let input = format!("{series}-{variant}-{role_str}-{stem}");
        let hash = short_hash(&input);
        format!("{series}-{variant}-{role_str}-{stem}-{hash}")
    }

    /// Generate a deterministic auto-id with collision handling.
    pub fn generate_auto_id_with_resolution(
        &self,
        series: &ModelSeries,
        variant: &ModelVariant,
        role: ModelRole,
        source: &ModelSource,
        fingerprint: Option<&Fingerprint>,
    ) -> AutoIdResult {
        let base_id = self.generate_auto_id(series, variant, role, source);

        match self.existing.iter().find(|d| d.id().as_str() == base_id) {
            None => AutoIdResult::new(
                ModelId::new(&base_id),
                IdResolution::NoConflict,
                OperationReport::new(),
            ),
            Some(existing) => {
                let same_source = existing.source() == source;
                let same_fp = match (fingerprint, existing.fingerprint()) {
                    (Some(fp), Some(ep)) => fp == ep,
                    _ => false,
                };

                if same_source && same_fp {
                    AutoIdResult::new(
                        ModelId::new(&base_id),
                        IdResolution::SameIdentity,
                        OperationReport::new(),
                    )
                } else {
                    let suffixed =
                        self.resolve_suffixed_id(&base_id, series, variant, role, source);
                    let mut report = OperationReport::new();
                    report.push_diagnostic(Diagnostic::new(
                        reimagine_core::model::DiagnosticId::new(format!(
                            "model_manager:auto_id_collision:{suffixed}"
                        )),
                        DiagnosticCode::new("MODEL_MANAGER/AUTO_ID_COLLISION_RESOLVED"),
                        DiagnosticSeverity::Warning,
                        DiagnosticSourceName::new("model-manager"),
                        format!(
                            "auto-generated id `{base_id}` collided and was resolved to `{suffixed}`"
                        ),
                        DiagnosticTarget::new(DiagnosticTargetDomain::new("model-manager"))
                            .with_id(suffixed.clone()),
                    ));
                    AutoIdResult::new(ModelId::new(suffixed), IdResolution::SuffixAppended, report)
                }
            }
        }
    }

    fn resolve_suffixed_id(
        &self,
        base_id: &str,
        series: &ModelSeries,
        variant: &ModelVariant,
        role: ModelRole,
        source: &ModelSource,
    ) -> String {
        let suffix_input = format!("{series}-{variant}-{role:?}-{}", source_identity(source));
        let suffix = long_hash(&suffix_input);
        let candidate = format!("{base_id}-{suffix}");

        if !self.id_exists(&candidate) {
            return candidate;
        }

        for counter in 2_u32.. {
            let candidate = format!("{base_id}-{suffix}-{counter}");
            if !self.id_exists(&candidate) {
                return candidate;
            }
        }

        unreachable!("unbounded counter must find an unused model id")
    }

    fn id_exists(&self, id: &str) -> bool {
        self.existing.iter().any(|d| d.id().as_str() == id)
    }
}

fn short_hash(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    hex_encode_short(&digest, 8)
}

fn long_hash(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    hex_encode_short(&digest, 16)
}

fn hex_encode_short(digest: &[u8], len: usize) -> String {
    let mut hex: String = digest
        .iter()
        .take(len.div_ceil(2))
        .map(|b| format!("{b:02x}"))
        .collect();
    hex.truncate(len);
    hex
}

fn stem_from_path(path: &str) -> String {
    let filename = path.rsplit('/').next().unwrap_or(path);
    let stem = filename
        .rsplit_once('.')
        .map(|(s, _)| s)
        .unwrap_or(filename);
    normalize_stem(stem)
}

fn normalize_stem(stem: &str) -> String {
    let mut result = String::with_capacity(stem.len());
    let mut prev_underscore = false;
    for ch in stem.to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            result.push(ch);
            prev_underscore = false;
        } else if !prev_underscore {
            result.push('_');
            prev_underscore = true;
        }
    }
    result.trim_end_matches('_').to_owned()
}

fn source_identity(source: &ModelSource) -> String {
    match source {
        ModelSource::LocalFileRelative { root_id, path } => {
            format!("relative:{root_id}:{path}")
        }
        ModelSource::LocalFileAbsolute { path } => format!("absolute:{path}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stem_from_path_extracts_filename_stem() {
        assert_eq!(
            stem_from_path("checkpoints/sdxl_base_1.0.safetensors"),
            "sdxl_base_1_0"
        );
        assert_eq!(stem_from_path("model.gguf"), "model");
    }

    #[test]
    fn normalize_stem_lowercases_and_replaces_non_alnum() {
        assert_eq!(normalize_stem("My Cool Model (v2)"), "my_cool_model_v2");
        assert_eq!(normalize_stem("sdxl_base_1.0"), "sdxl_base_1_0");
        assert_eq!(normalize_stem("hello__world"), "hello_world");
    }

    #[test]
    fn short_hash_is_eight_hex_chars() {
        let h = short_hash("test-input");
        assert_eq!(h.len(), 8);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn short_hash_is_deterministic() {
        assert_eq!(short_hash("abc"), short_hash("abc"));
        assert_ne!(short_hash("abc"), short_hash("def"));
    }
}
