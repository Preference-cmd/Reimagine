use burn_store::{ApplyError, ApplyResult};

use crate::error::BurnBackendError;

#[derive(Debug, Clone, Copy)]
pub(crate) struct SdxlLoadPolicy {
    component_role: &'static str,
    partial_load_policy: &'static str,
    required_snapshots: &'static [&'static str],
    required_prefixes: &'static [&'static str],
    optional_snapshots: &'static [&'static str],
    generated_snapshot_contains: &'static [&'static str],
    deferred_snapshot_families: &'static [&'static str],
    remapped_key_patterns: &'static [&'static str],
}

impl SdxlLoadPolicy {
    pub(crate) const fn new(component_role: &'static str) -> Self {
        Self {
            component_role,
            partial_load_policy: "partial load policy: allowed only for the current scaffold/module role; required snapshots are still enforced",
            required_snapshots: &[],
            required_prefixes: &[],
            optional_snapshots: &[],
            generated_snapshot_contains: &[],
            deferred_snapshot_families: &[],
            remapped_key_patterns: &[],
        }
    }

    pub(crate) const fn with_required_snapshots(
        mut self,
        required_snapshots: &'static [&'static str],
    ) -> Self {
        self.required_snapshots = required_snapshots;
        self
    }

    pub(crate) const fn with_required_prefixes(
        mut self,
        required_prefixes: &'static [&'static str],
    ) -> Self {
        self.required_prefixes = required_prefixes;
        self
    }

    pub(crate) const fn with_optional_snapshots(
        mut self,
        optional_snapshots: &'static [&'static str],
    ) -> Self {
        self.optional_snapshots = optional_snapshots;
        self
    }

    pub(crate) const fn with_generated_snapshot_contains(
        mut self,
        generated_snapshot_contains: &'static [&'static str],
    ) -> Self {
        self.generated_snapshot_contains = generated_snapshot_contains;
        self
    }

    pub(crate) const fn with_remapped_key_patterns(
        mut self,
        remapped_key_patterns: &'static [&'static str],
    ) -> Self {
        self.remapped_key_patterns = remapped_key_patterns;
        self
    }
}

pub(crate) fn validate_apply_result(
    policy: SdxlLoadPolicy,
    result: &ApplyResult,
) -> Result<(), BurnBackendError> {
    let report = SdxlLoadReport::from_apply_result(policy, result);
    if report.has_failures() {
        return Err(BurnBackendError::InvalidRequest(report.to_string()));
    }

    Ok(())
}

#[cfg(test)]
pub(crate) fn format_apply_report(policy: SdxlLoadPolicy, result: &ApplyResult) -> String {
    SdxlLoadReport::from_apply_result(policy, result).to_string()
}

#[derive(Debug)]
struct SdxlLoadReport {
    policy: SdxlLoadPolicy,
    required_missing: Vec<String>,
    optional_missing: Vec<String>,
    unexpected_source: Vec<String>,
    generated_snapshots: Vec<String>,
    errors: Vec<ApplyError>,
    applied_count: usize,
    missing_count: usize,
}

impl SdxlLoadReport {
    fn from_apply_result(policy: SdxlLoadPolicy, result: &ApplyResult) -> Self {
        let mut required_missing = result
            .missing
            .iter()
            .map(|(path, _)| path)
            .filter(|path| matches_required_prefix(path, policy.required_prefixes))
            .cloned()
            .collect::<Vec<_>>();

        for required in policy.required_snapshots {
            if !result.applied.iter().any(|applied| applied == required)
                && !required_missing.iter().any(|missing| missing == required)
            {
                required_missing.push((*required).to_string());
            }
        }
        required_missing.sort();
        required_missing.dedup();

        let optional_missing = policy
            .optional_snapshots
            .iter()
            .filter(|optional| !result.applied.iter().any(|applied| applied == **optional))
            .map(|optional| (*optional).to_string())
            .collect::<Vec<_>>();

        let generated_snapshots = result
            .applied
            .iter()
            .filter(|applied| {
                policy
                    .generated_snapshot_contains
                    .iter()
                    .any(|needle| applied.contains(needle))
            })
            .cloned()
            .collect::<Vec<_>>();

        Self {
            policy,
            required_missing,
            optional_missing,
            unexpected_source: result.unused.clone(),
            generated_snapshots,
            errors: result.errors.clone(),
            applied_count: result.applied.len(),
            missing_count: result.missing.len(),
        }
    }

    fn has_failures(&self) -> bool {
        !self.required_missing.is_empty() || !self.errors.is_empty()
    }
}

impl std::fmt::Display for SdxlLoadReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Burn-native SDXL load report component_role={}",
            self.policy.component_role
        )?;
        writeln!(f, "{}", self.policy.partial_load_policy)?;
        writeln!(f, "applied snapshots: {}", self.applied_count)?;
        writeln!(f, "missing snapshots observed: {}", self.missing_count)?;

        for pattern in self.policy.remapped_key_patterns {
            writeln!(f, "remapped source key pattern: {pattern}")?;
        }
        for generated in &self.generated_snapshots {
            writeln!(f, "generated snapshot: {generated}")?;
        }
        for missing in &self.required_missing {
            writeln!(f, "required snapshot missing: {missing}")?;
        }
        for missing in &self.optional_missing {
            writeln!(f, "optional snapshot missing: {missing}")?;
        }
        for family in self.policy.deferred_snapshot_families {
            writeln!(f, "deferred snapshot family: {family}")?;
        }
        for unexpected in &self.unexpected_source {
            writeln!(f, "unexpected source snapshot: {unexpected}")?;
        }
        for error in &self.errors {
            match error {
                ApplyError::ShapeMismatch {
                    path,
                    expected,
                    found,
                } => {
                    writeln!(
                        f,
                        "shape mismatch: {path}; expected={expected:?}; found={found:?}"
                    )?;
                }
                ApplyError::DTypeMismatch {
                    path,
                    expected,
                    found,
                } => {
                    writeln!(
                        f,
                        "dtype mismatch: {path}; expected={expected:?}; found={found:?}"
                    )?;
                }
                ApplyError::AdapterError { path, message } => {
                    writeln!(f, "adapter error: {path}; {message}")?;
                }
                ApplyError::LoadError { path, message } => {
                    writeln!(f, "load error: {path}; {message}")?;
                }
            }
        }

        Ok(())
    }
}

fn matches_required_prefix(path: &str, required_prefixes: &[&str]) -> bool {
    required_prefixes
        .iter()
        .any(|prefix| path.starts_with(prefix))
}
