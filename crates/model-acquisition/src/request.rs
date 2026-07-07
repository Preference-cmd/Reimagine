use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// A HuggingFace repository identifier in `namespace/name` form.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RepoId(String);

impl RepoId {
    /// Validate and construct a `RepoId`.
    ///
    /// Returns `None` if the string is empty, or does not contain exactly one `/`.
    pub fn new(id: impl Into<String>) -> Option<Self> {
        let s = id.into();
        if s.is_empty() {
            return None;
        }
        // Must contain exactly one '/' separating namespace and name.
        let mut parts = s.split('/');
        match (parts.next(), parts.next(), parts.next()) {
            (Some(ns), Some(name), None) if !ns.is_empty() && !name.is_empty() => Some(Self(s)),
            _ => None,
        }
    }

    /// The full `namespace/name` string.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The namespace component (before `/`).
    pub fn namespace(&self) -> &str {
        self.0.split('/').next().unwrap_or("")
    }

    /// The name component (after `/`).
    pub fn name(&self) -> &str {
        self.0.split('/').nth(1).unwrap_or("")
    }
}

impl std::fmt::Display for RepoId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A git-style revision (branch, tag, or commit hash).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Revision(String);

impl Revision {
    /// The default branch (`main`).
    pub const MAIN: &'static str = "main";

    pub fn new(rev: impl Into<String>) -> Self {
        Self(rev.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for Revision {
    fn default() -> Self {
        Self(Self::MAIN.to_owned())
    }
}

/// File pattern allow-list for download filtering.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct AllowPatterns(Vec<String>);

impl AllowPatterns {
    /// Empty allow patterns means all files are included.
    pub fn new(patterns: Vec<String>) -> Self {
        Self(patterns)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn as_slice(&self) -> &[String] {
        &self.0
    }
}

impl From<Vec<String>> for AllowPatterns {
    fn from(v: Vec<String>) -> Self {
        Self(v)
    }
}

/// Controls what happens when the target directory already exists.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OverwritePolicy {
    /// Leave existing files and skip the acquisition.
    Skip,
    /// Overwrite existing files.
    Overwrite,
    /// Report an error if the target exists.
    Fail,
}

impl Default for OverwritePolicy {
    fn default() -> Self {
        Self::Skip
    }
}

/// A validated relative path under `<base_path>/models/`.
///
/// Guaranteed to be:
/// - Relative (no leading `/` or drive letter)
/// - Free of `.` and `..` segments
/// - Free of the `converted/` prefix
/// - Non-empty
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetRelativeDir(PathBuf);

impl TargetRelativeDir {
    /// Validate and construct a `TargetRelativeDir`.
    ///
    /// Returns the error message if validation fails.
    pub fn new(path: PathBuf) -> Result<Self, String> {
        Self::validate(&path).map(|_| Self(path))
    }

    fn validate(path: &std::path::Path) -> Result<(), String> {
        if path.as_os_str().is_empty() {
            return Err("target relative dir must not be empty".to_owned());
        }
        if path.has_root() {
            return Err("target relative dir must be relative".to_owned());
        }
        for component in path.components() {
            match component {
                std::path::Component::ParentDir => {
                    return Err("target relative dir must not contain `..`".to_owned());
                }
                std::path::Component::CurDir => {
                    return Err("target relative dir must not contain `.`".to_owned());
                }
                std::path::Component::Normal(seg) => {
                    // Prevent `converted/` as the first segment.
                    if seg == "converted" && path.components().next() == Some(component) {
                        return Err(
                            "target relative dir must not start with `converted/`".to_owned()
                        );
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub fn as_path(&self) -> &std::path::Path {
        &self.0
    }

    pub fn as_os_str(&self) -> &std::ffi::OsStr {
        self.0.as_os_str()
    }
}

/// A provider of model weights.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcquireProvider {
    HuggingFace,
}

/// Complete description of a model acquisition request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAcquisitionRequest {
    pub provider: AcquireProvider,
    pub repo_id: RepoId,
    #[serde(default)]
    pub revision: Revision,
    #[serde(default)]
    pub allow_patterns: AllowPatterns,
    pub target_relative_dir: TargetRelativeDir,
    #[serde(default)]
    pub overwrite_policy: OverwritePolicy,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_id_valid() {
        let id = RepoId::new("runwayml/stable-diffusion-v1-5").unwrap();
        assert_eq!(id.namespace(), "runwayml");
        assert_eq!(id.name(), "stable-diffusion-v1-5");
        assert_eq!(id.as_str(), "runwayml/stable-diffusion-v1-5");
    }

    #[test]
    fn repo_id_empty_rejected() {
        assert!(RepoId::new("").is_none());
    }

    #[test]
    fn repo_id_no_slash_rejected() {
        assert!(RepoId::new("justname").is_none());
    }

    #[test]
    fn repo_id_triple_slash_rejected() {
        assert!(RepoId::new("a/b/c").is_none());
    }

    #[test]
    fn target_relative_dir_valid() {
        let dir = TargetRelativeDir::new(PathBuf::from("sdxl/base")).unwrap();
        assert_eq!(dir.as_path(), PathBuf::from("sdxl/base"));
    }

    #[test]
    fn target_relative_dir_empty_rejected() {
        assert!(TargetRelativeDir::new(PathBuf::new()).is_err());
    }

    #[test]
    fn target_relative_dir_absolute_rejected() {
        assert!(TargetRelativeDir::new(PathBuf::from("/absolute/path")).is_err());
    }

    #[test]
    fn target_relative_dir_parent_dotdot_rejected() {
        assert!(TargetRelativeDir::new(PathBuf::from("../escape")).is_err());
    }

    #[test]
    fn target_relative_dir_curdir_rejected() {
        assert!(TargetRelativeDir::new(PathBuf::from("./rel")).is_err());
    }

    #[test]
    fn target_relative_dir_converted_prefix_rejected() {
        assert!(TargetRelativeDir::new(PathBuf::from("converted/sdxl")).is_err());
    }

    #[test]
    fn revision_default_is_main() {
        let rev = Revision::default();
        assert_eq!(rev.as_str(), "main");
    }
}
