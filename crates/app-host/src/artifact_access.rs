//! Artifact access resolution and path safety validation.

use std::path::{Path, PathBuf};

use reimagine_config::AppPaths;
use reimagine_core::model::{ArtifactId, ArtifactRef, NodeId};

/// Resolved artifact access information.
#[derive(Debug, Clone)]
pub struct ArtifactAccess {
    pub artifact_id: ArtifactId,
    pub node_id: NodeId,
    pub reference: ArtifactRef,
    pub path: PathBuf,
    pub media_type: String,
}

/// Errors that can occur when resolving artifact access.
#[derive(Debug)]
pub enum ArtifactAccessError {
    /// The artifact id is not known to any active or terminal run.
    UnknownArtifact,
    /// The artifact reference failed path safety validation.
    UnsafeReference,
    /// The artifact record exists but the file no longer exists.
    FileGone,
    /// The artifact media type is not supported (V1: PNG only).
    UnsupportedMedia,
}

impl std::fmt::Display for ArtifactAccessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownArtifact => write!(f, "unknown artifact"),
            Self::UnsafeReference => write!(f, "unsafe artifact reference"),
            Self::FileGone => write!(f, "artifact file gone"),
            Self::UnsupportedMedia => write!(f, "unsupported media type"),
        }
    }
}

impl std::error::Error for ArtifactAccessError {}

/// Normalize a path by resolving `.` and `..` components without requiring
/// the path to exist on disk. This is a pure lexical operation — it does
/// not follow symlinks.
fn normalize_path(path: &Path) -> PathBuf {
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                components.pop();
            }
            std::path::Component::CurDir => {}
            other => components.push(other),
        }
    }
    components.iter().collect()
}

/// Validate an artifact reference and resolve it to an absolute path
/// under the output directory.
///
/// Path safety rules:
/// - Reject absolute paths
/// - Reject empty paths
/// - Reject `..` components
/// - Reject non-normal path components
/// - Require the reference to start with `output/`
/// - Join the safe relative suffix with `AppPaths::output_dir()`
/// - Canonicalize the output directory
/// - If the file exists, canonicalize and verify it stays under output dir
pub fn resolve_artifact_path(
    reference: &ArtifactRef,
    paths: &AppPaths,
) -> Result<PathBuf, ArtifactAccessError> {
    let ref_str = reference.as_str();

    // Reject empty paths
    if ref_str.is_empty() {
        return Err(ArtifactAccessError::UnsafeReference);
    }

    // Reject absolute paths
    if ref_str.starts_with('/') || ref_str.starts_with('\\') {
        return Err(ArtifactAccessError::UnsafeReference);
    }

    // Reject Windows absolute paths (e.g., C:\...)
    if ref_str.len() >= 2 && ref_str.as_bytes()[1] == b':' {
        return Err(ArtifactAccessError::UnsafeReference);
    }

    // Parse as path and check components
    let path = Path::new(ref_str);

    // Check each component
    for component in path.components() {
        match component {
            std::path::Component::Normal(_) => {}
            std::path::Component::ParentDir => {
                return Err(ArtifactAccessError::UnsafeReference);
            }
            _ => {
                return Err(ArtifactAccessError::UnsafeReference);
            }
        }
    }

    // Require the reference to start with "output/"
    let suffix = ref_str
        .strip_prefix("output/")
        .ok_or(ArtifactAccessError::UnsafeReference)?;

    // Reject if suffix is empty
    if suffix.is_empty() {
        return Err(ArtifactAccessError::UnsafeReference);
    }

    // Canonicalize the output directory. If it doesn't exist, no artifacts
    // can exist either, so return an error rather than weakening the
    // starts_with check with a raw path.
    let output_dir = paths
        .output_dir()
        .canonicalize()
        .map_err(|_| ArtifactAccessError::UnknownArtifact)?;

    // Join the suffix with output_dir
    let file_path = output_dir.join(suffix);

    // If the file exists, canonicalize and verify it stays under output dir
    if file_path.exists() {
        let canonical_file = file_path
            .canonicalize()
            .map_err(|_| ArtifactAccessError::UnsafeReference)?;

        if !canonical_file.starts_with(&output_dir) {
            return Err(ArtifactAccessError::UnsafeReference);
        }

        return Ok(canonical_file);
    }

    // File doesn't exist yet — verify the joined path normalizes under
    // output_dir. This prevents a crafted suffix like "foo/../../../etc/passwd"
    // from escaping when the file is later created.
    let normalized = normalize_path(&file_path);
    if !normalized.starts_with(&output_dir) {
        return Err(ArtifactAccessError::UnsafeReference);
    }

    Ok(file_path)
}

/// Determine the media type for an artifact reference.
/// V1 supports PNG only.
pub fn media_type_for_reference(reference: &ArtifactRef) -> Result<String, ArtifactAccessError> {
    let ref_str = reference.as_str();
    if ref_str.ends_with(".png") {
        Ok("image/png".to_string())
    } else {
        Err(ArtifactAccessError::UnsupportedMedia)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_paths() -> AppPaths {
        let base = std::env::temp_dir().join(format!(
            "reimagine-artifact-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        AppPaths::new(base)
    }

    #[test]
    fn reject_empty_reference() {
        let paths = temp_paths();
        let reference = ArtifactRef::new("");
        assert!(matches!(
            resolve_artifact_path(&reference, &paths),
            Err(ArtifactAccessError::UnsafeReference)
        ));
    }

    #[test]
    fn reject_absolute_path() {
        let paths = temp_paths();
        let reference = ArtifactRef::new("/etc/passwd");
        assert!(matches!(
            resolve_artifact_path(&reference, &paths),
            Err(ArtifactAccessError::UnsafeReference)
        ));
    }

    #[test]
    fn reject_parent_dir_traversal() {
        let paths = temp_paths();
        let reference = ArtifactRef::new("output/../../../etc/passwd");
        assert!(matches!(
            resolve_artifact_path(&reference, &paths),
            Err(ArtifactAccessError::UnsafeReference)
        ));
    }

    #[test]
    fn reject_missing_output_prefix() {
        let paths = temp_paths();
        let reference = ArtifactRef::new("foo/bar.png");
        assert!(matches!(
            resolve_artifact_path(&reference, &paths),
            Err(ArtifactAccessError::UnsafeReference)
        ));
    }

    #[test]
    fn reject_non_png_media() {
        let reference = ArtifactRef::new("output/foo.txt");
        assert!(matches!(
            media_type_for_reference(&reference),
            Err(ArtifactAccessError::UnsupportedMedia)
        ));
    }

    #[test]
    fn accept_valid_png_reference() {
        let reference = ArtifactRef::new("output/foo.png");
        assert_eq!(media_type_for_reference(&reference).unwrap(), "image/png");
    }
}
