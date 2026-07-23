use std::path::{Path, PathBuf};

pub fn is_safe_target_relative_dir(path: &Path) -> Result<(), String> {
    if path.as_os_str().is_empty() {
        return Err("path must not be empty".to_owned());
    }
    if path.has_root() {
        return Err("path must be relative, not absolute".to_owned());
    }
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                return Err("path must not contain `..` components".to_owned());
            }
            std::path::Component::CurDir => {
                return Err("path must not contain `.` components".to_owned());
            }
            std::path::Component::Normal(seg) => {
                if seg == "converted" && path.components().next() == Some(component) {
                    return Err("path must not start with `converted/`".to_owned());
                }
            }
            _ => {
                return Err("path contains unexpected component type".to_owned());
            }
        }
    }
    Ok(())
}

/// Validate that `resolved` is a child of `base` (within the workspace models/ dir).
pub fn is_child_of(base: &Path, resolved: &Path) -> bool {
    resolved.starts_with(base)
}

/// Resolve a relative target dir against the workspace models base path.
///
/// Validates safety constraints before returning.
pub fn resolve_models_path(base_models_dir: &Path, relative: &Path) -> Result<PathBuf, String> {
    is_safe_target_relative_dir(relative)?;
    let resolved = base_models_dir.join(relative);
    // Canonicalize to protect against symlink escapes if path exists.
    if resolved.exists() {
        let canonical = resolved
            .canonicalize()
            .map_err(|e| format!("failed to canonicalize path: {e}"))?;
        if !canonical.starts_with(
            &base_models_dir
                .canonicalize()
                .map_err(|e| format!("failed to canonicalize base: {e}"))?,
        ) {
            return Err("resolved path escapes the models directory".to_owned());
        }
        Ok(canonical)
    } else {
        Ok(resolved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn safe_relative_dir_ok() {
        assert!(is_safe_target_relative_dir(Path::new("sdxl/base")).is_ok());
        assert!(is_safe_target_relative_dir(Path::new("foo/bar/baz")).is_ok());
    }

    #[test]
    fn safe_relative_dir_empty_rejected() {
        assert!(is_safe_target_relative_dir(Path::new("")).is_err());
    }

    #[test]
    fn safe_relative_dir_absolute_rejected() {
        assert!(is_safe_target_relative_dir(Path::new("/abs")).is_err());
    }

    #[test]
    fn safe_relative_dir_dotdot_rejected() {
        assert!(is_safe_target_relative_dir(Path::new("a/../b")).is_err());
    }

    #[test]
    fn safe_relative_dir_dot_rejected() {
        assert!(is_safe_target_relative_dir(Path::new("./foo")).is_err());
    }

    #[test]
    fn safe_relative_dir_converted_prefix_rejected() {
        assert!(is_safe_target_relative_dir(Path::new("converted/sdxl")).is_err());
    }

    #[test]
    fn safe_relative_dir_deep_converted_ok() {
        assert!(is_safe_target_relative_dir(Path::new("models/converted/sdxl")).is_ok());
    }

    #[test]
    fn resolve_simple_path() {
        let base = PathBuf::from("/workspace/models");
        let rel = Path::new("sdxl/base");
        let resolved = resolve_models_path(&base, rel).unwrap();
        assert_eq!(resolved, base.join("sdxl/base"));
    }

    #[test]
    fn resolve_escapes_models_rejected() {
        let base = PathBuf::from("/workspace/models");
        let rel = Path::new("../outside");
        assert!(resolve_models_path(&base, rel).is_err());
    }
}
