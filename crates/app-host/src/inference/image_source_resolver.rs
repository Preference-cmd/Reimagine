//! Workspace-safe image source resolver.
//!
//! [`InputImageSourceResolver`] implements
//! [`reimagine_inference::ImageSourceResolver`]. It enforces the V1
//! rule that user-supplied image inputs (the `image` param of
//! `builtin.load_image`) must live under
//! `<base_path>/input/`. Absolute paths, parent escapes, and
//! symlinks that point outside the input directory are rejected
//! with a precise [`NodeExecutorError`]. The inference layer never
//! inspects raw paths; the executor receives only the already
//! authorized [`ResolvedImageSource`].
//!
//! V1 recognized media types are `image/png`, `image/jpeg`,
//! `image/webp`. Other extensions are rejected before reaching the
//! backend so the executor surfaces a deterministic, non-retryable
//! failure instead of letting the backend silently mis-import.

use std::path::{Path, PathBuf};

use reimagine_config::AppPaths;
use reimagine_inference::{NodeExecutorError, ResolvedImageSource};

/// Workspace-safe image source resolver.
#[derive(Debug, Clone)]
pub struct InputImageSourceResolver {
    input_dir: PathBuf,
}

impl InputImageSourceResolver {
    /// Construct a resolver rooted at `<base_path>/input`.
    pub fn new(paths: &AppPaths) -> Self {
        Self {
            input_dir: paths.input_dir().to_path_buf(),
        }
    }

    /// Construct a resolver rooted at an explicit input directory.
    /// Primarily used by tests that do not have a full
    /// [`AppPaths`] available.
    #[cfg(test)]
    pub fn with_input_dir(input_dir: impl Into<PathBuf>) -> Self {
        Self {
            input_dir: input_dir.into(),
        }
    }

    fn input_dir(&self) -> &Path {
        &self.input_dir
    }
}

impl reimagine_inference::ImageSourceResolver for InputImageSourceResolver {
    fn resolve(&self, path: &Path) -> Result<ResolvedImageSource, NodeExecutorError> {
        // Reject empty / whitespace-only paths up front so the
        // diagnostic doesn't read like an OS-level "no such file"
        // error from a downstream syscall.
        if path.as_os_str().is_empty() {
            return Err(NodeExecutorError::Failed {
                message: "builtin.load_image `image` param must be a non-empty path".to_string(),
            });
        }

        // Reject absolute paths: V1 inputs live under
        // `<base_path>/input/` and must be specified as workspace-
        // relative paths.
        if path.is_absolute() {
            return Err(NodeExecutorError::Failed {
                message: format!(
                    "builtin.load_image rejects absolute path `{}`; V1 inputs must be workspace-relative under `<base_path>/input/`",
                    path.display()
                ),
            });
        }

        // Reject parent escapes. `..` components anywhere in the
        // path are an unambiguous escape attempt.
        for component in path.components() {
            if matches!(component, std::path::Component::ParentDir) {
                return Err(NodeExecutorError::Failed {
                    message: format!(
                        "builtin.load_image rejects parent escape `{}`; V1 inputs must stay under `<base_path>/input/`",
                        path.display()
                    ),
                });
            }
        }

        let canonical_input =
            std::fs::canonicalize(self.input_dir()).map_err(|err| NodeExecutorError::Failed {
                message: format!(
                    "builtin.load_image could not resolve workspace input directory `{}`: {err}",
                    self.input_dir().display()
                ),
            })?;
        let candidate = canonical_input.join(path);
        let canonical_candidate = std::fs::canonicalize(&candidate).map_err(|err| {
            NodeExecutorError::Failed {
                message: format!(
                    "builtin.load_image could not resolve `{}` against workspace input directory `{}`: {err}",
                    path.display(),
                    self.input_dir().display()
                ),
            }
        })?;
        if !canonical_candidate.starts_with(&canonical_input) {
            return Err(NodeExecutorError::Failed {
                message: format!(
                    "builtin.load_image resolved path `{}` escapes workspace input directory `{}`; V1 inputs must stay under `<base_path>/input/`",
                    canonical_candidate.display(),
                    self.input_dir().display()
                ),
            });
        }

        // V1 inputs are user-supplied image files. Reject
        // directories and other non-files precisely so the backend
        // does not have to.
        let metadata =
            std::fs::metadata(&canonical_candidate).map_err(|err| NodeExecutorError::Failed {
                message: format!(
                    "builtin.load_image could not stat resolved path `{}`: {err}",
                    canonical_candidate.display()
                ),
            })?;
        if !metadata.is_file() {
            return Err(NodeExecutorError::Failed {
                message: format!(
                    "builtin.load_image resolved path `{}` is not a regular file; V1 inputs must be regular files",
                    canonical_candidate.display()
                ),
            });
        }

        let media_type = media_type_for_extension(
            canonical_candidate
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or(""),
        )
        .ok_or_else(|| NodeExecutorError::Failed {
            message: format!(
                "builtin.load_image rejects unsupported media type for `{}`; V1 supports png, jpg, jpeg, webp",
                canonical_candidate.display()
            ),
        })?;

        let display_name = canonical_candidate
            .file_name()
            .and_then(|n| n.to_str())
            .map(|s| s.to_string());

        Ok(ResolvedImageSource::new(
            canonical_candidate,
            media_type,
            display_name,
        ))
    }
}

/// V1 supported image media types by file extension.
fn media_type_for_extension(extension: &str) -> Option<&'static str> {
    match extension.to_ascii_lowercase().as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "webp" => Some("image/webp"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_file(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn accepts_workspace_relative_path_under_input_dir() {
        let base = tempdir("accept");
        let input = base.join("input");
        write_file(&input, "cat.png", b"png");
        let resolver = InputImageSourceResolver::with_input_dir(&input);

        let resolved =
            <InputImageSourceResolver as reimagine_inference::ImageSourceResolver>::resolve(
                &resolver,
                Path::new("cat.png"),
            )
            .expect("relative path under input dir should resolve");

        assert_eq!(resolved.media_type(), "image/png");
        assert_eq!(resolved.display_name(), Some("cat.png"));
        assert!(resolved.path().ends_with("cat.png"));

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn rejects_absolute_path() {
        let base = tempdir("absolute");
        let input = base.join("input");
        std::fs::create_dir_all(&input).unwrap();
        let resolver = InputImageSourceResolver::with_input_dir(&input);

        let err = <InputImageSourceResolver as reimagine_inference::ImageSourceResolver>::resolve(
            &resolver,
            Path::new("/etc/passwd"),
        )
        .expect_err("absolute path must be rejected");
        let msg = match err {
            NodeExecutorError::Failed { message } => message,
            other => panic!("expected Failed, got {other:?}"),
        };
        assert!(msg.contains("absolute"), "{msg}");
    }

    #[test]
    fn rejects_parent_escape() {
        let base = tempdir("escape");
        let input = base.join("input");
        std::fs::create_dir_all(&input).unwrap();
        let resolver = InputImageSourceResolver::with_input_dir(&input);

        let err = <InputImageSourceResolver as reimagine_inference::ImageSourceResolver>::resolve(
            &resolver,
            Path::new("../escape.txt"),
        )
        .expect_err("parent escape must be rejected");
        let msg = match err {
            NodeExecutorError::Failed { message } => message,
            other => panic!("expected Failed, got {other:?}"),
        };
        assert!(msg.contains("parent escape"), "{msg}");

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn rejects_unknown_extension() {
        let base = tempdir("unknown");
        let input = base.join("input");
        write_file(&input, "weird.txt", b"hello");
        let resolver = InputImageSourceResolver::with_input_dir(&input);

        let err = <InputImageSourceResolver as reimagine_inference::ImageSourceResolver>::resolve(
            &resolver,
            Path::new("weird.txt"),
        )
        .expect_err("unknown media type must be rejected");
        let msg = match err {
            NodeExecutorError::Failed { message } => message,
            other => panic!("expected Failed, got {other:?}"),
        };
        assert!(msg.contains("unsupported media type"), "{msg}");

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn rejects_directory_paths() {
        let base = tempdir("dir");
        let input = base.join("input");
        std::fs::create_dir_all(&input).unwrap();
        std::fs::create_dir_all(input.join("nested.png")).unwrap();
        let resolver = InputImageSourceResolver::with_input_dir(&input);

        let err = <InputImageSourceResolver as reimagine_inference::ImageSourceResolver>::resolve(
            &resolver,
            Path::new("nested.png"),
        )
        .expect_err("directory path must be rejected");
        let msg = match err {
            NodeExecutorError::Failed { message } => message,
            other => panic!("expected Failed, got {other:?}"),
        };
        assert!(msg.contains("not a regular file"), "{msg}");

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn accepts_jpeg_extension_with_image_jpeg_media_type() {
        let base = tempdir("jpeg");
        let input = base.join("input");
        write_file(&input, "photo.jpeg", b"jpeg-bytes");
        let resolver = InputImageSourceResolver::with_input_dir(&input);

        let resolved =
            <InputImageSourceResolver as reimagine_inference::ImageSourceResolver>::resolve(
                &resolver,
                Path::new("photo.jpeg"),
            )
            .expect("jpeg should resolve");

        assert_eq!(resolved.media_type(), "image/jpeg");
        assert_eq!(resolved.display_name(), Some("photo.jpeg"));

        let _ = std::fs::remove_dir_all(&base);
    }

    fn tempdir(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "reimagine-app-host-input-resolver-{name}-{nonce}-{}",
            std::process::id()
        ))
    }
}
