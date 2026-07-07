use std::path::{Path, PathBuf};

use crate::error::{ModelAcquisitionError, ModelAcquisitionResult};
use crate::request::{AcquireProvider, OverwritePolicy, RepoId, Revision};

/// Compute the staging directory path for a given acquisition.
///
/// Format: `<base_models_dir>/.staging/<provider>/<namespace>-<name>@<revision>/`
pub fn staging_dir(
    base_models_dir: &Path,
    provider: &AcquireProvider,
    repo_id: &RepoId,
    revision: &Revision,
) -> PathBuf {
    let provider_str = match provider {
        AcquireProvider::HuggingFace => "huggingface",
    };
    let repo_slug = format!("{}-{}", repo_id.namespace(), repo_id.name());
    base_models_dir
        .join(".staging")
        .join(provider_str)
        .join(format!("{repo_slug}@{}", revision.as_str()))
}

/// Compute a backup directory path (sibling to `target` with `.backup` suffix).
fn backup_dir(target: &Path) -> PathBuf {
    let name = target
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("target");
    target.with_file_name(format!(".{name}.backup"))
}

/// Atomically promote a completed staging directory to its final target path.
///
/// This function follows the same pattern as Burn's `publish_staged_package`:
///
/// 1. If the target does not exist, atomically rename staging → target.
/// 2. If the target exists and policy is `Overwrite`, rename target → `.target.backup`,
///    then staging → target, then delete backup. On failure, restore from backup.
/// 3. If the target exists and policy is `Fail`, return `TargetExists`.
/// 4. If the target exists and policy is `Skip`, return `TargetExists`.
///
/// The caller is responsible for ensuring the staging directory is fully written
/// and that no other process is concurrently writing to either path.
pub async fn promote_staged(
    staging: &Path,
    target: &Path,
    overwrite_policy: &OverwritePolicy,
) -> ModelAcquisitionResult<()> {
    use crate::ModelAcquisitionError;

    if !target.exists() {
        // Ensure parent directory exists — rename requires it.
        if let Some(parent) = target.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|source| {
                ModelAcquisitionError::Io {
                    path: parent.to_path_buf(),
                    message: source.to_string(),
                }
            })?;
        }
        tokio::fs::rename(staging, target)
            .await
            .map_err(|source| ModelAcquisitionError::Io {
                path: target.to_path_buf(),
                message: source.to_string(),
            })?;
        return Ok(());
    }

    match overwrite_policy {
        OverwritePolicy::Skip | OverwritePolicy::Fail => {
            return Err(ModelAcquisitionError::TargetExists {
                path: target.to_path_buf(),
            });
        }
        OverwritePolicy::Overwrite => {
            // Proceed with backup-and-replace.
        }
    }

    let backup = backup_dir(target);
    remove_dir_if_exists(&backup).await?;

    // Rename existing target to backup.
    tokio::fs::rename(target, &backup)
        .await
        .map_err(|source| ModelAcquisitionError::Io {
            path: target.to_path_buf(),
            message: format!("failed to back up existing target: {source}"),
        })?;

    // Attempt to promote staging to target.
    match tokio::fs::rename(staging, target).await {
        Ok(()) => {
            // Success — remove backup.
            remove_dir_if_exists(&backup).await?;
            Ok(())
        }
        Err(source) => {
            // Failed — try to restore backup.
            let err = ModelAcquisitionError::Io {
                path: target.to_path_buf(),
                message: format!("failed to promote staging: {source}"),
            };
            let _ = tokio::fs::rename(&backup, target).await;
            Err(err)
        }
    }
}

async fn remove_dir_if_exists(path: &Path) -> ModelAcquisitionResult<()> {
    if path.exists() {
        tokio::fs::remove_dir_all(path)
            .await
            .map_err(|source| ModelAcquisitionError::Io {
                path: path.to_path_buf(),
                message: source.to_string(),
            })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[tokio::test]
    async fn test_staging_dir_format() {
        let base = PathBuf::from("/workspace/models");
        let repo = RepoId::new("runwayml/stable-diffusion-v1-5").unwrap();
        let rev = Revision::new("main");

        let dir = staging_dir(&base, &AcquireProvider::HuggingFace, &repo, &rev);
        assert_eq!(
            dir,
            PathBuf::from(
                "/workspace/models/.staging/huggingface/runwayml-stable-diffusion-v1-5@main"
            )
        );
    }

    #[tokio::test]
    async fn test_promote_new_target() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join(".staging");
        let target = tmp.path().join("models/sdxl/base");

        fs::create_dir_all(&staging).unwrap();
        fs::write(staging.join("model.safetensors"), b"fake").unwrap();

        assert!(!target.exists());
        promote_staged(&staging, &target, &OverwritePolicy::Skip)
            .await
            .unwrap();
        assert!(target.join("model.safetensors").exists());
    }

    #[tokio::test]
    async fn test_promote_skip_existing() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join(".staging");
        let target = tmp.path().join("models/sdxl/base");

        fs::create_dir_all(&staging).unwrap();
        fs::write(staging.join("new.bin"), b"new").unwrap();
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("old.bin"), b"old").unwrap();

        let err = promote_staged(&staging, &target, &OverwritePolicy::Skip)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            crate::ModelAcquisitionError::TargetExists { .. }
        ));
        // Target should be unchanged.
        assert!(target.join("old.bin").exists());
        assert!(!target.join("new.bin").exists());
    }

    #[tokio::test]
    async fn test_promote_overwrite() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join(".staging");
        let target = tmp.path().join("models/sdxl/base");

        fs::create_dir_all(&staging).unwrap();
        fs::write(staging.join("new.bin"), b"new").unwrap();
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("old.bin"), b"old").unwrap();

        promote_staged(&staging, &target, &OverwritePolicy::Overwrite)
            .await
            .unwrap();
        assert!(target.join("new.bin").exists());
        assert!(!target.join("old.bin").exists());
        // Backup should have been cleaned up.
        assert!(!target.with_file_name(".models.backup").exists());
    }
}
