use tokio::fs;

use crate::{AppPaths, ConfigError, ConfigResult};

/// Migrate flat workflow directory to project-based layout.
///
/// This migration:
/// 1. Creates projects/default/ directory structure
/// 2. Copies workflows/*.json to projects/default/workflows/
/// 3. Preserves original files as backup (renamed with .backup extension)
/// 4. Is idempotent - repeated calls don't duplicate files
pub async fn migrate_to_project_layout(app_paths: &AppPaths) -> ConfigResult<()> {
    let default_project_dir = app_paths.default_project_dir();
    let default_workflows_dir = app_paths.default_project_workflows_dir();

    // Create default project directory structure
    fs::create_dir_all(&default_project_dir)
        .await
        .map_err(|error| ConfigError::WriteFailed {
            path: default_project_dir.clone(),
            message: format!("Failed to create default project directory: {}", error),
        })?;

    fs::create_dir_all(&default_workflows_dir)
        .await
        .map_err(|error| ConfigError::WriteFailed {
            path: default_workflows_dir.clone(),
            message: format!(
                "Failed to create default project workflows directory: {}",
                error
            ),
        })?;

    // Create agent_threads directory for default project
    let default_agent_threads_dir = app_paths.default_project_agent_threads_dir();
    fs::create_dir_all(&default_agent_threads_dir)
        .await
        .map_err(|error| ConfigError::WriteFailed {
            path: default_agent_threads_dir.clone(),
            message: format!(
                "Failed to create default project agent_threads directory: {}",
                error
            ),
        })?;

    // Check if the old workflows directory exists and has files to migrate
    let old_workflows_dir = app_paths.workflows_dir();
    if !old_workflows_dir.exists() {
        // Nothing to migrate
        return Ok(());
    }

    // Read all files in the old workflows directory
    let mut entries =
        fs::read_dir(old_workflows_dir)
            .await
            .map_err(|error| ConfigError::ReadFailed {
                path: old_workflows_dir.to_path_buf(),
                message: format!("Failed to read workflows directory: {}", error),
            })?;

    while let Some(entry) = entries
        .next_entry()
        .await
        .map_err(|error| ConfigError::ReadFailed {
            path: old_workflows_dir.to_path_buf(),
            message: format!("Failed to read directory entry: {}", error),
        })?
    {
        let file_name = entry.file_name();
        let file_name_str = file_name.to_string_lossy();

        // Only migrate .json files
        if !file_name_str.ends_with(".json") {
            continue;
        }

        let source_path = entry.path();
        let dest_path = default_workflows_dir.join(&file_name);
        let backup_path = old_workflows_dir.join(format!("{}.backup", file_name_str));

        // Check if destination already exists (idempotent check)
        if dest_path.exists() {
            // Skip - already migrated
            continue;
        }

        // Copy the file to the new location
        fs::copy(&source_path, &dest_path)
            .await
            .map_err(|error| ConfigError::WriteFailed {
                path: dest_path.clone(),
                message: format!("Failed to copy workflow file: {}", error),
            })?;

        // Create backup by renaming the original
        fs::rename(&source_path, &backup_path)
            .await
            .map_err(|error| ConfigError::WriteFailed {
                path: backup_path.clone(),
                message: format!("Failed to create backup: {}", error),
            })?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_migrate_to_project_layout() {
        let temp_dir = tempdir().unwrap();
        let app_paths = AppPaths::new(temp_dir.path());

        // Create old workflows directory with some files
        let old_workflows_dir = app_paths.workflows_dir();
        fs::create_dir_all(old_workflows_dir).unwrap();

        // Create test workflow files
        let workflow1 = old_workflows_dir.join("workflow1.json");
        let workflow2 = old_workflows_dir.join("workflow2.json");
        fs::write(&workflow1, r#"{"name": "test1"}"#).unwrap();
        fs::write(&workflow2, r#"{"name": "test2"}"#).unwrap();

        // Run migration
        migrate_to_project_layout(&app_paths).await.unwrap();

        // Check that default project directories were created
        assert!(app_paths.default_project_dir().exists());
        assert!(app_paths.default_project_workflows_dir().exists());
        assert!(app_paths.default_project_agent_threads_dir().exists());

        // Check that files were copied to new location
        let dest1 = app_paths
            .default_project_workflows_dir()
            .join("workflow1.json");
        let dest2 = app_paths
            .default_project_workflows_dir()
            .join("workflow2.json");
        assert!(dest1.exists());
        assert!(dest2.exists());

        // Check that content is the same
        let content1 = fs::read_to_string(&dest1).unwrap();
        let content2 = fs::read_to_string(&dest2).unwrap();
        assert_eq!(content1, r#"{"name": "test1"}"#);
        assert_eq!(content2, r#"{"name": "test2"}"#);

        // Check that backups were created
        let backup1 = old_workflows_dir.join("workflow1.json.backup");
        let backup2 = old_workflows_dir.join("workflow2.json.backup");
        assert!(backup1.exists());
        assert!(backup2.exists());

        // Original files should no longer exist
        assert!(!workflow1.exists());
        assert!(!workflow2.exists());
    }

    #[tokio::test]
    async fn test_migration_is_idempotent() {
        let temp_dir = tempdir().unwrap();
        let app_paths = AppPaths::new(temp_dir.path());

        // Create old workflows directory with a file
        let old_workflows_dir = app_paths.workflows_dir();
        fs::create_dir_all(old_workflows_dir).unwrap();

        let workflow1 = old_workflows_dir.join("workflow1.json");
        fs::write(&workflow1, r#"{"name": "test1"}"#).unwrap();

        // Run migration twice
        migrate_to_project_layout(&app_paths).await.unwrap();
        migrate_to_project_layout(&app_paths).await.unwrap();

        // Check that only one copy exists
        let dest1 = app_paths
            .default_project_workflows_dir()
            .join("workflow1.json");
        assert!(dest1.exists());

        // Check that backup exists
        let backup1 = old_workflows_dir.join("workflow1.json.backup");
        assert!(backup1.exists());

        // Count files in destination directory
        let mut count = 0;
        let entries = fs::read_dir(app_paths.default_project_workflows_dir()).unwrap();
        for _ in entries {
            count += 1;
        }
        assert_eq!(count, 1); // Only one workflow file
    }

    #[tokio::test]
    async fn test_migration_preserves_original_files() {
        let temp_dir = tempdir().unwrap();
        let app_paths = AppPaths::new(temp_dir.path());

        // Create old workflows directory with a file
        let old_workflows_dir = app_paths.workflows_dir();
        fs::create_dir_all(old_workflows_dir).unwrap();

        let workflow1 = old_workflows_dir.join("workflow1.json");
        let original_content = r#"{"name": "original", "version": 1}"#;
        fs::write(&workflow1, original_content).unwrap();

        // Run migration
        migrate_to_project_layout(&app_paths).await.unwrap();

        // Check that backup contains original content
        let backup1 = old_workflows_dir.join("workflow1.json.backup");
        assert!(backup1.exists());
        let backup_content = fs::read_to_string(&backup1).unwrap();
        assert_eq!(backup_content, original_content);

        // Check that new file also has the content
        let dest1 = app_paths
            .default_project_workflows_dir()
            .join("workflow1.json");
        let dest_content = fs::read_to_string(&dest1).unwrap();
        assert_eq!(dest_content, original_content);
    }

    #[tokio::test]
    async fn test_migration_handles_missing_workflows_dir() {
        let temp_dir = tempdir().unwrap();
        let app_paths = AppPaths::new(temp_dir.path());

        // Don't create workflows directory
        // Run migration - should succeed
        migrate_to_project_layout(&app_paths).await.unwrap();

        // Check that default project directories were still created
        assert!(app_paths.default_project_dir().exists());
        assert!(app_paths.default_project_workflows_dir().exists());
        assert!(app_paths.default_project_agent_threads_dir().exists());
    }

    #[tokio::test]
    async fn test_migration_handles_non_json_files() {
        let temp_dir = tempdir().unwrap();
        let app_paths = AppPaths::new(temp_dir.path());

        // Create old workflows directory with mixed files
        let old_workflows_dir = app_paths.workflows_dir();
        fs::create_dir_all(old_workflows_dir).unwrap();

        let workflow1 = old_workflows_dir.join("workflow1.json");
        let readme = old_workflows_dir.join("README.txt");
        let hidden = old_workflows_dir.join(".hidden");

        fs::write(&workflow1, r#"{"name": "test1"}"#).unwrap();
        fs::write(&readme, "This is a readme").unwrap();
        fs::write(&hidden, "hidden content").unwrap();

        // Run migration
        migrate_to_project_layout(&app_paths).await.unwrap();

        // Check that only .json file was migrated
        let dest1 = app_paths
            .default_project_workflows_dir()
            .join("workflow1.json");
        assert!(dest1.exists());

        let dest_readme = app_paths.default_project_workflows_dir().join("README.txt");
        let dest_hidden = app_paths.default_project_workflows_dir().join(".hidden");
        assert!(!dest_readme.exists());
        assert!(!dest_hidden.exists());

        // Check that non-json files still exist in original location
        assert!(readme.exists());
        assert!(hidden.exists());
    }
}
