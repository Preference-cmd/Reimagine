use std::path::{Path, PathBuf};

use crate::{ConfigError, ConfigResult};

/// Workspace directory layout for V1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppPaths {
    base_path: PathBuf,
    models_dir: PathBuf,
    input_dir: PathBuf,
    output_dir: PathBuf,
    workflows_dir: PathBuf,
    config_dir: PathBuf,
    projects_dir: PathBuf,
}

impl AppPaths {
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        let base_path = base_path.into();
        Self {
            models_dir: base_path.join("models"),
            input_dir: base_path.join("input"),
            output_dir: base_path.join("output"),
            workflows_dir: base_path.join("workflows"),
            config_dir: base_path.join("config"),
            projects_dir: base_path.join("projects"),
            base_path,
        }
    }

    pub async fn ensure_all(&self) -> ConfigResult<()> {
        for dir in [
            &self.base_path,
            &self.models_dir,
            &self.input_dir,
            &self.output_dir,
            &self.workflows_dir,
            &self.config_dir,
            &self.projects_dir,
        ] {
            tokio::fs::create_dir_all(dir)
                .await
                .map_err(|error| ConfigError::WriteFailed {
                    path: dir.clone(),
                    message: error.to_string(),
                })?;
        }
        Ok(())
    }

    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    pub fn input_dir(&self) -> &Path {
        &self.input_dir
    }

    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }

    pub fn workflows_dir(&self) -> &Path {
        &self.workflows_dir
    }

    pub fn config_dir(&self) -> &Path {
        &self.config_dir
    }

    pub fn projects_dir(&self) -> &Path {
        &self.projects_dir
    }

    /// Get the directory for a specific project.
    pub fn project_dir(&self, project_id: &str) -> PathBuf {
        self.projects_dir.join(project_id)
    }

    /// Get the workflows directory for a specific project.
    pub fn project_workflows_dir(&self, project_id: &str) -> PathBuf {
        self.project_dir(project_id).join("workflows")
    }

    /// Get the agent threads directory for a specific project.
    pub fn project_agent_threads_dir(&self, project_id: &str) -> PathBuf {
        self.project_dir(project_id).join("agent_threads")
    }

    /// Get the default project directory.
    pub fn default_project_dir(&self) -> PathBuf {
        self.project_dir("default")
    }

    /// Get the default project workflows directory.
    pub fn default_project_workflows_dir(&self) -> PathBuf {
        self.project_workflows_dir("default")
    }

    /// Get the default project agent threads directory.
    pub fn default_project_agent_threads_dir(&self) -> PathBuf {
        self.project_agent_threads_dir("default")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_app_paths_project_methods() {
        let temp_dir = tempdir().unwrap();
        let app_paths = AppPaths::new(temp_dir.path());

        // Test projects_dir
        assert_eq!(app_paths.projects_dir(), temp_dir.path().join("projects"));

        // Test project_dir
        assert_eq!(
            app_paths.project_dir("my-project"),
            temp_dir.path().join("projects/my-project")
        );

        // Test project_workflows_dir
        assert_eq!(
            app_paths.project_workflows_dir("my-project"),
            temp_dir.path().join("projects/my-project/workflows")
        );

        // Test project_agent_threads_dir
        assert_eq!(
            app_paths.project_agent_threads_dir("my-project"),
            temp_dir.path().join("projects/my-project/agent_threads")
        );

        // Test default project methods
        assert_eq!(
            app_paths.default_project_dir(),
            temp_dir.path().join("projects/default")
        );
        assert_eq!(
            app_paths.default_project_workflows_dir(),
            temp_dir.path().join("projects/default/workflows")
        );
        assert_eq!(
            app_paths.default_project_agent_threads_dir(),
            temp_dir.path().join("projects/default/agent_threads")
        );
    }

    #[test]
    fn test_app_paths_new() {
        let temp_dir = tempdir().unwrap();
        let app_paths = AppPaths::new(temp_dir.path());

        assert_eq!(app_paths.base_path(), temp_dir.path());
        assert_eq!(app_paths.models_dir(), temp_dir.path().join("models"));
        assert_eq!(app_paths.input_dir(), temp_dir.path().join("input"));
        assert_eq!(app_paths.output_dir(), temp_dir.path().join("output"));
        assert_eq!(app_paths.workflows_dir(), temp_dir.path().join("workflows"));
        assert_eq!(app_paths.config_dir(), temp_dir.path().join("config"));
        assert_eq!(app_paths.projects_dir(), temp_dir.path().join("projects"));
    }

    #[tokio::test]
    async fn test_ensure_all_creates_directories() {
        let temp_dir = tempdir().unwrap();
        let app_paths = AppPaths::new(temp_dir.path());

        // Ensure all directories are created
        app_paths.ensure_all().await.unwrap();

        assert!(temp_dir.path().exists());
        assert!(app_paths.models_dir().exists());
        assert!(app_paths.input_dir().exists());
        assert!(app_paths.output_dir().exists());
        assert!(app_paths.workflows_dir().exists());
        assert!(app_paths.config_dir().exists());
        assert!(app_paths.projects_dir().exists());
    }

    #[test]
    fn test_project_methods_with_different_ids() {
        let temp_dir = tempdir().unwrap();
        let app_paths = AppPaths::new(temp_dir.path());

        // Test with different project IDs
        let project1 = app_paths.project_dir("project-1");
        let project2 = app_paths.project_dir("project-2");

        assert_eq!(project1, temp_dir.path().join("projects/project-1"));
        assert_eq!(project2, temp_dir.path().join("projects/project-2"));

        // Ensure they're different
        assert_ne!(project1, project2);
    }
}
