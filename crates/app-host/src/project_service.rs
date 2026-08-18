//! Project service: CRUD for the Project domain aggregate root.
//!
//! Layout: `projects/{project_id}/project.json` plus the project-owned
//! board at `projects/{project_id}/board.json`. Creation writes both
//! documents atomically; deletion removes the whole project directory
//! (workflows, agent threads, and the board), which is the V1 cascade
//! strategy until workflow/thread services own their indices.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

use reimagine_config::{AppPaths, atomic_write};
use reimagine_core::model::ProjectId;
use reimagine_core::project::{Project, ProjectMetadata};

use crate::board_service::{BoardService, ensure_board_file};
use crate::{AppHostError, AppHostResult};

pub struct ProjectService {
    paths: AppPaths,
    projects: RwLock<BTreeMap<ProjectId, Project>>,
    board_service: Arc<BoardService>,
}

impl std::fmt::Debug for ProjectService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let project_count = self
            .projects
            .read()
            .map(|projects| projects.len())
            .unwrap_or_default();
        f.debug_struct("ProjectService")
            .field("paths", &self.paths)
            .field("project_count", &project_count)
            .field("board_service", &self.board_service)
            .finish()
    }
}

impl ProjectService {
    pub fn new(paths: AppPaths, board_service: Arc<BoardService>) -> Self {
        Self {
            paths,
            projects: RwLock::new(BTreeMap::new()),
            board_service,
        }
    }

    pub fn paths(&self) -> &AppPaths {
        &self.paths
    }

    pub fn board_service(&self) -> &Arc<BoardService> {
        &self.board_service
    }

    pub fn project_dir(&self, project_id: &ProjectId) -> std::path::PathBuf {
        self.paths.project_dir(project_id.as_str())
    }

    pub fn project_file(&self, project_id: &ProjectId) -> std::path::PathBuf {
        self.project_dir(project_id).join("project.json")
    }

    /// Create a project, persisting `project.json` and an empty
    /// `board.json` before the project becomes visible.
    pub async fn create_project(
        &self,
        project_id: ProjectId,
        metadata: ProjectMetadata,
    ) -> AppHostResult<Project> {
        {
            let projects = self.projects.read().expect("project registry poisoned");
            if projects.contains_key(&project_id) {
                return Err(AppHostError::ProjectAlreadyExists {
                    project_id: project_id.clone(),
                });
            }
        }
        let project_dir = self.project_dir(&project_id);
        if project_dir.exists() {
            return Err(AppHostError::ProjectAlreadyExists {
                project_id: project_id.clone(),
            });
        }

        tokio::fs::create_dir_all(&project_dir)
            .await
            .map_err(|error| AppHostError::Io {
                path: project_dir.clone(),
                message: error.to_string(),
            })?;

        let project = Project::new(project_id.clone(), metadata);
        write_project_atomic(&self.project_file(&project_id), &project).await?;
        ensure_board_file(&self.paths, &project_id).await?;

        self.projects
            .write()
            .expect("project registry poisoned")
            .insert(project_id, project.clone());
        Ok(project)
    }

    /// Load a project from disk, caching it for subsequent reads.
    pub async fn load_project(&self, project_id: &ProjectId) -> AppHostResult<Project> {
        if let Some(project) = self
            .projects
            .read()
            .expect("project registry poisoned")
            .get(project_id)
            .cloned()
        {
            return Ok(project);
        }

        let project = read_project_atomic_source(&self.project_file(project_id)).await?;
        self.projects
            .write()
            .expect("project registry poisoned")
            .insert(project_id.clone(), project.clone());
        Ok(project)
    }

    /// List every project that has a `project.json` on disk, loading
    /// not-yet-cached entries. Projects are returned sorted by id.
    pub async fn list_projects(&self) -> AppHostResult<Vec<Project>> {
        let mut entries = tokio::fs::read_dir(self.paths.projects_dir())
            .await
            .map_err(|error| AppHostError::Io {
                path: self.paths.projects_dir().to_path_buf(),
                message: error.to_string(),
            })?;

        let mut loaded = BTreeMap::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|error| AppHostError::Io {
                path: self.paths.projects_dir().to_path_buf(),
                message: error.to_string(),
            })?
        {
            let Ok(file_type) = entry.file_type().await else {
                continue;
            };
            if !file_type.is_dir() {
                continue;
            }
            let Some(project_id) = entry
                .file_name()
                .to_str()
                .map(|name| ProjectId::new(name.to_owned()))
            else {
                continue;
            };
            let project_file = self.project_file(&project_id);
            if !project_file.is_file() {
                continue;
            }
            if let Some(project) = self
                .projects
                .read()
                .expect("project registry poisoned")
                .get(&project_id)
                .cloned()
            {
                loaded.insert(project_id, project);
                continue;
            }
            match read_project_atomic_source(&project_file).await {
                Ok(project) => {
                    loaded.insert(project_id, project);
                }
                Err(error) => {
                    tracing::warn!(
                        path = %project_file.display(),
                        %error,
                        "skipping unreadable project"
                    );
                }
            }
        }

        *self.projects.write().expect("project registry poisoned") = loaded.clone();
        Ok(loaded.into_values().collect())
    }

    /// Replace a project's metadata and persist it. Creation timestamp
    /// is preserved from the loaded project; callers supply the new
    /// `updated_at` inside `metadata`.
    pub async fn update_project(
        &self,
        project_id: &ProjectId,
        metadata: ProjectMetadata,
    ) -> AppHostResult<Project> {
        let mut project = self.load_project(project_id).await?;
        *project.metadata_mut() = metadata;
        write_project_atomic(&self.project_file(project_id), &project).await?;
        self.projects
            .write()
            .expect("project registry poisoned")
            .insert(project_id.clone(), project.clone());
        Ok(project)
    }

    /// Delete a project and everything under its directory (board,
    /// workflows, agent threads, memory). In-memory registries are
    /// cleared first so a later create with the same id starts fresh.
    pub async fn delete_project(&self, project_id: &ProjectId) -> AppHostResult<()> {
        self.load_project(project_id).await?;
        self.projects
            .write()
            .expect("project registry poisoned")
            .remove(project_id);
        self.board_service.remove_project(project_id);
        tokio::fs::remove_dir_all(self.project_dir(project_id))
            .await
            .map_err(|error| AppHostError::Io {
                path: self.project_dir(project_id),
                message: error.to_string(),
            })
    }
}

async fn read_project_atomic_source(path: &Path) -> AppHostResult<Project> {
    match tokio::fs::read(path).await {
        Ok(bytes) => serde_json::from_slice(&bytes).map_err(|error| AppHostError::Io {
            path: path.to_path_buf(),
            message: format!("invalid project document: {error}"),
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let project_id = path
                .parent()
                .and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str())
                .map(|name| ProjectId::new(name.to_owned()))
                .unwrap_or_else(|| ProjectId::new("unknown-project"));
            Err(AppHostError::UnknownProject { project_id })
        }
        Err(error) => Err(AppHostError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        }),
    }
}

pub(crate) async fn write_project_atomic(path: &Path, project: &Project) -> AppHostResult<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|error| AppHostError::Io {
                path: parent.to_path_buf(),
                message: error.to_string(),
            })?;
    }
    let bytes = serde_json::to_vec_pretty(project).map_err(|error| AppHostError::Io {
        path: path.to_path_buf(),
        message: format!("failed to serialize project: {error}"),
    })?;
    atomic_write(path, bytes)
        .await
        .map_err(|error| AppHostError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    Ok(())
}
