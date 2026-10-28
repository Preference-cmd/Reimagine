//! Project service: CRUD for the Project domain aggregate root.
//!
//! Layout: `projects/{project_id}/project.json` plus the project-owned
//! board at `projects/{project_id}/board.json`. Creation writes both
//! documents atomically; deletion removes the whole project directory
//! (workflows, the board, and project memory), which is the V1 cascade
//! strategy until the workflow service owns its index.

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use reimagine_config::AppPaths;
use reimagine_core::model::ProjectId;
use reimagine_core::project::{Project, ProjectMetadata};

use crate::board_service::{BoardService, ensure_board_file};
use crate::collection_service::CollectionService;
use crate::document_store::{read_document, write_document};
use crate::{AppHostError, AppHostResult};

pub struct ProjectService {
    paths: AppPaths,
    projects: RwLock<BTreeMap<ProjectId, Project>>,
    board_service: Arc<BoardService>,
    collection_service: Arc<CollectionService>,
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
            .field("collection_service", &self.collection_service)
            .finish()
    }
}

impl ProjectService {
    pub fn new(
        paths: AppPaths,
        board_service: Arc<BoardService>,
        collection_service: Arc<CollectionService>,
    ) -> Self {
        Self {
            paths,
            projects: RwLock::new(BTreeMap::new()),
            board_service,
            collection_service,
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

    /// Return whether the project is known to the in-memory registry or has
    /// a durable project document on disk.
    pub fn contains_project(&self, project_id: &ProjectId) -> bool {
        self.projects
            .read()
            .expect("project registry poisoned")
            .contains_key(project_id)
            || self.project_file(project_id).is_file()
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
        tokio::fs::create_dir_all(self.paths.project_workflows_dir(project_id.as_str()))
            .await
            .map_err(|error| AppHostError::Io {
                path: self.paths.project_workflows_dir(project_id.as_str()),
                message: error.to_string(),
            })?;

        let project = Project::new(project_id.clone(), metadata);
        write_document(&self.project_file(&project_id), &project, "project").await?;
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

        let project: Project = read_document(&self.project_file(project_id), "project")
            .await?
            .ok_or_else(|| AppHostError::UnknownProject {
                project_id: project_id.clone(),
            })?;
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
            match read_document::<Project>(&project_file, "project").await {
                Ok(Some(project)) => {
                    loaded.insert(project_id, project);
                }
                Ok(None) => {}
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
        write_document(&self.project_file(project_id), &project, "project").await?;
        self.projects
            .write()
            .expect("project registry poisoned")
            .insert(project_id.clone(), project.clone());
        Ok(project)
    }

    /// Delete a project and everything under its directory (board,
    /// workflows, memory). In-memory registries are cleared first so a
    /// later create with the same id starts fresh.
    pub async fn delete_project(&self, project_id: &ProjectId) -> AppHostResult<()> {
        self.load_project(project_id).await?;
        self.projects
            .write()
            .expect("project registry poisoned")
            .remove(project_id);
        self.board_service.remove_project(project_id);
        self.collection_service.remove_project(project_id);
        tokio::fs::remove_dir_all(self.project_dir(project_id))
            .await
            .map_err(|error| AppHostError::Io {
                path: self.project_dir(project_id),
                message: error.to_string(),
            })
    }
}
