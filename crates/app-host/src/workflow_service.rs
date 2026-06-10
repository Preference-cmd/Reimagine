use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use reimagine_config::{AppPaths, atomic_write};
use reimagine_core::command::{CommandBatch, CommandResult};
use reimagine_core::model::{NodeCatalog, WorkflowId};
use reimagine_core::session::WorkflowSession;
use reimagine_core::workflow::Workflow;

use crate::{AppHostError, AppHostResult};

pub struct WorkflowService {
    app_paths: AppPaths,
    sessions: RwLock<BTreeMap<WorkflowId, Arc<Mutex<WorkflowSession>>>>,
}

impl std::fmt::Debug for WorkflowService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let workflow_count = self
            .sessions
            .read()
            .map(|sessions| sessions.len())
            .unwrap_or_default();
        f.debug_struct("WorkflowService")
            .field("workflows_dir", &self.app_paths.workflows_dir())
            .field("workflow_count", &workflow_count)
            .finish()
    }
}

impl WorkflowService {
    pub fn new(app_paths: AppPaths) -> Self {
        Self {
            app_paths,
            sessions: RwLock::new(BTreeMap::new()),
        }
    }

    pub fn register_workflow(&self, workflow: Workflow) -> WorkflowId {
        let workflow_id = workflow.id().clone();
        let session = Arc::new(Mutex::new(WorkflowSession::new(workflow)));
        self.sessions
            .write()
            .expect("workflow registry poisoned")
            .insert(workflow_id.clone(), session);
        workflow_id
    }

    pub fn contains(&self, workflow_id: &WorkflowId) -> bool {
        self.sessions
            .read()
            .expect("workflow registry poisoned")
            .contains_key(workflow_id)
    }

    pub fn list_workflow_ids(&self) -> Vec<WorkflowId> {
        self.sessions
            .read()
            .expect("workflow registry poisoned")
            .keys()
            .cloned()
            .collect()
    }

    pub fn snapshot(&self, workflow_id: &WorkflowId) -> AppHostResult<Workflow> {
        let session = self.session(workflow_id)?;
        let guard = session.lock().expect("workflow session poisoned");
        Ok(guard.workflow().clone())
    }

    pub fn preview_batch(
        &self,
        workflow_id: &WorkflowId,
        node_catalog: &impl NodeCatalog,
        batch: CommandBatch,
    ) -> AppHostResult<CommandResult> {
        let session = self.session(workflow_id)?;
        let guard = session.lock().expect("workflow session poisoned");
        Ok(guard.preview_batch(node_catalog, batch))
    }

    pub fn apply_batch(
        &self,
        workflow_id: &WorkflowId,
        node_catalog: &impl NodeCatalog,
        batch: CommandBatch,
    ) -> AppHostResult<CommandResult> {
        let session = self.session(workflow_id)?;
        let mut guard = session.lock().expect("workflow session poisoned");
        Ok(guard.apply_batch(node_catalog, batch))
    }

    pub async fn save_workflow(&self, workflow_id: &WorkflowId) -> AppHostResult<PathBuf> {
        let workflow = self.snapshot(workflow_id)?;
        self.save_workflow_snapshot(&workflow).await
    }

    pub async fn save_workflow_snapshot(&self, workflow: &Workflow) -> AppHostResult<PathBuf> {
        let path = self.path_for_workflow_id(workflow.id())?;
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|error| AppHostError::Io {
                    path: parent.to_path_buf(),
                    message: error.to_string(),
                })?;
        }
        let bytes =
            serde_json::to_vec_pretty(workflow).map_err(|error| AppHostError::WorkflowJson {
                path: path.clone(),
                message: error.to_string(),
            })?;
        atomic_write(&path, bytes)
            .await
            .map_err(|error| AppHostError::Io {
                path: path.clone(),
                message: error.to_string(),
            })?;
        Ok(path)
    }

    pub async fn load_workflow(&self, workflow_id: &WorkflowId) -> AppHostResult<WorkflowId> {
        let path = self.path_for_workflow_id(workflow_id)?;
        let bytes = tokio::fs::read(&path)
            .await
            .map_err(|error| AppHostError::Io {
                path: path.clone(),
                message: error.to_string(),
            })?;
        let workflow = serde_json::from_slice::<Workflow>(&bytes).map_err(|error| {
            AppHostError::WorkflowJson {
                path: path.clone(),
                message: error.to_string(),
            }
        })?;
        if workflow.id() != workflow_id {
            return Err(AppHostError::WorkflowJson {
                path,
                message: format!(
                    "workflow file id `{}` does not match requested id `{workflow_id}`",
                    workflow.id()
                ),
            });
        }
        Ok(self.register_workflow(workflow))
    }

    pub fn workflows_dir(&self) -> &Path {
        self.app_paths.workflows_dir()
    }

    pub fn path_for_workflow_id(&self, workflow_id: &WorkflowId) -> AppHostResult<PathBuf> {
        ensure_safe_file_stem(workflow_id)?;
        Ok(self
            .app_paths
            .workflows_dir()
            .join(format!("{workflow_id}.json")))
    }

    fn session(&self, workflow_id: &WorkflowId) -> AppHostResult<Arc<Mutex<WorkflowSession>>> {
        self.sessions
            .read()
            .expect("workflow registry poisoned")
            .get(workflow_id)
            .cloned()
            .ok_or_else(|| AppHostError::UnknownWorkflow {
                workflow_id: workflow_id.clone(),
            })
    }
}

fn ensure_safe_file_stem(workflow_id: &WorkflowId) -> AppHostResult<()> {
    let id = workflow_id.as_str();
    let unsafe_id =
        id.is_empty() || id.contains('/') || id.contains('\\') || id == "." || id == "..";
    if unsafe_id {
        return Err(AppHostError::WorkflowIdPathUnsafe {
            workflow_id: workflow_id.clone(),
        });
    }
    Ok(())
}
