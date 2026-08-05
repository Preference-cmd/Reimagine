use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::UNIX_EPOCH;

use reimagine_config::{AppPaths, atomic_write};
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandBatch, CommandProvenance, CommandResult,
    CommandResultStatus,
};
use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::model::{DiagnosticId, NodeCatalog, WorkflowId};
use reimagine_core::session::WorkflowSession;
use reimagine_core::workflow::Workflow;

use crate::proposal::WorkflowProposal;
use crate::{AppHostError, AppHostResult};

/// Summary of a workflow file persisted in the workspace `workflows/` dir.
///
/// Kept crate-internal: Tauri/Axum adapters serialize it to JSON via serde
/// without needing to name the type.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SavedWorkflowInfo {
    id: String,
    modified_millis: u64,
}

pub struct WorkflowService {
    app_paths: AppPaths,
    sessions: RwLock<BTreeMap<WorkflowId, Arc<Mutex<WorkflowSession>>>>,
    proposals: RwLock<BTreeMap<WorkflowId, WorkflowProposal>>,
}

impl std::fmt::Debug for WorkflowService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let workflow_count = self
            .sessions
            .read()
            .map(|sessions| sessions.len())
            .unwrap_or_default();
        let proposal_count = self
            .proposals
            .read()
            .map(|proposals| proposals.len())
            .unwrap_or_default();
        f.debug_struct("WorkflowService")
            .field("workflows_dir", &self.app_paths.workflows_dir())
            .field("workflow_count", &workflow_count)
            .field("proposal_count", &proposal_count)
            .finish()
    }
}

impl WorkflowService {
    pub fn new(app_paths: AppPaths) -> Self {
        Self {
            app_paths,
            sessions: RwLock::new(BTreeMap::new()),
            proposals: RwLock::new(BTreeMap::new()),
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

    /// List workflow files persisted on disk (newest first).
    ///
    /// Only lists files whose stem is a safe, ASCII workflow id; skips
    /// anything else that might have been dropped into the directory.
    #[allow(private_interfaces)]
    pub fn list_saved_workflows(&self) -> AppHostResult<Vec<SavedWorkflowInfo>> {
        let dir = self.app_paths.workflows_dir();
        let entries = match std::fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(error) => {
                return Err(AppHostError::Io {
                    path: dir.to_path_buf(),
                    message: error.to_string(),
                });
            }
        };
        let mut out = Vec::new();
        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    return Err(AppHostError::Io {
                        path: dir.to_path_buf(),
                        message: error.to_string(),
                    });
                }
            };
            let file_name = entry.file_name();
            let Some(stem) = file_name
                .to_str()
                .and_then(|name| name.strip_suffix(".json"))
            else {
                continue;
            };
            if stem.is_empty()
                || !stem.is_ascii()
                || stem == "."
                || stem == ".."
                || stem.contains('/')
                || stem.contains('\\')
            {
                continue;
            }
            let modified_millis = entry
                .metadata()
                .ok()
                .and_then(|metadata| metadata.modified().ok())
                .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
                .map(|duration| duration.as_millis() as u64)
                .unwrap_or(0);
            out.push(SavedWorkflowInfo {
                id: stem.to_string(),
                modified_millis,
            });
        }
        out.sort_by_key(|b| std::cmp::Reverse(b.modified_millis));
        Ok(out)
    }

    pub fn path_for_workflow_id(&self, workflow_id: &WorkflowId) -> AppHostResult<PathBuf> {
        ensure_safe_file_stem(workflow_id)?;
        Ok(self
            .app_paths
            .workflows_dir()
            .join(format!("{workflow_id}.json")))
    }

    pub fn store_proposal(&self, proposal: WorkflowProposal) -> AppHostResult<()> {
        let workflow_id = proposal.workflow_id().clone();
        let mut proposals = self.proposals.write().expect("proposal registry poisoned");
        proposals.insert(workflow_id, proposal);
        Ok(())
    }

    pub fn apply_pending_proposal(
        &self,
        workflow_id: &WorkflowId,
        node_catalog: &impl NodeCatalog,
        approved_by: Option<reimagine_core::command::CommandActor>,
    ) -> AppHostResult<CommandResult> {
        let proposal = self.get_pending_proposal(workflow_id).ok_or_else(|| {
            AppHostError::NoPendingProposal {
                workflow_id: workflow_id.clone(),
            }
        })?;

        // Stale-approval guard: reject if the workflow has advanced since the proposal was created.
        {
            let session = self.session(workflow_id)?;
            let guard = session.lock().expect("workflow session poisoned");
            let current_version = guard.version();
            if current_version != proposal.base_version() {
                let diagnostic = Diagnostic::new(
                    DiagnosticId::new(format!(
                        "app-host:proposal-stale:{}",
                        proposal.proposal_id().as_str()
                    )),
                    DiagnosticCode::new("APP_HOST/PROPOSAL_STALE"),
                    DiagnosticSeverity::Error,
                    DiagnosticSourceName::new("app-host"),
                    format!(
                        "proposal base version {} does not match current workflow version {}; \
                         refusing stale approval",
                        proposal.base_version().get(),
                        current_version.get(),
                    ),
                    DiagnosticTarget::new(DiagnosticTargetDomain::new("workflow"))
                        .with_id(workflow_id.as_str()),
                );
                return Ok(CommandResult::new(
                    CommandResultStatus::Rejected,
                    current_version,
                    Vec::new(),
                    vec![diagnostic],
                    None,
                ));
            }
        }

        let mut batch = CommandBatch::new(
            proposal.command_batch().id().clone(),
            CommandActor::new(CommandActorKind::Agent)
                .with_id(proposal.agent_session_id().as_str()),
            proposal.base_version(),
            CommandProvenance::AgentProposal {
                proposal_id: proposal.proposal_id().clone(),
                approved_by,
            },
            proposal.command_batch().created_at().clone(),
            proposal.command_batch().commands().to_vec(),
        );
        if let Some(cid) = proposal.command_batch().correlation_id() {
            batch = batch.with_correlation_id(cid.clone());
        }

        let result = self.apply_batch(workflow_id, node_catalog, batch)?;
        if result.status() != CommandResultStatus::Rejected {
            self.remove_proposal(workflow_id);
        }
        Ok(result)
    }

    pub fn get_pending_proposal(&self, workflow_id: &WorkflowId) -> Option<WorkflowProposal> {
        self.proposals
            .read()
            .expect("proposal registry poisoned")
            .get(workflow_id)
            .cloned()
            .filter(|p| p.status() == crate::proposal::ProposalStatus::Pending)
    }

    pub fn list_proposals(&self) -> Vec<WorkflowProposal> {
        self.proposals
            .read()
            .expect("proposal registry poisoned")
            .values()
            .cloned()
            .collect()
    }

    pub fn remove_proposal(&self, workflow_id: &WorkflowId) -> Option<WorkflowProposal> {
        self.proposals
            .write()
            .expect("proposal registry poisoned")
            .remove(workflow_id)
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
