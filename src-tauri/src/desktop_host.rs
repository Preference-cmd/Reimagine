use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use mod_sdk::{ModRegistry, ProcessMod, ProcessModSpec};
use reimagine_app_host::WorkspaceScope;
use reimagine_app_host::dto::{
    ArtifactMetadataDto, BoardCommandResultDto, BoardSnapshotDto, HealthResponse,
    NodeCatalogResponse, ProjectDto, ProjectMetadataInputDto, RunWorkflowResponse,
};
#[cfg(feature = "aigc")]
use reimagine_app_host::dto::{ComputeProfileDto, ModelInfoDto};
use reimagine_app_host::{
    AppHost, AppHostError, InvokeRequest, InvokeResult, OperationInfo, WorkspaceHost,
};
#[cfg(feature = "aigc")]
use reimagine_app_host::{
    BackendSelection, WorkerBackendCandidate, WorkerInstallationDto, WorkerManagementService,
    WorkerSelectionHandle, WorkerSwitchError,
};
#[cfg(feature = "aigc")]
use reimagine_backend_worker_host::{WorkerLaunchSpec, WorkerLimits};
#[cfg(feature = "aigc")]
use reimagine_backend_worker_protocol::ProtocolRange;
use reimagine_config::AppPaths;
use reimagine_core::board::BoardCommand;
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandBatch, CommandProvenance, CommandResult,
    OperationContext, WorkflowChange, WorkflowCommand,
};
use reimagine_core::event::{Event, EventBus};
use reimagine_core::model::{HistoryEntryId, ProjectId};
use reimagine_core::task::CancelToken;
use reimagine_core::workflow::Workflow;
#[cfg(feature = "aigc")]
use serde::Serialize;
use tauri::ipc::Channel;

use crate::event_bridge::TauriEventBridge;

const WORKSPACE_SCOPE: &str = "reimagine-tauri-host";
const APP_HOST_LOCK: &str = "desktop app_host lock poisoned";

/// Names an explicit `text-mod` binary; must point at an existing file.
const TEXT_MOD_ENV: &str = "REIMAGINE_TEXT_MOD";
/// The mod binary shipped beside the desktop executable.
const TEXT_MOD_NAME: &str = "text-mod";
/// Bounds the mod's initialize handshake and `mod/start` round-trip.
const TEXT_MOD_STARTUP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Resolve the text-mod binary: `REIMAGINE_TEXT_MOD` when it names a file,
/// otherwise `text-mod` beside the current executable (`.exe` on Windows).
fn text_mod_candidate(env: Option<&std::ffi::OsStr>, exe_dir: Option<&Path>) -> Option<PathBuf> {
    if let Some(value) = env {
        let path = PathBuf::from(value);
        if path.is_file() {
            return Some(path);
        }
        eprintln!(
            "[reimagine] {TEXT_MOD_ENV} (`{}`) is not a file; looking beside the executable",
            path.display()
        );
    }
    let candidate = exe_dir?.join(format!("{TEXT_MOD_NAME}{}", std::env::consts::EXE_SUFFIX));
    candidate.is_file().then_some(candidate)
}

fn resolve_text_mod_path() -> Option<PathBuf> {
    let env = std::env::var_os(TEXT_MOD_ENV);
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf));
    text_mod_candidate(env.as_deref(), exe_dir.as_deref())
}

/// Start the text mod when its binary is present.
///
/// A missing binary or a failed spawn never blocks application startup: the
/// host is built with an empty registry and the text operations are simply
/// absent from `op_list`. A mod that spawned but failed to start stays
/// registered as unavailable, so `op_list` reports its operations with that
/// health instead of silently hiding them. The child is stopped by
/// [`WorkspaceHost::shutdown`] on exit.
async fn start_text_mod(event_bus: &EventBus) -> Arc<ModRegistry> {
    let Some(program) = resolve_text_mod_path() else {
        eprintln!("[reimagine] text mod not found; text operations are unavailable");
        return Arc::new(ModRegistry::empty());
    };
    let spec = ProcessModSpec {
        startup_timeout: TEXT_MOD_STARTUP_TIMEOUT,
        ..ProcessModSpec::new(&program)
    };
    let process_mod = match ProcessMod::spawn(spec).await {
        Ok(process_mod) => process_mod,
        Err(diagnostic) => {
            eprintln!(
                "[reimagine] failed to spawn the text mod at `{}`: {}",
                program.display(),
                diagnostic.message()
            );
            return Arc::new(ModRegistry::empty());
        }
    };
    let registry = ModRegistry::start(vec![Arc::new(process_mod)], event_bus).await;
    if !registry
        .list()
        .first()
        .is_some_and(|entry| entry.health.is_available())
    {
        eprintln!(
            "[reimagine] the text mod at `{}` did not start; text operations are unavailable",
            program.display()
        );
    }
    Arc::new(registry)
}

/// Create the workspace event bus and the bridge that forwards its events to
/// subscribed Tauri channels.
fn event_pipeline() -> (EventBus, TauriEventBridge) {
    let event_bus = EventBus::new();
    let event_bridge = TauriEventBridge::new();

    let mut event_rx = event_bus.subscribe();
    let bridge_forwarder = event_bridge.clone();
    tokio::spawn(async move {
        loop {
            match event_rx.recv().await {
                Ok(event) => bridge_forwarder.broadcast(&event),
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    (event_bus, event_bridge)
}

#[derive(Debug)]
pub struct DesktopHostState {
    app_host: RwLock<AppHost>,
    event_bridge: TauriEventBridge,
    active_project: RwLock<ProjectId>,
    #[cfg(feature = "aigc")]
    worker_management: Arc<WorkerManagementService>,
}

impl Clone for DesktopHostState {
    fn clone(&self) -> Self {
        Self {
            app_host: RwLock::new(self.app_host.read().expect(APP_HOST_LOCK).clone()),
            event_bridge: self.event_bridge.clone(),
            active_project: RwLock::new(self.active_project.read().expect(APP_HOST_LOCK).clone()),
            #[cfg(feature = "aigc")]
            worker_management: self.worker_management.clone(),
        }
    }
}

impl DesktopHostState {
    /// Bootstrap without resolving or spawning a text mod.
    ///
    /// Test-only: the production path discovers `text-mod` beside the
    /// executable or through `REIMAGINE_TEXT_MOD`, which would couple tests to
    /// the developer's environment and could start a real child process.
    #[cfg(test)]
    pub async fn bootstrap(workspace_base_path: impl Into<PathBuf>) -> Result<Self, AppHostError> {
        let workspace_base_path = workspace_base_path.into();
        let app_data_root = workspace_base_path.with_extension("app-data");
        AppPaths::new(&workspace_base_path).ensure_all().await?;
        let (event_bus, event_bridge) = event_pipeline();
        Self::from_parts(
            app_data_root,
            workspace_base_path,
            event_bus,
            event_bridge,
            Arc::new(ModRegistry::empty()),
        )
        .await
    }

    pub async fn bootstrap_with_app_data_root(
        app_data_root: impl Into<PathBuf>,
        workspace_base_path: impl Into<PathBuf>,
    ) -> Result<Self, AppHostError> {
        let app_data_root = app_data_root.into();
        let workspace_base_path = workspace_base_path.into();
        AppPaths::new(&workspace_base_path).ensure_all().await?;
        let (event_bus, event_bridge) = event_pipeline();
        let mod_registry = start_text_mod(&event_bus).await;
        Self::from_parts(
            app_data_root,
            workspace_base_path,
            event_bus,
            event_bridge,
            mod_registry,
        )
        .await
    }

    /// Build the host from an ensured workspace, a running event pipeline and
    /// a mod registry.
    async fn from_parts(
        app_data_root: PathBuf,
        workspace_base_path: PathBuf,
        event_bus: EventBus,
        event_bridge: TauriEventBridge,
        mod_registry: Arc<ModRegistry>,
    ) -> Result<Self, AppHostError> {
        let workspace = WorkspaceHost::try_with_app_data_root_and_event_bus_and_mod_registry(
            WorkspaceScope::new(WORKSPACE_SCOPE),
            &workspace_base_path,
            &app_data_root,
            event_bus,
            mod_registry,
        )
        .await?;

        #[cfg(feature = "aigc")]
        let worker_management = Arc::new(WorkerManagementService::offline(&app_data_root)?);

        Ok(Self {
            app_host: RwLock::new(AppHost::new(workspace)),
            event_bridge,
            active_project: RwLock::new(ProjectId::new("default")),
            #[cfg(feature = "aigc")]
            worker_management,
        })
    }

    #[cfg(test)]
    pub fn workspace_base_path(&self) -> PathBuf {
        self.app_host
            .read()
            .expect(APP_HOST_LOCK)
            .workspace()
            .base_path()
            .to_path_buf()
    }

    #[cfg(feature = "aigc")]
    #[allow(dead_code)]
    pub fn worker_management(&self) -> &WorkerManagementService {
        &self.worker_management
    }

    pub fn active_project_id(&self) -> ProjectId {
        self.active_project.read().expect(APP_HOST_LOCK).clone()
    }

    pub fn subscribe_events(&self, channel: Channel<Event>) {
        self.event_bridge.subscribe(channel);
    }

    pub async fn list_projects(&self) -> Result<Vec<ProjectDto>, AppHostError> {
        let workspace = self.workspace_arc();
        Ok(workspace
            .project_service()
            .list_projects()
            .await?
            .into_iter()
            .map(ProjectDto::from)
            .collect())
    }

    pub async fn create_project(
        &self,
        project_id: String,
        metadata: ProjectMetadataInputDto,
    ) -> Result<ProjectDto, AppHostError> {
        let project_id = parse_project_id(&project_id)?;
        let now = chrono::Utc::now().to_rfc3339();
        let workspace = self.workspace_arc();
        let project = workspace
            .project_service()
            .create_project(project_id, metadata.into_domain(&now))
            .await?;
        Ok(project.into())
    }

    pub async fn load_project(&self, project_id: &str) -> Result<ProjectDto, AppHostError> {
        let project_id = parse_project_id(project_id)?;
        let workspace = self.workspace_arc();
        Ok(workspace
            .project_service()
            .load_project(&project_id)
            .await?
            .into())
    }

    pub async fn update_project(
        &self,
        project_id: &str,
        metadata: ProjectMetadataInputDto,
    ) -> Result<ProjectDto, AppHostError> {
        let project_id = parse_project_id(project_id)?;
        let now = chrono::Utc::now().to_rfc3339();
        let workspace = self.workspace_arc();
        let current = workspace
            .project_service()
            .load_project(&project_id)
            .await?;
        Ok(workspace
            .project_service()
            .update_project(
                &project_id,
                metadata.into_updated_domain(current.metadata().created_at().clone(), &now),
            )
            .await?
            .into())
    }

    pub async fn delete_project(&self, project_id: &str) -> Result<(), AppHostError> {
        let project_id = parse_project_id(project_id)?;
        let workspace = self.workspace_arc();
        workspace
            .project_service()
            .load_project(&project_id)
            .await?;
        workspace.delete_project(&project_id).await?;
        if self.active_project_id() == project_id {
            let fallback = workspace
                .project_service()
                .list_projects()
                .await?
                .into_iter()
                .next()
                .map(|project| project.id().clone())
                .unwrap_or_else(|| ProjectId::new("default"));
            *self.active_project.write().expect(APP_HOST_LOCK) = fallback;
        }
        Ok(())
    }

    pub async fn set_active_project(&self, project_id: &str) -> Result<ProjectDto, AppHostError> {
        let project_id = parse_project_id(project_id)?;
        let workspace = self.workspace_arc();
        let project = workspace
            .project_service()
            .load_project(&project_id)
            .await?;
        *self.active_project.write().expect(APP_HOST_LOCK) = project_id;
        Ok(project.into())
    }

    pub async fn board_snapshot(&self, project_id: &str) -> Result<BoardSnapshotDto, AppHostError> {
        let project_id = self.require_project(project_id).await?;
        let workspace = self.workspace_arc();
        Ok(workspace
            .board_service()
            .snapshot(&project_id)
            .await?
            .into())
    }

    pub async fn preview_board_commands(
        &self,
        project_id: &str,
        batch: serde_json::Value,
    ) -> Result<BoardCommandResultDto, AppHostError> {
        let project_id = self.require_project(project_id).await?;
        let batch: CommandBatch<BoardCommand> =
            serde_json::from_value(batch).map_err(|error| AppHostError::WorkflowJson {
                path: PathBuf::new(),
                message: format!("invalid board command batch: {error}"),
            })?;
        let workspace = self.workspace_arc();
        Ok(workspace
            .board_service()
            .preview_batch(&project_id, batch)
            .await?
            .into())
    }

    pub async fn apply_board_commands(
        &self,
        project_id: &str,
        batch: serde_json::Value,
    ) -> Result<BoardCommandResultDto, AppHostError> {
        let project_id = self.require_project(project_id).await?;
        let batch: CommandBatch<BoardCommand> =
            serde_json::from_value(batch).map_err(|error| AppHostError::WorkflowJson {
                path: PathBuf::new(),
                message: format!("invalid board command batch: {error}"),
            })?;
        let workspace = self.workspace_arc();
        Ok(workspace
            .board_service()
            .apply_batch(&project_id, batch)
            .await?
            .into())
    }

    pub async fn undo_board(
        &self,
        project_id: &str,
        expected_entry_id: Option<&str>,
    ) -> Result<Option<BoardCommandResultDto>, AppHostError> {
        let project_id = self.require_project(project_id).await?;
        let expected = expected_entry_id.map(HistoryEntryId::from);
        let workspace = self.workspace_arc();
        Ok(workspace
            .board_service()
            .undo(
                &project_id,
                CommandActor::new(CommandActorKind::Human),
                expected.as_ref(),
            )
            .await?
            .map(Into::into))
    }

    pub async fn redo_board(
        &self,
        project_id: &str,
        expected_entry_id: Option<&str>,
    ) -> Result<Option<BoardCommandResultDto>, AppHostError> {
        let project_id = self.require_project(project_id).await?;
        let expected = expected_entry_id.map(HistoryEntryId::from);
        let workspace = self.workspace_arc();
        Ok(workspace
            .board_service()
            .redo(
                &project_id,
                CommandActor::new(CommandActorKind::Human),
                expected.as_ref(),
            )
            .await?
            .map(Into::into))
    }

    async fn require_project(&self, project_id: &str) -> Result<ProjectId, AppHostError> {
        let project_id = parse_project_id(project_id)?;
        let workspace = self.workspace_arc();
        workspace
            .project_service()
            .load_project(&project_id)
            .await?;
        Ok(project_id)
    }

    pub async fn op_list(&self, project_id: &str) -> Result<Vec<OperationInfo>, AppHostError> {
        let project_id = self.require_project(project_id).await?;
        let workspace = self.workspace_arc();
        Ok(workspace.operation_service().list(&project_id).await)
    }

    pub async fn op_invoke(
        &self,
        project_id: &str,
        name: String,
        input: serde_json::Value,
    ) -> Result<InvokeResult, AppHostError> {
        let project_id = self.require_project(project_id).await?;
        let workspace = self.workspace_arc();
        let context = OperationContext::new(
            CommandActor::new(CommandActorKind::Human),
            CommandProvenance::Direct,
        );
        let cancel = CancelToken::new();
        let request = InvokeRequest {
            name,
            input,
            context,
            cancel,
            run: None,
        };
        workspace
            .operation_service()
            .invoke(&project_id, request)
            .await
            .map_err(AppHostError::from)
    }

    pub async fn op_preview(
        &self,
        project_id: &str,
        name: String,
        input: serde_json::Value,
        preview_key: String,
    ) -> Result<serde_json::Value, AppHostError> {
        let project_id = self.require_project(project_id).await?;
        let workspace = self.workspace_arc();
        let cancel = CancelToken::new();
        workspace
            .operation_service()
            .preview(&project_id, &name, input, &preview_key, cancel)
            .await
            .map_err(AppHostError::from)
    }

    /// Cancel the in-flight preview registered under `preview_key`.
    ///
    /// The key is global to the operation service, so this needs no project and
    /// is a no-op when no preview is running under it.
    pub async fn op_preview_cancel(&self, preview_key: &str) -> Result<(), AppHostError> {
        let workspace = self.workspace_arc();
        workspace
            .operation_service()
            .cancel_preview(preview_key)
            .await
            .map_err(AppHostError::from)
    }

    pub(crate) fn require_project_sync(&self, project_id: &str) -> Result<ProjectId, AppHostError> {
        let project_id = parse_project_id(project_id)?;
        let workspace = self.workspace_arc();
        if !workspace.project_service().contains_project(&project_id) {
            return Err(AppHostError::UnknownProject { project_id });
        }
        Ok(project_id)
    }

    pub(crate) fn workspace_arc(&self) -> Arc<WorkspaceHost> {
        Arc::clone(self.app_host.read().expect(APP_HOST_LOCK).workspace())
    }

    pub fn health(&self) -> HealthResponse {
        HealthResponse::from_workspace(self.app_host.read().expect(APP_HOST_LOCK).workspace())
    }

    #[cfg(feature = "aigc")]
    pub fn compute_profile(&self) -> ComputeProfileDto {
        self.app_host
            .read()
            .expect(APP_HOST_LOCK)
            .workspace()
            .compute_profile_dto()
    }

    /// Returns the workspace node catalog as a host‑neutral DTO.
    pub fn list_node_defs(&self) -> NodeCatalogResponse {
        use reimagine_app_host::dto::NodeDefDto;
        let defs = self
            .app_host
            .read()
            .expect(APP_HOST_LOCK)
            .workspace()
            .list_node_defs();
        NodeCatalogResponse {
            nodes: defs.into_iter().map(NodeDefDto::from).collect(),
        }
    }

    /// Returns the model list as a host‑neutral DTO.
    #[cfg(feature = "aigc")]
    pub async fn list_models(&self) -> Result<Vec<ModelInfoDto>, AppHostError> {
        use reimagine_app_host::dto::ModelInfoDto;
        let workspace = {
            let app_host = self.app_host.read().expect(APP_HOST_LOCK);
            Arc::clone(app_host.workspace())
        };
        let descriptors = workspace.model_service().list_models().await?;
        Ok(descriptors.into_iter().map(ModelInfoDto::from).collect())
    }

    #[allow(dead_code)]
    pub async fn run_workflow(
        &self,
        workflow_value: serde_json::Value,
    ) -> Result<RunWorkflowResponse, AppHostError> {
        self.run_workflow_for_project("default", workflow_value)
            .await
    }

    pub async fn run_workflow_for_project(
        &self,
        project_id: &str,
        workflow_value: serde_json::Value,
    ) -> Result<RunWorkflowResponse, AppHostError> {
        use reimagine_app_host::RunWorkflowRequest;

        let project_id = self.require_project(project_id).await?;

        // 1. Deserialize and register the workflow
        let workflow: Workflow =
            serde_json::from_value(workflow_value).map_err(|e| AppHostError::WorkflowJson {
                path: std::path::PathBuf::new(),
                message: e.to_string(),
            })?;
        let workflow_id = {
            let app_host = self.app_host.read().expect(APP_HOST_LOCK);
            app_host
                .workspace()
                .workflow_service_for_project(&project_id)
                .register_workflow(workflow)
        };

        // 2. Build run request
        let request = RunWorkflowRequest::for_project(
            project_id,
            workflow_id.clone(),
            reimagine_core::readiness::RunTargetSelection::AllDefaultTargets,
        );

        // 3. Run the workflow
        let workspace = {
            let app_host = self.app_host.read().expect(APP_HOST_LOCK);
            Arc::clone(app_host.workspace())
        };
        let result = workspace.run_workflow(request).await?;

        match result {
            reimagine_app_host::RunWorkflowResult::Started {
                handle,
                initial_snapshot,
                diagnostics,
            } => {
                let run_id = handle.run_id().clone();

                Ok(RunWorkflowResponse::Started {
                    run_id,
                    workflow_id: handle.workflow_id().clone(),
                    workflow_version: handle.workflow_version(),
                    initial_snapshot: Box::new((*initial_snapshot).into()),
                    diagnostics: diagnostics.into_iter().map(Into::into).collect(),
                })
            }
            reimagine_app_host::RunWorkflowResult::Blocked { diagnostics } => {
                Ok(RunWorkflowResponse::Blocked {
                    workflow_id,
                    diagnostics: diagnostics.into_iter().map(Into::into).collect(),
                })
            }
        }
    }

    pub fn cancel_run(&self, run_id_str: &str) -> Result<(), AppHostError> {
        use reimagine_core::model::RunId;
        let run_id = RunId::new(run_id_str);
        self.app_host
            .read()
            .expect(APP_HOST_LOCK)
            .workspace()
            .runtime_service()
            .cancel(&run_id)
            .map_err(|_e| AppHostError::UnknownRun {
                run_id: run_id.clone(),
            })
    }

    /// Resolve an artifact id to metadata (path-safe, validated).
    pub fn resolve_artifact(&self, artifact_id: &str) -> Result<ArtifactMetadataDto, AppHostError> {
        use reimagine_core::model::ArtifactId;
        let id = ArtifactId::new(artifact_id);
        let access = self
            .app_host
            .read()
            .expect(APP_HOST_LOCK)
            .workspace()
            .resolve_artifact(&id)?;
        Ok(ArtifactMetadataDto::from(access))
    }

    /// Open an artifact file with the desktop system handler.
    pub fn open_artifact(&self, artifact_id: &str) -> Result<(), AppHostError> {
        use reimagine_core::model::ArtifactId;
        let id = ArtifactId::new(artifact_id);
        let access = self
            .app_host
            .read()
            .expect(APP_HOST_LOCK)
            .workspace()
            .resolve_artifact(&id)?;
        opener::open(access.path.as_path()).map_err(|e| AppHostError::Io {
            path: access.path,
            message: format!("failed to open artifact: {e}"),
        })
    }

    pub fn preview_workflow_commands(
        &self,
        project_id: String,
        workflow_id: String,
        command_batch: serde_json::Value,
    ) -> Result<CommandResult<WorkflowChange>, AppHostError> {
        let batch: CommandBatch<WorkflowCommand> =
            serde_json::from_value(command_batch).map_err(|e| AppHostError::WorkflowJson {
                path: std::path::PathBuf::new(),
                message: format!("invalid command batch: {e}"),
            })?;

        let project_id = self.require_project_sync(&project_id)?;
        let (node_catalog, workflow_service) = {
            let app_host = self.app_host.read().expect(APP_HOST_LOCK);
            let workspace = app_host.workspace();
            (
                Arc::clone(workspace.node_catalog()),
                workspace.workflow_service_for_project(&project_id),
            )
        };

        workflow_service.preview_batch(
            &reimagine_core::model::WorkflowId::new(workflow_id),
            node_catalog.as_ref(),
            batch,
        )
    }

    /// Apply a command batch directly.
    ///
    /// Returns the applied `CommandResult` with changes and diagnostics.
    pub async fn apply_workflow_commands(
        &self,
        project_id: String,
        workflow_id: String,
        command_batch: serde_json::Value,
        _approved_by: Option<serde_json::Value>,
    ) -> Result<CommandResult<WorkflowChange>, AppHostError> {
        let batch: CommandBatch<WorkflowCommand> =
            serde_json::from_value(command_batch).map_err(|e| AppHostError::WorkflowJson {
                path: std::path::PathBuf::new(),
                message: format!("invalid command batch: {e}"),
            })?;

        let project_id = self.require_project(&project_id).await?;
        let (node_catalog, workflow_service) = {
            let app_host = self.app_host.read().expect(APP_HOST_LOCK);
            let workspace = app_host.workspace();
            (
                Arc::clone(workspace.node_catalog()),
                workspace.workflow_service_for_project(&project_id),
            )
        };

        workflow_service
            .apply_commands(
                &reimagine_core::model::WorkflowId::new(workflow_id),
                node_catalog.as_ref(),
                batch,
            )
            .await
    }

    /// Approve a pending workflow proposal.
    ///
    /// Calls `WorkflowService::apply_pending_proposal()` and returns the
    /// resulting `CommandResult`. Returns an error if no pending proposal
    /// exists for the workflow.
    pub async fn approve_proposal(
        &self,
        project_id: String,
        workflow_id: String,
    ) -> Result<CommandResult<WorkflowChange>, AppHostError> {
        let project_id = self.require_project(&project_id).await?;
        let (node_catalog, workflow_service) = {
            let app_host = self.app_host.read().expect(APP_HOST_LOCK);
            let workspace = app_host.workspace();
            (
                Arc::clone(workspace.node_catalog()),
                workspace.workflow_service_for_project(&project_id),
            )
        };

        workflow_service
            .apply_pending_proposal_persisted(
                &reimagine_core::model::WorkflowId::new(workflow_id),
                node_catalog.as_ref(),
                None, // approved_by — Tauri thin shell, no human actor identity
            )
            .await
    }

    /// List pending proposals from all workflows.
    #[allow(dead_code)]
    pub fn list_pending_proposals(&self) -> Result<Vec<serde_json::Value>, AppHostError> {
        let workflow_service = {
            let app_host = self.app_host.read().expect(APP_HOST_LOCK);
            Arc::clone(app_host.workspace().workflow_service())
        };
        let proposals = workflow_service.list_proposals();
        proposals
            .into_iter()
            .map(|p| {
                serde_json::to_value(p).map_err(|e| AppHostError::WorkflowJson {
                    path: std::path::PathBuf::new(),
                    message: format!("failed to serialize proposal: {e}"),
                })
            })
            .collect()
    }

    #[allow(dead_code)]
    pub async fn save_workflow(
        &self,
        workflow_id: &str,
        workflow_value: serde_json::Value,
    ) -> Result<PathBuf, AppHostError> {
        self.save_workflow_for_project("default", workflow_id, workflow_value)
            .await
    }

    pub async fn save_workflow_for_project(
        &self,
        project_id: &str,
        workflow_id: &str,
        workflow_value: serde_json::Value,
    ) -> Result<PathBuf, AppHostError> {
        use reimagine_core::workflow::Workflow;

        let project_id = self.require_project(project_id).await?;
        workflow_id_from_str(workflow_id)?;
        let workflow: Workflow =
            serde_json::from_value(workflow_value).map_err(|e| AppHostError::WorkflowJson {
                path: std::path::PathBuf::new(),
                message: format!("invalid workflow json: {e}"),
            })?;

        let workflow_service = {
            let app_host = self.app_host.read().expect(APP_HOST_LOCK);
            let workspace = app_host.workspace();
            workspace.workflow_service_for_project(&project_id)
        };
        workflow_service.save_external_workflow(workflow).await
    }

    /// Load a workflow (JSON) from the workspace `workflows/` directory.
    ///
    /// Registers the loaded workflow as a session and returns its JSON.
    #[allow(dead_code)]
    pub async fn load_workflow_json(
        &self,
        workflow_id: &str,
    ) -> Result<serde_json::Value, AppHostError> {
        self.load_workflow_json_for_project("default", workflow_id)
            .await
    }

    pub async fn load_workflow_json_for_project(
        &self,
        project_id: &str,
        workflow_id: &str,
    ) -> Result<serde_json::Value, AppHostError> {
        let project_id = self.require_project(project_id).await?;
        let id = workflow_id_from_str(workflow_id)?;
        let workflow_service = {
            let app_host = self.app_host.read().expect(APP_HOST_LOCK);
            let workspace = app_host.workspace();
            workspace.workflow_service_for_project(&project_id)
        };
        workflow_service.load_workflow(&id).await?;
        let workflow = workflow_service.snapshot(&id)?;
        serde_json::to_value(workflow).map_err(|e| AppHostError::WorkflowJson {
            path: std::path::PathBuf::new(),
            message: format!("failed to serialize workflow: {e}"),
        })
    }

    /// List workflow files persisted on disk (newest first).
    ///
    /// Returns JSON summaries `{ id, modified_millis }` so the adapter layer
    /// never has to name the (crate-private) summary type.
    #[allow(dead_code)]
    pub fn list_saved_workflows(&self) -> Result<Vec<serde_json::Value>, AppHostError> {
        self.list_saved_workflows_for_project("default")
    }

    pub fn list_saved_workflows_for_project(
        &self,
        project_id: &str,
    ) -> Result<Vec<serde_json::Value>, AppHostError> {
        let project_id = self.require_project_sync(project_id)?;
        let workflow_service = {
            let app_host = self.app_host.read().expect(APP_HOST_LOCK);
            let workspace = app_host.workspace();
            workspace.workflow_service_for_project(&project_id)
        };
        let infos = workflow_service.list_saved_workflows()?;
        infos
            .into_iter()
            .map(|info| {
                serde_json::to_value(info).map_err(|e| AppHostError::WorkflowJson {
                    path: std::path::PathBuf::new(),
                    message: format!("failed to serialize workflow summary: {e}"),
                })
            })
            .collect()
    }

    #[cfg(feature = "aigc")]
    /// Search HuggingFace models via the catalog.
    pub async fn search_models(
        &self,
        query: &reimagine_model_acquisition::ModelSearchQuery,
    ) -> Result<Vec<reimagine_app_host::dto::ModelCatalogEntryDto>, AppHostError> {
        let workspace = {
            let app_host = self.app_host.read().expect(APP_HOST_LOCK);
            Arc::clone(app_host.workspace())
        };
        let entries = workspace.model_service().search(query).await?;
        Ok(entries
            .into_iter()
            .map(reimagine_app_host::dto::ModelCatalogEntryDto::from)
            .collect())
    }

    #[cfg(feature = "aigc")]
    /// Fetch the full model card for a repository.
    pub async fn get_model_card(
        &self,
        repo_id: &str,
    ) -> Result<reimagine_app_host::dto::ModelCardDto, AppHostError> {
        let workspace = {
            let app_host = self.app_host.read().expect(APP_HOST_LOCK);
            Arc::clone(app_host.workspace())
        };
        let card = workspace.model_service().model_card(repo_id).await?;
        Ok(reimagine_app_host::dto::ModelCardDto::from(card))
    }

    /// Download a HuggingFace model with progress streaming.
    ///
    /// Returns an `AcquisitionReportDto` through the Tauri IPC channel.
    /// When `from_catalog` is true, uses the smart download strategy from BE-42
    /// which fetches repository metadata, detects the format, and builds
    /// optimal download patterns automatically.
    #[cfg(feature = "aigc")]
    #[allow(clippy::too_many_arguments)]
    pub async fn download_huggingface_model(
        &self,
        download_id: String,
        repo_id: String,
        revision: Option<String>,
        allow_patterns: Option<Vec<String>>,
        target_relative_dir: String,
        overwrite: Option<String>,
        auto_detect: Option<bool>,
        from_catalog: Option<bool>,
    ) -> Result<reimagine_app_host::dto::ModelDownloadOutput, AppHostError> {
        use reimagine_model_acquisition::{
            AcquireProvider, AllowPatterns, ModelAcquisitionRequest, OverwritePolicy, RepoId,
            Revision, TargetRelativeDir,
        };

        let repo_id = RepoId::new(&repo_id).ok_or_else(|| AppHostError::Io {
            path: std::path::PathBuf::new(),
            message: format!("invalid repo_id: expected `namespace/name` format, got `{repo_id}`"),
        })?;

        let revision = revision.map(Revision::new).unwrap_or_default();
        let allow_patterns = allow_patterns.map(AllowPatterns::new).unwrap_or_default();

        let target_relative_dir =
            TargetRelativeDir::new(target_relative_dir.into()).map_err(|e| AppHostError::Io {
                path: std::path::PathBuf::new(),
                message: format!("invalid target_relative_dir: {e}"),
            })?;

        let overwrite_policy = match overwrite.as_deref() {
            Some("overwrite") => OverwritePolicy::Overwrite,
            Some("fail") => OverwritePolicy::Fail,
            _ => OverwritePolicy::Skip,
        };

        let use_smart_download = from_catalog.unwrap_or(false);
        let effective_auto_detect = if use_smart_download {
            true
        } else {
            auto_detect.unwrap_or(true)
        };

        let request = ModelAcquisitionRequest {
            provider: AcquireProvider::HuggingFace,
            repo_id,
            revision,
            allow_patterns,
            target_relative_dir,
            overwrite_policy,
            auto_detect: effective_auto_detect,
        };

        let acq = {
            let app_host = self.app_host.read().expect(APP_HOST_LOCK);
            Arc::clone(app_host.workspace().services().model_acquisition_service())
        };

        // The catalog card, when available, gives the expected size before
        // the download reports one.
        let estimated_bytes = if use_smart_download {
            let workspace = {
                let app_host = self.app_host.read().expect(APP_HOST_LOCK);
                Arc::clone(app_host.workspace())
            };
            workspace
                .model_service()
                .model_card(request.repo_id.as_str())
                .await
                .ok()
                .map(|card| card.estimated_download_size)
        } else {
            None
        };

        let report = acq
            .acquire_with_progress_events(download_id, request, estimated_bytes)
            .await?;

        Ok(reimagine_app_host::dto::ModelDownloadOutput::from(report))
    }

    /// Shut down the active inference worker and the registered mods, if any.
    ///
    /// This is the application-level shutdown hook. Call it when the host
    /// is exiting so that child worker and mod processes are cleaned up
    /// instead of becoming orphans.
    pub async fn shutdown(&self) {
        let app_host = self.app_host.read().expect(APP_HOST_LOCK).clone();
        app_host.shutdown().await;
    }

    // ─── Re-bootstrap (BE-38 / B4-8) ──────────────────────────────

    #[cfg(feature = "aigc")]
    /// Drain in-flight runs and re-bootstrap the workspace with a new
    /// backend selection.
    ///
    /// Returns the compute profile of the rebuilt workspace. The selection
    /// is not persisted, so a restarted app boots with the configured
    /// backend again.
    pub async fn rebootstrap_backend(
        &self,
        selection: BackendSelection,
    ) -> Result<ComputeProfileDto, AppHostError> {
        let current = {
            let app_host = self.app_host.read().expect(APP_HOST_LOCK);
            Arc::clone(app_host.workspace())
        };
        let rebuilt = WorkspaceHost::rebuild_workspace(&current, selection).await?;
        let profile = rebuilt.compute_profile_dto();
        *self.app_host.write().expect(APP_HOST_LOCK) = AppHost::new(rebuilt);
        Ok(profile)
    }

    // ─── Worker switching (BE-32) ────────────────────────────────

    #[cfg(feature = "aigc")]
    /// Drain in-flight runs (waiting up to `deadline`) and switch the active
    /// worker to the installed worker for `target_instance`.
    pub async fn drain_and_switch_worker(
        &self,
        target_instance: &str,
        deadline: std::time::Duration,
    ) -> Result<WorkerSwitchResultDto, AppHostError> {
        let candidate = self.resolve_switch_candidate(target_instance)?;
        let workspace = {
            let app_host = self.app_host.read().expect(APP_HOST_LOCK);
            Arc::clone(app_host.workspace())
        };
        let handle = workspace
            .drain_and_switch_worker(candidate, deadline)
            .await?;
        Ok(WorkerSwitchResultDto::from(handle))
    }

    #[cfg(feature = "aigc")]
    /// Cancel in-flight runs and switch the active worker to the installed
    /// worker for `target_instance`.
    pub async fn cancel_and_switch_worker(
        &self,
        target_instance: &str,
        deadline: std::time::Duration,
    ) -> Result<WorkerSwitchResultDto, AppHostError> {
        let candidate = self.resolve_switch_candidate(target_instance)?;
        let workspace = {
            let app_host = self.app_host.read().expect(APP_HOST_LOCK);
            Arc::clone(app_host.workspace())
        };
        let handle = workspace
            .cancel_and_switch_worker(candidate, deadline)
            .await?;
        Ok(WorkerSwitchResultDto::from(handle))
    }

    #[cfg(feature = "aigc")]
    /// List installed workers that can be used as switch targets.
    pub fn list_worker_switch_targets(&self) -> Result<Vec<WorkerInstallationDto>, AppHostError> {
        Ok(self.worker_management.list_installed()?)
    }

    #[cfg(feature = "aigc")]
    fn resolve_switch_candidate(
        &self,
        target_instance: &str,
    ) -> Result<WorkerBackendCandidate, WorkerSwitchError> {
        let snapshot = self.worker_management.inventory().list().map_err(|error| {
            WorkerSwitchError::Startup {
                message: format!("worker inventory unavailable: {error}"),
            }
        })?;
        let record = snapshot
            .records
            .iter()
            .find(|record| record.identity.backend_instance_id.0 == target_instance)
            .ok_or_else(|| WorkerSwitchError::Startup {
                message: format!("no installed worker for backend instance `{target_instance}`"),
            })?;
        let executable = installed_worker_executable(&record.install_path).ok_or_else(|| {
            WorkerSwitchError::Startup {
                message: format!(
                    "no executable found inside installed worker directory `{}`",
                    record.install_path
                ),
            }
        })?;
        let manifest_profile = record.manifest_profile.clone().ok_or_else(|| {
            WorkerSwitchError::Startup {
                message: format!(
                    "installed worker `{target_instance}` has no manifest profile; reinstall to populate it"
                ),
            }
        })?;
        let launch = WorkerLaunchSpec {
            executable,
            expected: record.identity.clone(),
            supported_protocols: ProtocolRange::new(2, 2),
            limits: WorkerLimits::default(),
            environment: Vec::new(),
        };
        WorkerBackendCandidate::try_new(launch, manifest_profile).map_err(|error| {
            WorkerSwitchError::Startup {
                message: format!("installed worker `{target_instance}` is not launchable: {error}"),
            }
        })
    }
}

/// Result of a worker switch, adapted from [`WorkerSelectionHandle`] for IPC.
#[cfg(feature = "aigc")]
#[derive(Debug, Clone, Serialize)]
pub struct WorkerSwitchResultDto {
    pub instance: String,
    pub incarnation_id: String,
}

#[cfg(feature = "aigc")]
impl From<WorkerSelectionHandle> for WorkerSwitchResultDto {
    fn from(handle: WorkerSelectionHandle) -> Self {
        Self {
            instance: handle.instance().to_string(),
            incarnation_id: handle.incarnation_id().0.clone(),
        }
    }
}

/// Find the worker executable inside an installed worker directory.
///
/// Installed packages contain exactly one executable (validated at install
/// time), so the first executable file found is the worker binary.
#[cfg(feature = "aigc")]
fn installed_worker_executable(install_path: &str) -> Option<std::path::PathBuf> {
    fn visit(dir: &std::path::Path, found: &mut Option<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                visit(&path, found);
            } else if is_executable(&path) && found.is_none() {
                *found = Some(path);
            }
        }
    }
    let mut found = None;
    visit(std::path::Path::new(install_path), &mut found);
    found
}

#[cfg(feature = "aigc")]
fn is_executable(path: &std::path::Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path)
            .map(|metadata| metadata.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        path.is_file()
    }
}

pub fn default_workspace_path(app_data_dir: impl AsRef<Path>) -> PathBuf {
    app_data_dir.as_ref().join("workspace")
}

/// Validate a user-supplied workflow id before it reaches `WorkflowId::new`
/// (which asserts — and would panic — on invalid input).
fn parse_project_id(id: &str) -> Result<ProjectId, AppHostError> {
    let safe = !id.is_empty()
        && id.is_ascii()
        && !id.contains('/')
        && !id.contains('\\')
        && id != "."
        && id != "..";
    if !safe {
        return Err(AppHostError::ProjectIdPathUnsafe {
            project_id: id.to_owned(),
        });
    }
    Ok(ProjectId::new(id))
}

fn workflow_id_from_str(id: &str) -> Result<reimagine_core::model::WorkflowId, AppHostError> {
    let safe = !id.is_empty()
        && id.is_ascii()
        && !id.contains('/')
        && !id.contains('\\')
        && id != "."
        && id != "..";
    if !safe {
        return Err(AppHostError::WorkflowIdPathUnsafe {
            workflow_id: reimagine_core::model::WorkflowId::new("invalid-workflow-id"),
        });
    }
    Ok(reimagine_core::model::WorkflowId::new(id))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    fn write_file(path: &Path) {
        fs::write(path, b"#!/bin/sh\n").expect("write fixture");
    }

    #[test]
    fn an_existing_env_path_wins_over_the_sibling() {
        let dir = tempfile::tempdir().expect("temp dir");
        let explicit = dir.path().join("custom-mod");
        write_file(&explicit);
        let sibling = dir
            .path()
            .join(format!("{TEXT_MOD_NAME}{}", std::env::consts::EXE_SUFFIX));
        write_file(&sibling);

        assert_eq!(
            text_mod_candidate(Some(explicit.as_os_str()), Some(dir.path())),
            Some(explicit)
        );
    }

    #[test]
    fn a_missing_env_path_falls_back_to_the_sibling() {
        let dir = tempfile::tempdir().expect("temp dir");
        let sibling = dir
            .path()
            .join(format!("{TEXT_MOD_NAME}{}", std::env::consts::EXE_SUFFIX));
        write_file(&sibling);

        assert_eq!(
            text_mod_candidate(
                Some(dir.path().join("missing").as_os_str()),
                Some(dir.path())
            ),
            Some(sibling)
        );
    }

    #[test]
    fn without_an_env_path_only_an_existing_sibling_is_used() {
        let dir = tempfile::tempdir().expect("temp dir");
        assert_eq!(text_mod_candidate(None, Some(dir.path())), None);

        let sibling = dir
            .path()
            .join(format!("{TEXT_MOD_NAME}{}", std::env::consts::EXE_SUFFIX));
        write_file(&sibling);
        assert_eq!(
            text_mod_candidate(None, Some(dir.path())),
            Some(sibling.clone())
        );
        assert_eq!(text_mod_candidate(None, None), None);
    }
}
