use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use reimagine_agent_harness::{AgentEventSink, WorkspaceScope};
use reimagine_app_host::TurnRunResult;
use reimagine_app_host::dto::{
    AgentEventPayload, AgentSessionInfo, ArtifactMetadataDto, BoardCommandResultDto,
    BoardSnapshotDto, ComputeProfileDto, HealthResponse, ModelInfoDto, NodeCatalogResponse,
    ProjectDto, ProjectMetadataInputDto, RunWorkflowResponse,
};
use reimagine_app_host::{
    AgentServiceTurnRequest, AppHost, AppHostError, BackendSelection, CanvasContext,
    ProjectContextAssembler, WorkerBackendCandidate, WorkerInstallationDto,
    WorkerManagementService, WorkerSelectionHandle, WorkerSwitchError, WorkspaceHost,
};
use reimagine_backend_worker_host::{WorkerLaunchSpec, WorkerLimits};
use reimagine_backend_worker_protocol::ProtocolRange;
use reimagine_config::AppPaths;
use reimagine_core::board::BoardCommandBatch;
use reimagine_core::command::CommandResult;
use reimagine_core::model::ProjectId;
use reimagine_core::workflow::Workflow;
use reimagine_runtime::BoxedRunEventSink;
use serde::Serialize;
use tauri::ipc::Channel;

use crate::agent_event_hub::TauriAgentEventHub;
use crate::document_event_hub::TauriDocumentEventHub;
use crate::download_event_hub::TauriDownloadEventHub;
use crate::event_hub::{RunEventPayload, TauriRunEventHub};

const WORKSPACE_SCOPE: &str = "reimagine-tauri-host";
const APP_HOST_LOCK: &str = "desktop app_host lock poisoned";

#[derive(Debug)]
pub struct DesktopHostState {
    app_host: RwLock<AppHost>,
    event_hub: Arc<TauriRunEventHub>,
    agent_event_hub: Arc<TauriAgentEventHub>,
    document_event_hub: Arc<TauriDocumentEventHub>,
    active_project: RwLock<ProjectId>,
    download_event_hub: Arc<TauriDownloadEventHub>,
    worker_management: Arc<WorkerManagementService>,
}

impl Clone for DesktopHostState {
    fn clone(&self) -> Self {
        Self {
            app_host: RwLock::new(self.app_host.read().expect(APP_HOST_LOCK).clone()),
            event_hub: self.event_hub.clone(),
            agent_event_hub: self.agent_event_hub.clone(),
            document_event_hub: self.document_event_hub.clone(),
            active_project: RwLock::new(self.active_project.read().expect(APP_HOST_LOCK).clone()),
            download_event_hub: self.download_event_hub.clone(),
            worker_management: self.worker_management.clone(),
        }
    }
}

impl DesktopHostState {
    #[cfg(test)]
    pub async fn bootstrap(workspace_base_path: impl Into<PathBuf>) -> Result<Self, AppHostError> {
        let workspace_base_path = workspace_base_path.into();
        let app_data_root = workspace_base_path.with_extension("app-data");
        Self::bootstrap_with_app_data_root(app_data_root, workspace_base_path).await
    }

    pub async fn bootstrap_with_app_data_root(
        app_data_root: impl Into<PathBuf>,
        workspace_base_path: impl Into<PathBuf>,
    ) -> Result<Self, AppHostError> {
        let app_data_root = app_data_root.into();
        let workspace_base_path = workspace_base_path.into();
        AppPaths::new(&workspace_base_path).ensure_all().await?;
        let event_hub = Arc::new(TauriRunEventHub::new());
        let event_sink: BoxedRunEventSink = event_hub.clone();
        // AR-03: agent events stream through the TauriChannel hub (the
        // embedded TurnEventSink), not the frozen daemon JSON-RPC path.
        let agent_event_hub = Arc::new(TauriAgentEventHub::new());
        let agent_event_sink: Arc<dyn AgentEventSink> = agent_event_hub.clone();
        let workspace = WorkspaceHost::try_with_app_data_root_and_event_sinks(
            WorkspaceScope::new(WORKSPACE_SCOPE),
            &workspace_base_path,
            &app_data_root,
            event_sink,
            agent_event_sink,
        )
        .await?;

        let document_event_hub = Arc::new(TauriDocumentEventHub::new());
        let mut document_events = workspace.document_events().subscribe();
        let event_forwarder = Arc::clone(&document_event_hub);
        tokio::spawn(async move {
            loop {
                match document_events.recv().await {
                    Ok(event) => event_forwarder.send(event),
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
        });

        let download_event_hub = Arc::new(TauriDownloadEventHub::new());
        let worker_management = Arc::new(WorkerManagementService::offline(&app_data_root)?);

        Ok(Self {
            app_host: RwLock::new(AppHost::new(workspace)),
            event_hub,
            agent_event_hub,
            document_event_hub,
            active_project: RwLock::new(ProjectId::new("default")),
            download_event_hub,
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

    #[allow(dead_code)]
    pub fn worker_management(&self) -> &WorkerManagementService {
        &self.worker_management
    }

    pub fn active_project_id(&self) -> ProjectId {
        self.active_project.read().expect(APP_HOST_LOCK).clone()
    }

    pub fn subscribe_document_events(
        &self,
        channel: Channel<reimagine_app_host::DocumentChangedEvent>,
    ) {
        self.document_event_hub.subscribe(channel);
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
        let batch: BoardCommandBatch =
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
        let batch: BoardCommandBatch =
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
    ) -> Result<Option<BoardCommandResultDto>, AppHostError> {
        let project_id = self.require_project(project_id).await?;
        let workspace = self.workspace_arc();
        Ok(workspace
            .board_service()
            .undo(&project_id)
            .await?
            .map(Into::into))
    }

    pub async fn redo_board(
        &self,
        project_id: &str,
    ) -> Result<Option<BoardCommandResultDto>, AppHostError> {
        let project_id = self.require_project(project_id).await?;
        let workspace = self.workspace_arc();
        Ok(workspace
            .board_service()
            .redo(&project_id)
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

    fn require_project_sync(&self, project_id: &str) -> Result<ProjectId, AppHostError> {
        let project_id = parse_project_id(project_id)?;
        let workspace = self.workspace_arc();
        if !workspace.project_service().contains_project(&project_id) {
            return Err(AppHostError::UnknownProject { project_id });
        }
        Ok(project_id)
    }

    fn workspace_arc(&self) -> Arc<WorkspaceHost> {
        Arc::clone(self.app_host.read().expect(APP_HOST_LOCK).workspace())
    }

    pub fn health(&self) -> HealthResponse {
        HealthResponse::ok(
            self.app_host
                .read()
                .expect(APP_HOST_LOCK)
                .workspace()
                .workspace_scope()
                .as_str(),
        )
    }

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
        channel: Channel<RunEventPayload>,
    ) -> Result<RunWorkflowResponse, AppHostError> {
        self.run_workflow_for_project("default", workflow_value, channel)
            .await
    }

    pub async fn run_workflow_for_project(
        &self,
        project_id: &str,
        workflow_value: serde_json::Value,
        channel: Channel<RunEventPayload>,
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
                report,
            } => {
                let run_id = handle.run_id().clone();

                // 4. Subscribe the channel for live events
                self.event_hub.subscribe(&run_id, channel);

                // 5. Replay any events already emitted before we subscribed.
                self.event_hub.replay(&run_id);

                Ok(RunWorkflowResponse::Started {
                    run_id,
                    workflow_id: handle.workflow_id().clone(),
                    workflow_version: handle.workflow_version(),
                    initial_snapshot: Box::new((*initial_snapshot).into()),
                    diagnostics: report
                        .diagnostics()
                        .iter()
                        .map(|d| d.clone().into())
                        .collect(),
                })
            }
            reimagine_app_host::RunWorkflowResult::Blocked { report } => {
                Ok(RunWorkflowResponse::Blocked {
                    workflow_id,
                    diagnostics: report
                        .diagnostics()
                        .iter()
                        .map(|d| d.clone().into())
                        .collect(),
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

    /// Create a new agent session through the daemon bridge.
    ///
    /// `mode` must be "Agent" or "Build"; `provider` must be registered in
    /// the catalog. Sessions live on the daemon; `AgentService` is kept for
    /// backward compatibility but is not used by the desktop host anymore.
    pub async fn create_agent_session(
        &self,
        mode: String,
        provider: String,
    ) -> Result<AgentSessionInfo, AppHostError> {
        // V1 wire mode names are lowercase; the UI contract uses the
        // `AgentMode` debug names.
        let wire_mode = match mode.as_str() {
            "Agent" => "agent",
            "Build" => "build",
            other => {
                return Err(AppHostError::UnknownAgentMode {
                    mode: format!("unknown agent mode `{other}`; expected `Agent` or `Build`"),
                });
            }
        };

        // Validate provider exists in the catalog
        let provider_known = {
            let app_host = self.app_host.read().expect(APP_HOST_LOCK);
            app_host
                .workspace()
                .agent_service()
                .providers()
                .contains(&reimagine_agent_harness::ProviderName::new(&provider))
        };
        if !provider_known {
            return Err(AppHostError::UnknownAgentSession {
                session_id: reimagine_agent_harness::AgentSessionId::new(
                    "unknown-provider".to_string(),
                ),
            });
        }

        // Create session directly using AgentService
        let session_id = reimagine_agent_harness::AgentSessionId::new(format!(
            "session-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        ));
        let mode = match wire_mode {
            "agent" => reimagine_agent_harness::AgentMode::Agent,
            "build" => reimagine_agent_harness::AgentMode::Build,
            _ => unreachable!(),
        };
        let provider_name = reimagine_agent_harness::ProviderName::new(&provider);
        let started_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .to_string();

        let session = {
            let app_host = self.app_host.read().expect(APP_HOST_LOCK);
            app_host.workspace().agent_service().create_session(
                session_id.clone(),
                mode,
                provider_name,
                started_at,
            )
        };

        Ok(AgentSessionInfo {
            session_id: session.id().to_string(),
            mode: match session.mode() {
                reimagine_agent_harness::AgentMode::Agent => "Agent".to_owned(),
                reimagine_agent_harness::AgentMode::Build => "Build".to_owned(),
            },
            provider: session.provider().to_string(),
            started_at: session.started_at().to_string(),
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub async fn agent_turn(
        &self,
        session_id: String,
        turn_id: String,
        model: String,
        input: serde_json::Value,
        output_schema: Option<serde_json::Value>,
        timeout_ms: Option<u64>,
        context: Option<serde_json::Value>,
        channel: Channel<AgentEventPayload>,
    ) -> Result<TurnRunResult, AppHostError> {
        let input_text = turn_input_text(&input)?;

        // Parse session_id and turn_id
        let session_id = reimagine_agent_harness::AgentSessionId::new(session_id);
        let turn_id = reimagine_agent_harness::AgentTurnId::new(turn_id);

        // Get the session from AgentService
        let session = {
            let app_host = self.app_host.read().expect(APP_HOST_LOCK);
            app_host
                .workspace()
                .agent_service()
                .get_session(&session_id)?
        };

        let input_text = match context {
            Some(value) => match serde_json::from_value::<CanvasContext>(value) {
                Ok(canvas) => {
                    let compatible = canvas.project_id.as_str()
                        == session.project_id().map(|id| id.as_str()).unwrap_or("");
                    if compatible {
                        format!(
                            "{}\n\n{}",
                            ProjectContextAssembler::assemble(&canvas),
                            input_text
                        )
                    } else {
                        eprintln!(
                            "[reimagine] dropping canvas context hint for session with different project"
                        );
                        input_text
                    }
                }
                Err(error) => {
                    eprintln!("[reimagine] dropping malformed canvas context hint: {error}");
                    input_text
                }
            },
            None => input_text,
        };

        // AR-03: subscribe the channel so embedded AgentEvents stream to
        // the UI, then drop the subscription (backend drops its sender)
        // when the turn finishes.
        self.agent_event_hub.subscribe(&session_id, channel);

        // Create turn request
        let mut request = AgentServiceTurnRequest::from_user_text(
            session_id.clone(),
            turn_id,
            model.into(),
            input_text,
        );
        if let Some(output_schema) = output_schema {
            request = request.with_output_schema(output_schema);
        }
        if let Some(timeout_ms) = timeout_ms {
            request = request.with_turn_timeout(std::time::Duration::from_millis(timeout_ms));
        }

        // Run the turn using AgentService
        let result = {
            let agent_service = {
                let app_host = self.app_host.read().expect(APP_HOST_LOCK);
                app_host.workspace().agent_service().clone()
            };
            agent_service.run_turn(request).await
        };

        // On success, emit the synthetic turn_completed marker, then
        // disconnect the channel. On error, the harness already emitted
        // provider_error through the hub; still unsubscribing so a failed
        // turn cannot leak a live sender.
        let completed = result.as_ref().map(agent_turn_completed_message).ok();
        let duration_ms = result
            .as_ref()
            .ok()
            .and_then(|turn| turn.duration().map(|duration| duration.as_millis() as u64));
        let estimated_cost = result
            .as_ref()
            .ok()
            .and_then(|turn| turn.estimated_cost(3.0, 15.0));
        let usage = result
            .as_ref()
            .ok()
            .and_then(|turn| turn.usage().cloned())
            .map(Into::into);
        self.agent_event_hub.send_turn_completed_with_observability(
            &session_id,
            completed.unwrap_or_else(|| "turn failed".to_string()),
            duration_ms,
            estimated_cost,
            usage,
        );
        self.agent_event_hub.unsubscribe(&session_id);
        let result = result?;

        // Convert result to TurnRunResult
        Ok(TurnRunResult {
            status: reimagine_app_host::TurnRunStatus::Accepted,
            session_id: result.session_id().to_string(),
            turn_id: result.turn_id().to_string(),
        })
    }
    pub fn preview_workflow_commands(
        &self,
        project_id: String,
        workflow_id: String,
        command_batch: serde_json::Value,
    ) -> Result<CommandResult, AppHostError> {
        use reimagine_core::command::CommandBatch;

        let batch: CommandBatch =
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
    ) -> Result<CommandResult, AppHostError> {
        use reimagine_core::command::CommandBatch;

        let batch: CommandBatch =
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
    ) -> Result<CommandResult, AppHostError> {
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

    /// List available provider names for the agent UI selector.
    pub async fn list_agent_providers(&self) -> Result<Vec<String>, AppHostError> {
        let providers: Vec<String> = {
            let app_host = self.app_host.read().expect(APP_HOST_LOCK);
            app_host
                .workspace()
                .agent_service()
                .providers()
                .provider_names()
                .into_iter()
                .map(|p| p.to_string())
                .collect()
        };

        Ok(providers.into_iter().map(|p| p.to_string()).collect())
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
    #[allow(clippy::too_many_arguments)]
    pub async fn download_huggingface_model(
        &self,
        repo_id: String,
        revision: Option<String>,
        allow_patterns: Option<Vec<String>>,
        target_relative_dir: String,
        overwrite: Option<String>,
        auto_detect: Option<bool>,
        from_catalog: Option<bool>,
        channel: Channel<reimagine_app_host::dto::DownloadEventPayload>,
    ) -> Result<reimagine_app_host::dto::ModelDownloadOutput, AppHostError> {
        use reimagine_model_acquisition::{
            AcquireProvider, AllowPatterns, ModelAcquisitionRequest, OverwritePolicy, RepoId,
            Revision, TargetRelativeDir,
        };

        let download_id = format!(
            "dl-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );

        // Subscribe the channel for progress events.
        self.download_event_hub.subscribe(&download_id, channel);

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
        let progress_sink = self.download_event_hub.sink_for(&download_id);

        // Notify started with catalog metadata when from_catalog.
        if use_smart_download {
            if let Ok(card) = {
                let workspace = {
                    let app_host = self.app_host.read().expect(APP_HOST_LOCK);
                    Arc::clone(app_host.workspace())
                };
                workspace
                    .model_service()
                    .model_card(request.repo_id.as_str())
                    .await
            } {
                let model_name = card.entry.id.clone();
                let detected_format = format!("{:?}", card.detected_format);
                let estimated_size = card.estimated_download_size;
                progress_sink.started(request.repo_id.as_str(), request.revision.as_str());
                // Enrich the started event with catalog metadata via a custom event.
                self.download_event_hub.send_enriched_started(
                    &download_id,
                    request.repo_id.as_str(),
                    request.revision.as_str(),
                    Some(model_name),
                    Some(detected_format),
                    Some(estimated_size),
                );
            } else {
                progress_sink.started(request.repo_id.as_str(), request.revision.as_str());
            }
        } else {
            progress_sink.started(request.repo_id.as_str(), request.revision.as_str());
        }

        let report = acq.acquire(request, Some(progress_sink)).await?;

        Ok(reimagine_app_host::dto::ModelDownloadOutput::from(report))
    }

    /// Shut down the active inference worker, if any.
    ///
    /// This is the application-level shutdown hook. Call it when the host
    /// is exiting so that child worker processes are cleaned up instead of
    /// becoming orphans.
    pub async fn shutdown(&self) {
        let app_host = self.app_host.read().expect(APP_HOST_LOCK).clone();
        app_host.shutdown().await;
    }

    // ─── Re-bootstrap (BE-38 / B4-8) ──────────────────────────────

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

    /// Drain in-flight runs (waiting up to `deadline`) and switch the active
    /// worker to the installed worker for `target_instance`.
    pub async fn drain_and_switch_worker(
        &self,
        target_instance: &str,
        deadline: std::time::Duration,
    ) -> Result<WorkerSwitchResultDto, WorkerSwitchError> {
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

    /// Cancel in-flight runs and switch the active worker to the installed
    /// worker for `target_instance`.
    pub async fn cancel_and_switch_worker(
        &self,
        target_instance: &str,
        deadline: std::time::Duration,
    ) -> Result<WorkerSwitchResultDto, WorkerSwitchError> {
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

    /// List installed workers that can be used as switch targets.
    pub fn list_worker_switch_targets(&self) -> Result<Vec<WorkerInstallationDto>, AppHostError> {
        Ok(self.worker_management.list_installed()?)
    }

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
            supported_protocols: ProtocolRange::new(1, 1),
            limits: WorkerLimits::default(),
            environment: Vec::new(),
            transport: Default::default(),
        };
        WorkerBackendCandidate::try_new(launch, manifest_profile).map_err(|error| {
            WorkerSwitchError::Startup {
                message: format!("installed worker `{target_instance}` is not launchable: {error}"),
            }
        })
    }
}

/// Result of a worker switch, adapted from [`WorkerSelectionHandle`] for IPC.
#[derive(Debug, Clone, Serialize)]
pub struct WorkerSwitchResultDto {
    pub instance: String,
    pub incarnation_id: String,
}

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

/// Translate the `agent_turn` `input` argument into the daemon's prompt.
///
/// The UI sends a JSON array of `{ role, content }` messages (the shape
/// the in-process `AgentService` accepted); the daemon V1 protocol takes
/// a plain string or a `{ "text": ... }` object. The last user message's
/// content is the prompt; other shapes are passed through unchanged.
fn turn_input_text(input: &serde_json::Value) -> Result<String, AppHostError> {
    use serde_json::Value;
    match input {
        Value::String(text) => Ok(text.clone()),
        Value::Object(map) => match map.get("text").and_then(Value::as_str) {
            Some(text) => Ok(text.to_owned()),
            None => Err(AppHostError::WorkflowJson {
                path: std::path::PathBuf::new(),
                message: "invalid input messages: expected a `text` field".to_owned(),
            }),
        },
        Value::Array(messages) => {
            // Lenient extraction: UI messages are `{ role, content }`
            // objects; the last user message is the new prompt.
            let text = messages
                .iter()
                .rev()
                .find(|message| message.get("role").and_then(Value::as_str) == Some("user"))
                .and_then(|message| message.get("content").and_then(Value::as_str))
                .map(str::to_owned);
            text.ok_or_else(|| AppHostError::WorkflowJson {
                path: std::path::PathBuf::new(),
                message: "invalid input messages: expected at least one user message with \
                          string content"
                    .to_owned(),
            })
        }
        _ => Err(AppHostError::WorkflowJson {
            path: std::path::PathBuf::new(),
            message: "invalid input messages: expected an array of messages, a string, \
                      or an object with a `text` field"
                .to_owned(),
        }),
    }
}

/// Translate a daemon JSON-RPC notification envelope into the
/// `AgentEventPayload` shape the UI already consumes.
///
/// Unknown or malformed envelopes yield `None` and are dropped so the
/// stream stays resilient to protocol additions. Retained for the
/// mapping tests until AR-03 replaces it with the embedded event sink.
#[cfg(test)]
fn agent_event_payload_from_envelope(envelope: &serde_json::Value) -> Option<AgentEventPayload> {
    use serde_json::Value;
    let method = envelope.get("method").and_then(Value::as_str)?;
    let params = envelope.get("params")?;
    let session_id = params.get("session_id").and_then(Value::as_str)?.to_owned();
    let (kind, tool_name, tool_call_id, code, message) = match method {
        "agent.session_started" => {
            let provider = params
                .get("provider")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let mode = params
                .get("mode")
                .and_then(Value::as_str)
                .unwrap_or_default();
            (
                "session_started",
                None,
                None,
                None,
                Some(format!("provider={provider} mode={mode}")),
            )
        }
        "agent.session_stopped" => (
            "session_stopped",
            None,
            None,
            None,
            params
                .get("reason")
                .and_then(Value::as_str)
                .map(str::to_owned),
        ),
        "agent.tool_invoked" => tool_event_payload("tool_invoked", params, None),
        "agent.tool_completed" => tool_event_payload("tool_completed", params, None),
        "agent.tool_failed" => tool_event_payload(
            "tool_failed",
            params,
            params
                .get("error")
                .and_then(Value::as_str)
                .map(str::to_owned),
        ),
        "agent.content_delta" => (
            "content_delta",
            None,
            None,
            None,
            params
                .get("text")
                .and_then(Value::as_str)
                .map(str::to_owned),
        ),
        "agent.proposal_ready" => (
            "proposal_ready",
            None,
            None,
            None,
            params
                .get("proposal_id")
                .and_then(Value::as_str)
                .map(|id| format!("proposal_id={id}")),
        ),
        "agent.error" => (
            "provider_error",
            None,
            None,
            params
                .get("code")
                .and_then(Value::as_str)
                .map(str::to_owned),
            params
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_owned),
        ),
        "agent.turn_completed" => (
            "turn_completed",
            None,
            None,
            None,
            params.get("result").and_then(turn_completed_message),
        ),
        _ => return None,
    };
    Some(AgentEventPayload {
        session_id,
        kind: kind.to_owned(),
        tool_name,
        tool_call_id,
        code,
        message,
        ..Default::default()
    })
}

/// Build a tool-event payload row from `agent.tool_*` notification params.
#[cfg(test)]
fn tool_event_payload(
    kind: &'static str,
    params: &serde_json::Value,
    error: Option<String>,
) -> (
    &'static str,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
) {
    (
        kind,
        params
            .get("tool")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        params
            .get("tool_call_id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        None,
        error,
    )
}

/// Project a `turn_completed` result into the payload message: the final
/// response text when present, otherwise the serialized result.
#[cfg(test)]
fn turn_completed_message(result: &serde_json::Value) -> Option<String> {
    match result {
        serde_json::Value::String(text) => Some(text.clone()),
        serde_json::Value::Object(map) => {
            match map
                .get("final_response")
                .and_then(serde_json::Value::as_str)
            {
                Some(text) if !text.is_empty() => Some(text.to_owned()),
                _ => Some(result.to_string()),
            }
        }
        _ => Some(result.to_string()),
    }
}

/// Build the end-of-turn marker message for the turn_completed payload:
/// the final response text when present, otherwise a status summary.
fn agent_turn_completed_message(result: &reimagine_agent_harness::AgentTurnResult) -> String {
    match result.final_response() {
        Some(message) if !message.content().is_empty() => message.content().to_string(),
        _ => format!(
            "status={:?} stop_reason={:?}",
            result.status(),
            result.stop_reason()
        ),
    }
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
    use super::*;
    use serde_json::json;

    // ─── agent bridge input translation ──────────────────────────

    #[test]
    fn turn_input_passes_through_text_shapes() {
        assert_eq!(turn_input_text(&json!("hello")).expect("string"), "hello");
        assert_eq!(
            turn_input_text(&json!({ "text": "draw a cat" })).expect("text object"),
            "draw a cat"
        );
    }

    #[test]
    fn turn_input_uses_last_user_message_from_ui_array() {
        let input = json!([
            { "role": "system", "content": "be concise" },
            { "role": "user", "content": "first" },
            { "role": "assistant", "content": "ok" },
            { "role": "user", "content": "the prompt" },
        ]);
        assert_eq!(
            turn_input_text(&input).expect("message array"),
            "the prompt"
        );
    }

    #[test]
    fn turn_input_rejects_bad_shapes() {
        assert!(turn_input_text(&json!(42)).is_err());
        assert!(turn_input_text(&json!({ "role": "user" })).is_err());
        assert!(turn_input_text(&json!([{ "role": "assistant", "content": "no user" }])).is_err());
        assert!(turn_input_text(&json!([{ "role": "user" }])).is_err());
    }

    // ─── daemon notification → AgentEventPayload translation ─────

    fn envelope(method: &str, params: serde_json::Value) -> serde_json::Value {
        json!({ "jsonrpc": "2.0", "method": method, "params": params })
    }

    #[test]
    fn envelope_translates_content_delta() {
        let payload = agent_event_payload_from_envelope(&envelope(
            "agent.content_delta",
            json!({ "session_id": "s1", "turn_id": "t1", "text": "hello" }),
        ))
        .expect("translated");
        assert_eq!(payload.session_id, "s1");
        assert_eq!(payload.kind, "content_delta");
        assert_eq!(payload.message.as_deref(), Some("hello"));
        assert!(payload.tool_name.is_none());
    }

    #[test]
    fn envelope_translates_tool_events() {
        let invoked = agent_event_payload_from_envelope(&envelope(
            "agent.tool_invoked",
            json!({ "session_id": "s1", "turn_id": "t1", "tool": "workflow.read", "tool_call_id": "call-1" }),
        ))
        .expect("translated");
        assert_eq!(invoked.kind, "tool_invoked");
        assert_eq!(invoked.tool_name.as_deref(), Some("workflow.read"));
        assert_eq!(invoked.tool_call_id.as_deref(), Some("call-1"));

        let failed = agent_event_payload_from_envelope(&envelope(
            "agent.tool_failed",
            json!({ "session_id": "s1", "turn_id": "t1", "tool": "workflow.read", "tool_call_id": "call-1", "error": "timeout" }),
        ))
        .expect("translated");
        assert_eq!(failed.kind, "tool_failed");
        assert_eq!(failed.message.as_deref(), Some("timeout"));
    }

    #[test]
    fn envelope_translates_lifecycle_notifications() {
        let started = agent_event_payload_from_envelope(&envelope(
            "agent.session_started",
            json!({ "session_id": "s1", "mode": "agent", "provider": "openai" }),
        ))
        .expect("translated");
        assert_eq!(started.kind, "session_started");
        assert_eq!(
            started.message.as_deref(),
            Some("provider=openai mode=agent")
        );

        let stopped = agent_event_payload_from_envelope(&envelope(
            "agent.session_stopped",
            json!({ "session_id": "s1", "reason": "user_requested" }),
        ))
        .expect("translated");
        assert_eq!(stopped.kind, "session_stopped");
        assert_eq!(stopped.message.as_deref(), Some("user_requested"));

        let proposal = agent_event_payload_from_envelope(&envelope(
            "agent.proposal_ready",
            json!({ "session_id": "s1", "proposal_id": "p1" }),
        ))
        .expect("translated");
        assert_eq!(proposal.kind, "proposal_ready");
        assert_eq!(proposal.message.as_deref(), Some("proposal_id=p1"));
    }

    #[test]
    fn envelope_translates_error_and_turn_completed() {
        let error = agent_event_payload_from_envelope(&envelope(
            "agent.error",
            json!({ "session_id": "s1", "code": "provider_error", "message": "upstream 429" }),
        ))
        .expect("translated");
        assert_eq!(error.kind, "provider_error");
        assert_eq!(error.code.as_deref(), Some("provider_error"));
        assert_eq!(error.message.as_deref(), Some("upstream 429"));

        let completed = agent_event_payload_from_envelope(&envelope(
            "agent.turn_completed",
            json!({ "session_id": "s1", "turn_id": "t1", "result": { "status": "completed", "final_response": "done" } }),
        ))
        .expect("translated");
        assert_eq!(completed.kind, "turn_completed");
        assert_eq!(completed.message.as_deref(), Some("done"));
    }

    #[test]
    fn envelope_drops_unknown_or_malformed_messages() {
        assert!(
            agent_event_payload_from_envelope(
                &json!({ "method": "agent.unknown", "params": { "session_id": "s1" } })
            )
            .is_none()
        );
        assert!(
            agent_event_payload_from_envelope(&json!({ "method": "agent.content_delta" }))
                .is_none()
        );
        assert!(
            agent_event_payload_from_envelope(
                &json!({ "method": "agent.content_delta", "params": {} })
            )
            .is_none()
        );
    }
}
