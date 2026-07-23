use std::path::{Path, PathBuf};
use std::sync::Arc;

use reimagine_agent::{AgentEventSink, WorkspaceScope};
use reimagine_app_host::dto::{
    AgentEventPayload, AgentSessionInfo, AgentTurnResponse, ArtifactMetadataDto, ComputeProfileDto,
    HealthResponse, ModelInfoDto, NodeCatalogResponse, RunWorkflowResponse,
};
use reimagine_app_host::{
    AgentServiceTurnRequest, AppHost, AppHostError, WorkerManagementService, WorkspaceHost,
};
use reimagine_config::AppPaths;
use reimagine_core::command::CommandResult;
use reimagine_core::workflow::Workflow;
use reimagine_runtime::BoxedRunEventSink;
use tauri::ipc::Channel;

use crate::agent_event_hub::TauriAgentEventHub;
use crate::download_event_hub::TauriDownloadEventHub;
use crate::event_hub::{RunEventPayload, TauriRunEventHub};

const WORKSPACE_SCOPE: &str = "reimagine-tauri-host";

#[derive(Debug, Clone)]
pub struct DesktopHostState {
    app_host: AppHost,
    event_hub: Arc<TauriRunEventHub>,
    agent_event_hub: Arc<TauriAgentEventHub>,
    download_event_hub: Arc<TauriDownloadEventHub>,
    worker_management: Arc<WorkerManagementService>,
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
        let agent_event_hub = Arc::new(TauriAgentEventHub::new());
        let agent_event_sink: Arc<dyn AgentEventSink> = agent_event_hub.clone();
        let workspace = WorkspaceHost::try_with_defaults_and_event_sinks(
            WorkspaceScope::new(WORKSPACE_SCOPE),
            &workspace_base_path,
            event_sink,
            agent_event_sink,
        )
        .await?;

        let download_event_hub = Arc::new(TauriDownloadEventHub::new());
        let worker_management = Arc::new(WorkerManagementService::offline(app_data_root)?);

        Ok(Self {
            app_host: AppHost::new(workspace),
            event_hub,
            agent_event_hub,
            download_event_hub,
            worker_management,
        })
    }

    #[cfg(test)]
    pub fn workspace_base_path(&self) -> &Path {
        self.app_host.workspace().base_path()
    }

    #[allow(dead_code)]
    pub fn worker_management(&self) -> &WorkerManagementService {
        &self.worker_management
    }

    pub fn health(&self) -> HealthResponse {
        HealthResponse::ok(self.app_host.workspace().workspace_scope().as_str())
    }

    pub fn compute_profile(&self) -> ComputeProfileDto {
        self.app_host.workspace().compute_profile_dto()
    }

    /// Returns the workspace node catalog as a host‑neutral DTO.
    pub fn list_node_defs(&self) -> NodeCatalogResponse {
        use reimagine_app_host::dto::NodeDefDto;
        let defs = self.app_host.workspace().list_node_defs();
        NodeCatalogResponse {
            nodes: defs.into_iter().map(NodeDefDto::from).collect(),
        }
    }

    /// Returns the model list as a host‑neutral DTO.
    pub async fn list_models(&self) -> Result<Vec<ModelInfoDto>, AppHostError> {
        use reimagine_app_host::dto::ModelInfoDto;
        let descriptors = self
            .app_host
            .workspace()
            .model_service()
            .list_models()
            .await?;
        Ok(descriptors.into_iter().map(ModelInfoDto::from).collect())
    }

    pub async fn run_workflow(
        &self,
        workflow_value: serde_json::Value,
        channel: Channel<RunEventPayload>,
    ) -> Result<RunWorkflowResponse, AppHostError> {
        use reimagine_app_host::RunWorkflowRequest;

        // 1. Deserialize and register the workflow
        let workflow: Workflow =
            serde_json::from_value(workflow_value).map_err(|e| AppHostError::WorkflowJson {
                path: std::path::PathBuf::new(),
                message: e.to_string(),
            })?;
        let workflow_id = self
            .app_host
            .workspace()
            .workflow_service()
            .register_workflow(workflow);

        // 2. Build run request
        let request = RunWorkflowRequest::new(
            workflow_id.clone(),
            reimagine_core::readiness::RunTargetSelection::AllDefaultTargets,
        );

        // 3. Run the workflow
        let result = self.app_host.workspace().run_workflow(request).await?;

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
        let access = self.app_host.workspace().resolve_artifact(&id)?;
        Ok(ArtifactMetadataDto::from(access))
    }

    /// Open an artifact file with the desktop system handler.
    pub fn open_artifact(&self, artifact_id: &str) -> Result<(), AppHostError> {
        use reimagine_core::model::ArtifactId;
        let id = ArtifactId::new(artifact_id);
        let access = self.app_host.workspace().resolve_artifact(&id)?;
        opener::open(access.path.as_path()).map_err(|e| AppHostError::Io {
            path: access.path,
            message: format!("failed to open artifact: {e}"),
        })
    }

    /// Create a new agent session.
    ///
    /// Returns `AppHostError::UnknownAgentProvider` if the named provider
    /// is not registered in the catalog.
    pub fn create_agent_session(
        &self,
        mode: String,
        provider: String,
    ) -> Result<AgentSessionInfo, AppHostError> {
        use reimagine_app_host::dto::AgentSessionInfo as Dto;

        let agent_mode = match mode.as_str() {
            "Agent" => reimagine_agent::AgentMode::Agent,
            "Build" => reimagine_agent::AgentMode::Build,
            other => {
                return Err(AppHostError::UnknownAgentMode {
                    mode: other.to_string(),
                });
            }
        };

        // Validate provider exists in catalog before creating session
        let agent_service = self.app_host.workspace().agent_service();
        let catalog = agent_service.providers();
        let provider_name = reimagine_agent::ProviderName::new(&provider);
        if !catalog.contains(&provider_name) {
            return Err(AppHostError::UnknownAgentProvider {
                provider: provider_name.clone(),
            });
        }

        let session_id = reimagine_agent::AgentSessionId::new(format!(
            "sess-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));

        let started_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .to_string();

        let session =
            agent_service.create_session(session_id.clone(), agent_mode, provider_name, started_at);

        Ok(Dto::from(session))
    }

    /// Execute a single agent turn.
    ///
    /// The `model` field is the model name string as understood by the
    /// registered provider. The `input` is a JSON array of user messages.
    /// Subscribes the provided channel for live `AgentEvent` streaming.
    pub async fn agent_turn(
        &self,
        session_id: String,
        turn_id: String,
        model: String,
        input: serde_json::Value,
        channel: Channel<AgentEventPayload>,
    ) -> Result<AgentTurnResponse, AppHostError> {
        // Parse messages from JSON array
        let messages: Vec<reimagine_agent::Message> =
            serde_json::from_value(input).map_err(|e| AppHostError::WorkflowJson {
                path: std::path::PathBuf::new(),
                message: format!("invalid input messages: {e}"),
            })?;

        let agent_service = self.app_host.workspace().agent_service();

        // Validate session exists (fail-fast before subscription)
        let _session =
            agent_service.get_session(&reimagine_agent::AgentSessionId::new(&session_id))?;

        // Subscribe the channel before starting the turn (no replay needed)
        self.agent_event_hub.subscribe(&session_id, channel);

        let turn_request = AgentServiceTurnRequest::new(
            reimagine_agent::AgentSessionId::new(session_id.clone()),
            reimagine_agent::AgentTurnId::new(turn_id.clone()),
            reimagine_agent::ModelName::new(model),
            messages,
        );

        let result = agent_service.run_turn(turn_request).await?;

        Ok(AgentTurnResponse::from(result))
    }

    /// Preview a command batch (dry-run).
    ///
    /// Returns diagnostics and change preview without mutating the workflow.
    pub fn preview_workflow_commands(
        &self,
        workflow_id: String,
        command_batch: serde_json::Value,
    ) -> Result<CommandResult, AppHostError> {
        use reimagine_core::command::CommandBatch;

        let batch: CommandBatch =
            serde_json::from_value(command_batch).map_err(|e| AppHostError::WorkflowJson {
                path: std::path::PathBuf::new(),
                message: format!("invalid command batch: {e}"),
            })?;

        let node_catalog = self.app_host.workspace().node_catalog();
        let workflow_service = self.app_host.workspace().workflow_service();

        workflow_service.preview_batch(
            &reimagine_core::model::WorkflowId::new(workflow_id),
            node_catalog.as_ref(),
            batch,
        )
    }

    /// Apply a command batch directly.
    ///
    /// Returns the applied `CommandResult` with changes and diagnostics.
    pub fn apply_workflow_commands(
        &self,
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

        let node_catalog = self.app_host.workspace().node_catalog();
        let workflow_service = self.app_host.workspace().workflow_service();

        workflow_service.apply_batch(
            &reimagine_core::model::WorkflowId::new(workflow_id),
            node_catalog.as_ref(),
            batch,
        )
    }

    /// Approve a pending workflow proposal.
    ///
    /// Calls `WorkflowService::apply_pending_proposal()` and returns the
    /// resulting `CommandResult`. Returns an error if no pending proposal
    /// exists for the workflow.
    pub fn approve_proposal(&self, workflow_id: String) -> Result<CommandResult, AppHostError> {
        let node_catalog = self.app_host.workspace().node_catalog();
        let workflow_service = self.app_host.workspace().workflow_service();

        workflow_service.apply_pending_proposal(
            &reimagine_core::model::WorkflowId::new(workflow_id),
            node_catalog.as_ref(),
            None, // approved_by — Tauri thin shell, no human actor identity
        )
    }

    /// List pending proposals from all workflows.
    #[allow(dead_code)]
    pub fn list_pending_proposals(&self) -> Result<Vec<serde_json::Value>, AppHostError> {
        let workflow_service = self.app_host.workspace().workflow_service();
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
    pub fn list_agent_providers(&self) -> Result<Vec<String>, AppHostError> {
        let catalog = self.app_host.workspace().agent_service().providers();
        Ok(catalog
            .provider_names()
            .into_iter()
            .map(|p| p.to_string())
            .collect())
    }

    /// Download a HuggingFace model with progress streaming.
    ///
    /// Returns an `AcquisitionReportDto` through the Tauri IPC channel.
    pub async fn download_huggingface_model(
        &self,
        repo_id: String,
        revision: Option<String>,
        allow_patterns: Option<Vec<String>>,
        target_relative_dir: String,
        overwrite: Option<String>,
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

        let request = ModelAcquisitionRequest {
            provider: AcquireProvider::HuggingFace,
            repo_id,
            revision,
            allow_patterns,
            target_relative_dir,
            overwrite_policy,
        };

        let acq = self
            .app_host
            .workspace()
            .services()
            .model_acquisition_service()
            .clone();
        let progress_sink = self.download_event_hub.sink_for(&download_id);

        // Notify started.
        progress_sink.started(request.repo_id.as_str(), request.revision.as_str());

        let report = acq.acquire(request, Some(progress_sink)).await?;

        Ok(reimagine_app_host::dto::ModelDownloadOutput::from(report))
    }
}

pub fn default_workspace_path(app_data_dir: impl AsRef<Path>) -> PathBuf {
    app_data_dir.as_ref().join("workspace")
}
