use std::path::{Path, PathBuf};
use std::sync::Arc;

use reimagine_agent::{AgentEventSink, WorkspaceScope};
use reimagine_app_host::dto::{
    AgentEventPayload, AgentSessionInfo, AgentTurnResponse, ArtifactMetadataDto, ComputeProfileDto,
    HealthResponse, ModelInfoDto, NodeCatalogResponse, RunWorkflowResponse,
};
use reimagine_app_host::{
    AgentServiceTurnRequest, AppHost, AppHostError, WorkerBackendCandidate, WorkerInstallationDto,
    WorkerManagementService, WorkerSelectionHandle, WorkerSwitchError, WorkspaceHost,
};
use reimagine_backend_worker_host::{WorkerLaunchSpec, WorkerLimits};
use reimagine_backend_worker_protocol::ProtocolRange;
use reimagine_config::AppPaths;
use reimagine_core::command::CommandResult;
use reimagine_core::workflow::Workflow;
use reimagine_runtime::BoxedRunEventSink;
use serde::Serialize;
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

    // ─── Workflow persistence ─────────────────────────────────────

    /// Persist a workflow (JSON) to the workspace `workflows/` directory.
    ///
    /// The JSON must deserialize into `reimagine_core::workflow::Workflow`.
    /// Registers the workflow as a session before writing, so it is also
    /// immediately runnable.
    pub async fn save_workflow(
        &self,
        workflow_id: &str,
        workflow_value: serde_json::Value,
    ) -> Result<PathBuf, AppHostError> {
        use reimagine_core::workflow::Workflow;

        workflow_id_from_str(workflow_id)?;
        let workflow: Workflow =
            serde_json::from_value(workflow_value).map_err(|e| AppHostError::WorkflowJson {
                path: std::path::PathBuf::new(),
                message: format!("invalid workflow json: {e}"),
            })?;

        let workflow_service = self.app_host.workspace().workflow_service();
        workflow_service.register_workflow(workflow.clone());
        workflow_service.save_workflow_snapshot(&workflow).await
    }

    /// Load a workflow (JSON) from the workspace `workflows/` directory.
    ///
    /// Registers the loaded workflow as a session and returns its JSON.
    pub async fn load_workflow_json(
        &self,
        workflow_id: &str,
    ) -> Result<serde_json::Value, AppHostError> {
        let id = workflow_id_from_str(workflow_id)?;
        let workflow_service = self.app_host.workspace().workflow_service();
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
    pub fn list_saved_workflows(&self) -> Result<Vec<serde_json::Value>, AppHostError> {
        let infos = self
            .app_host
            .workspace()
            .workflow_service()
            .list_saved_workflows()?;
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
        let entries = self
            .app_host
            .workspace()
            .model_service()
            .search(query)
            .await?;
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
        let card = self
            .app_host
            .workspace()
            .model_service()
            .model_card(repo_id)
            .await?;
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

        let acq = self
            .app_host
            .workspace()
            .services()
            .model_acquisition_service()
            .clone();
        let progress_sink = self.download_event_hub.sink_for(&download_id);

        // Notify started with catalog metadata when from_catalog.
        if use_smart_download {
            if let Ok(card) = self
                .app_host
                .workspace()
                .model_service()
                .model_card(request.repo_id.as_str())
                .await
            {
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
        self.app_host.shutdown().await;
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
        let handle = self
            .app_host
            .workspace()
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
        let handle = self
            .app_host
            .workspace()
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

/// Validate a user-supplied workflow id before it reaches `WorkflowId::new`
/// (which asserts — and would panic — on invalid input).
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
