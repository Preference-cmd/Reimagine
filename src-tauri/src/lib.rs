mod agent_bridge;
mod desktop_host;
mod download_event_hub;
mod event_hub;

pub use agent_bridge::{AgentBridge, AgentBridgeError};

use desktop_host::{DesktopHostState, WorkerSwitchResultDto, default_workspace_path};
use event_hub::RunEventPayload;
use reimagine_agent_daemon::protocol::TurnRunResult;
use reimagine_app_host::{
    AppHostError, AppHostErrorCode, BackendSelection, WorkerInstallationDto, WorkerSwitchError,
    dto::{
        AgentEventPayload, AgentSessionInfo, ArtifactMetadataDto, ComputeProfileDto,
        DownloadEventPayload, HealthResponse, ModelCardDto, ModelCatalogEntryDto,
        ModelDownloadOutput, ModelFilters, ModelInfoDto, NodeDefDto, RunWorkflowResponse,
    },
};
use reimagine_core::command::CommandResult;
use serde::Serialize;
use tauri::{Manager, ipc::Channel};

#[derive(Debug, Clone, Serialize)]
struct TauriCommandError {
    code: AppHostErrorCode,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<serde_json::Value>,
}

impl TauriCommandError {
    fn bootstrap(message: impl Into<String>) -> Self {
        Self {
            code: AppHostErrorCode::BootstrapFailed,
            message: message.into(),
            details: None,
        }
    }

    fn command(message: impl Into<String>) -> Self {
        Self {
            code: AppHostErrorCode::CommandFailed,
            message: message.into(),
            details: None,
        }
    }

    fn unknown_provider(provider: impl Into<String>) -> Self {
        Self {
            code: AppHostErrorCode::UnknownProvider,
            message: format!(
                "Provider '{}' is not configured. Add a provider in Settings.",
                provider.into()
            ),
            details: None,
        }
    }
}

impl std::fmt::Display for TauriCommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code.as_str(), self.message)
    }
}

impl std::error::Error for TauriCommandError {}

fn app_host_command_error(error: AppHostError) -> TauriCommandError {
    if let AppHostError::UnknownAgentProvider { provider } = &error {
        return TauriCommandError::unknown_provider(provider.to_string());
    }
    TauriCommandError {
        code: error.code(),
        message: error.to_string(),
        details: error.details(),
    }
}

fn agent_bridge_command_error(error: AgentBridgeError) -> TauriCommandError {
    // Protocol errors carry a clean client-facing message; everything else
    // falls back to the display text.
    match error {
        AgentBridgeError::Protocol { message, .. } => TauriCommandError::command(message),
        other => TauriCommandError::command(other.to_string()),
    }
}

fn worker_switch_command_error(error: WorkerSwitchError) -> TauriCommandError {
    TauriCommandError {
        code: reimagine_app_host::worker_switch_error_code(&error),
        message: error.to_string(),
        details: reimagine_app_host::worker_switch_error_details(&error),
    }
}

fn switch_deadline(deadline_secs: Option<f64>) -> Result<std::time::Duration, TauriCommandError> {
    match deadline_secs {
        None => Ok(std::time::Duration::from_secs(30)),
        Some(secs) if secs > 0.0 => Ok(std::time::Duration::from_secs_f64(secs)),
        Some(_) => Err(TauriCommandError::command(
            "deadline_secs must be a positive number of seconds",
        )),
    }
}

// ─── Existing commands ───────────────────────────────────────────

#[tauri::command]
fn health(state: tauri::State<'_, DesktopHostState>) -> Result<HealthResponse, TauriCommandError> {
    Ok(state.health())
}

#[tauri::command]
fn get_compute_profile(
    state: tauri::State<'_, DesktopHostState>,
) -> Result<ComputeProfileDto, TauriCommandError> {
    Ok(state.compute_profile())
}

#[tauri::command]
fn get_node_defs(
    state: tauri::State<'_, DesktopHostState>,
) -> Result<Vec<NodeDefDto>, TauriCommandError> {
    let response = state.list_node_defs();
    Ok(response.nodes)
}

#[tauri::command]
async fn list_models(
    state: tauri::State<'_, DesktopHostState>,
) -> Result<Vec<ModelInfoDto>, TauriCommandError> {
    state.list_models().await.map_err(app_host_command_error)
}

#[tauri::command]
async fn run_workflow(
    state: tauri::State<'_, DesktopHostState>,
    workflow: serde_json::Value,
    channel: Channel<RunEventPayload>,
) -> Result<RunWorkflowResponse, TauriCommandError> {
    state
        .run_workflow(workflow, channel)
        .await
        .map_err(app_host_command_error)
}

#[tauri::command]
async fn cancel_run(
    state: tauri::State<'_, DesktopHostState>,
    run_id: String,
) -> Result<(), TauriCommandError> {
    state.cancel_run(&run_id).map_err(app_host_command_error)
}

#[tauri::command]
async fn resolve_artifact(
    state: tauri::State<'_, DesktopHostState>,
    artifact_id: String,
) -> Result<ArtifactMetadataDto, TauriCommandError> {
    state
        .resolve_artifact(&artifact_id)
        .map_err(app_host_command_error)
}

#[tauri::command]
async fn open_artifact(
    state: tauri::State<'_, DesktopHostState>,
    artifact_id: String,
) -> Result<(), TauriCommandError> {
    state
        .open_artifact(&artifact_id)
        .map_err(app_host_command_error)
}

// ─── Agent commands ──────────────────────────────────────────────

/// Create a new agent session on the daemon.
///
/// `mode` must be "Agent" or "Build".
/// `provider` must match a registered provider in the catalog.
#[tauri::command]
async fn create_agent_session(
    state: tauri::State<'_, DesktopHostState>,
    mode: String,
    provider: String,
) -> Result<AgentSessionInfo, TauriCommandError> {
    state
        .create_agent_session(mode, provider)
        .await
        .map_err(agent_bridge_command_error)
}

/// Execute a single agent turn with live event streaming via the daemon.
///
/// `session_id` must be a valid existing session on the daemon.
/// `turn_id` is a caller-generated id for this turn (idempotent retries).
/// `model` is the model name string for the registered provider.
/// `input` is a JSON array of `{ role, content }` message objects.
/// Resolves once the daemon accepts the turn; daemon notifications stream
/// `AgentEventPayload` events through the channel.
#[tauri::command]
async fn agent_turn(
    state: tauri::State<'_, DesktopHostState>,
    session_id: String,
    turn_id: String,
    model: String,
    input: serde_json::Value,
    channel: Channel<AgentEventPayload>,
) -> Result<TurnRunResult, TauriCommandError> {
    state
        .agent_turn(session_id, turn_id, model, input, channel)
        .await
        .map_err(agent_bridge_command_error)
}

/// List available agent providers for the UI selector.
#[tauri::command]
async fn list_agent_providers(
    state: tauri::State<'_, DesktopHostState>,
) -> Result<Vec<String>, TauriCommandError> {
    state
        .list_agent_providers()
        .await
        .map_err(agent_bridge_command_error)
}

// ─── Model download commands ───────────────────────────────────────

/// Download a HuggingFace model with progress streaming.
///
/// Streams `DownloadEventPayload` events through the provided channel.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
async fn download_huggingface_model(
    state: tauri::State<'_, DesktopHostState>,
    repo_id: String,
    revision: Option<String>,
    allow_patterns: Option<Vec<String>>,
    target_relative_dir: String,
    overwrite: Option<String>,
    auto_detect: Option<bool>,
    from_catalog: Option<bool>,
    channel: tauri::ipc::Channel<DownloadEventPayload>,
) -> Result<ModelDownloadOutput, TauriCommandError> {
    state
        .download_huggingface_model(
            repo_id,
            revision,
            allow_patterns,
            target_relative_dir,
            overwrite,
            auto_detect,
            from_catalog,
            channel,
        )
        .await
        .map_err(app_host_command_error)
}

/// Search HuggingFace models via the catalog.
///
/// Returns a list of model catalog entries matching the query and filters.
#[tauri::command]
async fn search_models(
    state: tauri::State<'_, DesktopHostState>,
    query: String,
    filters: Option<ModelFilters>,
) -> Result<Vec<ModelCatalogEntryDto>, TauriCommandError> {
    use reimagine_model_acquisition::{ModelSearchQuery, SortBy};

    let sort = match filters.as_ref().map(|f| f.sort.as_str()) {
        Some("likes") => SortBy::Likes,
        Some("trending") => SortBy::Trending,
        Some("lastModified") => SortBy::LastModified,
        _ => SortBy::Downloads,
    };

    let search_query = ModelSearchQuery {
        search: if query.is_empty() { None } else { Some(query) },
        pipeline_tag: filters.as_ref().and_then(|f| f.pipeline_tag.clone()),
        library_name: filters.as_ref().and_then(|f| f.library_name.clone()),
        tags: filters.as_ref().map(|f| f.tags.clone()).unwrap_or_default(),
        sort,
        limit: filters.map(|f| f.limit).unwrap_or(20),
    };

    state
        .search_models(&search_query)
        .await
        .map_err(app_host_command_error)
}

/// Fetch the full model card for a HuggingFace repository.
#[tauri::command]
async fn get_model_card(
    state: tauri::State<'_, DesktopHostState>,
    repo_id: String,
) -> Result<ModelCardDto, TauriCommandError> {
    state
        .get_model_card(&repo_id)
        .await
        .map_err(app_host_command_error)
}

// ─── Worker switch commands (BE-32) ─────────────────────────────

/// Drain in-flight runs (waiting up to `deadline_secs`) and switch the
/// active worker to the installed worker for `target` (a backend instance
/// id such as `burn:wgpu:default`).
#[tauri::command]
async fn drain_and_switch_worker(
    state: tauri::State<'_, DesktopHostState>,
    target: String,
    deadline_secs: Option<f64>,
) -> Result<WorkerSwitchResultDto, TauriCommandError> {
    let deadline = switch_deadline(deadline_secs)?;
    state
        .drain_and_switch_worker(&target, deadline)
        .await
        .map_err(worker_switch_command_error)
}

/// Cancel in-flight runs, then switch the active worker to the installed
/// worker for `target`.
#[tauri::command]
async fn cancel_and_switch_worker(
    state: tauri::State<'_, DesktopHostState>,
    target: String,
    deadline_secs: Option<f64>,
) -> Result<WorkerSwitchResultDto, TauriCommandError> {
    let deadline = switch_deadline(deadline_secs)?;
    state
        .cancel_and_switch_worker(&target, deadline)
        .await
        .map_err(worker_switch_command_error)
}

/// List installed workers usable as switch targets.
#[tauri::command]
fn list_worker_switch_targets(
    state: tauri::State<'_, DesktopHostState>,
) -> Result<Vec<WorkerInstallationDto>, TauriCommandError> {
    state
        .list_worker_switch_targets()
        .map_err(app_host_command_error)
}

/// Drain in-flight runs (waiting up to the re-bootstrap deadline) and
/// re-run the workspace bootstrap with `selection` (`"burn"` | `"candle"`).
///
/// Resolves to the compute profile of the rebuilt workspace.
#[tauri::command]
async fn rebootstrap_backend(
    state: tauri::State<'_, DesktopHostState>,
    selection: BackendSelection,
) -> Result<ComputeProfileDto, TauriCommandError> {
    state
        .rebootstrap_backend(selection)
        .await
        .map_err(app_host_command_error)
}

// ─── Workflow command commands ───────────────────────────────────

/// Preview a command batch (dry-run). Returns diagnostics without mutating.
#[tauri::command]
fn preview_workflow_commands(
    state: tauri::State<'_, DesktopHostState>,
    workflow_id: String,
    command_batch: serde_json::Value,
) -> Result<CommandResult, TauriCommandError> {
    state
        .preview_workflow_commands(workflow_id, command_batch)
        .map_err(app_host_command_error)
}

/// Apply a command batch directly.
#[tauri::command]
fn apply_workflow_commands(
    state: tauri::State<'_, DesktopHostState>,
    workflow_id: String,
    command_batch: serde_json::Value,
    _approved_by: Option<serde_json::Value>,
) -> Result<CommandResult, TauriCommandError> {
    state
        .apply_workflow_commands(workflow_id, command_batch, _approved_by)
        .map_err(app_host_command_error)
}

/// Approve a pending workflow proposal (human approval of build-mode output).
#[tauri::command]
fn approve_proposal(
    state: tauri::State<'_, DesktopHostState>,
    workflow_id: String,
) -> Result<CommandResult, TauriCommandError> {
    state
        .approve_proposal(workflow_id)
        .map_err(app_host_command_error)
}

// ─── Workflow persistence commands ──────────────────────────────

/// Save a workflow to the workspace `workflows/` directory as JSON.
///
/// `workflow_json` must be a full `reimagine_core::workflow::Workflow`
/// payload. Returns the path the file was written to.
#[tauri::command]
async fn save_workflow(
    state: tauri::State<'_, DesktopHostState>,
    workflow_id: String,
    workflow_json: serde_json::Value,
) -> Result<String, TauriCommandError> {
    state
        .save_workflow(&workflow_id, workflow_json)
        .await
        .map(|path| path.display().to_string())
        .map_err(app_host_command_error)
}

/// Load a workflow (JSON) from the workspace `workflows/` directory.
#[tauri::command]
async fn load_workflow(
    state: tauri::State<'_, DesktopHostState>,
    workflow_id: String,
) -> Result<serde_json::Value, TauriCommandError> {
    state
        .load_workflow_json(&workflow_id)
        .await
        .map_err(app_host_command_error)
}

/// List saved workflows (newest first) as `{ id, modified_millis }` objects.
#[tauri::command]
fn list_workflows(
    state: tauri::State<'_, DesktopHostState>,
) -> Result<Vec<serde_json::Value>, TauriCommandError> {
    state.list_saved_workflows().map_err(app_host_command_error)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    match tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let app_data_dir = app
                .path()
                .app_data_dir()
                .map_err(|error| TauriCommandError::bootstrap(error.to_string()))?;
            let workspace_path = default_workspace_path(&app_data_dir);
            let state = tauri::async_runtime::block_on(
                DesktopHostState::bootstrap_with_app_data_root(&app_data_dir, &workspace_path),
            )
            .map_err(|error| TauriCommandError::bootstrap(error.to_string()))?;
            app.manage(state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // Existing commands
            health,
            get_compute_profile,
            get_node_defs,
            list_models,
            run_workflow,
            cancel_run,
            resolve_artifact,
            open_artifact,
            // Agent commands
            create_agent_session,
            agent_turn,
            list_agent_providers,
            // Workflow command commands
            preview_workflow_commands,
            apply_workflow_commands,
            approve_proposal,
            // Workflow persistence commands
            save_workflow,
            load_workflow,
            list_workflows,
            // Model download commands
            download_huggingface_model,
            // Catalog commands
            search_models,
            get_model_card,
            // Worker switch commands
            drain_and_switch_worker,
            cancel_and_switch_worker,
            list_worker_switch_targets,
            // Re-bootstrap command
            rebootstrap_backend,
        ])
        .build(tauri::generate_context!())
    {
        Ok(app) => app.run(|app_handle, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                let state = app_handle.state::<DesktopHostState>();
                let state = state.inner().clone();
                let _ = std::thread::scope(|s| {
                    s.spawn(|| {
                        let rt = tokio::runtime::Handle::current();
                        rt.block_on(async {
                            let shutdown = tokio::time::timeout(
                                std::time::Duration::from_secs(5),
                                state.shutdown(),
                            );
                            if shutdown.await.is_err() {
                                eprintln!("[tauri] worker shutdown timed out after 5s; exiting");
                                std::process::exit(1);
                            }
                        });
                    })
                    .join()
                });
            }
        }),
        Err(error) => {
            eprintln!("[reimagine] failed to build the application: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::desktop_host::{DesktopHostState, default_workspace_path};
    use super::{
        TauriCommandError, agent_bridge_command_error, app_host_command_error,
        worker_switch_command_error,
    };

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("reimagine-tauri-host-{prefix}-{nonce}"))
    }

    #[test]
    fn workflow_save_list_load_roundtrip() {
        let base_path = temp_dir("workflow-persistence");
        let state = tauri::async_runtime::block_on(DesktopHostState::bootstrap(&base_path))
            .expect("desktop host state should bootstrap");

        let workflow_json = serde_json::json!({
            "schema_version": "reimagine.workflow.v1",
            "id": "test-workflow",
            "version": 1,
            "metadata": { "name": "Roundtrip" },
            "interface": { "inputs": [], "outputs": [] },
            "nodes": [{
                "id": "n1",
                "type_id": "model",
                "label": "Model",
                "params": { "seed": { "type": "integer", "value": 42 } }
            }],
            "edges": [],
            "layout": { "nodes": { "n1": { "x": 1.0, "y": 2.0 } } }
        });

        let saved_path =
            tauri::async_runtime::block_on(state.save_workflow("test-workflow", workflow_json))
                .expect("save workflow");
        assert!(
            saved_path.ends_with("test-workflow.json"),
            "expected path to end with test-workflow.json, got {}",
            saved_path.display()
        );

        let summaries = state.list_saved_workflows().expect("list workflows");
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0]["id"], "test-workflow");

        let loaded = tauri::async_runtime::block_on(state.load_workflow_json("test-workflow"))
            .expect("load workflow");
        assert_eq!(loaded["id"], "test-workflow");
        assert_eq!(loaded["metadata"]["name"], "Roundtrip");
        assert_eq!(loaded["layout"]["nodes"]["n1"]["x"], 1.0);
        assert_eq!(
            loaded["nodes"][0]["params"]["seed"],
            serde_json::json!({ "type": "integer", "value": 42 })
        );

        let _ = std::fs::remove_dir_all(&base_path);
    }

    #[test]
    fn default_workspace_path_uses_app_data_workspace_child() {
        let app_data_dir = temp_dir("app-data");

        assert_eq!(
            default_workspace_path(&app_data_dir),
            app_data_dir.join("workspace")
        );
    }

    #[test]
    fn desktop_host_bootstrap_creates_workspace_layout_and_health() {
        let base_path = temp_dir("bootstrap");
        let state = tauri::async_runtime::block_on(DesktopHostState::bootstrap(&base_path))
            .expect("desktop host state should bootstrap");

        assert_eq!(state.workspace_base_path(), base_path);
        assert!(
            state
                .worker_management()
                .list_installed()
                .expect("worker inventory")
                .is_empty()
        );
        assert!(base_path.join("models").is_dir());
        assert!(base_path.join("input").is_dir());
        assert!(base_path.join("output").is_dir());
        assert!(base_path.join("workflows").is_dir());
        assert!(base_path.join("config").is_dir());

        let health = state.health();
        assert_eq!(health.status, "ok");
        assert_eq!(health.workspace, "reimagine-tauri-host");

        let _ = std::fs::remove_dir_all(&base_path);
    }

    #[test]
    fn desktop_host_compute_profile_comes_from_workspace_host() {
        let base_path = temp_dir("compute-profile");
        let state = tauri::async_runtime::block_on(DesktopHostState::bootstrap(&base_path))
            .expect("desktop host state should bootstrap");

        let profile = state.compute_profile();
        assert!(
            profile
                .backend_profiles
                .iter()
                .any(|backend| backend.backend == "candle"),
            "expected compute profile to include the app-host candle profile"
        );
        assert!(
            profile
                .backend_profiles
                .iter()
                .flat_map(|backend| backend.instances.iter())
                .any(|instance| instance.instance == "candle:cpu"),
            "expected compute profile to include the app-host candle CPU instance"
        );

        let _ = std::fs::remove_dir_all(&base_path);
    }

    #[test]
    fn desktop_host_rebootstrap_swaps_backend_and_keeps_workspace() {
        use reimagine_app_host::BackendSelection;

        let base_path = temp_dir("rebootstrap");
        let state = tauri::async_runtime::block_on(DesktopHostState::bootstrap(&base_path))
            .expect("desktop host state should bootstrap");

        let profile =
            tauri::async_runtime::block_on(state.rebootstrap_backend(BackendSelection::Burn))
                .expect("rebootstrap to burn");

        assert!(
            !profile.backend_profiles.is_empty(),
            "rebootstrap must return the rebuilt compute profile"
        );
        assert_eq!(state.workspace_base_path(), base_path);
        assert_eq!(state.health().status, "ok");

        let _ = std::fs::remove_dir_all(&base_path);
    }

    // ─── Catalog command parameter validation tests ────────────────

    #[test]
    fn search_models_query_parsing_default_filters() {
        use reimagine_app_host::dto::ModelFilters;
        use reimagine_model_acquisition::{ModelSearchQuery, SortBy};

        let filters = ModelFilters::default();
        let sort = match filters.sort.as_str() {
            "likes" => SortBy::Likes,
            "trending" => SortBy::Trending,
            "lastModified" => SortBy::LastModified,
            _ => SortBy::Downloads,
        };

        let query = ModelSearchQuery {
            search: None,
            pipeline_tag: filters.pipeline_tag,
            library_name: filters.library_name,
            tags: filters.tags,
            sort,
            limit: filters.limit,
        };

        assert!(query.search.is_none());
        assert!(query.pipeline_tag.is_none());
        assert!(query.library_name.is_none());
        assert!(query.tags.is_empty());
        assert_eq!(query.sort, SortBy::Downloads);
        assert_eq!(query.limit, 20);
    }

    #[test]
    fn search_models_query_parsing_with_text_search() {
        use reimagine_app_host::dto::ModelFilters;
        use reimagine_model_acquisition::{ModelSearchQuery, SortBy};

        let query_text = "stable diffusion".to_string();
        let filters = ModelFilters {
            pipeline_tag: Some("text-to-image".to_string()),
            sort: "likes".to_string(),
            limit: 5,
            ..Default::default()
        };

        let sort = match filters.sort.as_str() {
            "likes" => SortBy::Likes,
            "trending" => SortBy::Trending,
            "lastModified" => SortBy::LastModified,
            _ => SortBy::Downloads,
        };

        let query = ModelSearchQuery {
            search: if query_text.is_empty() {
                None
            } else {
                Some(query_text)
            },
            pipeline_tag: filters.pipeline_tag,
            library_name: filters.library_name,
            tags: filters.tags,
            sort,
            limit: filters.limit,
        };

        assert_eq!(query.search.as_deref(), Some("stable diffusion"));
        assert_eq!(query.pipeline_tag.as_deref(), Some("text-to-image"));
        assert_eq!(query.sort, SortBy::Likes);
        assert_eq!(query.limit, 5);
    }

    #[test]
    fn search_models_query_parsing_empty_search_becomes_none() {
        use reimagine_app_host::dto::ModelFilters;
        use reimagine_model_acquisition::{ModelSearchQuery, SortBy};

        let query_text = String::new();
        let filters = ModelFilters::default();
        let sort = SortBy::Downloads;

        let query = ModelSearchQuery {
            search: if query_text.is_empty() {
                None
            } else {
                Some(query_text)
            },
            pipeline_tag: filters.pipeline_tag,
            library_name: filters.library_name,
            tags: filters.tags,
            sort,
            limit: filters.limit,
        };

        assert!(query.search.is_none());
    }

    #[test]
    fn model_card_dto_conversion() {
        use reimagine_app_host::dto::ModelCardDto;
        use reimagine_model_acquisition::{
            ModelCard, ModelCardData, ModelCatalogEntry, ModelRepoFormat,
        };

        let card = ModelCard {
            entry: ModelCatalogEntry {
                id: "org/model".to_string(),
                author: Some("org".to_string()),
                pipeline_tag: Some("text-to-image".to_string()),
                tags: vec!["diffusers".to_string()],
                downloads: 1000,
                likes: 50,
                last_modified: None,
                private: false,
            },
            siblings: vec![],
            card_data: Some(ModelCardData {
                model_summary: Some("A test model".to_string()),
                ..Default::default()
            }),
            detected_format: ModelRepoFormat::Diffusers,
            component_mapping: None,
            estimated_download_size: 1_000_000,
        };

        let dto = ModelCardDto::from(card);
        assert_eq!(dto.entry.id, "org/model");
        assert_eq!(dto.detected_format, "Diffusers");
        assert_eq!(dto.estimated_download_size, 1_000_000);
        assert_eq!(dto.model_summary.as_deref(), Some("A test model"));
        assert_eq!(dto.file_count, 0);
        assert!(dto.components.is_empty());
    }

    #[test]
    fn model_catalog_entry_dto_conversion() {
        use reimagine_app_host::dto::ModelCatalogEntryDto;
        use reimagine_model_acquisition::ModelCatalogEntry;

        let entry = ModelCatalogEntry {
            id: "org/model".to_string(),
            author: Some("org".to_string()),
            pipeline_tag: Some("text-to-image".to_string()),
            tags: vec!["diffusers".to_string(), "safetensors".to_string()],
            downloads: 5000,
            likes: 100,
            last_modified: Some("2025-01-01T00:00:00Z".to_string()),
            private: false,
        };

        let dto = ModelCatalogEntryDto::from(entry);
        assert_eq!(dto.id, "org/model");
        assert_eq!(dto.author.as_deref(), Some("org"));
        assert_eq!(dto.downloads, 5000);
        assert_eq!(dto.likes, 100);
        assert_eq!(dto.tags.len(), 2);
        assert!(!dto.private);
    }

    #[test]
    fn model_filters_default_values() {
        use reimagine_app_host::dto::ModelFilters;

        let filters = ModelFilters::default();
        assert!(filters.pipeline_tag.is_none());
        assert!(filters.library_name.is_none());
        assert!(filters.tags.is_empty());
        assert_eq!(filters.sort, "downloads");
        assert_eq!(filters.limit, 20);
    }

    // ─── BE-31 structured error payload tests ────────────────────

    #[test]
    fn command_error_serializes_structured_payload() {
        let error = TauriCommandError::command("boom");
        let value = serde_json::to_value(&error).expect("serialize");
        assert_eq!(value["code"], "command_failed");
        assert_eq!(value["message"], "boom");
        assert!(
            value.get("details").is_none(),
            "details should be omitted when absent"
        );
    }

    #[test]
    fn bootstrap_error_serializes_with_bootstrap_code() {
        let error = TauriCommandError::bootstrap("setup failed");
        let value = serde_json::to_value(&error).expect("serialize");
        assert_eq!(value["code"], "bootstrap_failed");
    }

    #[test]
    fn app_host_errors_map_to_structured_codes() {
        use reimagine_app_host::AppHostError;
        use reimagine_app_host::AppHostErrorCode;
        use reimagine_core::model::RunId;

        let error = AppHostError::UnknownRun {
            run_id: RunId::new("run_x"),
        };
        let mapped = app_host_command_error(error);
        assert_eq!(mapped.code, AppHostErrorCode::NotFound);
        assert_eq!(
            mapped.details.as_ref().expect("run details")["run_id"],
            "run_x"
        );

        let mapped = app_host_command_error(AppHostError::WorkflowJson {
            path: std::path::PathBuf::new(),
            message: "bad json".to_owned(),
        });
        assert_eq!(mapped.code, AppHostErrorCode::WorkflowInvalid);
    }

    #[test]
    fn unknown_agent_provider_keeps_friendly_message_and_code() {
        use reimagine_app_host::AppHostError;
        use reimagine_app_host::AppHostErrorCode;

        let mapped = app_host_command_error(AppHostError::UnknownAgentProvider {
            provider: reimagine_agent_harness::ProviderName::new("openai"),
        });
        assert_eq!(mapped.code, AppHostErrorCode::UnknownProvider);
        assert!(mapped.message.contains("Settings"));
    }

    #[test]
    fn agent_bridge_errors_map_to_command_failed() {
        use super::AgentBridgeError;
        use reimagine_app_host::AppHostErrorCode;

        let protocol = agent_bridge_command_error(AgentBridgeError::Protocol {
            code: -32602,
            message: "Provider 'openai' is not configured. Add a provider in Settings.".to_owned(),
        });
        assert_eq!(protocol.code, AppHostErrorCode::CommandFailed);
        assert_eq!(
            protocol.message,
            "Provider 'openai' is not configured. Add a provider in Settings."
        );

        let not_found =
            agent_bridge_command_error(AgentBridgeError::DaemonNotFound { searched: vec![] });
        assert_eq!(not_found.code, AppHostErrorCode::CommandFailed);
        assert!(not_found.message.contains("not installed"));
    }

    #[test]
    fn worker_switch_errors_map_to_worker_unavailable() {
        use reimagine_app_host::AppHostErrorCode;
        use reimagine_app_host::WorkerSwitchError;

        let mapped = worker_switch_command_error(WorkerSwitchError::NoActiveWorker);
        assert_eq!(mapped.code, AppHostErrorCode::WorkerUnavailable);
        let value = serde_json::to_value(&mapped).expect("serialize");
        assert_eq!(value["code"], "worker_unavailable");
    }

    #[test]
    fn switch_deadline_validates_input() {
        assert!(super::switch_deadline(None).is_ok());
        assert!(super::switch_deadline(Some(5.0)).is_ok());
        assert!(super::switch_deadline(Some(0.0)).is_err());
        assert!(super::switch_deadline(Some(-1.0)).is_err());
    }
}
