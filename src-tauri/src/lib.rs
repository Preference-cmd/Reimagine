mod desktop_host;
mod agent_event_hub;
mod event_hub;

use desktop_host::{DesktopHostState, default_workspace_path};
use event_hub::RunEventPayload;
use reimagine_app_host::{AppHostError, dto::{
    AgentEventPayload, AgentSessionInfo, AgentTurnResponse, ArtifactMetadataDto,
    ComputeProfileDto, HealthResponse, ModelInfoDto, NodeDefDto, RunWorkflowResponse,
}};
use reimagine_core::command::CommandResult;
use serde::Serialize;
use tauri::{Manager, ipc::Channel};

#[derive(Debug, Clone, Serialize)]
struct TauriCommandError {
    code: &'static str,
    message: String,
}

impl TauriCommandError {
    fn bootstrap(message: impl Into<String>) -> Self {
        Self {
            code: "bootstrap_failed",
            message: message.into(),
        }
    }

    fn command(message: impl Into<String>) -> Self {
        Self {
            code: "command_failed",
            message: message.into(),
        }
    }

    fn unknown_provider(provider: impl Into<String>) -> Self {
        Self {
            code: "unknown_provider",
            message: format!("Provider '{}' is not configured. Add a provider in Settings.", provider.into()),
        }
    }
}

impl std::fmt::Display for TauriCommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for TauriCommandError {}

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
    state
        .list_models()
        .await
        .map_err(|e| TauriCommandError::command(e.to_string()))
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
        .map_err(|e| TauriCommandError::command(e.to_string()))
}

#[tauri::command]
async fn cancel_run(
    state: tauri::State<'_, DesktopHostState>,
    run_id: String,
) -> Result<(), TauriCommandError> {
    state
        .cancel_run(&run_id)
        .map_err(|e| TauriCommandError::command(e.to_string()))
}

#[tauri::command]
async fn resolve_artifact(
    state: tauri::State<'_, DesktopHostState>,
    artifact_id: String,
) -> Result<ArtifactMetadataDto, TauriCommandError> {
    state
        .resolve_artifact(&artifact_id)
        .map_err(|e| TauriCommandError::command(e.to_string()))
}

#[tauri::command]
async fn open_artifact(
    state: tauri::State<'_, DesktopHostState>,
    artifact_id: String,
) -> Result<(), TauriCommandError> {
    state
        .open_artifact(&artifact_id)
        .map_err(|e| TauriCommandError::command(e.to_string()))
}

// ─── Agent commands ──────────────────────────────────────────────

/// Create a new agent session.
///
/// `mode` must be "Agent" or "Build".
/// `provider` must match a registered provider in the catalog.
#[tauri::command]
fn create_agent_session(
    state: tauri::State<'_, DesktopHostState>,
    mode: String,
    provider: String,
) -> Result<AgentSessionInfo, TauriCommandError> {
    state
        .create_agent_session(mode, provider)
        .map_err(|e| match e {
            AppHostError::UnknownAgentProvider { provider } => {
                TauriCommandError::unknown_provider(provider.to_string())
            }
            AppHostError::UnknownAgentMode { mode } => {
                TauriCommandError::command(format!("unknown agent mode: {mode}"))
            }
            _ => TauriCommandError::command(e.to_string()),
        })
}

/// Execute a single agent turn with live event streaming.
///
/// `session_id` must be a valid existing session.
/// `turn_id` is a caller-generated id for this turn (idempotent retries).
/// `model` is the model name string for the registered provider.
/// `input` is a JSON array of `{ role, content }` message objects.
#[tauri::command]
async fn agent_turn(
    state: tauri::State<'_, DesktopHostState>,
    session_id: String,
    turn_id: String,
    model: String,
    input: serde_json::Value,
    channel: Channel<AgentEventPayload>,
) -> Result<AgentTurnResponse, TauriCommandError> {
    state
        .agent_turn(session_id, turn_id, model, input, channel)
        .await
        .map_err(|e| TauriCommandError::command(e.to_string()))
}

/// List available agent providers for the UI selector.
#[tauri::command]
fn list_agent_providers(
    state: tauri::State<'_, DesktopHostState>,
) -> Result<Vec<String>, TauriCommandError> {
    state
        .list_agent_providers()
        .map_err(|e| TauriCommandError::command(e.to_string()))
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
        .map_err(|e| TauriCommandError::command(e.to_string()))
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
        .map_err(|e| TauriCommandError::command(e.to_string()))
}

/// Approve a pending workflow proposal (human approval of build-mode output).
#[tauri::command]
fn approve_proposal(
    state: tauri::State<'_, DesktopHostState>,
    workflow_id: String,
) -> Result<CommandResult, TauriCommandError> {
    state
        .approve_proposal(workflow_id)
        .map_err(|e| TauriCommandError::command(e.to_string()))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let workspace_path = default_workspace_path(
                app.path()
                    .app_data_dir()
                    .map_err(|error| TauriCommandError::bootstrap(error.to_string()))?,
            );
            let state =
                tauri::async_runtime::block_on(DesktopHostState::bootstrap(&workspace_path))
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::desktop_host::{DesktopHostState, default_workspace_path};

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("reimagine-tauri-host-{prefix}-{nonce}"))
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

        assert_eq!(state.workspace_base_path(), base_path.as_path());
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
}
