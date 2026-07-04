mod desktop_host;
mod event_hub;

use desktop_host::{default_workspace_path, DesktopHostState};
use event_hub::RunEventPayload;
use reimagine_app_host::dto::{
    ArtifactMetadataDto, ComputeProfileDto, HealthResponse, ModelInfoDto, NodeDefDto,
    RunWorkflowResponse,
};
use serde::Serialize;
use tauri::{ipc::Channel, Manager};

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
}

impl std::fmt::Display for TauriCommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for TauriCommandError {}

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
            health,
            get_compute_profile,
            get_node_defs,
            list_models,
            run_workflow,
            cancel_run,
            resolve_artifact,
            open_artifact,
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
