mod desktop_host;
mod event_hub;

use desktop_host::{DesktopHostState, default_workspace_path};
use reimagine_app_host::dto::{ComputeProfileDto, HealthResponse};
use serde::Serialize;
use tauri::Manager;

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
        .invoke_handler(tauri::generate_handler![health, get_compute_profile])
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
