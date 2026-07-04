use std::path::{Path, PathBuf};

use reimagine_agent::WorkspaceScope;
use reimagine_app_host::dto::{
    ComputeProfileDto, HealthResponse, ModelInfoDto, NodeCatalogResponse,
};
use reimagine_app_host::{AppHost, AppHostError, WorkspaceHost};
use reimagine_config::AppPaths;
use reimagine_runtime::{RunEventSink, VecRunEventSink};

const WORKSPACE_SCOPE: &str = "reimagine-tauri-host";

#[derive(Debug, Clone)]
pub struct DesktopHostState {
    app_host: AppHost,
}

impl DesktopHostState {
    pub async fn bootstrap(workspace_base_path: impl Into<PathBuf>) -> Result<Self, AppHostError> {
        let workspace_base_path = workspace_base_path.into();
        AppPaths::new(&workspace_base_path).ensure_all().await?;
        let event_sink: std::sync::Arc<dyn RunEventSink> =
            std::sync::Arc::new(VecRunEventSink::new());
        let workspace = WorkspaceHost::try_with_defaults_and_event_sink(
            WorkspaceScope::new(WORKSPACE_SCOPE),
            &workspace_base_path,
            event_sink,
        )
        .await?;

        Ok(Self {
            app_host: AppHost::new(workspace),
        })
    }

    #[cfg(test)]
    pub fn workspace_base_path(&self) -> &Path {
        self.app_host.workspace().base_path()
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
    ///
    /// Loads the manifest from disk on first call; subsequent calls
    /// return the cached manifest until it is explicitly refreshed.
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
}

pub fn default_workspace_path(app_data_dir: impl AsRef<Path>) -> PathBuf {
    app_data_dir.as_ref().join("workspace")
}
