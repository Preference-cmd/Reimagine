use std::path::{Path, PathBuf};
use std::sync::Arc;

use reimagine_agent::WorkspaceScope;
use reimagine_app_host::dto::{
    ComputeProfileDto, HealthResponse, ModelInfoDto, NodeCatalogResponse, RunWorkflowResponse,
};
use reimagine_app_host::{AppHost, AppHostError, WorkspaceHost};
use reimagine_config::AppPaths;
use reimagine_core::workflow::Workflow;
use reimagine_runtime::{BoxedRunEventSink, RunEventSink};
use tauri::ipc::Channel;

use crate::event_hub::{RunEventPayload, TauriRunEventHub};

const WORKSPACE_SCOPE: &str = "reimagine-tauri-host";

#[derive(Debug, Clone)]
pub struct DesktopHostState {
    app_host: AppHost,
    event_hub: Arc<TauriRunEventHub>,
}

impl DesktopHostState {
    pub async fn bootstrap(workspace_base_path: impl Into<PathBuf>) -> Result<Self, AppHostError> {
        let workspace_base_path = workspace_base_path.into();
        AppPaths::new(&workspace_base_path).ensure_all().await?;
        let event_hub = Arc::new(TauriRunEventHub::new());
        let event_sink: BoxedRunEventSink = event_hub.clone();
        let workspace = WorkspaceHost::try_with_defaults_and_event_sink(
            WorkspaceScope::new(WORKSPACE_SCOPE),
            &workspace_base_path,
            event_sink,
        )
        .await?;

        Ok(Self {
            app_host: AppHost::new(workspace),
            event_hub,
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
        let workflow: Workflow = serde_json::from_value(workflow_value)
            .map_err(|e| AppHostError::WorkflowJson {
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
}

pub fn default_workspace_path(app_data_dir: impl AsRef<Path>) -> PathBuf {
    app_data_dir.as_ref().join("workspace")
}
