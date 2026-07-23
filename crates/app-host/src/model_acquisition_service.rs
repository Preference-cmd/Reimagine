use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use reimagine_config::{AppConfig, AppPaths, ConfigDocument, ConfigHandle, ConfigReport};
use reimagine_model_acquisition::hf::provider::AcquisitionProgressSink;
use reimagine_model_acquisition::{
    AcquisitionReport, ModelAcquisitionConfig, ModelAcquisitionRequest,
};

use crate::AppHostResult;

/// Orchestrates model downloads from HuggingFace Hub.
#[derive(Debug, Clone)]
pub struct ModelAcquisitionService {
    models_dir: PathBuf,
    config_handle: ConfigHandle<ModelAcquisitionConfig>,
}

impl ModelAcquisitionService {
    pub fn new(paths: AppPaths, app_config: &AppConfig) -> Self {
        let config_handle = app_config
            .config::<ModelAcquisitionConfig>()
            .expect("ModelAcquisitionConfig handle creation should always succeed");
        Self {
            models_dir: paths.models_dir().to_path_buf(),
            config_handle,
        }
    }

    pub async fn load_config(&self) -> AppHostResult<ModelAcquisitionConfig> {
        let (cfg, _report) = self
            .config_handle
            .load()
            .await
            .map_err(crate::AppHostError::BootstrapConfig)?;
        Ok(cfg)
    }

    pub async fn save_config(&self, cfg: &ModelAcquisitionConfig) -> AppHostResult<ConfigReport> {
        let report = self
            .config_handle
            .save(cfg)
            .await
            .map_err(crate::AppHostError::BootstrapConfig)?;
        Ok(report)
    }

    /// Download a HuggingFace model.
    ///
    /// Returns a `Pin<Box<dyn Future + Send>>` so the caller does not need to
    /// worry about hf-hub's internal !Send borrows.
    pub fn acquire(
        self: Arc<Self>,
        request: ModelAcquisitionRequest,
        progress_sink: Option<Arc<dyn AcquisitionProgressSink>>,
    ) -> Pin<Box<dyn Future<Output = AppHostResult<AcquisitionReport>> + Send>> {
        let models_dir = self.models_dir.clone();
        let cfg = match load_config_sync(&self.config_handle) {
            Ok(c) => c,
            Err(e) => return Box::pin(std::future::ready(Err(e))),
        };
        let models_dir2 = models_dir.clone();
        Box::pin(async move {
            // Run the download on a dedicated spawned task.  hf-hub uses
            // std::sync::Mutex internally which makes the borrow-chain
            // !Send when &references cross .await points.  By moving all
            // owned values into spawn_blocking we side-step the issue.
            let report = tokio::task::spawn_blocking(move || {
                // Build a sync client using hf_hub's blocking API.
                let mut builder =
                    hf_hub::HFClientBuilder::new().endpoint(cfg.huggingface.endpoint.clone());
                if let Some(ref token) = cfg.huggingface.token {
                    builder = builder.token(token.clone());
                }
                let sync_client = builder.build_sync().map_err(|e| {
                    reimagine_model_acquisition::ModelAcquisitionError::ConfigInvalid {
                        key: "huggingface".to_owned(),
                        reason: e.to_string(),
                    }
                })?;
                let stg_dir = reimagine_model_acquisition::staging_dir(
                    &models_dir,
                    &request.provider,
                    &request.repo_id,
                    &request.revision,
                );
                let target_dir = models_dir.join(request.target_relative_dir.as_path());
                std::fs::create_dir_all(&stg_dir).map_err(|e| {
                    reimagine_model_acquisition::ModelAcquisitionError::Io {
                        path: stg_dir.clone(),
                        message: e.to_string(),
                    }
                })?;

                // Wrap the progress sink as an hf-hub ProgressHandler.
                let progress: Option<hf_hub::progress::Progress> = progress_sink.map(|sink| {
                    hf_hub::progress::Progress::new(
                        reimagine_model_acquisition::ProgressSinkBridge::new(sink),
                    )
                });

                let repo = sync_client.model(request.repo_id.namespace(), request.repo_id.name());
                let allow_patterns: Option<Vec<String>> = if request.allow_patterns.is_empty() {
                    None
                } else {
                    Some(request.allow_patterns.as_slice().to_vec())
                };
                let snapshot_path = repo
                    .snapshot_download()
                    .maybe_revision(Some(request.revision.as_str().to_string()))
                    .maybe_allow_patterns(allow_patterns)
                    .maybe_local_dir(Some(stg_dir.clone()))
                    .force_download(matches!(
                        request.overwrite_policy,
                        reimagine_model_acquisition::OverwritePolicy::Overwrite
                    ))
                    .max_workers(8)
                    .maybe_progress(progress)
                    .send()
                    .map_err(
                        |e| reimagine_model_acquisition::ModelAcquisitionError::Hub {
                            repo: request.repo_id.to_string(),
                            message: e.to_string(),
                        },
                    )?;
                // Promote staging to target.
                let rt = tokio::runtime::Handle::current();
                rt.block_on(reimagine_model_acquisition::promote_staged(
                    &stg_dir,
                    &target_dir,
                    &request.overwrite_policy,
                ))?;
                // Build report.
                let mut report = AcquisitionReport::new(
                    "huggingface",
                    request.repo_id.to_string(),
                    request.revision.as_str(),
                    request.target_relative_dir.as_os_str().to_string_lossy(),
                );
                if snapshot_path.exists() {
                    for entry in std::fs::read_dir(&snapshot_path).map_err(|e| {
                        reimagine_model_acquisition::ModelAcquisitionError::Io {
                            path: snapshot_path.clone(),
                            message: e.to_string(),
                        }
                    })? {
                        let entry = entry.map_err(|e| {
                            reimagine_model_acquisition::ModelAcquisitionError::Io {
                                path: snapshot_path.clone(),
                                message: e.to_string(),
                            }
                        })?;
                        let md = entry.metadata().map_err(|e| {
                            reimagine_model_acquisition::ModelAcquisitionError::Io {
                                path: entry.path(),
                                message: e.to_string(),
                            }
                        })?;
                        if md.is_file() {
                            let relative = entry.file_name().to_string_lossy().to_string();
                            report.add_file(reimagine_model_acquisition::AcquisitionFileEntry {
                                relative_path: relative,
                                bytes: md.len(),
                                outcome:
                                    reimagine_model_acquisition::AcquisitionOutcome::Downloaded,
                            });
                        }
                    }
                }
                Ok::<_, reimagine_model_acquisition::ModelAcquisitionError>(report)
            })
            .await
            .map_err(|join| crate::AppHostError::Io {
                path: models_dir2,
                message: format!("model download task panicked: {join}"),
            })?;
            report.map_err(crate::AppHostError::from)
        })
    }
}

fn load_config_sync(
    handle: &ConfigHandle<ModelAcquisitionConfig>,
) -> AppHostResult<ModelAcquisitionConfig> {
    use std::io::Read;
    let path = handle.path();
    match std::fs::File::open(path) {
        Ok(mut file) => {
            let mut contents = String::new();
            file.read_to_string(&mut contents)
                .map_err(|e| crate::AppHostError::Io {
                    path: path.to_path_buf(),
                    message: e.to_string(),
                })?;
            serde_json::from_str(&contents).map_err(|e| {
                crate::AppHostError::BootstrapConfig(reimagine_config::ConfigError::JsonInvalid {
                    key: Some(ModelAcquisitionConfig::KEY.to_owned()),
                    path: path.to_path_buf(),
                    message: e.to_string(),
                })
            })
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(ModelAcquisitionConfig::default()),
        Err(e) => Err(crate::AppHostError::Io {
            path: path.to_path_buf(),
            message: e.to_string(),
        }),
    }
}
