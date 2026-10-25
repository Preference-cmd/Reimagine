use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use reimagine_core::event::{Event, EventBus};
use serde::Serialize;

use reimagine_config::{AppConfig, AppPaths, ConfigDocument, ConfigHandle, ConfigReport};
use reimagine_model_acquisition::hf::provider::AcquisitionProgressSink;
use reimagine_model_acquisition::{
    AcquisitionReport, ModelAcquisitionConfig, ModelAcquisitionRequest,
};

use crate::AigcResult;

/// Orchestrates model downloads from HuggingFace Hub.
#[derive(Debug, Clone)]
pub struct ModelAcquisitionService {
    models_dir: PathBuf,
    config_handle: ConfigHandle<ModelAcquisitionConfig>,
    event_bus: EventBus,
}

impl ModelAcquisitionService {
    pub fn new(paths: AppPaths, app_config: &AppConfig) -> Self {
        Self::with_event_bus(paths, app_config, EventBus::new())
    }

    pub fn with_event_bus(paths: AppPaths, app_config: &AppConfig, event_bus: EventBus) -> Self {
        let config_handle = app_config
            .config::<ModelAcquisitionConfig>()
            .expect("ModelAcquisitionConfig handle creation should always succeed");
        Self {
            models_dir: paths.models_dir().to_path_buf(),
            config_handle,
            event_bus,
        }
    }

    pub async fn load_config(&self) -> AigcResult<ModelAcquisitionConfig> {
        let (cfg, _report) = self
            .config_handle
            .load()
            .await
            .map_err(crate::AigcError::BootstrapConfig)?;
        Ok(cfg)
    }

    pub async fn save_config(&self, cfg: &ModelAcquisitionConfig) -> AigcResult<ConfigReport> {
        let report = self
            .config_handle
            .save(cfg)
            .await
            .map_err(crate::AigcError::BootstrapConfig)?;
        Ok(report)
    }

    /// Download a HuggingFace model and publish its progress on the event bus
    /// as `aigc.download.{started,progress,completed,failed}` module events
    /// whose payload carries `download_id`.
    ///
    /// `estimated_bytes` seeds `totalBytes` until the download reports the
    /// real size.
    pub fn acquire_with_progress_events(
        self: Arc<Self>,
        download_id: String,
        request: ModelAcquisitionRequest,
        estimated_bytes: Option<u64>,
    ) -> Pin<Box<dyn Future<Output = AigcResult<AcquisitionReport>> + Send>> {
        let sink = Arc::new(DownloadEvents::new(
            self.event_bus.clone(),
            download_id,
            estimated_bytes,
        ));
        sink.publish("started", None);
        let download = Arc::clone(&self).acquire(request, Some(sink.clone()));
        Box::pin(async move {
            let result = download.await;
            match &result {
                Ok(report) => sink.done(report),
                Err(error) => sink.publish("failed", Some(error.to_string())),
            }
            result
        })
    }

    /// Download a HuggingFace model.
    ///
    /// Returns a `Pin<Box<dyn Future + Send>>` so the caller does not need to
    /// worry about hf-hub's internal !Send borrows.
    ///
    /// When `request.auto_detect` is true and `allow_patterns` is empty,
    /// automatically detects the repository format and builds optimal download
    /// patterns before downloading.
    pub fn acquire(
        self: Arc<Self>,
        request: ModelAcquisitionRequest,
        progress_sink: Option<Arc<dyn AcquisitionProgressSink>>,
    ) -> Pin<Box<dyn Future<Output = AigcResult<AcquisitionReport>> + Send>> {
        let models_dir = self.models_dir.clone();
        let cfg = match load_config_sync(&self.config_handle) {
            Ok(c) => c,
            Err(e) => return Box::pin(std::future::ready(Err(e))),
        };
        let models_dir2 = models_dir.clone();
        let auto_detect = request.auto_detect;
        let explicit_patterns_empty = request.allow_patterns.is_empty();
        let should_resolve = auto_detect && explicit_patterns_empty;
        Box::pin(async move {
            // If auto-detect is enabled and no explicit patterns, resolve them first.
            let (resolved_format, request) = if should_resolve {
                let async_client = reimagine_model_acquisition::build_hf_client(&cfg);
                let resolved = reimagine_model_acquisition::resolve_download_patterns(
                    &async_client,
                    request.repo_id.as_str(),
                    request.revision.as_str(),
                    &request.allow_patterns,
                )
                .await
                .map_err(crate::AigcError::from)?;
                let format_name = Some(format!("{:?}", resolved.format));
                let mut req = request;
                req.allow_patterns = resolved.patterns;
                (format_name, req)
            } else {
                (None, request)
            };

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
                // The snapshot lands in `stg_dir` (its `local_dir`).
                repo.snapshot_download()
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
                report.detected_format = resolved_format;
                for file in reimagine_model_acquisition::promoted_files(&target_dir)? {
                    report.add_file(file);
                }
                Ok::<_, reimagine_model_acquisition::ModelAcquisitionError>(report)
            })
            .await
            .map_err(|join| crate::AigcError::Io {
                path: models_dir2,
                message: format!("model download task panicked: {join}"),
            })?;
            report.map_err(crate::AigcError::from)
        })
    }
}

fn load_config_sync(
    handle: &ConfigHandle<ModelAcquisitionConfig>,
) -> AigcResult<ModelAcquisitionConfig> {
    use std::io::Read;
    let path = handle.path();
    match std::fs::File::open(path) {
        Ok(mut file) => {
            let mut contents = String::new();
            file.read_to_string(&mut contents)
                .map_err(|e| crate::AigcError::Io {
                    path: path.to_path_buf(),
                    message: e.to_string(),
                })?;
            serde_json::from_str(&contents).map_err(|e| {
                crate::AigcError::BootstrapConfig(reimagine_config::ConfigError::JsonInvalid {
                    key: Some(ModelAcquisitionConfig::KEY.to_owned()),
                    path: path.to_path_buf(),
                    message: e.to_string(),
                })
            })
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(ModelAcquisitionConfig::default()),
        Err(e) => Err(crate::AigcError::Io {
            path: path.to_path_buf(),
            message: e.to_string(),
        }),
    }
}

/// Payload of an `aigc.download.*` event.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DownloadProgress<'a> {
    download_id: &'a str,
    /// Bytes transferred so far, summed over files.
    bytes_downloaded: u64,
    total_bytes: Option<u64>,
    message: Option<String>,
}

/// Publishes one download's progress on the event bus.
struct DownloadEvents {
    event_bus: EventBus,
    download_id: String,
    /// Latest transferred byte count per file; a file reports many times.
    file_bytes: Mutex<HashMap<String, u64>>,
    total_bytes: AtomicU64,
}

impl DownloadEvents {
    fn new(event_bus: EventBus, download_id: String, estimated_bytes: Option<u64>) -> Self {
        Self {
            event_bus,
            download_id,
            file_bytes: Mutex::new(HashMap::new()),
            total_bytes: AtomicU64::new(estimated_bytes.unwrap_or(0)),
        }
    }

    fn bytes_downloaded(&self) -> u64 {
        self.file_bytes
            .lock()
            .expect("download progress lock")
            .values()
            .sum()
    }

    fn publish(&self, stage: &str, message: Option<String>) {
        let total_bytes = self.total_bytes.load(Ordering::Relaxed);
        let payload = DownloadProgress {
            download_id: &self.download_id,
            bytes_downloaded: self.bytes_downloaded(),
            total_bytes: (total_bytes > 0).then_some(total_bytes),
            message,
        };
        let payload = serde_json::to_value(payload).expect("download progress serializes");
        self.event_bus
            .publish(Event::module(format!("aigc.download.{stage}"), payload));
    }
}

impl AcquisitionProgressSink for DownloadEvents {
    fn file_done(&self, relative_path: &str, bytes: u64, outcome: &str) {
        {
            let mut files = self.file_bytes.lock().expect("download progress lock");
            let seen = files.entry(relative_path.to_owned()).or_default();
            *seen = (*seen).max(bytes);
        }
        self.publish("progress", Some(format!("file {outcome}: {relative_path}")));
    }

    fn done(&self, report: &AcquisitionReport) {
        let bytes = self.bytes_downloaded().max(report.total_bytes);
        self.total_bytes.store(bytes, Ordering::Relaxed);
        self.publish("completed", None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next_module_event(
        rx: &mut tokio::sync::broadcast::Receiver<Event>,
    ) -> (String, serde_json::Value) {
        match rx.try_recv().expect("an event was published") {
            Event::Module { kind, payload } => (kind, payload),
            other => panic!("expected a module event, got {other:?}"),
        }
    }

    #[test]
    fn download_progress_is_published_as_aigc_module_events() {
        let bus = EventBus::new();
        let mut rx = bus.subscribe();
        let events = DownloadEvents::new(bus, "dl-1".to_owned(), Some(1000));

        events.publish("started", None);
        events.file_done("a.safetensors", 300, "downloaded");
        events.file_done("b.safetensors", 100, "downloading");
        events.file_done("b.safetensors", 200, "downloaded");
        // The report lists no files: completion keeps the bytes seen so far.
        let report = AcquisitionReport::new("huggingface", "org/model", "main", "org/model");
        events.done(&report);

        let (kind, payload) = next_module_event(&mut rx);
        assert_eq!(kind, "aigc.download.started");
        assert_eq!(payload["downloadId"], "dl-1");
        assert_eq!(payload["totalBytes"], 1000);

        let (kind, payload) = next_module_event(&mut rx);
        assert_eq!(kind, "aigc.download.progress");
        assert_eq!(payload["bytesDownloaded"], 300);
        let (_, payload) = next_module_event(&mut rx);
        assert_eq!(payload["bytesDownloaded"], 400);
        let (_, payload) = next_module_event(&mut rx);
        assert_eq!(
            payload["bytesDownloaded"], 500,
            "a file's repeated reports are not summed"
        );

        let (kind, payload) = next_module_event(&mut rx);
        assert_eq!(kind, "aigc.download.completed");
        assert_eq!(payload["bytesDownloaded"], 500);
        assert_eq!(payload["totalBytes"], 500);
    }
}
