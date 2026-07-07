use std::sync::Arc;

use hf_hub::progress::{self, DownloadEvent, FileStatus, Progress, ProgressHandler};

use crate::error::ModelAcquisitionResult;
use crate::report::{AcquisitionFileEntry, AcquisitionOutcome, AcquisitionReport};
use crate::request::ModelAcquisitionRequest;
use crate::staging::{self, staging_dir};

/// A progress callback that the app-host can implement to receive progress updates.
pub trait AcquisitionProgressSink: Send + Sync {
    /// Called when the download begins.
    fn started(&self, _repo_id: &str, _revision: &str) {}
    /// Called per-file as each file download completes or is skipped.
    fn file_done(&self, relative_path: &str, bytes: u64, outcome: &str);
    /// Called once when the acquisition finishes.
    fn done(&self, report: &AcquisitionReport);
}

/// Bridges an `AcquisitionProgressSink` to hf-hub's `ProgressHandler` trait.
pub struct ProgressSinkBridge {
    sink: Arc<dyn AcquisitionProgressSink>,
}

impl ProgressSinkBridge {
    pub fn new(sink: Arc<dyn AcquisitionProgressSink>) -> Self {
        Self { sink }
    }
}

impl ProgressHandler for ProgressSinkBridge {
    fn on_progress(&self, event: &progress::ProgressEvent) {
        if let progress::ProgressEvent::Download(DownloadEvent::Progress { files }) = event {
            for f in files {
                let outcome = match f.status {
                    FileStatus::Complete => "downloaded",
                    _ => "cached", // Started / InProgress are transient
                };
                self.sink.file_done(&f.filename, f.total_bytes, outcome);
            }
        }
    }
}

/// Downloads a model from HuggingFace Hub.
pub struct HuggingFaceProvider {
    client: hf_hub::HFClient,
}

impl HuggingFaceProvider {
    pub fn new(config: &crate::config::ModelAcquisitionConfig) -> Self {
        let client = crate::hf::client::build_hf_client(config);
        Self { client }
    }

    /// Download a HuggingFace model according to the request.
    ///
    /// Takes ownership of `self`, `base_models_dir`, and `request` so the
    /// returned future is unconditionally `Send` (no borrowed references
    /// captured across `.await` points).
    pub async fn download(
        self,
        base_models_dir: std::path::PathBuf,
        request: ModelAcquisitionRequest,
        sink: Option<Arc<dyn AcquisitionProgressSink>>,
    ) -> ModelAcquisitionResult<AcquisitionReport> {
        let target_dir = base_models_dir.join(request.target_relative_dir.as_path());
        let stg_dir = staging_dir(
            &base_models_dir,
            &request.provider,
            &request.repo_id,
            &request.revision,
        );

        tokio::fs::create_dir_all(&stg_dir).await.map_err(|e| {
            crate::ModelAcquisitionError::Io {
                path: stg_dir.clone(),
                message: e.to_string(),
            }
        })?;

        let repo = self
            .client
            .model(request.repo_id.namespace(), request.repo_id.name());

        let progress: Option<Progress> = sink
            .clone()
            .map(|s| Progress::new(ProgressSinkBridge { sink: s }));

        let allow_patterns: Option<Vec<String>> = if request.allow_patterns.is_empty() {
            None
        } else {
            Some(request.allow_patterns.as_slice().to_vec())
        };

        let snapshot_path: std::path::PathBuf = repo
            .snapshot_download()
            .maybe_revision(Some(request.revision.as_str().to_string()))
            .maybe_allow_patterns(allow_patterns)
            .maybe_local_dir(Some(stg_dir.clone()))
            .force_download(matches!(
                request.overwrite_policy,
                crate::OverwritePolicy::Overwrite
            ))
            .maybe_progress(progress)
            .send()
            .await
            .map_err(|e| crate::ModelAcquisitionError::Hub {
                repo: request.repo_id.to_string(),
                message: e.to_string(),
            })?;

        staging::promote_staged(&snapshot_path, &target_dir, &request.overwrite_policy).await?;

        let mut report = AcquisitionReport::new(
            "huggingface",
            request.repo_id.to_string(),
            request.revision.as_str(),
            request.target_relative_dir.as_os_str().to_string_lossy(),
        );

        if snapshot_path.exists() {
            let mut entries = tokio::fs::read_dir(&snapshot_path).await.map_err(|e| {
                crate::ModelAcquisitionError::Io {
                    path: snapshot_path.clone(),
                    message: e.to_string(),
                }
            })?;

            while let Some(entry) =
                entries
                    .next_entry()
                    .await
                    .map_err(|e| crate::ModelAcquisitionError::Io {
                        path: snapshot_path.clone(),
                        message: e.to_string(),
                    })?
            {
                let md = entry
                    .metadata()
                    .await
                    .map_err(|e| crate::ModelAcquisitionError::Io {
                        path: entry.path(),
                        message: e.to_string(),
                    })?;
                if md.is_file() {
                    let relative = entry.file_name().to_string_lossy().to_string();
                    report.add_file(AcquisitionFileEntry {
                        relative_path: relative,
                        bytes: md.len(),
                        outcome: AcquisitionOutcome::Downloaded,
                    });
                }
            }
        }

        report
            .write(target_dir.parent().unwrap_or(&base_models_dir))
            .await?;

        if let Some(s) = sink {
            s.done(&report);
        }

        Ok(report)
    }
}
