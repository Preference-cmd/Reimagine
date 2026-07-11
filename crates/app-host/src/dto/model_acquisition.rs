use serde::{Deserialize, Serialize};

/// Progress event payload streamed during model download.
///
/// Mirrors the `RunEventPayload` / `AgentEventPayload` naming convention for
/// Tauri Channel streaming.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadEventPayload {
    pub id: String,
    pub status: String,
    pub repo_id: String,
    pub revision: String,
    pub bytes_downloaded: u64,
    pub total_bytes: Option<u64>,
    pub message: Option<String>,
}

/// Input to the `model.download` agent tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDownloadInput {
    /// HuggingFace repo ID in `namespace/name` format (e.g. `stabilityai/stable-diffusion-xl-base-1.0`).
    pub repo_id: String,
    /// Git revision (branch, tag, or commit hash). Defaults to `"main"`.
    #[serde(default)]
    pub revision: Option<String>,
    /// Glob patterns to filter files to download. When empty, all files are downloaded.
    #[serde(default)]
    pub allow_patterns: Option<Vec<String>>,
    /// Relative target directory under `<base>/models/`. Must not use `..`, `.`, or
    /// start with `converted/`.
    pub target_relative_dir: String,
    /// Overwrite policy when the target already exists. One of `"skip"`, `"overwrite"`, `"fail"`.
    /// Defaults to `"skip"`.
    #[serde(default)]
    pub overwrite: Option<String>,
}

/// Single file record in the download report.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileEntryDto {
    /// Relative path within the target directory.
    pub relative_path: String,
    /// Size in bytes.
    pub bytes: u64,
    /// Outcome: `"downloaded"`, `"skipped"`, `"overwritten"`, or `"failed"`.
    pub outcome: String,
}

/// Output of a completed model download.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDownloadOutput {
    /// Whether the tool completed successfully.
    pub effective: bool,
    /// The provider that performed the download.
    pub provider: String,
    /// The repository identifier.
    pub repo_id: String,
    /// The revision that was fetched.
    pub revision: String,
    /// Target directory relative to the workspace base path.
    pub target_dir: String,
    /// Per-file records.
    pub files: Vec<FileEntryDto>,
    /// Total bytes downloaded.
    pub total_bytes: u64,
    /// ISO 8601 timestamp of completion.
    pub finished_at: String,
}

impl From<reimagine_model_acquisition::AcquisitionReport> for ModelDownloadOutput {
    fn from(report: reimagine_model_acquisition::AcquisitionReport) -> Self {
        Self {
            effective: true,
            provider: report.provider,
            repo_id: report.repo_id,
            revision: report.revision,
            target_dir: report.target_dir,
            files: report
                .files
                .into_iter()
                .map(|f| FileEntryDto {
                    relative_path: f.relative_path,
                    bytes: f.bytes,
                    outcome: format!("{:?}", f.outcome),
                })
                .collect(),
            total_bytes: report.total_bytes,
            finished_at: report.finished_at,
        }
    }
}

	/// Input to the `POST /models/acquire` endpoint.
///
/// Downloads a HuggingFace model, converts it to Burn-native
/// components, and registers it in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelAcquireInput {
    /// HuggingFace repo ID (e.g. `stabilityai/stable-diffusion-xl-base-1.0`).
    pub repo_id: String,
    /// Git revision (defaults to `"main"`).
    #[serde(default)]
    pub revision: Option<String>,
    /// Target backend: `"burn"` or `"candle"`. Defaults to `"burn"`.
    #[serde(default)]
    pub target_backend: Option<String>,
    /// Overwrite policy: `"skip"`, `"overwrite"`, `"fail"`. Defaults to `"skip"`.
    #[serde(default)]
    pub overwrite: Option<String>,
}

/// Summary of the download step.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelAcquireDownloadReport {
    /// The HuggingFace repo identifier.
    pub repo_id: String,
    /// The git revision fetched.
    pub revision: String,
    /// Number of downloaded files.
    pub file_count: usize,
    /// Total bytes transferred.
    pub total_bytes: u64,
}

/// Output of `POST /models/acquire`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelAcquireOutput {
    /// The outcome: `"acquired"` on success.
    pub outcome: String,
    /// The source model ID (derived from repo_id).
    pub model_id: String,
    /// The import result model ID (e.g. `<model_id>-burn`).
    pub imported_model_id: String,
    /// Download step summary.
    pub acquisition: ModelAcquireDownloadReport,
    /// Conversion report summary.
    pub conversion: ModelAcquireConversionReport,
}

/// Summary of the conversion step.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelAcquireConversionReport {
    /// Target backend (`"burn"` or `"candle"`).
    pub backend: String,
    /// Number of tensors mapped.
    pub mapped_tensor_count: usize,
    /// Number of output components written.
    pub component_count: usize,
    /// Source layout detected.
    pub source_layout: String,
}
