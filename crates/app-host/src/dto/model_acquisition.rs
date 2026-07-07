use serde::{Deserialize, Serialize};

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
