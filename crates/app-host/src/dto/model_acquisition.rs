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
    /// Model display name from catalog metadata (e.g., "Stable Diffusion XL").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    /// Detected repository format (e.g., "Diffusers", "SingleFileSafetensors").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detected_format: Option<String>,
    /// Estimated total download size in bytes from catalog metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_size: Option<u64>,
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
    /// When true (the default), if `allow_patterns` is empty the download strategy
    /// will fetch repository metadata, detect the model format, and build optimal
    /// download patterns automatically.
    #[serde(default)]
    pub auto_detect: Option<bool>,
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
    /// The detected repository format (e.g., "Diffusers", "SingleFileSafetensors").
    /// Present only when auto-detect was used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detected_format: Option<String>,
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
            detected_format: report.detected_format,
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

// ─── Catalog search DTOs ───────────────────────────────────────────

/// Optional filters for catalog search.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelFilters {
    /// Pipeline tag filter (e.g., "text-to-image", "text-generation").
    #[serde(default)]
    pub pipeline_tag: Option<String>,
    /// Library name filter (e.g., "diffusers", "transformers").
    #[serde(default)]
    pub library_name: Option<String>,
    /// Additional tag filters.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Sort order: "downloads", "likes", "trending", "lastModified".
    #[serde(default = "default_sort_downloads")]
    pub sort: String,
    /// Maximum number of results.
    #[serde(default = "default_limit_20")]
    pub limit: usize,
}

impl Default for ModelFilters {
    fn default() -> Self {
        Self {
            pipeline_tag: None,
            library_name: None,
            tags: Vec::new(),
            sort: default_sort_downloads(),
            limit: default_limit_20(),
        }
    }
}

fn default_sort_downloads() -> String {
    "downloads".to_string()
}

fn default_limit_20() -> usize {
    20
}

/// A single model catalog entry returned from search.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCatalogEntryDto {
    /// Repository ID in `owner/name` form.
    pub id: String,
    /// Repository author/owner.
    pub author: Option<String>,
    /// Primary pipeline tag (e.g., "text-to-image").
    pub pipeline_tag: Option<String>,
    /// Hub tags associated with this model.
    pub tags: Vec<String>,
    /// Number of downloads in the last 30 days.
    pub downloads: u64,
    /// Number of likes.
    pub likes: u64,
    /// ISO-8616 timestamp of the most recent commit.
    pub last_modified: Option<String>,
    /// Whether the repository is private.
    pub private: bool,
}

impl From<reimagine_model_acquisition::ModelCatalogEntry> for ModelCatalogEntryDto {
    fn from(entry: reimagine_model_acquisition::ModelCatalogEntry) -> Self {
        Self {
            id: entry.id,
            author: entry.author,
            pipeline_tag: entry.pipeline_tag,
            tags: entry.tags,
            downloads: entry.downloads,
            likes: entry.likes,
            last_modified: entry.last_modified,
            private: entry.private,
        }
    }
}

/// Full model card with detailed metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCardDto {
    /// Basic catalog entry information.
    pub entry: ModelCatalogEntryDto,
    /// Detected model repository format.
    pub detected_format: String,
    /// Estimated total download size in bytes.
    pub estimated_download_size: u64,
    /// Model summary from the model card.
    pub model_summary: Option<String>,
    /// Number of files in the repository.
    pub file_count: usize,
    /// Names of key components detected (e.g., "unet", "vae", "text_encoder").
    pub components: Vec<String>,
}

impl From<reimagine_model_acquisition::ModelCard> for ModelCardDto {
    fn from(card: reimagine_model_acquisition::ModelCard) -> Self {
        let components = card
            .component_mapping
            .as_ref()
            .map(|cm| {
                let mut names = Vec::new();
                if cm.unet.is_some() {
                    names.push("unet".to_string());
                }
                if cm.text_encoder.is_some() {
                    names.push("text_encoder".to_string());
                }
                if cm.text_encoder_2.is_some() {
                    names.push("text_encoder_2".to_string());
                }
                if cm.vae.is_some() {
                    names.push("vae".to_string());
                }
                names
            })
            .unwrap_or_default();

        Self {
            entry: ModelCatalogEntryDto::from(card.entry),
            detected_format: format!("{:?}", card.detected_format),
            estimated_download_size: card.estimated_download_size,
            model_summary: card
                .card_data
                .as_ref()
                .and_then(|cd| cd.model_summary.clone()),
            file_count: card.siblings.len(),
            components,
        }
    }
}
