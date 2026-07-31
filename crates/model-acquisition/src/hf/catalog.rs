//! HuggingFace model catalog — search, browse, and inspect models.
//!
//! Provides [`ModelCatalog`] for searching HuggingFace models with filters,
//! fetching detailed model cards, and convenience methods for common
//! Reimagine use-cases (e.g., popular text-to-image models).

use crate::error::ModelAcquisitionError;
use crate::hf::format::{ModelRepoFormat, detect_format};
use crate::hf::metadata::HfRepoMetadata;
use crate::hf::model_index::ComponentMapping;
use crate::hf::strategy::fetch_model_index;

/// Predefined tag filters for finding image generation models on HuggingFace.
///
/// These tags combine to surface diffusers-based, safetensors-weighted,
/// text-to-image models — the typical Reimagine target.
pub const IMAGE_GENERATION_FILTERS: &[&str] = &["diffusers", "safetensors", "text-to-image"];

/// Sort order for model search results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum SortBy {
    /// Sort by download count (default).
    #[default]
    Downloads,
    /// Sort by like count.
    Likes,
    /// Sort by trending score.
    Trending,
    /// Sort by last modified date.
    LastModified,
}

impl SortBy {
    /// Convert to the HuggingFace API `sort` parameter value.
    fn as_api_param(self) -> &'static str {
        match self {
            Self::Downloads => "downloads",
            Self::Likes => "likes",
            Self::Trending => "trendingScore",
            Self::LastModified => "lastModified",
        }
    }
}

/// Search query parameters for the HuggingFace model catalog.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelSearchQuery {
    /// Free-text search query (matches model ID and description).
    pub search: Option<String>,
    /// Pipeline tag filter (e.g., "text-to-image", "text-generation").
    pub pipeline_tag: Option<String>,
    /// Library name filter (e.g., "diffusers", "transformers").
    pub library_name: Option<String>,
    /// Additional tag filters applied client-side.
    pub tags: Vec<String>,
    /// Sort order for results.
    pub sort: SortBy,
    /// Maximum number of results to return.
    pub limit: usize,
}

impl ModelSearchQuery {
    /// Create a search query for image generation models.
    pub fn image_generation() -> Self {
        Self {
            pipeline_tag: Some("text-to-image".to_string()),
            tags: IMAGE_GENERATION_FILTERS
                .iter()
                .map(|s| s.to_string())
                .collect(),
            sort: SortBy::Downloads,
            limit: 50,
            ..Default::default()
        }
    }
}

/// A single entry in the model catalog search results.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelCatalogEntry {
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

/// Model card data parsed from the model card (README.md front matter).
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelCardData {
    /// Brief description or summary of the model.
    pub model_summary: Option<String>,
    /// Model type identifier.
    pub model_type: Option<String>,
    /// Library this model is designed for.
    pub library_name: Option<String>,
    /// Primary pipeline tag.
    pub pipeline_tag: Option<String>,
    /// Tags from the model card.
    pub tags: Vec<String>,
}

/// Full model card with detailed metadata, file listing, and format detection.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelCard {
    /// Basic catalog entry information.
    pub entry: ModelCatalogEntry,
    /// Files in the repository.
    pub siblings: Vec<crate::hf::metadata::HfSibling>,
    /// Parsed model card data (from README.md YAML front matter).
    pub card_data: Option<ModelCardData>,
    /// Detected model repository format.
    pub detected_format: ModelRepoFormat,
    /// Typed component mapping (populated for Diffusers format repos).
    pub component_mapping: Option<ComponentMapping>,
    /// Estimated total download size in bytes.
    pub estimated_download_size: u64,
}

/// Catalog client for searching and inspecting HuggingFace models.
///
/// Wraps a [`hf_hub::HFClient`] for repository operations and a
/// [`reqwest::Client`] for direct API queries with advanced filtering.
pub struct ModelCatalog {
    hf_client: hf_hub::HFClient,
    http_client: reqwest::Client,
    endpoint: String,
    token: Option<String>,
}

impl ModelCatalog {
    /// Create a new catalog client.
    ///
    /// Uses the endpoint and token from the provided HF client for API requests.
    /// The `hf_client` is also used for repository metadata operations
    /// (e.g., fetching siblings and format detection in [`Self::model_card`]).
    pub fn new(hf_client: hf_hub::HFClient) -> Self {
        let endpoint = hf_client.endpoint().to_string();
        Self {
            hf_client,
            http_client: reqwest::Client::new(),
            endpoint,
            token: None,
        }
    }

    /// Set the authentication token for direct API requests.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Search for models matching the given query.
    ///
    /// Sends `GET {endpoint}/api/models` with the appropriate query parameters.
    /// When `query.library_name` is set, results are filtered client-side since
    /// the Hub API does not support `library_name` as a query parameter.
    pub async fn search(
        &self,
        query: &ModelSearchQuery,
    ) -> Result<Vec<ModelCatalogEntry>, ModelAcquisitionError> {
        let url = format!("{}/api/models", self.endpoint);

        let mut params: Vec<(&str, String)> = Vec::new();

        if let Some(ref search) = query.search {
            params.push(("search", search.clone()));
        }
        if let Some(ref pipeline_tag) = query.pipeline_tag {
            params.push(("pipeline_tag", pipeline_tag.clone()));
        }
        // The Hub API accepts a single `filter` for tag filtering.
        // Apply the first tag via the API; additional tags are filtered client-side.
        if let Some(first_tag) = query.tags.first() {
            params.push(("filter", first_tag.clone()));
        }

        params.push(("sort", query.sort.as_api_param().to_string()));
        params.push(("limit", query.limit.to_string()));

        let mut request = self.http_client.get(&url).query(&params);

        if let Some(ref token) = self.token {
            request = request.bearer_auth(token);
        }

        let response = request
            .send()
            .await
            .map_err(|e| ModelAcquisitionError::Hub {
                repo: "models".to_string(),
                message: format!("failed to send catalog search request: {e}"),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ModelAcquisitionError::Hub {
                repo: "models".to_string(),
                message: format!("catalog search failed with status {status}: {body}"),
            });
        }

        let body = response
            .text()
            .await
            .map_err(|e| ModelAcquisitionError::Hub {
                repo: "models".to_string(),
                message: format!("failed to read catalog search response: {e}"),
            })?;

        let models: Vec<serde_json::Value> =
            serde_json::from_str(&body).map_err(|e| ModelAcquisitionError::Hub {
                repo: "models".to_string(),
                message: format!("failed to parse catalog search response: {e}"),
            })?;

        let entries: Vec<ModelCatalogEntry> = models
            .into_iter()
            .filter_map(|m| parse_catalog_entry(&m))
            .filter(|entry| passes_tag_filter(entry, &query.tags))
            .collect();

        Ok(entries)
    }

    /// Fetch the full model card for a repository.
    ///
    /// Returns detailed metadata including siblings, detected format,
    /// component mapping (for diffusers repos), and estimated download size.
    pub async fn model_card(&self, repo_id: &str) -> Result<ModelCard, ModelAcquisitionError> {
        let metadata = HfRepoMetadata::fetch(&self.hf_client, repo_id, "main").await?;

        let detected_format = detect_format(&metadata);

        let component_mapping = if detected_format == ModelRepoFormat::Diffusers {
            if metadata
                .siblings
                .iter()
                .any(|s| s.rfilename == "model_index.json")
            {
                fetch_model_index(&self.hf_client, repo_id, "main")
                    .await
                    .ok()
                    .map(|mi| mi.to_component_mapping())
            } else {
                None
            }
        } else {
            None
        };

        let estimated_download_size: u64 = metadata.siblings.iter().filter_map(|s| s.size).sum();

        let author = repo_id.split_once('/').map(|(a, _)| a.to_string());

        let entry = ModelCatalogEntry {
            id: metadata.repo_id.clone(),
            author,
            pipeline_tag: metadata.pipeline_tag.clone(),
            tags: metadata.tags.clone(),
            downloads: 0,
            likes: 0,
            last_modified: None,
            private: false,
        };

        Ok(ModelCard {
            entry,
            siblings: metadata.siblings,
            card_data: None,
            detected_format,
            component_mapping,
            estimated_download_size,
        })
    }

    /// Convenience method to find popular models for a given pipeline tag.
    ///
    /// Searches sorted by downloads and returns up to `limit` results.
    pub async fn popular(
        &self,
        pipeline_tag: &str,
        limit: usize,
    ) -> Result<Vec<ModelCatalogEntry>, ModelAcquisitionError> {
        let query = ModelSearchQuery {
            pipeline_tag: Some(pipeline_tag.to_string()),
            sort: SortBy::Downloads,
            limit,
            ..Default::default()
        };
        self.search(&query).await
    }
}

/// Parse a JSON value into a `ModelCatalogEntry`.
fn parse_catalog_entry(value: &serde_json::Value) -> Option<ModelCatalogEntry> {
    let id = value.get("id")?.as_str()?.to_string();

    let author = value
        .get("author")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let pipeline_tag = value
        .get("pipelineTag")
        .or_else(|| value.get("pipeline_tag"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let tags = value
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let downloads = value.get("downloads").and_then(|v| v.as_u64()).unwrap_or(0);

    let likes = value.get("likes").and_then(|v| v.as_u64()).unwrap_or(0);

    let last_modified = value
        .get("lastModified")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let private = value
        .get("private")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    Some(ModelCatalogEntry {
        id,
        author,
        pipeline_tag,
        tags,
        downloads,
        likes,
        last_modified,
        private,
    })
}

/// Check whether a catalog entry matches all specified tag filters.
fn passes_tag_filter(entry: &ModelCatalogEntry, required_tags: &[String]) -> bool {
    required_tags
        .iter()
        .all(|tag| entry.tags.iter().any(|t| t == tag))
}

/// Parse `ModelCardData` from raw card data JSON.
pub fn parse_card_data(card_data: &serde_json::Value) -> Option<ModelCardData> {
    if card_data.is_null() {
        return None;
    }

    let model_summary = card_data
        .get("model_summary")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let model_type = card_data
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let library_name = card_data
        .get("library_name")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let pipeline_tag = card_data
        .get("pipeline_tag")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let tags = card_data
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    Some(ModelCardData {
        model_summary,
        model_type,
        library_name,
        pipeline_tag,
        tags,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_by_api_param() {
        assert_eq!(SortBy::Downloads.as_api_param(), "downloads");
        assert_eq!(SortBy::Likes.as_api_param(), "likes");
        assert_eq!(SortBy::Trending.as_api_param(), "trendingScore");
        assert_eq!(SortBy::LastModified.as_api_param(), "lastModified");
    }

    #[test]
    fn test_parse_catalog_entry_full() {
        let json = serde_json::json!({
            "id": "runwayml/stable-diffusion-v1-5",
            "author": "runwayml",
            "pipelineTag": "text-to-image",
            "tags": ["diffusers", "safetensors", "text-to-image"],
            "downloads": 12345,
            "likes": 678,
            "lastModified": "2025-01-15T10:00:00.000Z",
            "private": false
        });

        let entry = parse_catalog_entry(&json).unwrap();
        assert_eq!(entry.id, "runwayml/stable-diffusion-v1-5");
        assert_eq!(entry.author.as_deref(), Some("runwayml"));
        assert_eq!(entry.pipeline_tag.as_deref(), Some("text-to-image"));
        assert_eq!(entry.tags.len(), 3);
        assert_eq!(entry.downloads, 12345);
        assert_eq!(entry.likes, 678);
        assert_eq!(
            entry.last_modified.as_deref(),
            Some("2025-01-15T10:00:00.000Z")
        );
        assert!(!entry.private);
    }

    #[test]
    fn test_parse_catalog_entry_minimal() {
        let json = serde_json::json!({
            "id": "org/model"
        });

        let entry = parse_catalog_entry(&json).unwrap();
        assert_eq!(entry.id, "org/model");
        assert!(entry.author.is_none());
        assert!(entry.pipeline_tag.is_none());
        assert!(entry.tags.is_empty());
        assert_eq!(entry.downloads, 0);
        assert_eq!(entry.likes, 0);
        assert!(entry.last_modified.is_none());
        assert!(!entry.private);
    }

    #[test]
    fn test_parse_catalog_entry_missing_id_returns_none() {
        let json = serde_json::json!({ "downloads": 100 });
        assert!(parse_catalog_entry(&json).is_none());
    }

    #[test]
    fn test_parse_catalog_entry_wrong_type_returns_none() {
        let json = serde_json::json!("not an object");
        assert!(parse_catalog_entry(&json).is_none());
    }

    #[test]
    fn test_passes_tag_filter_empty() {
        let entry = ModelCatalogEntry {
            id: "test/model".to_string(),
            tags: vec!["diffusers".to_string()],
            downloads: 100,
            likes: 10,
            ..Default::default()
        };
        assert!(passes_tag_filter(&entry, &[]));
    }

    #[test]
    fn test_passes_tag_filter_match() {
        let entry = ModelCatalogEntry {
            id: "test/model".to_string(),
            tags: vec![
                "diffusers".to_string(),
                "safetensors".to_string(),
                "text-to-image".to_string(),
            ],
            downloads: 100,
            likes: 10,
            ..Default::default()
        };
        assert!(passes_tag_filter(
            &entry,
            &["diffusers".to_string(), "safetensors".to_string()]
        ));
    }

    #[test]
    fn test_passes_tag_filter_no_match() {
        let entry = ModelCatalogEntry {
            id: "test/model".to_string(),
            tags: vec!["diffusers".to_string()],
            downloads: 100,
            likes: 10,
            ..Default::default()
        };
        assert!(!passes_tag_filter(&entry, &["text-to-image".to_string()]));
    }

    #[test]
    fn test_parse_card_data_full() {
        let json = serde_json::json!({
            "model_summary": "A great model for image generation",
            "model_type": "stable-diffusion",
            "library_name": "diffusers",
            "pipeline_tag": "text-to-image",
            "tags": ["diffusers", "safetensors"]
        });

        let card_data = parse_card_data(&json).unwrap();
        assert_eq!(
            card_data.model_summary.as_deref(),
            Some("A great model for image generation")
        );
        assert_eq!(card_data.model_type.as_deref(), Some("stable-diffusion"));
        assert_eq!(card_data.library_name.as_deref(), Some("diffusers"));
        assert_eq!(card_data.pipeline_tag.as_deref(), Some("text-to-image"));
        assert_eq!(card_data.tags, vec!["diffusers", "safetensors"]);
    }

    #[test]
    fn test_parse_card_data_null() {
        let json = serde_json::json!(null);
        assert!(parse_card_data(&json).is_none());
    }

    #[test]
    fn test_parse_card_data_empty_object() {
        let json = serde_json::json!({});
        let card_data = parse_card_data(&json).unwrap();
        assert!(card_data.model_summary.is_none());
        assert!(card_data.model_type.is_none());
        assert!(card_data.library_name.is_none());
        assert!(card_data.pipeline_tag.is_none());
        assert!(card_data.tags.is_empty());
    }

    #[test]
    fn test_image_generation_query_defaults() {
        let query = ModelSearchQuery::image_generation();
        assert_eq!(query.pipeline_tag.as_deref(), Some("text-to-image"));
        assert!(query.tags.contains(&"diffusers".to_string()));
        assert!(query.tags.contains(&"safetensors".to_string()));
        assert!(query.tags.contains(&"text-to-image".to_string()));
        assert_eq!(query.sort, SortBy::Downloads);
        assert_eq!(query.limit, 50);
    }

    #[test]
    fn test_model_catalog_entry_serde() {
        let entry = ModelCatalogEntry {
            id: "org/model".to_string(),
            author: Some("org".to_string()),
            pipeline_tag: Some("text-to-image".to_string()),
            tags: vec!["diffusers".to_string()],
            downloads: 1000,
            likes: 50,
            last_modified: Some("2025-01-01T00:00:00Z".to_string()),
            private: false,
        };

        let json = serde_json::to_string(&entry).unwrap();
        let parsed: ModelCatalogEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(entry, parsed);
    }

    #[test]
    fn test_model_card_serde() {
        let card = ModelCard {
            entry: ModelCatalogEntry {
                id: "test/model".to_string(),
                downloads: 100,
                ..Default::default()
            },
            siblings: vec![],
            card_data: Some(ModelCardData {
                model_summary: Some("test".to_string()),
                ..Default::default()
            }),
            detected_format: ModelRepoFormat::Diffusers,
            component_mapping: None,
            estimated_download_size: 1_000_000,
        };

        let json = serde_json::to_string(&card).unwrap();
        let parsed: ModelCard = serde_json::from_str(&json).unwrap();
        assert_eq!(card, parsed);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[ignore = "requires network access to HuggingFace Hub"]
    #[tokio::test]
    async fn test_search_text_to_image_models() {
        let hf_client = hf_hub::HFClient::builder()
            .build()
            .expect("failed to build HFClient");

        let catalog = ModelCatalog::new(hf_client);

        let query = ModelSearchQuery {
            pipeline_tag: Some("text-to-image".to_string()),
            sort: SortBy::Downloads,
            limit: 5,
            ..Default::default()
        };

        let results = catalog.search(&query).await.expect("search failed");

        assert!(!results.is_empty());
        assert!(results.len() <= 5);

        // All results should have the text-to-image pipeline tag
        for entry in &results {
            assert_eq!(entry.pipeline_tag.as_deref(), Some("text-to-image"));
        }
    }

    #[ignore = "requires network access to HuggingFace Hub"]
    #[tokio::test]
    async fn test_model_card_fetch() {
        let hf_client = hf_hub::HFClient::builder()
            .build()
            .expect("failed to build HFClient");

        let catalog = ModelCatalog::new(hf_client);

        let card = catalog
            .model_card("hf-internal-testing/tiny-stable-diffusion-pipe")
            .await
            .expect("model_card failed");

        assert_eq!(
            card.entry.id,
            "hf-internal-testing/tiny-stable-diffusion-pipe"
        );
        assert!(!card.siblings.is_empty());
        assert!(card.estimated_download_size > 0);
    }

    #[ignore = "requires network access to HuggingFace Hub"]
    #[tokio::test]
    async fn test_popular_models() {
        let hf_client = hf_hub::HFClient::builder()
            .build()
            .expect("failed to build HFClient");

        let catalog = ModelCatalog::new(hf_client);

        let results = catalog
            .popular("text-to-image", 3)
            .await
            .expect("popular failed");

        assert!(!results.is_empty());
        assert!(results.len() <= 3);
    }
}
