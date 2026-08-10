//! Model capability catalog.
//!
//! `LlmModelCatalog` holds a registry of `provider -> models` entries
//! sourced from models.dev and exposes lookups by provider and model
//! id. PV-02a / PV-04 / PV-05 consume it for model discovery, routing,
//! and cost estimation.
//!
//! # On-disk format
//!
//! The cache file `{workspace}/config/model-catalog.json` is a JSON
//! object keyed by provider id. Each value stores a
//! [`ProviderCatalogEntry`] — the models.dev display `name` plus a list
//! of `ModelInfo` values with the provider stamped in, so consumers
//! never translate twice:
//!
//! ```json
//! {
//!   "openai": {
//!     "name": "OpenAI",
//!     "models": [
//!       {
//!         "name": "gpt-4o-mini",
//!         "provider": "openai",
//!         "capabilities": [],
//!         "reasoning": false,
//!         "input_modalities": ["text", "image"],
//!         "context_window": 128000,
//!         "max_tokens": 16384,
//!         "cost": { "input": 0.15, "output": 0.6, "cache_read": 0.075, "cache_write": 0.15 }
//!       }
//!     ]
//!   }
//! }
//! ```
//!
//! # Override semantics
//!
//! An optional `{workspace}/config/model-catalog.override.json` uses
//! the same format. Each provider key in the override file replaces the
//! registry entry for that provider **wholesale** — the entry's `name`
//! and the complete `models` list are taken from the override, with no
//! deep merge per model. Providers not mentioned in the override are
//! left untouched. Overrides are applied by [`LlmModelCatalog::load`] and
//! re-applied after a successful [`LlmModelCatalog::refresh`].
//!
//! # Loading and fallback
//!
//! - [`LlmModelCatalog::load`]: reads the cache file; a missing cache file
//!   yields an empty catalog, a corrupt one returns
//!   [`LlmCatalogError::Parse`].
//! - [`LlmModelCatalog::refresh`]: fetches `https://models.dev/api.json`
//!   (with an optional provider filter), persists the result atomically,
//!   and replaces the in-memory catalog. On failure the catalog keeps
//!   its previous state, unless neither a cache file nor in-memory data
//!   exists, in which case the embedded snapshot
//!   (`assets/model-catalog-snapshot.json`, openai / anthropic /
//!   openrouter) is loaded as a fallback.

use std::collections::HashMap;
use std::io::ErrorKind;
use std::path::Path;
use std::time::Duration;

use crate::ids::{ModelName, ProviderName};
use crate::provider::{ModelCost, ModelInfo};
use reimagine_config::{AppPaths, atomic_write};
use serde::{Deserialize, Serialize};

/// Embedded fallback catalog used when a refresh fails and no cache
/// file exists yet.
const SNAPSHOT_JSON: &str = include_str!("../assets/model-catalog-snapshot.json");
const CATALOG_FILE: &str = "model-catalog.json";
const OVERRIDE_FILE: &str = "model-catalog.override.json";
const MODELS_DEV_URL: &str = "https://models.dev/api.json";
const REFRESH_TIMEOUT: Duration = Duration::from_secs(30);

/// Errors from catalog loading, syncing, and persistence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlmCatalogError {
    /// Network or HTTP failure reaching the upstream catalog.
    Fetch(String),
    /// A catalog file did not parse as valid JSON.
    Parse(String),
    /// A catalog file could not be read.
    Read(String),
    /// The catalog could not be written to disk.
    Write(String),
}

impl LlmCatalogError {
    pub fn fetch(message: impl Into<String>) -> Self {
        Self::Fetch(message.into())
    }

    pub fn parse(message: impl Into<String>) -> Self {
        Self::Parse(message.into())
    }

    pub fn read(message: impl Into<String>) -> Self {
        Self::Read(message.into())
    }

    pub fn write(message: impl Into<String>) -> Self {
        Self::Write(message.into())
    }
}

impl std::fmt::Display for LlmCatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fetch(m) => write!(f, "[FETCH] {m}"),
            Self::Parse(m) => write!(f, "[PARSE] {m}"),
            Self::Read(m) => write!(f, "[READ] {m}"),
            Self::Write(m) => write!(f, "[WRITE] {m}"),
        }
    }
}

impl std::error::Error for LlmCatalogError {}

/// A single provider's catalog entry as stored on disk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderCatalogEntry {
    /// Display name of the provider (e.g. "OpenAI").
    pub name: String,
    /// Models advertised by the provider.
    #[serde(default)]
    pub models: Vec<ModelInfo>,
}

/// Provider -> models registry.
///
/// Serializes as the flat `{ "provider": entry, ... }` map documented
/// in the module docs, with provider ids sorted so cache files are
/// written deterministically. [`LlmModelCatalog::providers`] returns ids
/// in the same sorted order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LlmModelCatalog {
    providers: HashMap<String, ProviderCatalogEntry>,
}

impl Serialize for LlmModelCatalog {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::Error as _;
        let mut map = serde_json::Map::new();
        for (provider, entry) in self.sorted_providers() {
            let value = serde_json::to_value(entry).map_err(S::Error::custom)?;
            map.insert(provider, value);
        }
        map.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for LlmModelCatalog {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let providers = HashMap::<String, ProviderCatalogEntry>::deserialize(deserializer)?;
        Ok(Self { providers })
    }
}

impl LlmModelCatalog {
    /// Provider ids in sorted order.
    fn sorted_providers(&self) -> Vec<(String, &ProviderCatalogEntry)> {
        let mut providers: Vec<_> = self.providers.iter().collect();
        providers.sort_by_key(|(provider, _)| *provider);
        providers
            .into_iter()
            .map(|(provider, entry)| (provider.clone(), entry))
            .collect()
    }
    /// Load the catalog from `{config_dir}/model-catalog.json`. A
    /// missing cache file yields an empty catalog; the optional override
    /// file is applied afterwards. Corrupt files return an error.
    pub async fn load(paths: &AppPaths) -> Result<Self, LlmCatalogError> {
        let config_dir = paths.config_dir();
        let cache_path = config_dir.join(CATALOG_FILE);
        let mut catalog = match tokio::fs::read_to_string(&cache_path).await {
            Ok(text) => {
                serde_json::from_str(&text).map_err(|e| LlmCatalogError::parse(e.to_string()))?
            }
            Err(error) if error.kind() == ErrorKind::NotFound => Self::default(),
            Err(error) => return Err(LlmCatalogError::read(error.to_string())),
        };
        catalog.apply_override(config_dir).await?;
        Ok(catalog)
    }

    /// Sync the catalog from models.dev. `providers`, when `Some`,
    /// restricts which provider entries are kept. On success the cache
    /// file is written atomically and the in-memory catalog is replaced
    /// (overrides re-applied). On failure the catalog keeps its
    /// previous state unless neither a cache file nor in-memory data
    /// exists, in which case the embedded snapshot is loaded.
    pub async fn refresh(
        &mut self,
        paths: &AppPaths,
        providers: Option<&[String]>,
    ) -> Result<(), LlmCatalogError> {
        let client = reqwest::Client::builder()
            .timeout(REFRESH_TIMEOUT)
            .build()
            .map_err(|error| LlmCatalogError::fetch(error.to_string()))?;
        self.refresh_with_http_client(paths, providers, &client, MODELS_DEV_URL)
            .await
    }

    /// `refresh` with an explicit HTTP client and endpoint (used by
    /// tests and hosts that need a custom transport).
    pub async fn refresh_with_http_client(
        &mut self,
        paths: &AppPaths,
        providers: Option<&[String]>,
        client: &reqwest::Client,
        endpoint: &str,
    ) -> Result<(), LlmCatalogError> {
        let config_dir = paths.config_dir();
        let fetched = match fetch_models_dev(client, endpoint, providers).await {
            Ok(fetched) => fetched,
            Err(error) => {
                let cache_path = config_dir.join(CATALOG_FILE);
                let has_cache = tokio::fs::try_exists(&cache_path).await.unwrap_or(true);
                if !has_cache && self.providers.is_empty() {
                    *self = Self::from_snapshot();
                }
                return Err(error);
            }
        };
        let json = serde_json::to_string_pretty(&fetched)
            .map_err(|error| LlmCatalogError::write(error.to_string()))?;
        atomic_write(config_dir.join(CATALOG_FILE), json.as_bytes())
            .await
            .map_err(|error| LlmCatalogError::write(error.to_string()))?;
        let mut effective = fetched;
        effective.apply_override(config_dir).await?;
        *self = effective;
        Ok(())
    }

    /// Look up a model by provider id and model id.
    pub fn model(&self, provider: &str, model_id: &str) -> Option<&ModelInfo> {
        self.provider(provider)?
            .models
            .iter()
            .find(|model| model.name().as_str() == model_id)
    }

    /// The catalog entry for `provider`, if present.
    pub fn provider(&self, provider: &str) -> Option<&ProviderCatalogEntry> {
        self.providers.get(provider)
    }

    /// Provider ids present in the catalog, sorted lexicographically.
    pub fn providers(&self) -> Vec<String> {
        self.sorted_providers()
            .into_iter()
            .map(|(provider, _)| provider)
            .collect()
    }

    /// `true` when no provider entries are registered.
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    /// Load the embedded snapshot (`assets/model-catalog-snapshot.json`).
    fn from_snapshot() -> Self {
        serde_json::from_str(SNAPSHOT_JSON).expect("bundled snapshot must parse")
    }

    /// Apply the optional override file: each provider key replaces the
    /// registry entry wholesale. A missing file is a no-op.
    async fn apply_override(&mut self, config_dir: &Path) -> Result<(), LlmCatalogError> {
        let path = config_dir.join(OVERRIDE_FILE);
        let text = match tokio::fs::read_to_string(&path).await {
            Ok(text) => text,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(LlmCatalogError::read(error.to_string())),
        };
        let overrides: HashMap<String, ProviderCatalogEntry> =
            serde_json::from_str(&text).map_err(|e| LlmCatalogError::parse(e.to_string()))?;
        for (provider, entry) in overrides {
            self.providers.insert(provider, entry);
        }
        Ok(())
    }
}

/// Lenient DTO for the models.dev API shape. Unknown fields and missing
/// optional fields are ignored.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawProvider {
    #[serde(default)]
    name: String,
    #[serde(default)]
    models: Vec<RawModel>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawModel {
    #[serde(default)]
    id: String,
    #[serde(default)]
    reasoning: bool,
    #[serde(default)]
    input: Vec<String>,
    #[serde(default)]
    context_window: Option<usize>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    cost: Option<RawCost>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawCost {
    #[serde(default)]
    input: f64,
    #[serde(default)]
    output: f64,
    #[serde(default)]
    cache_read: f64,
    #[serde(default)]
    cache_write: f64,
}

async fn fetch_models_dev(
    client: &reqwest::Client,
    endpoint: &str,
    providers: Option<&[String]>,
) -> Result<LlmModelCatalog, LlmCatalogError> {
    let response = client
        .get(endpoint)
        .send()
        .await
        .map_err(|error| LlmCatalogError::fetch(error.to_string()))?;
    let status = response.status();
    if !status.is_success() {
        return Err(LlmCatalogError::fetch(format!("HTTP {status}")));
    }
    let raw: HashMap<String, RawProvider> = response
        .json()
        .await
        .map_err(|error| LlmCatalogError::parse(error.to_string()))?;
    let mut catalog = LlmModelCatalog::default();
    for (provider_id, raw_provider) in raw {
        if let Some(filter) = providers
            && !filter.iter().any(|wanted| wanted == &provider_id)
        {
            continue;
        }
        let name = if raw_provider.name.is_empty() {
            provider_id.clone()
        } else {
            raw_provider.name
        };
        let models = raw_provider
            .models
            .into_iter()
            .filter(|model| !model.id.is_empty())
            .map(|model| to_model_info(&provider_id, model))
            .collect();
        catalog
            .providers
            .insert(provider_id, ProviderCatalogEntry { name, models });
    }
    Ok(catalog)
}

fn to_model_info(provider_id: &str, raw: RawModel) -> ModelInfo {
    ModelInfo::new(ModelName::new(raw.id))
        .with_provider(ProviderName::new(provider_id))
        .with_reasoning(raw.reasoning)
        .with_input_modalities(raw.input)
        .with_context_window(raw.context_window)
        .with_max_tokens(raw.max_tokens)
        .with_cost(
            raw.cost.map(|cost| {
                ModelCost::new(cost.input, cost.output, cost.cache_read, cost.cache_write)
            }),
        )
}
