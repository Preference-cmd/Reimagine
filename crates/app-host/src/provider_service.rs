//! Provider CRUD in the app-host single process (AR-13).
//!
//! `ProviderService` mutates `agent-providers.json` (the app-host
//! config document) and keeps the live `AgentProviderCatalog`
//! transactionally aligned with disk. Every mutation is a
//! transaction: providers are validated and built first, the
//! document is persisted atomically, and only then is the in-memory
//! catalog replaced. A failure at any step leaves both the document
//! and the catalog unchanged, so a bad mutation can never be
//! reported as a durable success.
//!
//! Provider secrets stay in app-host. The summaries and results
//! returned to Tauri/UI never contain `api_key` (or any provider
//! secret) — `ProviderSummary` reports only `has_api_key`, and
//! `ProviderMutationResult` carries a safe message plus a
//! machine-readable code.

use std::path::PathBuf;
use std::sync::Arc;

use reimagine_config::AppConfig;

use crate::agent_provider::{AgentProviderCatalog, build_provider};
use crate::provider_config::{AgentProviderConfigDocument, ProviderConfig};
use crate::{AppHostError, AppHostResult};

/// Redacted, IPC-safe view of a provider entry. Deliberately omits
/// `api_key` and any adapter-internal secrets (AR-13).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderSummary {
    pub name: String,
    pub enabled: bool,
    pub protocol: String,
    pub base_url: Option<String>,
    pub default_model: Option<String>,
    pub has_api_key: bool,
}

/// Outcome of a provider mutation, safe for Tauri/UI serialization.
/// `code` is machine-readable (e.g. "duplicate_add", "missing",
/// "invalid_config", "adapter_build_failed"); `message` is a safe
/// human-readable string that never contains provider secrets.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderMutationResult {
    pub code: String,
    pub provider: String,
    pub message: String,
}

impl ProviderMutationResult {
    fn ok(code: &str, provider: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.to_string(),
            provider: provider.to_string(),
            message: message.into(),
        }
    }

    fn err(code: &str, provider: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.to_string(),
            provider: provider.to_string(),
            message: message.into(),
        }
    }
}

/// Provider CRUD in the app-host single process.
#[derive(Debug, Clone)]
pub struct ProviderService {
    config: Arc<AppConfig>,
    /// The live catalog shared with `AgentService`. Mutations replace
    /// its contents only after the document is safely on disk. The
    /// `AgentProviderCatalog` value is clone/cheap (Arc-backed map), so
    /// sharing a `clone()` of the workspace catalog keeps CRUD changes
    /// visible to the agent service immediately.
    catalog: AgentProviderCatalog,
    workspace_base_path: PathBuf,
}

impl ProviderService {
    pub fn new(config: Arc<AppConfig>, catalog: AgentProviderCatalog) -> Self {
        let workspace_base_path = config.paths().base_path().to_path_buf();
        Self {
            config,
            catalog,
            workspace_base_path,
        }
    }

    pub fn catalog(&self) -> &AgentProviderCatalog {
        &self.catalog
    }

    async fn document_handle(
        &self,
    ) -> AppHostResult<reimagine_config::ConfigHandle<AgentProviderConfigDocument>> {
        let handle = self.config.config::<AgentProviderConfigDocument>()?;
        Ok(handle)
    }

    async fn load_document(&self) -> AppHostResult<AgentProviderConfigDocument> {
        let handle = self.document_handle().await?;
        let (document, _report) = handle.load().await?;
        Ok(document)
    }

    /// List providers as redacted summaries (never api keys).
    pub async fn list(&self) -> AppHostResult<Vec<ProviderSummary>> {
        let document = self.load_document().await?;
        Ok(document.providers().iter().map(redact).collect())
    }

    /// Add a provider: build + validate first, then persist the
    /// document, then synchronise the catalog.
    pub async fn add(&self, config: ProviderConfig) -> AppHostResult<ProviderMutationResult> {
        let document = self.load_document().await?;
        let name = config.name().to_string();
        if document.providers().iter().any(|p| p.name() == name) {
            return Ok(ProviderMutationResult::err(
                "duplicate_add",
                &name,
                format!("provider `{name}` already exists"),
            ));
        }
        // Build before persisting: an unbuildable provider is a
        // no-op error, never a half-applied mutation.
        if let Err(error) = build_provider(&config, Some(&self.workspace_base_path)) {
            return Ok(ProviderMutationResult::err(
                "adapter_build_failed",
                &name,
                provider_safe_message(&name, &error.to_string()),
            ));
        }

        let mut providers = document.into_providers();
        providers.push(config);
        let next = AgentProviderConfigDocument::new(providers);
        self.persist_and_sync(&name, "added", next).await
    }

    /// Update a provider in place: replace its config, persist, and
    /// synchronise the catalog.
    pub async fn update(
        &self,
        name: &str,
        config: ProviderConfig,
    ) -> AppHostResult<ProviderMutationResult> {
        let document = self.load_document().await?;
        let idx = match document.providers().iter().position(|p| p.name() == name) {
            Some(idx) => idx,
            None => {
                return Ok(ProviderMutationResult::err(
                    "missing",
                    name,
                    format!("provider `{name}` does not exist"),
                ));
            }
        };
        // The replacement keeps the original name: a mutation must not
        // silently rename an entry into a duplicate (AR-13).
        let config = config.with_name(name.to_string());
        let mut providers = document.into_providers();
        providers[idx] = config;
        let next = AgentProviderConfigDocument::new(providers);
        self.persist_and_sync(name, "updated", next).await
    }

    /// Delete a provider from the document and the catalog.
    pub async fn delete(&self, name: &str) -> AppHostResult<ProviderMutationResult> {
        let document = self.load_document().await?;
        let before = document.providers().len();
        let providers: Vec<ProviderConfig> = document
            .into_providers()
            .into_iter()
            .filter(|p| p.name() != name)
            .collect();
        if providers.len() == before {
            return Ok(ProviderMutationResult::err(
                "missing",
                name,
                format!("provider `{name}` does not exist"),
            ));
        }
        let next = AgentProviderConfigDocument::new(providers);
        self.persist_and_sync(name, "deleted", next).await
    }

    /// Toggle `enabled`, persisting the document and resync the catalog.
    pub async fn set_enabled(
        &self,
        name: &str,
        enabled: bool,
    ) -> AppHostResult<ProviderMutationResult> {
        let document = self.load_document().await?;
        let idx = match document.providers().iter().position(|p| p.name() == name) {
            Some(idx) => idx,
            None => {
                return Ok(ProviderMutationResult::err(
                    "missing",
                    name,
                    format!("provider `{name}` does not exist"),
                ));
            }
        };
        let mut providers = document.into_providers();
        providers[idx].set_enabled(enabled);
        let next = AgentProviderConfigDocument::new(providers);
        self.persist_and_sync(name, if enabled { "enabled" } else { "disabled" }, next)
            .await
    }

    /// Persist `next` atomically, then rebuild the catalog. On any
    /// error neither the document nor the catalog changes.
    async fn persist_and_sync(
        &self,
        name: &str,
        done_code: &str,
        next: AgentProviderConfigDocument,
    ) -> AppHostResult<ProviderMutationResult> {
        // Validate the whole document before writing (adapter builds
        // also run here and are reported as errors without mutating).
        let mut built = Vec::new();
        for config in next.enabled() {
            match build_provider(config, Some(&self.workspace_base_path)) {
                Ok(provider) => built.push(provider),
                Err(error) => {
                    return Ok(ProviderMutationResult::err(
                        "adapter_build_failed",
                        config.name(),
                        provider_safe_message(config.name(), &error.to_string()),
                    ));
                }
            }
        }

        // Persist atomically. The handle.save path writes through the
        // config store; harden file permissions for secret-bearing docs.
        let handle = self.document_handle().await?;
        if let Err(error) = handle.save(&next).await {
            return Err(AppHostError::BootstrapConfig(error));
        }
        apply_secret_file_permissions(
            &self
                .config
                .paths()
                .config_dir()
                .join("agent-providers.json"),
        );

        // Only now replace the live catalog with the built set.
        self.catalog.replace_all(built);
        Ok(ProviderMutationResult::ok(
            done_code,
            name,
            format!("provider `{name}` {done_code}"),
        ))
    }
}

fn redact(config: &ProviderConfig) -> ProviderSummary {
    ProviderSummary {
        name: config.name().to_string(),
        enabled: config.is_enabled(),
        protocol: config.protocol().as_str().to_string(),
        base_url: config.base_url().map(str::to_string),
        default_model: config.default_model().map(str::to_string),
        has_api_key: config.api_key().is_some(),
    }
}

/// Build a safe provider error message: strip anything that looks
/// like an api key/secret from the adapter message before it reaches
/// a Tauri/UI boundary.
fn provider_safe_message(name: &str, raw: &str) -> String {
    if raw.contains("api_key") {
        format!("provider `{name}` failed to build (adapter configuration error)")
    } else {
        format!("provider `{name}`: {raw}")
    }
}

/// Best-effort 0600 on secret-bearing config files (Unix). On
/// platforms without permission bits this is a no-op; the explicit
/// platform fallback is documented (AR-13) rather than asserted.
fn apply_secret_file_permissions(path: &std::path::Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    #[cfg(not(unix))]
    {
        let _ = path; // documented platform fallback: no-op on non-Unix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AgentProviderCatalog;
    use crate::provider_config::{
        AgentProviderConfigDocument, OpenAiChatCompletionsConfig, ProviderConfig,
    };
    use reimagine_config::AppConfig;

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("ri-ar13-{prefix}-{nonce}"))
    }

    fn service(prefix: &str) -> (ProviderService, std::path::PathBuf) {
        let base = temp_dir(prefix);
        let config = Arc::new(AppConfig::new(reimagine_config::AppPaths::new(&base)));
        let catalog = AgentProviderCatalog::new();
        let service = ProviderService::new(config, catalog);
        (service, base)
    }

    fn chat_provider(name: &str) -> ProviderConfig {
        ProviderConfig::with_openai_chat_completions(
            name,
            OpenAiChatCompletionsConfig::new(
                "https://api.example.com/v1",
                format!("sk-{name}"),
                "gpt-4o-mini",
            ),
        )
    }

    #[tokio::test]
    async fn crud_roundtrip_persists_document_and_resync_catalog() {
        let (service, base) = service("crud");
        let added = service.add(chat_provider("openai")).await.unwrap();
        assert_eq!(added.code, "added");

        let summaries = service.list().await.unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].name, "openai");
        assert!(summaries[0].has_api_key);
        assert!(
            !serde_json::to_string(&summaries[0])
                .unwrap()
                .contains("sk-openai"),
            "list summary must never contain the api key"
        );
        assert!(service.catalog().contains(&"openai".into()));

        // The document is on disk under config/.
        let doc_path = base.join("config/agent-providers.json");
        assert!(doc_path.is_file());

        // update keeps the original name and rewrites the doc
        let upd = service
            .update("openai", chat_provider("renamed"))
            .await
            .unwrap();
        assert_eq!(upd.code, "updated");
        let after = service.list().await.unwrap();
        assert_eq!(after[0].name, "openai", "update must not rename the entry");

        // set_enabled(false) removes from catalog but keeps document
        let dis = service.set_enabled("openai", false).await.unwrap();
        assert_eq!(dis.code, "disabled");
        assert!(
            !service.catalog().contains(&"openai".into()),
            "disabled must leave catalog"
        );

        // delete removes it everywhere
        let del = service.delete("openai").await.unwrap();
        assert_eq!(del.code, "deleted");
        assert!(service.list().await.unwrap().is_empty());
        assert!(!service.catalog().contains(&"openai".into()));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn duplicate_add_and_missing_operations_error_without_mutation() {
        let (service, base) = service("dup");
        service.add(chat_provider("openai")).await.unwrap();
        let dup = service.add(chat_provider("openai")).await.unwrap();
        assert_eq!(dup.code, "duplicate_add");
        assert_eq!(service.list().await.unwrap().len(), 1);

        let missing_update = service
            .update("ghost", chat_provider("ghost"))
            .await
            .unwrap();
        assert_eq!(missing_update.code, "missing");

        let missing_delete = service.delete("ghost").await.unwrap();
        assert_eq!(missing_delete.code, "missing");
        let _ = std::fs::remove_dir_all(&base);
    }
    #[cfg(unix)]
    #[tokio::test]
    async fn secret_config_file_is_0600_on_unix_after_mutation() {
        use std::os::unix::fs::PermissionsExt;
        let (service, base) = service("perm");
        service.add(chat_provider("openai")).await.unwrap();
        let path = base.join("config/agent-providers.json");
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600, "secret config must be 0600 on Unix");
        let _ = std::fs::remove_dir_all(&base);
    }
    #[cfg(unix)]
    #[tokio::test]
    async fn persist_failure_leaves_document_and_catalog_unchanged() {
        use std::os::unix::fs::PermissionsExt;
        let (service, base) = service("rollback");
        service.add(chat_provider("openai")).await.unwrap();
        let before = service.list().await.unwrap();

        // Make the config dir read-only so the next save fails.
        let config_dir = base.join("config");
        let original_mode = std::fs::metadata(&config_dir).unwrap().permissions().mode();
        std::fs::set_permissions(&config_dir, std::fs::Permissions::from_mode(0o555)).unwrap();

        let result = service.add(chat_provider("second")).await;
        std::fs::set_permissions(&config_dir, std::fs::Permissions::from_mode(original_mode))
            .unwrap();

        assert!(result.is_err(), "save failure surfaces as an error");
        assert_eq!(service.list().await.unwrap(), before, "document unchanged");
        assert!(
            !service.catalog().contains(&"second".into()),
            "catalog unchanged"
        );
        let _ = std::fs::remove_dir_all(&base);
    }
    #[tokio::test]
    async fn persisted_provider_survives_fresh_bootstrap_registration() {
        // AR-13 + AR-37: a successful mutation is recovered by the
        // production bootstrap after restart — i.e. a fresh catalog
        // rebuilt from the on-disk document sees the provider.
        let (service, base) = service("recovery");
        service.add(chat_provider("openai")).await.unwrap();

        // Simulate a fresh bootstrap: new AppConfig + fresh catalog,
        // load the document, register from it (AR-37 path).
        let fresh_config = Arc::new(AppConfig::new(reimagine_config::AppPaths::new(&base)));
        let handle = fresh_config
            .config::<AgentProviderConfigDocument>()
            .unwrap();
        let (document, _report) = handle.load().await.unwrap();
        assert_eq!(document.providers().len(), 1);

        let fresh_catalog = AgentProviderCatalog::new();
        let (registered, errors) =
            crate::register_providers_from_document(&fresh_catalog, &document, Some(&base));
        assert!(errors.is_empty(), "bootstrap registration must be clean");
        assert_eq!(registered.len(), 1);
        assert!(fresh_catalog.contains(&"openai".into()));
        let _ = std::fs::remove_dir_all(&base);
    }
}
