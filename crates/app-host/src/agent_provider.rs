use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

use reimagine_agent_harness::{AgentProvider, ProviderName};

use crate::provider_config::{
    AgentProviderConfigDocument, Protocol, ProviderConfig,
};

#[derive(Clone, Default)]
pub struct AgentProviderCatalog {
    providers: Arc<RwLock<BTreeMap<ProviderName, Arc<dyn AgentProvider>>>>,
}

impl std::fmt::Debug for AgentProviderCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names = self.provider_names();
        f.debug_struct("AgentProviderCatalog")
            .field("providers", &names)
            .finish()
    }
}

impl AgentProviderCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_provider(provider: Arc<dyn AgentProvider>) -> Self {
        let catalog = Self::new();
        catalog.register(provider);
        catalog
    }

    pub fn register(&self, provider: Arc<dyn AgentProvider>) -> ProviderName {
        let name = provider.name();
        self.providers
            .write()
            .expect("agent provider catalog poisoned")
            .insert(name.clone(), provider);
        name
    }

    pub fn get(&self, name: &ProviderName) -> Option<Arc<dyn AgentProvider>> {
        self.providers
            .read()
            .expect("agent provider catalog poisoned")
            .get(name)
            .cloned()
    }

    pub fn contains(&self, name: &ProviderName) -> bool {
        self.providers
            .read()
            .expect("agent provider catalog poisoned")
            .contains_key(name)
    }

    pub fn provider_names(&self) -> Vec<ProviderName> {
        self.providers
            .read()
            .expect("agent provider catalog poisoned")
            .keys()
            .cloned()
            .collect()
    }

    pub fn len(&self) -> usize {
        self.providers
            .read()
            .expect("agent provider catalog poisoned")
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Build an `Arc<dyn AgentProvider>` from a `ProviderConfig`.
///
/// The protocol discriminator selects which concrete adapter in
/// `reimagine-agent-provider` is constructed. Missing inner config is
/// rejected with
/// [`reimagine_agent_provider::ProviderAdapterError::MissingConfig`].
///
/// `workspace_dir`, when present, roots `FileSource::Url` resolution:
/// file blocks referencing workspace-relative paths are read from the
/// workspace and inlined as base64 before wire translation.
pub fn build_provider(
    config: &ProviderConfig,
    workspace_dir: Option<&Path>,
) -> Result<Arc<dyn AgentProvider>, reimagine_agent_provider::ProviderAdapterError> {
    use reimagine_agent_provider::{
        AnthropicMessagesProvider, OpenAiChatCompletionsProvider, OpenAiResponsesProvider,
    };
    let name = ProviderName::new(config.name().to_string());
    match config.protocol() {
        Protocol::OpenAiChatCompletions => {
            let inner = config.openai_chat_completions().ok_or_else(|| {
                reimagine_agent_provider::ProviderAdapterError::MissingConfig {
                    provider: config.name().to_string(),
                    protocol: Protocol::OpenAiChatCompletions,
                }
            })?;
            let provider = OpenAiChatCompletionsProvider::new(name, inner.clone());
            Ok(Arc::new(match workspace_dir {
                Some(dir) => provider.with_workspace_dir(dir),
                None => provider,
            }))
        }
        Protocol::AnthropicMessages => {
            let inner = config.anthropic_messages().ok_or_else(|| {
                reimagine_agent_provider::ProviderAdapterError::MissingConfig {
                    provider: config.name().to_string(),
                    protocol: Protocol::AnthropicMessages,
                }
            })?;
            let provider = AnthropicMessagesProvider::new(name, inner.clone());
            Ok(Arc::new(match workspace_dir {
                Some(dir) => provider.with_workspace_dir(dir),
                None => provider,
            }))
        }
        Protocol::OpenAiResponses => {
            let inner = config.openai_responses().ok_or_else(|| {
                reimagine_agent_provider::ProviderAdapterError::MissingConfig {
                    provider: config.name().to_string(),
                    protocol: Protocol::OpenAiResponses,
                }
            })?;
            let provider = OpenAiResponsesProvider::new(name, inner.clone());
            Ok(Arc::new(match workspace_dir {
                Some(dir) => provider.with_workspace_dir(dir),
                None => provider,
            }))
        }
    }
}

/// Register every enabled provider from a config document into a catalog.
///
/// Returns the provider names that were registered. Providers whose
/// config is missing its inner typed section are skipped and reported in
/// the returned error list; the remaining providers are still registered
/// so a partial document never bricks the whole catalog.
///
/// `workspace_dir`, when present, roots `FileSource::Url` resolution
/// for every registered provider.
pub fn register_providers_from_document(
    catalog: &AgentProviderCatalog,
    document: &AgentProviderConfigDocument,
    workspace_dir: Option<&Path>,
) -> (Vec<ProviderName>, Vec<String>) {
    let mut registered = Vec::new();
    let mut errors = Vec::new();
    for config in document.enabled() {
        match build_provider(config, workspace_dir) {
            Ok(provider) => {
                let name = catalog.register(provider);
                registered.push(name);
            }
            Err(error) => errors.push(format!("provider `{}`: {error}", config.name())),
        }
    }
    (registered, errors)
}
