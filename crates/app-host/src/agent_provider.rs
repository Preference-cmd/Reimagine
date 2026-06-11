use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use reimagine_agent::{AgentProvider, ProviderName};

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
