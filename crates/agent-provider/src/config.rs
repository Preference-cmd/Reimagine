//! Typed provider config shapes for V1.
//!
//! V1 stores provider config in a JSON document loaded by app-host. The
//! shapes here are the wire format: `enabled`, `kind` discriminator,
//! `base_url` (OpenAI-compatible only), `api_key`, and `default_model`.
//! `api_key` is held as a `String` in memory. The app-host is responsible
//! for sourcing the value from the workspace's secrets file before
//! constructing an adapter.

use serde::{Deserialize, Serialize};

/// Discriminator for the V1 provider kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProviderKind {
    #[serde(rename = "openai_compatible")]
    OpenAiCompatible,
    #[serde(rename = "anthropic")]
    Anthropic,
}

impl ProviderKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenAiCompatible => "openai_compatible",
            Self::Anthropic => "anthropic",
        }
    }
}

impl std::fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// OpenAI-compatible provider config. `base_url` is required because
/// V1 supports arbitrary OpenAI-compatible endpoints, not just
/// `api.openai.com`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiCompatibleConfig {
    base_url: String,
    api_key: String,
    default_model: String,
}

impl OpenAiCompatibleConfig {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        default_model: impl Into<String>,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            default_model: default_model.into(),
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    pub fn default_model(&self) -> &str {
        &self.default_model
    }
}

/// Anthropic provider config. V1 keeps this minimal: API key + default
/// model. The adapter hardcodes the Anthropic API base URL.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicConfig {
    api_key: String,
    default_model: String,
}

impl AnthropicConfig {
    pub fn new(api_key: impl Into<String>, default_model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            default_model: default_model.into(),
        }
    }

    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    pub fn default_model(&self) -> &str {
        &self.default_model
    }
}

/// A single provider entry. `kind` discriminates which inner config is
/// present. `enabled` defaults to `true`; hosts may disable a provider
/// without removing it from the config file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderConfig {
    name: String,
    #[serde(default = "default_enabled")]
    enabled: bool,
    kind: ProviderKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    api_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    default_model: Option<String>,
    /// Strongly-typed inner configs. Only the one matching `kind` is
    /// populated. The flat fields above exist so the on-disk JSON is
    /// readable; the typed fields are the source of truth at runtime.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    openai_compatible: Option<OpenAiCompatibleConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    anthropic: Option<AnthropicConfig>,
}

fn default_enabled() -> bool {
    true
}

impl ProviderConfig {
    pub fn with_openai_compatible(
        name: impl Into<String>,
        inner: OpenAiCompatibleConfig,
    ) -> Self {
        Self {
            name: name.into(),
            enabled: true,
            kind: ProviderKind::OpenAiCompatible,
            base_url: Some(inner.base_url().to_string()),
            api_key: Some(inner.api_key().to_string()),
            default_model: Some(inner.default_model().to_string()),
            openai_compatible: Some(inner),
            anthropic: None,
        }
    }

    pub fn with_anthropic(name: impl Into<String>, inner: AnthropicConfig) -> Self {
        Self {
            name: name.into(),
            enabled: true,
            kind: ProviderKind::Anthropic,
            base_url: None,
            api_key: Some(inner.api_key().to_string()),
            default_model: Some(inner.default_model().to_string()),
            openai_compatible: None,
            anthropic: Some(inner),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn kind(&self) -> ProviderKind {
        self.kind
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn set_enabled(&mut self, value: bool) {
        self.enabled = value;
    }

    pub fn openai_compatible(&self) -> Option<&OpenAiCompatibleConfig> {
        self.openai_compatible.as_ref()
    }

    pub fn anthropic(&self) -> Option<&AnthropicConfig> {
        self.anthropic.as_ref()
    }
}

/// On-disk V1 provider config document. The file shape is intentionally
/// flat per entry so a human can edit it. `AgentProviderConfigDocument`
/// is what `app-host` parses; the `build_provider` helper in `lib.rs`
/// turns each entry into an `Arc<dyn AgentProvider>`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentProviderConfigDocument {
    providers: Vec<ProviderConfig>,
}

impl AgentProviderConfigDocument {
    pub fn new(providers: Vec<ProviderConfig>) -> Self {
        Self { providers }
    }

    pub fn providers(&self) -> &[ProviderConfig] {
        &self.providers
    }

    pub fn into_providers(self) -> Vec<ProviderConfig> {
        self.providers
    }

    pub fn enabled(&self) -> impl Iterator<Item = &ProviderConfig> {
        self.providers.iter().filter(|p| p.is_enabled())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_kind_as_str() {
        assert_eq!(ProviderKind::OpenAiCompatible.as_str(), "openai_compatible");
        assert_eq!(ProviderKind::Anthropic.as_str(), "anthropic");
    }
}
