//! Typed provider config shapes for V1.
//!
//! V1 stores provider config in a JSON document loaded by app-host. The
//! shapes here are the wire format: `enabled`, `protocol`
//! discriminator, `base_url` (chat-completions protocols only),
//! `api_key`, and `default_model`. `api_key` is held as a `String` in
//! memory. The app-host is responsible for sourcing the value from the
//! workspace's secrets file before constructing an adapter.

use serde::{Deserialize, Serialize};

/// Discriminator for the message protocol a provider entry speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Protocol {
    #[serde(rename = "openai_chat_completions")]
    OpenAiChatCompletions,
    #[serde(rename = "anthropic_messages")]
    AnthropicMessages,
    #[serde(rename = "openai_responses")]
    OpenAiResponses,
}

impl Protocol {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenAiChatCompletions => "openai_chat_completions",
            Self::AnthropicMessages => "anthropic_messages",
            Self::OpenAiResponses => "openai_responses",
        }
    }
}

impl std::fmt::Display for Protocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// OpenAI-compatible provider config. `base_url` is required because
/// V1 supports arbitrary OpenAI-compatible endpoints, not just
/// `api.openai.com`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiChatCompletionsConfig {
    base_url: String,
    api_key: String,
    default_model: String,
}

impl OpenAiChatCompletionsConfig {
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

/// OpenAI Responses API provider config. `base_url` is required
/// because V1 supports arbitrary OpenAI-compatible endpoints, not just
/// `api.openai.com`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiResponsesConfig {
    base_url: String,
    api_key: String,
    default_model: String,
}

impl OpenAiResponsesConfig {
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
/// model. `base_url` is optional; when `None`, the adapter defaults to
/// `https://api.anthropic.com`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicMessagesConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base_url: Option<String>,
    api_key: String,
    default_model: String,
}

impl AnthropicMessagesConfig {
    pub fn new(api_key: impl Into<String>, default_model: impl Into<String>) -> Self {
        Self {
            base_url: None,
            api_key: api_key.into(),
            default_model: default_model.into(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref()
    }

    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    pub fn default_model(&self) -> &str {
        &self.default_model
    }
}

/// A single provider entry. `protocol` discriminates which inner config is
/// present. `enabled` defaults to `true`; hosts may disable a provider
/// without removing it from the config file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderConfig {
    name: String,
    #[serde(default = "default_enabled")]
    enabled: bool,
    protocol: Protocol,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    api_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    default_model: Option<String>,
    /// Strongly-typed inner configs. Only the one matching `protocol` is
    /// populated. The flat fields above exist so the on-disk JSON is
    /// readable; the typed fields are the source of truth at runtime.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    openai_chat_completions: Option<OpenAiChatCompletionsConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    anthropic_messages: Option<AnthropicMessagesConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    openai_responses: Option<OpenAiResponsesConfig>,
}

fn default_enabled() -> bool {
    true
}

impl ProviderConfig {
    pub fn with_openai_chat_completions(
        name: impl Into<String>,
        inner: OpenAiChatCompletionsConfig,
    ) -> Self {
        Self {
            name: name.into(),
            enabled: true,
            protocol: Protocol::OpenAiChatCompletions,
            base_url: Some(inner.base_url().to_string()),
            api_key: Some(inner.api_key().to_string()),
            default_model: Some(inner.default_model().to_string()),
            openai_chat_completions: Some(inner),
            anthropic_messages: None,
            openai_responses: None,
        }
    }

    pub fn with_anthropic_messages(
        name: impl Into<String>,
        inner: AnthropicMessagesConfig,
    ) -> Self {
        Self {
            name: name.into(),
            enabled: true,
            protocol: Protocol::AnthropicMessages,
            base_url: inner.base_url().map(|s| s.to_string()),
            api_key: Some(inner.api_key().to_string()),
            default_model: Some(inner.default_model().to_string()),
            openai_chat_completions: None,
            anthropic_messages: Some(inner),
            openai_responses: None,
        }
    }

    pub fn with_openai_responses(name: impl Into<String>, inner: OpenAiResponsesConfig) -> Self {
        Self {
            name: name.into(),
            enabled: true,
            protocol: Protocol::OpenAiResponses,
            base_url: Some(inner.base_url().to_string()),
            api_key: Some(inner.api_key().to_string()),
            default_model: Some(inner.default_model().to_string()),
            openai_chat_completions: None,
            anthropic_messages: None,
            openai_responses: Some(inner),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn protocol(&self) -> Protocol {
        self.protocol
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn set_enabled(&mut self, value: bool) {
        self.enabled = value;
    }

    pub fn openai_chat_completions(&self) -> Option<&OpenAiChatCompletionsConfig> {
        self.openai_chat_completions.as_ref()
    }

    pub fn anthropic_messages(&self) -> Option<&AnthropicMessagesConfig> {
        self.anthropic_messages.as_ref()
    }

    pub fn openai_responses(&self) -> Option<&OpenAiResponsesConfig> {
        self.openai_responses.as_ref()
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
    fn protocol_as_str() {
        assert_eq!(
            Protocol::OpenAiChatCompletions.as_str(),
            "openai_chat_completions"
        );
        assert_eq!(Protocol::AnthropicMessages.as_str(), "anthropic_messages");
        assert_eq!(Protocol::OpenAiResponses.as_str(), "openai_responses");
    }

    #[test]
    fn openai_responses_config_carries_base_url_api_key_and_default_model() {
        let cfg = OpenAiResponsesConfig::new("https://api.example.com/v1", "sk-test", "gpt-5-mini");
        assert_eq!(cfg.base_url(), "https://api.example.com/v1");
        assert_eq!(cfg.api_key(), "sk-test");
        assert_eq!(cfg.default_model(), "gpt-5-mini");
    }
}
