//! Typed provider config shapes for V1.
//!
//! V1 stores provider config in a JSON document loaded by app-host. The
//! shapes here are the wire format: `enabled`, `protocol`
//! discriminator, `base_url` (chat-completions protocols only),
//! `api_key`, and `default_model`. `api_key` is held as a `String` in
//! memory. The app-host is responsible for sourcing the value from the
//! workspace's secrets file before constructing an adapter.
//!
//! The `Protocol` discriminator lives in `reimagine-agent-provider` (it
//! is a provider-adapter concept) and is re-exported here for callers
//! of the config document.

pub use reimagine_agent_provider::{
    AnthropicMessagesConfig, OpenAiChatCompletionsConfig, OpenAiResponsesConfig, Protocol,
};

use serde::{Deserialize, Serialize};

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

impl reimagine_config::ConfigDocument for AgentProviderConfigDocument {
    const KEY: &'static str = "agent-providers.json";
    const SCHEMA_VERSION: &'static str = "1";

    fn validate(
        &self,
        _context: &reimagine_config::ConfigValidationContext,
    ) -> Vec<reimagine_core::diagnostic::Diagnostic> {
        use reimagine_core::diagnostic::{
            Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
            DiagnosticTargetDomain,
        };
        use reimagine_core::model::DiagnosticId;

        let mut diagnostics = Vec::new();
        for provider in &self.providers {
            let expected = provider.protocol();
            let has_inner = match expected {
                Protocol::OpenAiChatCompletions => provider.openai_chat_completions().is_some(),
                Protocol::AnthropicMessages => provider.anthropic_messages().is_some(),
                Protocol::OpenAiResponses => provider.openai_responses().is_some(),
            };
            if !has_inner {
                diagnostics.push(Diagnostic::new(
                    DiagnosticId::new(format!(
                        "config:agent_providers:{}:missing_config",
                        provider.name()
                    )),
                    DiagnosticCode::new("CONFIG/AGENT_PROVIDER_MISSING_CONFIG"),
                    DiagnosticSeverity::Error,
                    DiagnosticSourceName::new("config"),
                    format!(
                        "provider `{}` missing config for protocol `{expected}`",
                        provider.name()
                    ),
                    DiagnosticTarget::new(DiagnosticTargetDomain::new("config"))
                        .with_id("agent-providers.json"),
                ));
            }

            // Check for mixed protocols: provider has multiple inner configs populated
            let mut populated_protocols = Vec::new();
            if provider.openai_chat_completions().is_some() {
                populated_protocols.push(Protocol::OpenAiChatCompletions);
            }
            if provider.anthropic_messages().is_some() {
                populated_protocols.push(Protocol::AnthropicMessages);
            }
            if provider.openai_responses().is_some() {
                populated_protocols.push(Protocol::OpenAiResponses);
            }

            if populated_protocols.len() > 1 {
                let protocol_names: Vec<&str> =
                    populated_protocols.iter().map(|p| p.as_str()).collect();
                diagnostics.push(
                    Diagnostic::new(
                        DiagnosticId::new(format!(
                            "config:agent_providers:{}:mixed_protocols",
                            provider.name()
                        )),
                        DiagnosticCode::new("CONFIG/AGENT_PROVIDER_MIXED_PROTOCOL"),
                        DiagnosticSeverity::Error,
                        DiagnosticSourceName::new("config"),
                        format!(
                            "provider `{}` has multiple protocol configs: {}. A provider must use exactly one protocol.",
                            provider.name(),
                            protocol_names.join(", ")
                        ),
                        DiagnosticTarget::new(DiagnosticTargetDomain::new("config"))
                            .with_id("agent-providers.json"),
                    ),
                );
            }
        }
        diagnostics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_config::ConfigDocument;
    use reimagine_config::ConfigValidationContext;

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

    #[test]
    fn mixed_protocol_provider_rejected() {
        // Create a provider with multiple inner configs populated (mixed protocols)
        let openai_config = OpenAiChatCompletionsConfig::new(
            "https://api.example.com/v1",
            "sk-test",
            "gpt-4o-mini",
        );
        let anthropic_config = AnthropicMessagesConfig::new("sk-ant-test", "claude-3-5-sonnet");

        // Manually create a ProviderConfig with multiple inner configs
        let mixed_provider = ProviderConfig {
            name: "mixed-provider".to_string(),
            enabled: true,
            protocol: Protocol::OpenAiChatCompletions,
            base_url: Some("https://api.example.com/v1".to_string()),
            api_key: Some("sk-test".to_string()),
            default_model: Some("gpt-4o-mini".to_string()),
            openai_chat_completions: Some(openai_config),
            anthropic_messages: Some(anthropic_config),
            openai_responses: None,
        };

        let doc = AgentProviderConfigDocument::new(vec![mixed_provider]);
        let key = reimagine_config::ConfigKey::new("agent-providers.json").unwrap();
        let context = ConfigValidationContext::new(key, "/test/config");
        let diagnostics = doc.validate(&context);

        // Should have exactly one diagnostic for mixed protocols
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].code().as_str(),
            "CONFIG/AGENT_PROVIDER_MIXED_PROTOCOL"
        );
        assert!(diagnostics[0].message().contains("mixed-provider"));
        assert!(
            diagnostics[0]
                .message()
                .contains("multiple protocol configs")
        );
        assert!(diagnostics[0].message().contains("openai_chat_completions"));
        assert!(diagnostics[0].message().contains("anthropic_messages"));
    }

    #[test]
    fn single_protocol_provider_accepted() {
        // Create a provider with only one inner config populated
        let openai_config = OpenAiChatCompletionsConfig::new(
            "https://api.example.com/v1",
            "sk-test",
            "gpt-4o-mini",
        );

        let provider =
            ProviderConfig::with_openai_chat_completions("single-provider", openai_config);

        let doc = AgentProviderConfigDocument::new(vec![provider]);
        let key = reimagine_config::ConfigKey::new("agent-providers.json").unwrap();
        let context = ConfigValidationContext::new(key, "/test/config");
        let diagnostics = doc.validate(&context);

        // Should have no diagnostics (valid configuration)
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn mixed_protocol_with_all_three_protocols_rejected() {
        // Create a provider with all three inner configs populated
        let openai_config = OpenAiChatCompletionsConfig::new(
            "https://api.example.com/v1",
            "sk-test",
            "gpt-4o-mini",
        );
        let anthropic_config = AnthropicMessagesConfig::new("sk-ant-test", "claude-3-5-sonnet");
        let responses_config =
            OpenAiResponsesConfig::new("https://api.example.com/v1", "sk-test", "gpt-5-mini");

        let mixed_provider = ProviderConfig {
            name: "all-protocols-provider".to_string(),
            enabled: true,
            protocol: Protocol::OpenAiChatCompletions,
            base_url: Some("https://api.example.com/v1".to_string()),
            api_key: Some("sk-test".to_string()),
            default_model: Some("gpt-4o-mini".to_string()),
            openai_chat_completions: Some(openai_config),
            anthropic_messages: Some(anthropic_config),
            openai_responses: Some(responses_config),
        };

        let doc = AgentProviderConfigDocument::new(vec![mixed_provider]);
        let key = reimagine_config::ConfigKey::new("agent-providers.json").unwrap();
        let context = ConfigValidationContext::new(key, "/test/config");
        let diagnostics = doc.validate(&context);

        // Should have exactly one diagnostic for mixed protocols
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].code().as_str(),
            "CONFIG/AGENT_PROVIDER_MIXED_PROTOCOL"
        );
        assert!(diagnostics[0].message().contains("all-protocols-provider"));
        assert!(diagnostics[0].message().contains("openai_chat_completions"));
        assert!(diagnostics[0].message().contains("anthropic_messages"));
        assert!(diagnostics[0].message().contains("openai_responses"));
    }

    #[test]
    fn mixed_protocol_diagnostic_message_contains_protocol_names() {
        // Create a provider with two inner configs populated
        let openai_config = OpenAiChatCompletionsConfig::new(
            "https://api.example.com/v1",
            "sk-test",
            "gpt-4o-mini",
        );
        let responses_config =
            OpenAiResponsesConfig::new("https://api.example.com/v1", "sk-test", "gpt-5-mini");

        let mixed_provider = ProviderConfig {
            name: "test-provider".to_string(),
            enabled: true,
            protocol: Protocol::OpenAiChatCompletions,
            base_url: Some("https://api.example.com/v1".to_string()),
            api_key: Some("sk-test".to_string()),
            default_model: Some("gpt-4o-mini".to_string()),
            openai_chat_completions: Some(openai_config),
            anthropic_messages: None,
            openai_responses: Some(responses_config),
        };

        let doc = AgentProviderConfigDocument::new(vec![mixed_provider]);
        let key = reimagine_config::ConfigKey::new("agent-providers.json").unwrap();
        let context = ConfigValidationContext::new(key, "/test/config");
        let diagnostics = doc.validate(&context);

        // Should have exactly one diagnostic for mixed protocols
        assert_eq!(diagnostics.len(), 1);

        // Check that the message contains the expected format
        let message = diagnostics[0].message();
        assert!(message.contains("test-provider"));
        assert!(message.contains("A provider must use exactly one protocol"));
        assert!(message.contains("openai_chat_completions"));
        assert!(message.contains("openai_responses"));
    }

    #[test]
    fn multiple_providers_with_mixed_protocols_each_rejected() {
        // Create two providers, each with mixed protocols
        let openai_config1 = OpenAiChatCompletionsConfig::new(
            "https://api.example.com/v1",
            "sk-test",
            "gpt-4o-mini",
        );
        let anthropic_config1 = AnthropicMessagesConfig::new("sk-ant-test", "claude-3-5-sonnet");

        let mixed_provider1 = ProviderConfig {
            name: "mixed-provider-1".to_string(),
            enabled: true,
            protocol: Protocol::OpenAiChatCompletions,
            base_url: Some("https://api.example.com/v1".to_string()),
            api_key: Some("sk-test".to_string()),
            default_model: Some("gpt-4o-mini".to_string()),
            openai_chat_completions: Some(openai_config1),
            anthropic_messages: Some(anthropic_config1),
            openai_responses: None,
        };

        let openai_config2 =
            OpenAiChatCompletionsConfig::new("https://api2.example.com/v1", "sk-test2", "gpt-4o");
        let responses_config2 =
            OpenAiResponsesConfig::new("https://api2.example.com/v1", "sk-test2", "gpt-5");

        let mixed_provider2 = ProviderConfig {
            name: "mixed-provider-2".to_string(),
            enabled: true,
            protocol: Protocol::OpenAiChatCompletions,
            base_url: Some("https://api2.example.com/v1".to_string()),
            api_key: Some("sk-test2".to_string()),
            default_model: Some("gpt-4o".to_string()),
            openai_chat_completions: Some(openai_config2),
            anthropic_messages: None,
            openai_responses: Some(responses_config2),
        };

        let doc = AgentProviderConfigDocument::new(vec![mixed_provider1, mixed_provider2]);
        let key = reimagine_config::ConfigKey::new("agent-providers.json").unwrap();
        let context = ConfigValidationContext::new(key, "/test/config");
        let diagnostics = doc.validate(&context);

        // Should have exactly two diagnostics (one for each mixed provider)
        assert_eq!(diagnostics.len(), 2);

        // Both should have the mixed protocol error code
        for diag in &diagnostics {
            assert_eq!(diag.code().as_str(), "CONFIG/AGENT_PROVIDER_MIXED_PROTOCOL");
        }
    }

    #[test]
    fn mixed_protocol_provider_with_disabled_flag_still_validated() {
        // Create a disabled provider with mixed protocols
        let openai_config = OpenAiChatCompletionsConfig::new(
            "https://api.example.com/v1",
            "sk-test",
            "gpt-4o-mini",
        );
        let anthropic_config = AnthropicMessagesConfig::new("sk-ant-test", "claude-3-5-sonnet");

        let mixed_provider = ProviderConfig {
            name: "disabled-mixed-provider".to_string(),
            enabled: false,
            protocol: Protocol::OpenAiChatCompletions,
            base_url: Some("https://api.example.com/v1".to_string()),
            api_key: Some("sk-test".to_string()),
            default_model: Some("gpt-4o-mini".to_string()),
            openai_chat_completions: Some(openai_config),
            anthropic_messages: Some(anthropic_config),
            openai_responses: None,
        };

        let doc = AgentProviderConfigDocument::new(vec![mixed_provider]);
        let key = reimagine_config::ConfigKey::new("agent-providers.json").unwrap();
        let context = ConfigValidationContext::new(key, "/test/config");
        let diagnostics = doc.validate(&context);

        // Should still validate and reject even if disabled
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].code().as_str(),
            "CONFIG/AGENT_PROVIDER_MIXED_PROTOCOL"
        );
    }
}
