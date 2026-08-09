use reimagine_agent_provider::{
    AgentProviderConfigDocument, AnthropicMessagesConfig, OpenAiChatCompletionsConfig, Protocol,
    ProviderConfig,
};
use serde_json::json;

fn openai_compat(name: &str) -> ProviderConfig {
    ProviderConfig::with_openai_chat_completions(
        name,
        OpenAiChatCompletionsConfig::new("https://api.example.com/v1", "sk-test", "gpt-4o-mini"),
    )
}

fn anthropic(name: &str) -> ProviderConfig {
    ProviderConfig::with_anthropic_messages(
        name,
        AnthropicMessagesConfig::new("sk-ant-test", "claude-3-5-sonnet"),
    )
}

#[test]
fn openai_compatible_config_carries_base_url_api_key_and_default_model() {
    let cfg =
        OpenAiChatCompletionsConfig::new("https://api.example.com/v1", "sk-test", "gpt-4o-mini");
    assert_eq!(cfg.base_url(), "https://api.example.com/v1");
    assert_eq!(cfg.api_key(), "sk-test");
    assert_eq!(cfg.default_model(), "gpt-4o-mini");
}

#[test]
fn anthropic_config_carries_api_key_and_default_model() {
    let cfg = AnthropicMessagesConfig::new("sk-ant-test", "claude-3-5-sonnet");
    assert_eq!(cfg.api_key(), "sk-ant-test");
    assert_eq!(cfg.default_model(), "claude-3-5-sonnet");
}

#[test]
fn provider_config_openai_chat_completions_protocol_and_inner_config() {
    let pc = openai_compat("openai");
    assert_eq!(pc.name(), "openai");
    assert_eq!(pc.protocol(), Protocol::OpenAiChatCompletions);
    assert!(pc.is_enabled());
    let inner = pc.openai_chat_completions().expect("openai config present");
    assert_eq!(inner.base_url(), "https://api.example.com/v1");
    assert!(pc.anthropic_messages().is_none());
}

#[test]
fn provider_config_anthropic_kind_and_inner_config() {
    let pc = anthropic("anthropic_messages");
    assert_eq!(pc.name(), "anthropic_messages");
    assert_eq!(pc.protocol(), Protocol::AnthropicMessages);
    let inner = pc.anthropic_messages().expect("anthropic config present");
    assert_eq!(inner.default_model(), "claude-3-5-sonnet");
    assert!(pc.openai_chat_completions().is_none());
}

#[test]
fn provider_config_disabled_flag_round_trips() {
    let mut pc = openai_compat("openai");
    pc.set_enabled(false);
    assert!(!pc.is_enabled());
}

#[test]
fn document_round_trips_openai_compatible_entry() {
    let doc = AgentProviderConfigDocument::new(vec![openai_compat("openai")]);
    let v = serde_json::to_value(&doc).unwrap();
    assert_eq!(v["providers"][0]["name"], "openai");
    assert_eq!(v["providers"][0]["protocol"], "openai_chat_completions");
    assert_eq!(v["providers"][0]["base_url"], "https://api.example.com/v1");
    let parsed: AgentProviderConfigDocument = serde_json::from_value(v).unwrap();
    assert_eq!(parsed.providers().len(), 1);
    assert_eq!(
        parsed.providers()[0].protocol(),
        Protocol::OpenAiChatCompletions
    );
}

#[test]
fn document_round_trips_anthropic_entry() {
    let doc = AgentProviderConfigDocument::new(vec![anthropic("anthropic_messages")]);
    let v = serde_json::to_value(&doc).unwrap();
    assert_eq!(v["providers"][0]["name"], "anthropic_messages");
    assert_eq!(v["providers"][0]["protocol"], "anthropic_messages");
    assert_eq!(v["providers"][0]["default_model"], "claude-3-5-sonnet");
    let parsed: AgentProviderConfigDocument = serde_json::from_value(v).unwrap();
    assert_eq!(
        parsed.providers()[0].protocol(),
        Protocol::AnthropicMessages
    );
}

#[test]
fn document_rejects_unknown_protocol() {
    let v = json!({ "providers": [{ "name": "x", "protocol": "wat" }] });
    let res: Result<AgentProviderConfigDocument, _> = serde_json::from_value(v);
    assert!(res.is_err(), "unknown kind must not deserialize");
}
