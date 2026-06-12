use reimagine_agent_provider::{
    AgentProviderConfigDocument, AnthropicConfig, OpenAiCompatibleConfig, ProviderConfig,
    ProviderKind,
};
use serde_json::json;

fn openai_compat(name: &str) -> ProviderConfig {
    ProviderConfig::with_openai_compatible(
        name,
        OpenAiCompatibleConfig::new("https://api.example.com/v1", "sk-test", "gpt-4o-mini"),
    )
}

fn anthropic(name: &str) -> ProviderConfig {
    ProviderConfig::with_anthropic(
        name,
        AnthropicConfig::new("sk-ant-test", "claude-3-5-sonnet"),
    )
}

#[test]
fn openai_compatible_config_carries_base_url_api_key_and_default_model() {
    let cfg = OpenAiCompatibleConfig::new("https://api.example.com/v1", "sk-test", "gpt-4o-mini");
    assert_eq!(cfg.base_url(), "https://api.example.com/v1");
    assert_eq!(cfg.api_key(), "sk-test");
    assert_eq!(cfg.default_model(), "gpt-4o-mini");
}

#[test]
fn anthropic_config_carries_api_key_and_default_model() {
    let cfg = AnthropicConfig::new("sk-ant-test", "claude-3-5-sonnet");
    assert_eq!(cfg.api_key(), "sk-ant-test");
    assert_eq!(cfg.default_model(), "claude-3-5-sonnet");
}

#[test]
fn provider_config_openai_compatible_kind_and_inner_config() {
    let pc = openai_compat("openai");
    assert_eq!(pc.name(), "openai");
    assert_eq!(pc.kind(), ProviderKind::OpenAiCompatible);
    assert!(pc.is_enabled());
    let inner = pc.openai_compatible().expect("openai config present");
    assert_eq!(inner.base_url(), "https://api.example.com/v1");
    assert!(pc.anthropic().is_none());
}

#[test]
fn provider_config_anthropic_kind_and_inner_config() {
    let pc = anthropic("anthropic");
    assert_eq!(pc.name(), "anthropic");
    assert_eq!(pc.kind(), ProviderKind::Anthropic);
    let inner = pc.anthropic().expect("anthropic config present");
    assert_eq!(inner.default_model(), "claude-3-5-sonnet");
    assert!(pc.openai_compatible().is_none());
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
    assert_eq!(v["providers"][0]["kind"], "openai_compatible");
    assert_eq!(v["providers"][0]["base_url"], "https://api.example.com/v1");
    let parsed: AgentProviderConfigDocument = serde_json::from_value(v).unwrap();
    assert_eq!(parsed.providers().len(), 1);
    assert_eq!(parsed.providers()[0].kind(), ProviderKind::OpenAiCompatible);
}

#[test]
fn document_round_trips_anthropic_entry() {
    let doc = AgentProviderConfigDocument::new(vec![anthropic("anthropic")]);
    let v = serde_json::to_value(&doc).unwrap();
    assert_eq!(v["providers"][0]["name"], "anthropic");
    assert_eq!(v["providers"][0]["kind"], "anthropic");
    assert_eq!(v["providers"][0]["default_model"], "claude-3-5-sonnet");
    let parsed: AgentProviderConfigDocument = serde_json::from_value(v).unwrap();
    assert_eq!(parsed.providers()[0].kind(), ProviderKind::Anthropic);
}

#[test]
fn document_rejects_unknown_kind() {
    let v = json!({ "providers": [{ "name": "x", "kind": "wat" }] });
    let res: Result<AgentProviderConfigDocument, _> = serde_json::from_value(v);
    assert!(res.is_err(), "unknown kind must not deserialize");
}
