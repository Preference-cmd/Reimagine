//! Wiremock-driven integration tests for `ReqwestBackend`.
//!
//! These tests stand up a local `wiremock` server and point the
//! reqwest-backed client at it via the `base_url` config. They assert
//! the request shape (URL, method, auth header, body) and the
//! response translation back into `AgentResponse` / `Vec<ModelInfo>`.

use std::sync::Arc;

use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use reimagine_agent::{
    AgentRequest, AgentToolDefinition, Message, ModelCapability, ModelName, ProviderName,
};
use reimagine_agent_provider::{
    AnthropicMessagesConfig, CompletionBackend, OpenAiChatCompletionsConfig, ProviderAdapterError,
    ReqwestBackend, arc_real_openai_chat_completions_backend_with_http_client,
};

const OPENAI_KEY: &str = "sk-test-openai";
const ANTHROPIC_KEY: &str = "sk-test-anthropic";

fn openai_cfg_for(server: &MockServer) -> OpenAiChatCompletionsConfig {
    OpenAiChatCompletionsConfig::new(format!("{}/v1", server.uri()), OPENAI_KEY, "gpt-4o-mini")
}

fn anthropic_cfg_for(server: &MockServer) -> AnthropicMessagesConfig {
    AnthropicMessagesConfig::new(ANTHROPIC_KEY, "claude-3-5-sonnet-latest")
        .with_base_url(server.uri())
}

fn build_request(model: &str) -> AgentRequest {
    AgentRequest::new(ModelName::new(model), vec![Message::user("hi")]).with_tools(vec![
        AgentToolDefinition::new(
            "echo",
            "echo a string",
            json!({"type": "object", "properties": {"x": {"type": "number"}}}),
        ),
    ])
}

#[tokio::test]
async fn openai_complete_returns_translated_response() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", format!("Bearer {OPENAI_KEY}")))
        .and(body_partial_json(json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "echo a string",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "number"}}
                    }
                }
            }],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello back",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": "{\"x\": 42}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();

    let backend: Arc<dyn CompletionBackend> =
        arc_real_openai_chat_completions_backend_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        );

    let resp = backend
        .complete(build_request("gpt-4o-mini"))
        .await
        .expect("upstream returned 2xx");
    assert_eq!(resp.message().content(), "hello back");
    let calls = resp.message().tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id().as_str(), "call_1");
    assert_eq!(calls[0].name(), "echo");
    assert_eq!(calls[0].arguments()["x"], 42);
    assert_eq!(resp.stop_reason(), Some("tool_calls"));
    let usage = resp.usage().expect("usage present");
    assert_eq!(usage.input_tokens(), Some(7));
    assert_eq!(usage.output_tokens(), Some(4));
}

#[tokio::test]
async fn openai_complete_maps_non_2xx_to_api_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("invalid api key"))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        arc_real_openai_chat_completions_backend_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        );
    let err = backend
        .complete(build_request("gpt-4o-mini"))
        .await
        .expect_err("expected non-2xx response to surface as an error");
    match err {
        ProviderAdapterError::Api { code, message } => {
            assert_eq!(code, "401");
            assert!(message.contains("invalid api key"));
        }
        other => panic!("expected Api error, got {other:?}"),
    }
}

#[tokio::test]
async fn openai_list_models_returns_translated_listing() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("authorization", format!("Bearer {OPENAI_KEY}")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [
                { "id": "gpt-4o-mini", "object": "model" },
                { "id": "gpt-4o",       "object": "model" }
            ]
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        arc_real_openai_chat_completions_backend_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        );
    let models = backend.list_models().await.expect("list ok");
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].name().as_str(), "gpt-4o-mini");
    assert_eq!(models[1].name().as_str(), "gpt-4o");
    for m in &models {
        assert!(m.capabilities().contains(&ModelCapability::Chat));
        assert!(m.capabilities().contains(&ModelCapability::ToolUse));
        assert_eq!(m.provider().map(|p| p.as_str()), Some("openai-test"));
    }
}

#[tokio::test]
async fn anthropic_complete_returns_translated_response() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", ANTHROPIC_KEY))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello from claude"}],
            "model": "claude-3-5-sonnet-latest",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend = ReqwestBackend::anthropic_messages_with_http_client(
        ProviderName::new("anthropic-test"),
        anthropic_cfg_for(&server),
        http,
    );

    let resp = backend
        .complete(build_request("claude-3-5-sonnet-latest"))
        .await
        .expect("upstream returned 2xx");
    assert_eq!(resp.message().content(), "hello from claude");
    assert_eq!(resp.stop_reason(), Some("end_turn"));
    let usage = resp.usage().expect("usage present");
    assert_eq!(usage.input_tokens(), Some(10));
    assert_eq!(usage.output_tokens(), Some(20));
}

#[tokio::test]
async fn anthropic_complete_maps_non_2xx_to_api_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({
            "type": "error",
            "error": {"type": "authentication_error", "message": "invalid key"}
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend = ReqwestBackend::anthropic_messages_with_http_client(
        ProviderName::new("anthropic-test"),
        anthropic_cfg_for(&server),
        http,
    );

    let err = backend
        .complete(build_request("claude-3-5-sonnet-latest"))
        .await
        .expect_err("expected non-2xx");
    match err {
        ProviderAdapterError::Api { code, message } => {
            assert_eq!(code, "401");
            assert!(message.contains("authentication_error"));
        }
        other => panic!("expected Api error, got {other:?}"),
    }
}

#[tokio::test]
async fn anthropic_list_models_returns_translated_listing() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("x-api-key", ANTHROPIC_KEY))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                { "id": "claude-3-5-sonnet-latest", "type": "model" },
                { "id": "claude-3-haiku-latest", "type": "model" }
            ]
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend = ReqwestBackend::anthropic_messages_with_http_client(
        ProviderName::new("anthropic-test"),
        anthropic_cfg_for(&server),
        http,
    );

    let models = backend.list_models().await.expect("list ok");
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].name().as_str(), "claude-3-5-sonnet-latest");
    assert_eq!(models[1].name().as_str(), "claude-3-haiku-latest");
    for m in &models {
        assert!(m.capabilities().contains(&ModelCapability::Chat));
        assert!(m.capabilities().contains(&ModelCapability::ToolUse));
        assert_eq!(m.provider().map(|p| p.as_str()), Some("anthropic-test"));
    }
}
