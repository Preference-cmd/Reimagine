//! Wiremock-driven integration tests for `ReqwestBackend`.
//!
//! These tests stand up a local `wiremock` server and point the
//! reqwest-backed client at it via the `base_url` config. They assert
//! the request shape (URL, method, auth header, body) and the
//! response translation back into `AgentResponse` / `Vec<ModelInfo>`.

use std::sync::Arc;

use reimagine_agent::{
    AgentRequest, AgentToolDefinition, Message, ModelCapability, ModelName, ProviderName,
};
use reimagine_agent_provider::{
    AnthropicMessagesConfig, CompletionBackend, OpenAiChatCompletionsConfig, OpenAiResponsesConfig,
    ProviderAdapterError, ReqwestBackend,
    arc_real_openai_chat_completions_backend_with_http_client,
    arc_real_openai_responses_backend_with_http_client,
};
use serde_json::{Value, json};
use wiremock::matchers::BodyExactMatcher;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn body_exact_json(body: serde_json::Value) -> BodyExactMatcher {
    BodyExactMatcher::json(body)
}

const OPENAI_KEY: &str = "sk-test-openai";
const ANTHROPIC_KEY: &str = "sk-test-anthropic";
const RESPONSES_KEY: &str = "sk-test-responses";

fn openai_cfg_for(server: &MockServer) -> OpenAiChatCompletionsConfig {
    OpenAiChatCompletionsConfig::new(format!("{}/v1", server.uri()), OPENAI_KEY, "gpt-4o-mini")
}

fn anthropic_cfg_for(server: &MockServer) -> AnthropicMessagesConfig {
    AnthropicMessagesConfig::new(ANTHROPIC_KEY, "claude-3-5-sonnet-latest")
        .with_base_url(server.uri())
}

fn responses_cfg_for(server: &MockServer) -> OpenAiResponsesConfig {
    OpenAiResponsesConfig::new(format!("{}/v1", server.uri()), RESPONSES_KEY, "gpt-5-mini")
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

fn build_request_with_options(model: &str, options: Value) -> AgentRequest {
    build_request(model).with_options(options)
}

fn openai_completion_response() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": "cmpl-2",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    }))
}

fn anthropic_completion_response() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": "msg_2",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}],
        "model": "claude-3-5-sonnet-latest",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1}
    }))
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

fn responses_request() -> AgentRequest {
    AgentRequest::new(
        ModelName::new("gpt-5-mini"),
        vec![
            Message::system("sys"),
            Message::user("hi"),
            Message::assistant_with_tool_calls(
                "",
                vec![reimagine_agent::ToolCall::new(
                    reimagine_agent::ToolCallId::new("c1"),
                    "echo",
                    json!({"x": 1}),
                )],
            ),
            Message::tool_result(reimagine_agent::ToolCallId::new("c1"), "ok"),
        ],
    )
    .with_tools(vec![AgentToolDefinition::new(
        "echo",
        "echo a string",
        json!({"type": "object", "properties": {"x": {"type": "number"}}}),
    )])
}

#[tokio::test]
async fn responses_complete_assembles_full_input_array_and_prompt_cache_key() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(header("authorization", format!("Bearer {RESPONSES_KEY}")))
        .and(body_partial_json(json!({
            "model": "gpt-5-mini",
            "instructions": "sys",
            "input": [
                { "role": "user", "content": [{ "type": "input_text", "text": "hi" }] },
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "echo",
                    "arguments": "{\"x\":1}"
                },
                { "type": "function_call_output", "call_id": "c1", "output": "ok" }
            ],
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
            "prompt_cache_key": "session-42",
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_1",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "hello back" }]
            }],
            "usage": { "input_tokens": 9, "output_tokens": 3, "total_tokens": 12 }
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> = arc_real_openai_responses_backend_with_http_client(
        ProviderName::new("responses-test"),
        responses_cfg_for(&server),
        http,
    );

    let request = responses_request().with_options(json!({ "prompt_cache_key": "session-42" }));
    let resp = backend
        .complete(request)
        .await
        .expect("upstream returned 2xx");
    assert_eq!(resp.message().content(), "hello back");
    assert_eq!(resp.message().tool_calls().len(), 0);
    let usage = resp.usage().expect("usage present");
    assert_eq!(usage.input_tokens(), Some(9));
    assert_eq!(usage.output_tokens(), Some(3));
}

#[tokio::test]
async fn responses_complete_omits_prompt_cache_key_when_absent() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(header("authorization", format!("Bearer {RESPONSES_KEY}")))
        .and(body_exact_json(json!({
            "model": "gpt-5-mini",
            "instructions": "sys",
            "input": [
                { "role": "user", "content": [{ "type": "input_text", "text": "hi" }] },
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "echo",
                    "arguments": "{\"x\":1}"
                },
                { "type": "function_call_output", "call_id": "c1", "output": "ok" }
            ],
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
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_2",
            "output": [],
            "usage": { "input_tokens": 1, "output_tokens": 1, "total_tokens": 2 }
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> = arc_real_openai_responses_backend_with_http_client(
        ProviderName::new("responses-test"),
        responses_cfg_for(&server),
        http,
    );

    let resp = backend
        .complete(responses_request())
        .await
        .expect("upstream returned 2xx");
    assert!(resp.usage().is_some());
}

#[tokio::test]
async fn responses_complete_parses_function_call_items() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_3",
            "output": [{
                "type": "function_call",
                "call_id": "call_1",
                "name": "echo",
                "arguments": "{\"x\": 42}"
            }],
            "usage": { "input_tokens": 5, "output_tokens": 8, "total_tokens": 13 }
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> = arc_real_openai_responses_backend_with_http_client(
        ProviderName::new("responses-test"),
        responses_cfg_for(&server),
        http,
    );

    let resp = backend
        .complete(responses_request())
        .await
        .expect("upstream returned 2xx");
    let calls = resp.message().tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id().as_str(), "call_1");
    assert_eq!(calls[0].name(), "echo");
    assert_eq!(calls[0].arguments(), &json!({"x": 42}));
}

#[tokio::test]
async fn responses_list_models_reuses_models_path() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("authorization", format!("Bearer {RESPONSES_KEY}")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [
                { "id": "gpt-5-mini", "object": "model" },
                { "id": "gpt-5",       "object": "model" }
            ]
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> = arc_real_openai_responses_backend_with_http_client(
        ProviderName::new("responses-test"),
        responses_cfg_for(&server),
        http,
    );
    let models = backend.list_models().await.expect("list ok");
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].name().as_str(), "gpt-5-mini");
    assert_eq!(models[1].name().as_str(), "gpt-5");
    for m in &models {
        assert_eq!(m.provider().map(|p| p.as_str()), Some("responses-test"));
    }
}

#[tokio::test]
async fn responses_stream_returns_configuration_error_until_pv_01b() {
    let server = MockServer::start().await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> = arc_real_openai_responses_backend_with_http_client(
        ProviderName::new("responses-test"),
        responses_cfg_for(&server),
        http,
    );
    let err = match backend.stream(responses_request()).await {
        Err(err) => err,
        Ok(_) => panic!("expected streaming to be rejected"),
    };
    assert!(matches!(err, ProviderAdapterError::Configuration(_)));
}

#[tokio::test]
async fn openai_complete_forwards_sampling_params_and_unknown_keys() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "model": "gpt-4o-mini",
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["end"],
            "seed": 42,
            "user": "u1",
            "frequency_penalty": 0.2,
        })))
        .respond_with(openai_completion_response())
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        arc_real_openai_chat_completions_backend_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        );
    let req = build_request_with_options(
        "gpt-4o-mini",
        json!({
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": "end",
            "seed": 42,
            "user": "u1",
            "frequency_penalty": 0.2,
        }),
    );
    let resp = backend.complete(req).await.expect("complete ok");
    assert_eq!(resp.message().content(), "ok");
}

#[tokio::test]
async fn openai_complete_with_reasoning_strips_temperature_and_top_p() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(openai_completion_response())
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        arc_real_openai_chat_completions_backend_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        );
    let req = build_request_with_options(
        "gpt-4o-mini",
        json!({
            "reasoning_effort": "high",
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": "end",
            "seed": 42,
        }),
    );
    backend.complete(req).await.expect("complete ok");

    let requests = server.received_requests().await.expect("requests captured");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("body json");
    assert_eq!(body["reasoning_effort"], "high");
    assert_eq!(body["stop"], json!(["end"]));
    assert_eq!(body["seed"], 42);
    assert!(body.get("temperature").is_none());
    assert!(body.get("top_p").is_none());
}

#[tokio::test]
async fn openai_stream_forwards_sampling_params() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "model": "gpt-4o-mini",
            "stream": true,
            "max_tokens": 256,
            "temperature": 0.2,
            "seed": 7,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_string("data: [DONE]\n\n"))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        arc_real_openai_chat_completions_backend_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        );
    let req = build_request_with_options(
        "gpt-4o-mini",
        json!({"max_tokens": 256, "temperature": 0.2, "seed": 7}),
    );
    backend.stream(req).await.expect("stream starts");
}

#[tokio::test]
async fn anthropic_complete_forwards_sampling_params() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({
            "model": "claude-3-5-sonnet-latest",
            "max_tokens": 1024,
            "temperature": 0.4,
            "top_p": 0.8,
            "top_k": 32,
            "stop_sequences": ["a", "b"],
        })))
        .respond_with(anthropic_completion_response())
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend = ReqwestBackend::anthropic_messages_with_http_client(
        ProviderName::new("anthropic-test"),
        anthropic_cfg_for(&server),
        http,
    );
    let req = build_request_with_options(
        "claude-3-5-sonnet-latest",
        json!({
            "max_tokens": 1024,
            "temperature": 0.4,
            "top_p": 0.8,
            "top_k": 32,
            "stop": ["a", "b"],
        }),
    );
    let resp = backend.complete(req).await.expect("complete ok");
    assert_eq!(resp.message().content(), "ok");
}

#[tokio::test]
async fn anthropic_complete_with_reasoning_strips_inapplicable_params() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(anthropic_completion_response())
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend = ReqwestBackend::anthropic_messages_with_http_client(
        ProviderName::new("anthropic-test"),
        anthropic_cfg_for(&server),
        http,
    );
    let req = build_request_with_options(
        "claude-3-5-sonnet-latest",
        json!({
            "reasoning_effort": "high",
            "temperature": 0.4,
            "top_p": 0.8,
            "top_k": 32,
            "stop": ["a"],
            "seed": 9,
            "user": "u1",
        }),
    );
    backend.complete(req).await.expect("complete ok");

    let requests = server.received_requests().await.expect("requests captured");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("body json");
    assert_eq!(body["max_tokens"], 4096);
    assert_eq!(body["top_k"], 32);
    assert_eq!(body["stop_sequences"], json!(["a"]));
    assert!(body.get("temperature").is_none());
    assert!(body.get("top_p").is_none());
    assert!(body.get("reasoning_effort").is_none());
    assert!(body.get("seed").is_none());
    assert!(body.get("user").is_none());
}

#[tokio::test]
async fn anthropic_stream_forwards_sampling_params() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({
            "model": "claude-3-5-sonnet-latest",
            "stream": true,
            "max_tokens": 2048,
            "temperature": 0.4,
            "top_k": 32,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_string(""))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend = ReqwestBackend::anthropic_messages_with_http_client(
        ProviderName::new("anthropic-test"),
        anthropic_cfg_for(&server),
        http,
    );
    let req = build_request_with_options(
        "claude-3-5-sonnet-latest",
        json!({"max_tokens": 2048, "temperature": 0.4, "top_k": 32}),
    );
    backend.stream(req).await.expect("stream starts");
}
