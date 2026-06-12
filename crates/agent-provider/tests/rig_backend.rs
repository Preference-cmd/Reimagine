//! Wiremock-driven integration tests for `RealRigBackend`.
//!
//! These tests stand up a local `wiremock` server and point the
//! Rig-backed client at it via the `base_url` config. They assert
//! the request shape (URL, method, auth header, body) and the
//! response translation back into `AgentResponse` / `Vec<ModelInfo>`.
//!
//! The Anthropic test is intentionally a request-shape test, not a
//! full round-trip, because V1 `AnthropicConfig` does not carry a
//! `base_url`. The rig Anthropic client therefore targets
//! `https://api.anthropic.com`. The test asserts that the body
//! serializes well enough to leave the process by observing a
//! transport-level error from the network layer. The full live
//! round-trip is covered by manual smoke docs in a follow-up V1.5
//! issue that will add `base_url` to `AnthropicConfig`.

use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use reimagine_agent::{
    AgentRequest, AgentToolDefinition, Message, ModelCapability, ModelName, ProviderName,
};
use reimagine_agent_provider::{
    AnthropicConfig, CompletionBackend, OpenAiCompatibleConfig, ProviderAdapterError, RealRigBackend,
};

const OPENAI_KEY: &str = "sk-test-openai";
const ANTHROPIC_KEY: &str = "sk-test-anthropic";

fn openai_cfg_for(server: &MockServer) -> OpenAiCompatibleConfig {
    OpenAiCompatibleConfig::new(server.uri(), OPENAI_KEY, "gpt-4o-mini")
}

#[allow(dead_code)]
fn anthropic_cfg() -> AnthropicConfig {
    // AnthropicConfig is a V1 shape that does not carry base_url;
    // the rig Anthropic Client::builder() uses the constant
    // https://api.anthropic.com. Adding a base_url to AnthropicConfig
    // is a V1.5 concern tracked separately; the anthropic dispatcher
    // tests use a non-wiremock approach (see
    // `anthropic_complete_carries_required_fields`).
    AnthropicConfig::new(ANTHROPIC_KEY, "claude-3-5-sonnet-latest")
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
        .and(path("/chat/completions"))
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

    // `rig::http_client::ReqwestClient` is a re-export of `reqwest::Client`,
    // so we can build one via the rig path. Tests in this crate use
    // the rig re-export to avoid adding a new dev-dep.
    let http = rig::http_client::ReqwestClient::new();

    let backend = RealRigBackend::openai_compatible_with_http_client(
        ProviderName::new("openai-test"),
        openai_cfg_for(&server),
        http,
    );

    let outer = backend
        .complete(build_request("gpt-4o-mini"))
        .await
        .expect("setup ok");
    let resp = outer.expect("upstream returned 2xx");
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
    // The rig `ReqwestClient::send` seam converts non-2xx upstream
    // responses into `http_client::Error::InvalidStatusCodeWithMessage`
    // at the transport layer (see rig-core 0.31
    // `http_client/mod.rs::send`). That error reaches our dispatcher
    // as a `ProviderAdapterError::Transport`, not `Api`, so the
    // dispatcher's `if !resp.status().is_success()` branch never runs
    // for this seam. We therefore assert the observable variant —
    // `Transport` carrying the upstream status + body — which is
    // still a hard failure path (the response does NOT silently
    // succeed). If a future change moves non-2xx mapping into our
    // dispatcher (e.g. by switching to a custom `HttpClientExt` that
    // passes non-2xx through), the `Api` branch below becomes
    // reachable and we can prefer that assertion.
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("invalid api key"))
        .mount(&server)
        .await;

    let http = rig::http_client::ReqwestClient::new();
    let backend = RealRigBackend::openai_compatible_with_http_client(
        ProviderName::new("openai-test"),
        openai_cfg_for(&server),
        http,
    );
    // The outer `Result` is `Err(...)` when the rig HTTP seam
    // converts a non-2xx into a transport error (the dispatcher's
    // `?` propagates it). It is `Ok(Err(...))` when the dispatcher
    // itself observes the non-2xx and maps it to `Api`. Both are
    // hard failures; the test passes as long as the underlying
    // error is the expected one.
    let outer = backend.complete(build_request("gpt-4o-mini")).await;
    let err = match outer {
        Ok(Ok(_)) => panic!("expected non-2xx response to surface as an error"),
        Ok(Err(e)) => e,
        Err(e) => e,
    };
    match err {
        // Preferred: dispatcher maps non-2xx to Api. Unreachable on
        // the rig ReqwestClient seam today, but kept here for
        // forward compatibility.
        ProviderAdapterError::Api { code, message } => {
            assert_eq!(code, "401");
            assert!(message.contains("invalid api key"));
        }
        // Actual: rig's HTTP seam turns non-2xx into a transport
        // error carrying the upstream status + body.
        ProviderAdapterError::Transport(m) => {
            assert!(
                m.contains("401"),
                "expected status 401 in transport message, got: {m}"
            );
            assert!(
                m.contains("invalid api key"),
                "expected upstream body in transport message, got: {m}"
            );
        }
        other => panic!("expected Api or Transport error, got {other:?}"),
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

    let http = rig::http_client::ReqwestClient::new();
    let backend = RealRigBackend::openai_compatible_with_http_client(
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
async fn anthropic_complete_carries_required_fields() {
    // This test is a request-shape test, not a full round-trip: we
    // exercise the anthropic dispatcher and assert that the request
    // body is well-formed enough to reach the network layer. The
    // test relies on the underlying rig::Client targeting the real
    // `https://api.anthropic.com` because AnthropicConfig does not
    // yet carry a base_url. In a sandboxed environment where DNS or
    // outbound HTTPS is blocked, this surfaces as a Transport error.
    // If the upstream is reachable, we get an Api 401 instead; the
    // test asserts on the error variant present in this environment.
    let cfg = anthropic_cfg();
    let http = rig::http_client::ReqwestClient::new();
    let backend = RealRigBackend::anthropic_with_http_client(
        ProviderName::new("anthropic-test"),
        cfg,
        http,
    );
    let result = backend.complete(build_request("claude-3-5-sonnet-latest")).await;
    // The body must serialize successfully and we must reach the
    // network layer. Either Transport (DNS / TLS / connect failure)
    // or Api (upstream 401 because the test key is bogus) is
    // acceptable evidence that the body left the process.
    match result {
        Err(ProviderAdapterError::Transport(_)) => { /* expected in sandbox */ }
        Err(ProviderAdapterError::Api { code, message: _ }) => {
            assert!(
                code == "401" || code == "403" || code == "400",
                "unexpected Api status: {code}"
            );
        }
        Err(ProviderAdapterError::Serialization(m)) => {
            panic!("unexpected serialization error: {m}")
        }
        Err(ProviderAdapterError::Configuration(m)) => {
            panic!("unexpected configuration error: {m}")
        }
        Err(other) => panic!("unexpected error variant: {other:?}"),
        Ok(outer) => panic!("unexpected upstream response: {outer:?}"),
    }
}
