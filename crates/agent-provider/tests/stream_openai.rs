use reimagine_agent::{
    AgentProvider, AgentRequest, AgentResponse, AgentStreamEvent, Message, ModelName, ProviderName,
};
use reimagine_agent_provider::translation::streaming::OpenAiStreamAccumulator;
use reimagine_agent_provider::{
    CompletionBackend, FakeCompletionBackend, OpenAiCompatibleConfig, OpenAiCompatibleProvider,
    ScriptedBackendStep,
};
use serde_json::json;
use std::sync::Arc;

#[tokio::test]
async fn openai_adapter_complete_returns_response_and_maps_error() {
    let backend: Arc<dyn CompletionBackend> = Arc::new(FakeCompletionBackend::new(vec![
        ScriptedBackendStep::Complete(Ok(
            AgentResponse::new(Message::assistant("hi back")).with_stop_reason("stop")
        )),
    ]));
    let provider = OpenAiCompatibleProvider::with_backend(
        ProviderName::new("openai"),
        OpenAiCompatibleConfig::new("https://api.example.com/v1", "sk", "gpt-4o-mini"),
        backend,
    );
    let req = AgentRequest::new(ModelName::new("gpt-4o-mini"), vec![Message::user("hi")]);
    let resp = provider.complete(req).await.expect("complete ok");
    assert_eq!(resp.message().content(), "hi back");
    assert_eq!(resp.stop_reason(), Some("stop"));
}

#[tokio::test]
async fn openai_adapter_complete_maps_backend_error_to_provider_error() {
    let backend: Arc<dyn CompletionBackend> = Arc::new(FakeCompletionBackend::new(vec![
        ScriptedBackendStep::Complete(Err(
            reimagine_agent_provider::ProviderAdapterError::transport("connection refused"),
        )),
    ]));
    let provider = OpenAiCompatibleProvider::with_backend(
        ProviderName::new("openai"),
        OpenAiCompatibleConfig::new("https://api.example.com/v1", "sk", "gpt-4o-mini"),
        backend,
    );
    let req = AgentRequest::new(ModelName::new("gpt-4o-mini"), vec![Message::user("hi")]);
    let err = provider
        .complete(req)
        .await
        .expect_err("provider error expected");
    assert_eq!(err.code(), "TRANSPORT");
    assert!(err.message().contains("connection refused"));
    assert_eq!(err.provider().map(|p| p.as_str()), Some("openai"));
}

#[tokio::test]
async fn openai_adapter_stream_emits_deltas_complete_tool_call_and_done() {
    // Simulate the OpenAI chunk shape across three chunks.
    let chunks = vec![
        json!({
            "choices": [{
                "delta": { "role": "assistant", "content": "He" }
            }]
        }),
        json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "c1",
                        "function": { "name": "echo", "arguments": "{\"x\":" }
                    }]
                }
            }]
        }),
        json!({
            "choices": [{
                "delta": { "tool_calls": [{ "index": 0, "function": { "arguments": "1}" } }] },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 3, "completion_tokens": 4 }
        }),
    ];
    let mut acc = OpenAiStreamAccumulator::new();
    let mut events = Vec::new();
    for chunk in &chunks {
        events.extend(acc.ingest_chunk(chunk).unwrap());
    }
    let complete_calls = acc.flush_complete_tool_calls();
    events.extend(complete_calls);
    events.push(acc.finalize());

    // Expect: ContentDelta("He"), ToolCallDelta id, ToolCallDelta name,
    // ToolCallDelta args (×2), Usage, ToolCall, Done.
    let mut kinds: Vec<&'static str> = Vec::new();
    for e in &events {
        match e {
            AgentStreamEvent::ContentDelta(_) => kinds.push("content"),
            AgentStreamEvent::ToolCallDelta { .. } => kinds.push("delta"),
            AgentStreamEvent::ToolCall(_) => kinds.push("complete"),
            AgentStreamEvent::Usage(_) => kinds.push("usage"),
            AgentStreamEvent::Done { .. } => kinds.push("done"),
        }
    }
    assert!(kinds.contains(&"content"));
    assert!(kinds.contains(&"delta"));
    assert!(kinds.contains(&"complete"));
    assert!(kinds.contains(&"usage"));
    assert_eq!(kinds.last(), Some(&"done"));

    // Find the complete tool call and check it.
    let complete = events
        .iter()
        .find_map(|e| match e {
            AgentStreamEvent::ToolCall(c) => Some(c),
            _ => None,
        })
        .expect("complete tool call emitted");
    assert_eq!(complete.id().as_str(), "c1");
    assert_eq!(complete.name(), "echo");
    assert_eq!(complete.arguments(), &json!({"x": 1}));
}
