use reimagine_agent::AgentStreamEvent;
use reimagine_agent_provider::translation::streaming::AnthropicStreamAccumulator;
use serde_json::json;

#[tokio::test]
async fn anthropic_accumulator_emits_text_deltas_complete_tool_call_and_done() {
    let events = vec![
        json!({ "type": "message_start" }),
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": { "type": "text", "text": "" }
        }),
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": { "type": "text_delta", "text": "Hello" }
        }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "c1",
                "name": "echo",
                "input": {}
            }
        }),
        json!({
            "type": "content_block_delta",
            "index": 1,
            "delta": { "type": "input_json_delta", "partial_json": "{\"x\":" }
        }),
        json!({
            "type": "content_block_delta",
            "index": 1,
            "delta": { "type": "input_json_delta", "partial_json": "1}" }
        }),
        json!({ "type": "content_block_stop", "index": 1 }),
        json!({
            "type": "message_delta",
            "delta": { "stop_reason": "tool_use" },
            "usage": { "input_tokens": 1, "output_tokens": 2 }
        }),
        json!({ "type": "message_stop" }),
    ];
    let mut acc = AnthropicStreamAccumulator::new();
    let mut collected = Vec::new();
    for e in &events {
        collected.extend(acc.ingest_event(e).unwrap());
    }
    let mut kinds: Vec<&'static str> = Vec::new();
    for e in &collected {
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
    assert_eq!(*kinds.last().unwrap(), "done");

    // Find the complete tool call and check it.
    let complete = collected
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

#[tokio::test]
async fn anthropic_stream_missing_event_type_is_serialization_error() {
    let mut acc = AnthropicStreamAccumulator::new();
    let err = acc.ingest_event(&json!({})).unwrap_err();
    assert!(matches!(
        err,
        reimagine_agent_provider::ProviderAdapterError::Serialization(_)
    ));
}
