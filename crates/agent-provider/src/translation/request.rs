use serde_json::{Value, json};

use reimagine_agent::Message;

/// Build the `messages` array for an OpenAI-compatible chat completion request
/// from a slice of [`Message`]. Tool messages are mapped to role `"tool"` with
/// `tool_call_id` attached. Assistant messages that contain tool calls are
/// mapped to role `"assistant"` with a `tool_calls` array.
pub fn to_openai_messages(messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        let role = m.role();
        match role {
            "system" | "user" => {
                out.push(json!({ "role": role, "content": m.content() }));
            }
            "assistant" => {
                if m.tool_calls().is_empty() {
                    out.push(json!({ "role": "assistant", "content": m.content() }));
                } else {
                    let calls: Vec<Value> = m
                        .tool_calls()
                        .iter()
                        .map(|c| {
                            json!({
                                "id": c.id().as_str(),
                                "type": "function",
                                "function": {
                                    "name": c.name(),
                                    "arguments": c.arguments().to_string(),
                                }
                            })
                        })
                        .collect();
                    let mut obj = json!({ "role": "assistant", "tool_calls": calls });
                    if !m.content().is_empty() {
                        obj["content"] = json!(m.content());
                    } else {
                        obj["content"] = Value::Null;
                    }
                    out.push(obj);
                }
            }
            "tool" => {
                let id = m
                    .tool_call_id()
                    .map(|i| i.as_str().to_string())
                    .unwrap_or_default();
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": id,
                    "content": m.content(),
                }));
            }
            other => {
                out.push(
                    json!({ "role": "user", "content": format!("[{other}] {}", m.content()) }),
                );
            }
        }
    }
    out
}

/// Build the `messages` array for an Anthropic messages API call. System
/// content is returned as a separate `system` field; the caller is responsible
/// for putting it on the request envelope. Assistant tool calls become
/// `tool_use` content blocks; tool messages become `tool_result` content
/// blocks.
pub fn to_anthropic_messages(messages: &[Message]) -> (Option<String>, Vec<Value>) {
    let mut system: Option<String> = None;
    let mut out: Vec<Value> = Vec::with_capacity(messages.len());
    for m in messages {
        match m.role() {
            "system" => {
                system = Some(match system {
                    Some(existing) => format!("{existing}\n{}", m.content()),
                    None => m.content().to_string(),
                });
            }
            "user" => {
                out.push(json!({ "role": "user", "content": m.content() }));
            }
            "assistant" => {
                if m.tool_calls().is_empty() {
                    out.push(json!({ "role": "assistant", "content": m.content() }));
                } else {
                    let blocks: Vec<Value> = m
                        .tool_calls()
                        .iter()
                        .map(|c| {
                            json!({
                                "type": "tool_use",
                                "id": c.id().as_str(),
                                "name": c.name(),
                                "input": c.arguments(),
                            })
                        })
                        .collect();
                    let mut content: Vec<Value> = Vec::new();
                    if !m.content().is_empty() {
                        content.push(json!({ "type": "text", "text": m.content() }));
                    }
                    content.extend(blocks);
                    out.push(json!({ "role": "assistant", "content": content }));
                }
            }
            "tool" => {
                let id = m
                    .tool_call_id()
                    .map(|i| i.as_str().to_string())
                    .unwrap_or_default();
                out.push(json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": id,
                        "content": m.content(),
                    }],
                }));
            }
            other => {
                out.push(
                    json!({ "role": "user", "content": format!("[{other}] {}", m.content()) }),
                );
            }
        }
    }
    (system, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_agent::{ToolCall, ToolCallId};
    use serde_json::json;

    #[test]
    fn openai_messages_user_and_system() {
        let msgs = vec![Message::system("sys"), Message::user("hi")];
        let v = to_openai_messages(&msgs);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0]["role"], "system");
        assert_eq!(v[1]["role"], "user");
    }

    #[test]
    fn openai_messages_assistant_with_tool_calls_uses_function_shape() {
        let call = ToolCall::new(ToolCallId::new("c1"), "echo", json!({"x": 1}));
        let msgs = vec![Message::assistant_with_tool_calls("", vec![call])];
        let v = to_openai_messages(&msgs);
        assert_eq!(v[0]["role"], "assistant");
        assert_eq!(v[0]["tool_calls"][0]["function"]["name"], "echo");
        assert_eq!(v[0]["tool_calls"][0]["id"], "c1");
    }

    #[test]
    fn openai_messages_tool_role_carries_tool_call_id() {
        let msgs = vec![Message::tool_result(ToolCallId::new("c1"), "ok")];
        let v = to_openai_messages(&msgs);
        assert_eq!(v[0]["role"], "tool");
        assert_eq!(v[0]["tool_call_id"], "c1");
        assert_eq!(v[0]["content"], "ok");
    }

    #[test]
    fn anthropic_messages_splits_system_out() {
        let msgs = vec![Message::system("sys"), Message::user("hi")];
        let (system, v) = to_anthropic_messages(&msgs);
        assert_eq!(system.as_deref(), Some("sys"));
        assert_eq!(v.len(), 1);
        assert_eq!(v[0]["role"], "user");
    }

    #[test]
    fn anthropic_messages_assistant_tool_call_uses_tool_use_block() {
        let call = ToolCall::new(ToolCallId::new("c1"), "echo", json!({"x": 1}));
        let msgs = vec![Message::assistant_with_tool_calls("", vec![call])];
        let (_, v) = to_anthropic_messages(&msgs);
        assert_eq!(v[0]["role"], "assistant");
        assert_eq!(v[0]["content"][0]["type"], "tool_use");
        assert_eq!(v[0]["content"][0]["id"], "c1");
        assert_eq!(v[0]["content"][0]["name"], "echo");
        assert_eq!(v[0]["content"][0]["input"]["x"], 1);
    }

    #[test]
    fn anthropic_messages_tool_role_becomes_tool_result_block() {
        let msgs = vec![Message::tool_result(ToolCallId::new("c1"), "ok")];
        let (_, v) = to_anthropic_messages(&msgs);
        assert_eq!(v[0]["role"], "user");
        assert_eq!(v[0]["content"][0]["type"], "tool_result");
        assert_eq!(v[0]["content"][0]["tool_use_id"], "c1");
        assert_eq!(v[0]["content"][0]["content"], "ok");
    }
}
