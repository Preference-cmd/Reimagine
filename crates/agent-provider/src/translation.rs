//! Translation between Reimagine DTOs and provider-native DTOs.
//!
//! Sections:
//! 1. [`request`] — `AgentRequest` to OpenAI chat messages / Anthropic messages.
//! 2. [`response`] — OpenAI / Anthropic response JSON into `AgentResponse`.
//! 3. [`tools`] — `AgentToolDefinition` to OpenAI / Anthropic tool schemas.
//! 4. [`streaming`] — OpenAI / Anthropic streaming events into
//!    `AgentStreamEvent`, with a tool-call-delta accumulator state
//!    machine.
//!
//! The translation functions take `serde_json::Value` slices so this
//! module does not need to import the rig-core types directly; the
//! adapter layer is responsible for feeding in already-deserialized
//! request/response payloads.

use serde_json::{Value, json};

use reimagine_agent::{Message, ToolCall, ToolCallId};

use crate::error::ProviderAdapterError;

// ---------------------------------------------------------------------------
// Section 1: request translation
// ---------------------------------------------------------------------------
pub mod request {
    use super::*;

    /// Build the `messages` array for an OpenAI-compatible chat
    /// completion request from a slice of [`Message`]. Tool messages
    /// are mapped to role `"tool"` with `tool_call_id` attached.
    /// Assistant messages that contain tool calls are mapped to
    /// `role: "assistant"` with a `tool_calls` array.
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
                    let id = m.tool_call_id().map(|i| i.as_str().to_string()).unwrap_or_default();
                    out.push(json!({
                        "role": "tool",
                        "tool_call_id": id,
                        "content": m.content(),
                    }));
                }
                other => {
                    // Unknown role: fall back to user content so the
                    // provider still sees a coherent transcript.
                    out.push(json!({ "role": "user", "content": format!("[{other}] {}", m.content()) }));
                }
            }
        }
        out
    }

    /// Build the `messages` array for an Anthropic messages API call.
    /// System content is returned as a separate `system` field; the
    /// caller is responsible for putting it on the request envelope.
    /// Assistant tool calls become `tool_use` content blocks; tool
    /// messages become `tool_result` content blocks.
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
                    out.push(json!({ "role": "user", "content": format!("[{other}] {}", m.content()) }));
                }
            }
        }
        (system, out)
    }
}

// ---------------------------------------------------------------------------
// Section 2: response translation
// ---------------------------------------------------------------------------
pub mod response {
    use super::*;
    use reimagine_agent::{AgentResponse, Usage};

    /// Translate an OpenAI-compatible chat completion response JSON
    /// into an [`AgentResponse`]. The expected shape is:
    ///
    /// ```text
    /// { "choices": [ { "message": { "role": "assistant", "content": "...", "tool_calls": [...] }, "finish_reason": "..." } ],
    ///   "usage": { "prompt_tokens": N, "completion_tokens": M } }
    /// ```
    pub fn from_openai_response(value: &Value) -> Result<AgentResponse, ProviderAdapterError> {
        let choice0 = value
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .ok_or_else(|| ProviderAdapterError::serialization("missing choices[0]"))?;
        let message = choice0
            .get("message")
            .ok_or_else(|| ProviderAdapterError::serialization("missing choices[0].message"))?;
        let finish_reason = choice0
            .get("finish_reason")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let content = message
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let tool_calls = parse_openai_tool_calls(message.get("tool_calls"))?;

        let message = if tool_calls.is_empty() {
            Message::assistant(content)
        } else {
            Message::assistant_with_tool_calls(content, tool_calls)
        };

        let mut resp = AgentResponse::new(message);
        if let Some(reason) = finish_reason {
            resp = resp.with_stop_reason(reason);
        }
        if let Some(usage) = parse_openai_usage(value.get("usage"))? {
            resp = resp.with_usage(usage);
        }
        Ok(resp)
    }

    fn parse_openai_tool_calls(value: Option<&Value>) -> Result<Vec<ToolCall>, ProviderAdapterError> {
        let mut out = Vec::new();
        let Some(arr) = value.and_then(|v| v.as_array()) else {
            return Ok(out);
        };
        for (i, call) in arr.iter().enumerate() {
            let id = call
                .get("id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ProviderAdapterError::serialization(format!("tool_calls[{i}].id missing")))?
                .to_string();
            let name = call
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    ProviderAdapterError::serialization(format!("tool_calls[{i}].function.name missing"))
                })?
                .to_string();
            let args_str = call
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|v| v.as_str())
                .unwrap_or("{}");
            let arguments = serde_json::from_str(args_str)
                .map_err(|e| ProviderAdapterError::serialization(format!("tool_calls[{i}].function.arguments: {e}")))?;
            out.push(ToolCall::new(ToolCallId::new(id), name, arguments));
        }
        Ok(out)
    }

    fn parse_openai_usage(value: Option<&Value>) -> Result<Option<Usage>, ProviderAdapterError> {
        let Some(usage) = value else { return Ok(None) };
        let input = usage.get("prompt_tokens").and_then(|v| v.as_u64());
        let output = usage.get("completion_tokens").and_then(|v| v.as_u64());
        Ok(Some(Usage::new(input, output)))
    }

    /// Translate an Anthropic messages response JSON into an
    /// [`AgentResponse`]. Expected shape:
    ///
    /// ```text
    /// { "content": [ { "type": "text", "text": "..." }, { "type": "tool_use", "id": "...", "name": "...", "input": {...} } ],
    ///   "stop_reason": "...",
    ///   "usage": { "input_tokens": N, "output_tokens": M } }
    /// ```
    pub fn from_anthropic_response(value: &Value) -> Result<AgentResponse, ProviderAdapterError> {
        let content_arr = value
            .get("content")
            .and_then(|c| c.as_array())
            .ok_or_else(|| ProviderAdapterError::serialization("missing content array"))?;
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        for (i, block) in content_arr.iter().enumerate() {
            match block.get("type").and_then(|v| v.as_str()) {
                Some("text") => {
                    if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(t);
                    }
                }
                Some("tool_use") => {
                    let id = block
                        .get("id")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| {
                            ProviderAdapterError::serialization(format!("content[{i}].id missing"))
                        })?
                        .to_string();
                    let name = block
                        .get("name")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| {
                            ProviderAdapterError::serialization(format!("content[{i}].name missing"))
                        })?
                        .to_string();
                    let arguments = block
                        .get("input")
                        .cloned()
                        .unwrap_or(Value::Null);
                    tool_calls.push(ToolCall::new(ToolCallId::new(id), name, arguments));
                }
                _ => {
                    // Unknown block: skip.
                }
            }
        }
        let message = if tool_calls.is_empty() {
            Message::assistant(text)
        } else {
            Message::assistant_with_tool_calls(text, tool_calls)
        };
        let mut resp = AgentResponse::new(message);
        if let Some(reason) = value.get("stop_reason").and_then(|v| v.as_str()) {
            resp = resp.with_stop_reason(reason.to_string());
        }
        if let Some(usage) = value.get("usage") {
            let input = usage.get("input_tokens").and_then(|v| v.as_u64());
            let output = usage.get("output_tokens").and_then(|v| v.as_u64());
            resp = resp.with_usage(Usage::new(input, output));
        }
        Ok(resp)
    }
}

// ---------------------------------------------------------------------------
// Section 3: tool translation
// ---------------------------------------------------------------------------
pub mod tools {
    use super::*;
    use reimagine_agent::AgentToolDefinition;

    /// Translate a slice of `AgentToolDefinition` into the OpenAI
    /// `tools` array format. Each entry is a function tool with a
    /// JSON-Schema `parameters` object.
    pub fn to_openai_tools(defs: &[AgentToolDefinition]) -> Vec<Value> {
        defs.iter()
            .map(|d| {
                json!({
                    "type": "function",
                    "function": {
                        "name": d.name(),
                        "description": d.description(),
                        "parameters": d.parameters(),
                    }
                })
            })
            .collect()
    }

    /// Translate a slice of `AgentToolDefinition` into the Anthropic
    /// `tools` array format. Anthropic's shape is `{ name, description,
    /// input_schema }` per tool.
    pub fn to_anthropic_tools(defs: &[AgentToolDefinition]) -> Vec<Value> {
        defs.iter()
            .map(|d| {
                json!({
                    "name": d.name(),
                    "description": d.description(),
                    "input_schema": d.parameters(),
                })
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Section 4: streaming translation
// ---------------------------------------------------------------------------
pub mod streaming {
    //! Streaming event translation.
    //!
    //! The provider adapters use two accumulators:
    //! - [`OpenAiStreamAccumulator`] for OpenAI-compatible chunked
    //!   `tool_calls` deltas (the wire format streams function name and
    //!   arguments separately and the adapter correlates by index).
    //! - [`AnthropicStreamAccumulator`] for Anthropic's per-block event
    //!   vocabulary (`content_block_start`, `content_block_delta`,
    //!   `content_block_stop`, etc.).
    //!
    //! Both feed back into the `AgentStreamEvent` shape so the agent
    //! runtime does not see provider-native event types.

    use super::*;
    use reimagine_agent::{AgentStreamEvent, ToolCall, ToolCallId, Usage};

    /// OpenAI stream delta accumulator. Holds the partially-built
    /// tool calls by index and a per-call stable id and name. When the
    /// upstream emits a `finish_reason` (or the chunk stream ends), the
    /// caller flushes complete tool calls.
    #[derive(Debug, Default)]
    pub struct OpenAiStreamAccumulator {
        pub calls: Vec<PartialToolCall>,
        pub stop_reason: Option<String>,
        pub usage: Option<Usage>,
        pub done: bool,
    }

    #[derive(Debug, Default, Clone)]
    pub struct PartialToolCall {
        pub id: Option<String>,
        pub name: Option<String>,
        pub arguments: String,
    }

    impl OpenAiStreamAccumulator {
        pub fn new() -> Self {
            Self::default()
        }

        /// Ingest one OpenAI chat completion chunk. Returns the
        /// `AgentStreamEvent` values to emit for this chunk (excluding
        /// any `ToolCall` flushes, which the caller triggers via
        /// [`flush_complete_tool_calls`]).
        pub fn ingest_chunk(&mut self, chunk: &Value) -> Result<Vec<AgentStreamEvent>, ProviderAdapterError> {
            let mut out = Vec::new();
            // Content delta: `choices[0].delta.content`.
            if let Some(delta_content) = chunk
                .get("choices")
                .and_then(|c| c.as_array())
                .and_then(|a| a.first())
                .and_then(|c| c.get("delta"))
                .and_then(|d| d.get("content"))
                .and_then(|v| v.as_str())
            {
                if !delta_content.is_empty() {
                    out.push(AgentStreamEvent::ContentDelta(delta_content.to_string()));
                }
            }
            // Tool call deltas: `choices[0].delta.tool_calls[*]`.
            if let Some(tool_calls) = chunk
                .get("choices")
                .and_then(|c| c.as_array())
                .and_then(|a| a.first())
                .and_then(|c| c.get("delta"))
                .and_then(|d| d.get("tool_calls"))
                .and_then(|v| v.as_array())
            {
                for (i, call) in tool_calls.iter().enumerate() {
                    let index = call
                        .get("index")
                        .and_then(|v| v.as_u64())
                        .map(|i| i as usize)
                        .unwrap_or(i);
                    while self.calls.len() <= index {
                        self.calls.push(PartialToolCall::default());
                    }
                    let entry = &mut self.calls[index];
                    if let Some(id) = call.get("id").and_then(|v| v.as_str()) {
                        entry.id = Some(id.to_string());
                        out.push(AgentStreamEvent::ToolCallDelta {
                            index: index as u32,
                            id: Some(ToolCallId::new(id.to_string())),
                            name: None,
                            arguments_delta: None,
                        });
                    }
                    if let Some(name) = call
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|v| v.as_str())
                    {
                        entry.name = Some(name.to_string());
                        out.push(AgentStreamEvent::ToolCallDelta {
                            index: index as u32,
                            id: None,
                            name: Some(name.to_string()),
                            arguments_delta: None,
                        });
                    }
                    if let Some(args) = call
                        .get("function")
                        .and_then(|f| f.get("arguments"))
                        .and_then(|v| v.as_str())
                    {
                        entry.arguments.push_str(args);
                        out.push(AgentStreamEvent::ToolCallDelta {
                            index: index as u32,
                            id: None,
                            name: None,
                            arguments_delta: Some(args.to_string()),
                        });
                    }
                }
            }
            // Stop reason on the choice.
            if let Some(reason) = chunk
                .get("choices")
                .and_then(|c| c.as_array())
                .and_then(|a| a.first())
                .and_then(|c| c.get("finish_reason"))
                .and_then(|v| v.as_str())
            {
                self.stop_reason = Some(reason.to_string());
            }
            // Usage, when the upstream sends a final usage chunk.
            if let Some(usage) = chunk.get("usage") {
                let input = usage.get("prompt_tokens").and_then(|v| v.as_u64());
                let output = usage.get("completion_tokens").and_then(|v| v.as_u64());
                self.usage = Some(Usage::new(input, output));
                out.push(AgentStreamEvent::Usage(Usage::new(input, output)));
            }
            Ok(out)
        }

        /// Flush complete tool calls. Call when the stream is finished.
        /// Returns a `ToolCall` event for every entry that has at least
        /// an id and a name. Entries missing required fields are
        /// silently dropped (the upstream did not send enough info).
        pub fn flush_complete_tool_calls(&mut self) -> Vec<AgentStreamEvent> {
            let mut out = Vec::new();
            for partial in self.calls.drain(..) {
                if let (Some(id), Some(name)) = (partial.id, partial.name) {
                    let arguments = if partial.arguments.is_empty() {
                        Value::Null
                    } else {
                        serde_json::from_str(&partial.arguments).unwrap_or(Value::Null)
                    };
                    out.push(AgentStreamEvent::ToolCall(ToolCall::new(
                        ToolCallId::new(id),
                        name,
                        arguments,
                    )));
                }
            }
            out
        }

        /// Mark the stream as done. Returns the `Done` event.
        pub fn finalize(mut self) -> AgentStreamEvent {
            self.done = true;
            AgentStreamEvent::Done {
                stop_reason: self.stop_reason,
            }
        }
    }

    /// Anthropic stream accumulator. Anthropic streams emit a sequence
    /// of typed events; we accumulate text and tool-use content blocks
    /// keyed by `index`. A content block is "complete" when the
    /// upstream sends `content_block_stop` for that index.
    #[derive(Debug, Default)]
    pub struct AnthropicStreamAccumulator {
        pub text: String,
        pub tool_calls: Vec<PartialToolCall>,
        pub stop_reason: Option<String>,
        pub usage: Option<Usage>,
        pub done: bool,
    }

    impl AnthropicStreamAccumulator {
        pub fn new() -> Self {
            Self::default()
        }

        /// Ingest one Anthropic stream event JSON. Returns events to
        /// emit. Text deltas produce `ContentDelta`; tool-use deltas
        /// produce `ToolCallDelta`; complete tool blocks produce
        /// `ToolCall`. `message_stop` produces the final `Done`.
        pub fn ingest_event(&mut self, event: &Value) -> Result<Vec<AgentStreamEvent>, ProviderAdapterError> {
            let mut out = Vec::new();
            let event_type = event
                .get("type")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ProviderAdapterError::serialization("anthropic stream event missing type"))?;
            match event_type {
                "content_block_start" => {
                    let index = event
                        .get("index")
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| {
                            ProviderAdapterError::serialization("content_block_start missing index")
                        })? as usize;
                    let block = event.get("content_block");
                    while self.tool_calls.len() <= index {
                        self.tool_calls.push(PartialToolCall::default());
                    }
                    if let Some(b) = block {
                        if b.get("type").and_then(|v| v.as_str()) == Some("tool_use") {
                            if let Some(id) = b.get("id").and_then(|v| v.as_str()) {
                                self.tool_calls[index].id = Some(id.to_string());
                                out.push(AgentStreamEvent::ToolCallDelta {
                                    index: index as u32,
                                    id: Some(ToolCallId::new(id.to_string())),
                                    name: None,
                                    arguments_delta: None,
                                });
                            }
                            if let Some(name) = b.get("name").and_then(|v| v.as_str()) {
                                self.tool_calls[index].name = Some(name.to_string());
                                out.push(AgentStreamEvent::ToolCallDelta {
                                    index: index as u32,
                                    id: None,
                                    name: Some(name.to_string()),
                                    arguments_delta: None,
                                });
                            }
                        }
                    }
                }
                "content_block_delta" => {
                    let index = event
                        .get("index")
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| {
                            ProviderAdapterError::serialization("content_block_delta missing index")
                        })? as usize;
                    let delta = event.get("delta");
                    if let Some(d) = delta {
                        match d.get("type").and_then(|v| v.as_str()) {
                            Some("text_delta") => {
                                if let Some(t) = d.get("text").and_then(|v| v.as_str()) {
                                    self.text.push_str(t);
                                    out.push(AgentStreamEvent::ContentDelta(t.to_string()));
                                }
                            }
                            Some("input_json_delta") => {
                                if let Some(partial) = d.get("partial_json").and_then(|v| v.as_str()) {
                                    while self.tool_calls.len() <= index {
                                        self.tool_calls.push(PartialToolCall::default());
                                    }
                                    self.tool_calls[index].arguments.push_str(partial);
                                    out.push(AgentStreamEvent::ToolCallDelta {
                                        index: index as u32,
                                        id: None,
                                        name: None,
                                        arguments_delta: Some(partial.to_string()),
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                }
                "content_block_stop" => {
                    let index = event
                        .get("index")
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| {
                            ProviderAdapterError::serialization("content_block_stop missing index")
                        })? as usize;
                    if let Some(partial) = self.tool_calls.get_mut(index) {
                        if let (Some(id), Some(name)) = (partial.id.clone(), partial.name.clone()) {
                            let arguments = if partial.arguments.is_empty() {
                                Value::Null
                            } else {
                                serde_json::from_str(&partial.arguments).unwrap_or(Value::Null)
                            };
                            // Consume the entry so a duplicate stop
                            // event for the same index is a no-op.
                            *partial = PartialToolCall::default();
                            out.push(AgentStreamEvent::ToolCall(ToolCall::new(
                                ToolCallId::new(id),
                                name,
                                arguments,
                            )));
                        }
                    }
                }
                "message_delta" => {
                    if let Some(delta) = event.get("delta") {
                        if let Some(reason) = delta.get("stop_reason").and_then(|v| v.as_str()) {
                            self.stop_reason = Some(reason.to_string());
                        }
                    }
                    if let Some(usage) = event.get("usage") {
                        let input = usage.get("input_tokens").and_then(|v| v.as_u64());
                        let output = usage.get("output_tokens").and_then(|v| v.as_u64());
                        self.usage = Some(Usage::new(input, output));
                    }
                }
                "message_stop" => {
                    self.done = true;
                    out.push(AgentStreamEvent::Done {
                        stop_reason: self.stop_reason.take(),
                    });
                }
                _ => {
                    // Other event kinds (e.g. `ping`, `message_start`)
                    // are ignored.
                }
            }
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn openai_messages_user_and_system() {
        let msgs = vec![Message::system("sys"), Message::user("hi")];
        let v = request::to_openai_messages(&msgs);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0]["role"], "system");
        assert_eq!(v[1]["role"], "user");
    }

    #[test]
    fn openai_messages_assistant_with_tool_calls_uses_function_shape() {
        let call = ToolCall::new(
            ToolCallId::new("c1"),
            "echo",
            json!({"x": 1}),
        );
        let msgs = vec![Message::assistant_with_tool_calls("", vec![call])];
        let v = request::to_openai_messages(&msgs);
        assert_eq!(v[0]["role"], "assistant");
        assert_eq!(v[0]["tool_calls"][0]["function"]["name"], "echo");
        assert_eq!(v[0]["tool_calls"][0]["id"], "c1");
    }

    #[test]
    fn openai_messages_tool_role_carries_tool_call_id() {
        let msgs = vec![Message::tool_result(ToolCallId::new("c1"), "ok")];
        let v = request::to_openai_messages(&msgs);
        assert_eq!(v[0]["role"], "tool");
        assert_eq!(v[0]["tool_call_id"], "c1");
        assert_eq!(v[0]["content"], "ok");
    }

    #[test]
    fn anthropic_messages_splits_system_out() {
        let msgs = vec![Message::system("sys"), Message::user("hi")];
        let (system, v) = request::to_anthropic_messages(&msgs);
        assert_eq!(system.as_deref(), Some("sys"));
        assert_eq!(v.len(), 1);
        assert_eq!(v[0]["role"], "user");
    }

    #[test]
    fn anthropic_messages_assistant_tool_call_uses_tool_use_block() {
        let call = ToolCall::new(
            ToolCallId::new("c1"),
            "echo",
            json!({"x": 1}),
        );
        let msgs = vec![Message::assistant_with_tool_calls("", vec![call])];
        let (_, v) = request::to_anthropic_messages(&msgs);
        assert_eq!(v[0]["role"], "assistant");
        assert_eq!(v[0]["content"][0]["type"], "tool_use");
        assert_eq!(v[0]["content"][0]["id"], "c1");
        assert_eq!(v[0]["content"][0]["name"], "echo");
        assert_eq!(v[0]["content"][0]["input"]["x"], 1);
    }

    #[test]
    fn anthropic_messages_tool_role_becomes_tool_result_block() {
        let msgs = vec![Message::tool_result(ToolCallId::new("c1"), "ok")];
        let (_, v) = request::to_anthropic_messages(&msgs);
        assert_eq!(v[0]["role"], "user");
        assert_eq!(v[0]["content"][0]["type"], "tool_result");
        assert_eq!(v[0]["content"][0]["tool_use_id"], "c1");
        assert_eq!(v[0]["content"][0]["content"], "ok");
    }
}
