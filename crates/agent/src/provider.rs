//! Reimagine-owned provider boundary.
//!
//! `AgentProvider` is the trait every concrete provider adapter must
//! implement. The shapes are deliberately friendly to a future
//! Rig-backed OpenAI-compatible and Anthropic adapter, but the agent
//! crate does not depend on Rig, Cersei, or any concrete provider SDK.
//! `complete` and `stream` operate over `AgentRequest`/`AgentResponse`/
//! `AgentStreamEvent` payloads that are provider-agnostic.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ProviderError;
use crate::ids::{ModelName, ProviderName};

/// Stable, caller-supplied id for a tool call requested by the model.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolCallId(String);

impl ToolCallId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ToolCallId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for ToolCallId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for ToolCallId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

/// Provider-agnostic chat message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Role discriminator. V1 uses the conventional string names
    /// `"system"`, `"user"`, `"assistant"`, and `"tool"`. Provider
    /// adapters translate to OpenAI / Anthropic roles.
    role: String,
    /// Message content. Provider adapters split this into text or
    /// multi-part content as needed.
    content: String,
    /// Optional tool call id this message is responding to (for role =
    /// "tool").
    tool_call_id: Option<ToolCallId>,
    /// Optional tool calls the assistant is making (for role =
    /// "assistant").
    tool_calls: Vec<ToolCall>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    /// Build a `tool` message that delivers `content` as the result of
    /// the tool call with id `tool_call_id`.
    pub fn tool_result(tool_call_id: ToolCallId, content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: content.into(),
            tool_call_id: Some(tool_call_id),
            tool_calls: Vec::new(),
        }
    }

    /// Construct an assistant message that calls one or more tools.
    pub fn assistant_with_tool_calls(
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
            tool_call_id: None,
            tool_calls,
        }
    }

    pub fn role(&self) -> &str {
        &self.role
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn tool_call_id(&self) -> Option<&ToolCallId> {
        self.tool_call_id.as_ref()
    }

    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.tool_calls
    }
}

/// Tool call requested by the model. The `arguments` payload is raw JSON
/// so providers and tools can negotiate the exact shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    id: ToolCallId,
    /// Name of the tool the model wants to invoke. The agent runtime
    /// looks this up in the registry.
    name: String,
    /// JSON arguments for the tool, kept as a `serde_json::Value` so
    /// the registry can deserialize into the tool's input shape.
    arguments: Value,
}

impl ToolCall {
    pub fn new(id: ToolCallId, name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id,
            name: name.into(),
            arguments,
        }
    }

    pub fn id(&self) -> &ToolCallId {
        &self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn arguments(&self) -> &Value {
        &self.arguments
    }
}

/// Token-usage report from a provider. Optional, but every concrete
/// adapter should report usage when the upstream API does.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
}

impl Usage {
    pub fn new(input_tokens: Option<u64>, output_tokens: Option<u64>) -> Self {
        Self {
            input_tokens,
            output_tokens,
        }
    }

    pub fn input_tokens(&self) -> Option<u64> {
        self.input_tokens
    }

    pub fn output_tokens(&self) -> Option<u64> {
        self.output_tokens
    }
}

/// Information about a single model advertised by a provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    name: ModelName,
    /// Provider that owns this model. Optional because some catalogs
    /// are queried per provider.
    provider: Option<ProviderName>,
    capabilities: Vec<ModelCapability>,
}

impl ModelInfo {
    pub fn new(name: ModelName) -> Self {
        Self {
            name,
            provider: None,
            capabilities: Vec::new(),
        }
    }

    pub fn with_provider(mut self, provider: ProviderName) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn with_capability(mut self, cap: ModelCapability) -> Self {
        self.capabilities.push(cap);
        self
    }

    pub fn with_capabilities(mut self, caps: impl IntoIterator<Item = ModelCapability>) -> Self {
        self.capabilities.extend(caps);
        self
    }

    pub fn name(&self) -> &ModelName {
        &self.name
    }

    pub fn provider(&self) -> Option<&ProviderName> {
        self.provider.as_ref()
    }

    pub fn capabilities(&self) -> &[ModelCapability] {
        &self.capabilities
    }
}

/// Model capability hints surfaced by the provider. The set is
/// deliberately small in V1; provider adapters are free to advertise
/// only what the upstream API exposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelCapability {
    Chat,
    ToolUse,
    Vision,
    Streaming,
}

impl ModelCapability {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::ToolUse => "tool_use",
            Self::Vision => "vision",
            Self::Streaming => "streaming",
        }
    }
}

impl std::fmt::Display for ModelCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Request to a provider. Provider-agnostic on purpose so future
/// Rig-backed OpenAI-compatible and Anthropic adapters can serialize to
/// their native wire formats.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentRequest {
    model: ModelName,
    messages: Vec<Message>,
    /// Optional tools to advertise. Concrete adapters translate to
    /// OpenAI's `tools` or Anthropic's `tools` blocks.
    tools: Vec<AgentToolDefinition>,
    /// Provider-specific options blob, kept opaque at the boundary.
    options: Value,
}

impl AgentRequest {
    pub fn new(model: ModelName, messages: Vec<Message>) -> Self {
        Self {
            model,
            messages,
            tools: Vec::new(),
            options: Value::Null,
        }
    }

    pub fn with_tools(mut self, tools: Vec<AgentToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_options(mut self, options: Value) -> Self {
        self.options = options;
        self
    }

    pub fn model(&self) -> &ModelName {
        &self.model
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn tools(&self) -> &[AgentToolDefinition] {
        &self.tools
    }

    pub fn options(&self) -> &Value {
        &self.options
    }
}

/// Tool definition sent to the provider. The shape is provider-agnostic
/// (name, description, and a JSON-Schema for arguments) so adapters can
/// translate to OpenAI function-calling or Anthropic tool-use without
/// coupling the agent crate to either schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentToolDefinition {
    name: String,
    description: String,
    /// JSON Schema describing the tool's input shape.
    parameters: Value,
}

impl AgentToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn parameters(&self) -> &Value {
        &self.parameters
    }
}

/// Provider-agnostic response. Carries the assistant message, optional
/// usage report, and an optional `stop_reason` describing why the model
/// stopped (e.g. end-of-turn vs. tool use).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentResponse {
    message: Message,
    usage: Option<Usage>,
    stop_reason: Option<String>,
}

impl AgentResponse {
    pub fn new(message: Message) -> Self {
        Self {
            message,
            usage: None,
            stop_reason: None,
        }
    }

    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    pub fn with_stop_reason(mut self, reason: impl Into<String>) -> Self {
        self.stop_reason = Some(reason.into());
        self
    }

    pub fn message(&self) -> &Message {
        &self.message
    }

    pub fn usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }

    pub fn stop_reason(&self) -> Option<&str> {
        self.stop_reason.as_deref()
    }
}

/// Streamed event from a provider. The agent runtime consumes these to
/// surface incremental progress to the host (Tauri / Axum / tests).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AgentStreamEvent {
    /// A token-sized content delta for the assistant message.
    ContentDelta(String),
    /// A complete tool call the model wants to make. Providers that
    /// stream tool calls incrementally may emit one or more
    /// `ToolCallDelta` events first and finish with a `ToolCallComplete`
    /// event; the runtime treats both shapes uniformly.
    ToolCall(ToolCall),
    /// A partial tool call. `index` is the position in the final
    /// tool-call list. `id`, `name`, and `arguments_delta` are optional
    /// because providers may stream them at different times.
    ToolCallDelta {
        index: u32,
        id: Option<ToolCallId>,
        name: Option<String>,
        arguments_delta: Option<String>,
    },
    /// Final usage report. Optional because not every provider surfaces
    /// usage in the stream.
    Usage(Usage),
    /// Stream completed; the runtime stops reading.
    Done { stop_reason: Option<String> },
}

impl AgentStreamEvent {
    /// `true` for the terminal `Done` event.
    pub fn is_done(&self) -> bool {
        matches!(self, Self::Done { .. })
    }
}

/// Reimagine-owned provider trait.
///
/// Implementors adapt a specific upstream API (OpenAI-compatible,
/// Anthropic, future Rig-backed) to the agent runtime. Implementations
/// may be sync internally but expose async methods so the runtime can
/// compose them with other async services.
#[async_trait]
pub trait AgentProvider: Send + Sync {
    /// Provider identity. Used for diagnostics, audit, and for
    /// `list_models` payloads.
    fn name(&self) -> ProviderName;

    /// Send a single completion request. Returns the provider's full
    /// response, or a `ProviderError` on transport / API failure.
    async fn complete(&self, request: AgentRequest) -> Result<AgentResponse, ProviderError>;

    /// Stream a completion. Implementations are expected to return
    /// events in the order the upstream produces them, terminating with
    /// `AgentStreamEvent::Done`.
    async fn stream(&self, request: AgentRequest) -> Result<Box<dyn AgentStream>, ProviderError>;

    /// List the models this provider advertises. V1 expects this to be
    /// cheap; if the upstream needs an async call, implementations may
    /// cache internally.
    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError>;
}

/// Stream of provider events returned by `AgentProvider::stream`.
///
/// The trait is sealed to a small set of operations the agent runtime
/// needs: pull the next event, peek at the most recently yielded event
/// for diagnostics, and signal cancellation.
#[async_trait]
pub trait AgentStream: Send {
    /// Pull the next event. Returns `None` when the stream is
    /// exhausted; concrete adapters translate the upstream's "done"
    /// signal into either `None` or a final `AgentStreamEvent::Done`.
    async fn next_event(&mut self) -> Option<AgentStreamEvent>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn message_role_helpers() {
        let m = Message::system("you are a helpful assistant");
        assert_eq!(m.role(), "system");
        assert_eq!(m.content(), "you are a helpful assistant");

        let tool_id = ToolCallId::new("call-1");
        let m = Message::tool_result(tool_id.clone(), "ok");
        assert_eq!(m.role(), "tool");
        assert_eq!(m.tool_call_id(), Some(&tool_id));

        let call = ToolCall::new(ToolCallId::new("c1"), "echo", json!({"x": 1}));
        let m = Message::assistant_with_tool_calls("", vec![call.clone()]);
        assert_eq!(m.tool_calls().len(), 1);
        assert_eq!(m.tool_calls()[0], call);
    }

    #[test]
    fn request_response_shapes_roundtrip() {
        let model = ModelName::new("gpt-4o-mini");
        let req = AgentRequest::new(model.clone(), vec![Message::user("hi")])
            .with_tools(vec![AgentToolDefinition::new(
                "echo",
                "echo",
                json!({"type": "object"}),
            )])
            .with_options(json!({"temperature": 0.0}));
        assert_eq!(req.model(), &model);
        assert_eq!(req.messages().len(), 1);
        assert_eq!(req.tools().len(), 1);
        assert_eq!(req.options(), &json!({"temperature": 0.0}));

        let resp = AgentResponse::new(Message::assistant("hello"))
            .with_usage(Usage::new(Some(10), Some(20)))
            .with_stop_reason("end_turn");
        assert_eq!(resp.usage().unwrap().input_tokens(), Some(10));
        assert_eq!(resp.stop_reason(), Some("end_turn"));
    }

    #[test]
    fn model_info_carries_provider_and_capabilities() {
        let info = ModelInfo::new(ModelName::new("gpt-4o-mini"))
            .with_provider(ProviderName::new("openai"))
            .with_capabilities([ModelCapability::Chat, ModelCapability::ToolUse]);
        assert_eq!(info.provider().unwrap().as_str(), "openai");
        assert_eq!(info.capabilities().len(), 2);
    }

    #[test]
    fn stream_event_done_predicate() {
        assert!(AgentStreamEvent::Done { stop_reason: None }.is_done());
        assert!(!AgentStreamEvent::ContentDelta("hi".into()).is_done());
    }
}
