//! Agent turn lifecycle shapes.
//!
//! These types describe one turn inside an [`AgentSession`](crate::session::AgentSession):
//! the request that starts the turn, the result the harness returns when the
//! turn stops, and the per-tool-call records the harness produces along the
//! way.
//!
//! The actual orchestration lives in [`crate::loop`]. This module is
//! deliberately data-only: it owns no providers, no registries, no I/O. That
//! keeps the lifecycle model easy to unit-test and easy to reuse from
//! `app-host`, future Axum adapters, and tests.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ToolError;
use crate::ids::{AgentSessionId, ModelName, ProviderName, ToolName};
use crate::mode::AgentMode;
use crate::provider::{Message, ToolCallId, Usage};
use crate::session::AgentSession;

use reimagine_core::diagnostic::Diagnostic;

/// Conservative default for the V1 max-tool-step guard. Hosts and tests may
/// override via [`AgentTurnRequest::with_max_tool_steps`].
pub const DEFAULT_MAX_TOOL_STEPS: usize = 10;

/// Stable, caller-supplied id for an `AgentTurn`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentTurnId(String);

impl AgentTurnId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for AgentTurnId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for AgentTurnId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for AgentTurnId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

/// Lifecycle state of an `AgentTurn`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentTurnStatus {
    /// The turn is in flight. `AgentTurnResult` is in this state only as a
    /// transient builder value; the harness always returns a terminal
    /// status.
    Running,
    /// The turn stopped after the provider returned a final assistant
    /// response with no tool calls.
    Completed,
    /// The turn stopped because of a stop condition (max tool steps,
    /// provider error, future cancellation). Carries the reason.
    Stopped,
}

impl AgentTurnStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Stopped => "stopped",
        }
    }

    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running)
    }

    pub fn is_completed(&self) -> bool {
        matches!(self, Self::Completed)
    }

    pub fn is_stopped(&self) -> bool {
        matches!(self, Self::Stopped)
    }
}

impl std::fmt::Display for AgentTurnStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why the harness stopped a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentTurnStopReason {
    /// Provider returned an assistant message with no tool calls. The
    /// harness treats this as the natural end of a turn.
    FinalResponse,
    /// The configured `max_tool_steps` guard tripped before the provider
    /// returned a final response.
    MaxToolStepsExceeded,
    /// The provider returned `ProviderError`. The harness stops the turn;
    /// the error is also surfaced as a diagnostic on the turn result.
    ProviderError,
    /// Reserved for future turn cancellation. V1 never emits this
    /// variant; it exists so downstream callers can pattern-match without
    /// a `non_exhaustive` boundary.
    Cancelled,
}

impl AgentTurnStopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FinalResponse => "final_response",
            Self::MaxToolStepsExceeded => "max_tool_steps_exceeded",
            Self::ProviderError => "provider_error",
            Self::Cancelled => "cancelled",
        }
    }
}

impl std::fmt::Display for AgentTurnStopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Outcome category for a single tool invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToolCallStatus {
    /// Tool executed and returned a value.
    Succeeded,
    /// Policy denied the call before it ran.
    Rejected,
    /// Tool executed and returned `ToolError`.
    Failed,
}

impl ToolCallStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Rejected => "rejected",
            Self::Failed => "failed",
        }
    }
}

impl std::fmt::Display for ToolCallStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Per-tool-call record produced by the harness.
///
/// `ToolCallResult` is the host-neutral record of what happened when the
/// model asked for a tool. The provider only sees the serialized form of
/// `output` (or a structured diagnostic) via `Message::tool_result`; hosts
/// see the full record via `AgentTurnResult::tool_calls` and through the
/// emitted `AgentEvent` stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallResult {
    tool_call_id: ToolCallId,
    tool_name: ToolName,
    status: ToolCallStatus,
    output: Option<Value>,
    diagnostic: Option<ToolError>,
    effective: Option<bool>,
    session_id: AgentSessionId,
}

impl ToolCallResult {
    /// Start a new `ToolCallResult` for `tool_call_id` / `tool_name`. Use
    /// the `succeeded` / `rejected` / `failed` builders to set the outcome
    /// and `with_session` to attach correlation. `effective` stays `None`
    /// until `set_effective_from_output` (or `with_effective`) is called.
    pub fn new(tool_call_id: ToolCallId, tool_name: ToolName) -> Self {
        Self {
            tool_call_id,
            tool_name,
            status: ToolCallStatus::Succeeded,
            output: None,
            diagnostic: None,
            effective: None,
            session_id: AgentSessionId::new(""),
        }
    }

    /// Attach the agent session id for correlation.
    pub fn with_session(mut self, session_id: AgentSessionId) -> Self {
        self.session_id = session_id;
        self
    }

    /// Mark this result as a successful tool invocation carrying `output`.
    pub fn succeeded(mut self, output: Value) -> Self {
        self.status = ToolCallStatus::Succeeded;
        self.output = Some(output);
        self.diagnostic = None;
        self
    }

    /// Mark this result as a policy-rejected invocation carrying `error`.
    pub fn rejected(mut self, error: ToolError) -> Self {
        self.status = ToolCallStatus::Rejected;
        self.output = None;
        self.diagnostic = Some(error);
        self
    }

    /// Mark this result as a tool execution failure carrying `error`.
    pub fn failed(mut self, error: ToolError) -> Self {
        self.status = ToolCallStatus::Failed;
        self.output = None;
        self.diagnostic = Some(error);
        self
    }

    /// Extract `effective` from a top-level boolean field named
    /// `"effective"` in the successful output JSON. If the field is
    /// missing or non-boolean, `effective` stays `None` — the harness
    /// never infers side effects from tool name or status.
    pub fn set_effective_from_output(mut self) -> Self {
        if let Some(Value::Object(map)) = &self.output {
            if let Some(Value::Bool(b)) = map.get("effective") {
                self.effective = Some(*b);
            }
        }
        self
    }

    /// Explicit override for tests and cases where the harness already
    /// knows the effective flag from context.
    pub fn with_effective(mut self, value: bool) -> Self {
        self.effective = Some(value);
        self
    }

    pub fn tool_call_id(&self) -> &ToolCallId {
        &self.tool_call_id
    }

    pub fn tool_name(&self) -> &ToolName {
        &self.tool_name
    }

    pub fn status(&self) -> ToolCallStatus {
        self.status
    }

    pub fn output(&self) -> Option<&Value> {
        self.output.as_ref()
    }

    pub fn diagnostic(&self) -> Option<&ToolError> {
        self.diagnostic.as_ref()
    }

    pub fn effective(&self) -> Option<bool> {
        self.effective
    }

    pub fn session_id(&self) -> &AgentSessionId {
        &self.session_id
    }
}

/// Input to [`crate::loop::AgentLoop::run_turn`].
#[derive(Debug, Clone)]
pub struct AgentTurnRequest {
    session: AgentSession,
    turn_id: AgentTurnId,
    /// Model the harness should address in this turn. Reimagine-owned
    /// provider adapters translate this to the upstream's native model
    /// identifier.
    model: ModelName,
    input: Vec<Message>,
    max_tool_steps: usize,
}

impl AgentTurnRequest {
    /// Build a new turn request. `max_tool_steps = 0` means "use the V1
    /// default" (`DEFAULT_MAX_TOOL_STEPS`).
    pub fn new(
        session: AgentSession,
        turn_id: AgentTurnId,
        model: ModelName,
        input: Vec<Message>,
    ) -> Self {
        Self {
            session,
            turn_id,
            model,
            input,
            max_tool_steps: 0,
        }
    }

    /// Override the max-tool-step guard. `0` falls back to the default.
    pub fn with_max_tool_steps(mut self, max_tool_steps: usize) -> Self {
        self.max_tool_steps = max_tool_steps;
        self
    }

    pub fn session(&self) -> &AgentSession {
        &self.session
    }

    pub fn turn_id(&self) -> &AgentTurnId {
        &self.turn_id
    }

    pub fn model(&self) -> &ModelName {
        &self.model
    }

    pub fn input(&self) -> &[Message] {
        &self.input
    }

    /// Effective max-tool-step guard. Falls back to
    /// [`DEFAULT_MAX_TOOL_STEPS`] when the caller passed `0`.
    pub fn max_tool_steps(&self) -> usize {
        if self.max_tool_steps == 0 {
            DEFAULT_MAX_TOOL_STEPS
        } else {
            self.max_tool_steps
        }
    }
}

/// Output of [`crate::loop::AgentLoop::run_turn`].
#[derive(Debug, Clone, PartialEq)]
pub struct AgentTurnResult {
    turn_id: AgentTurnId,
    session_id: AgentSessionId,
    mode: AgentMode,
    provider: ProviderName,
    model: ModelName,
    status: AgentTurnStatus,
    stop_reason: AgentTurnStopReason,
    final_response: Option<Message>,
    tool_calls: Vec<ToolCallResult>,
    diagnostics: Vec<Diagnostic>,
    usage: Option<Usage>,
    messages: Vec<Message>,
}

impl Default for AgentTurnResult {
    fn default() -> Self {
        Self {
            turn_id: AgentTurnId::new(""),
            session_id: AgentSessionId::new(""),
            mode: AgentMode::Agent,
            provider: ProviderName::new(""),
            model: ModelName::new(""),
            status: AgentTurnStatus::Running,
            stop_reason: AgentTurnStopReason::FinalResponse,
            final_response: None,
            tool_calls: Vec::new(),
            diagnostics: Vec::new(),
            usage: None,
            messages: Vec::new(),
        }
    }
}

impl AgentTurnResult {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_turn_id(mut self, turn_id: AgentTurnId) -> Self {
        self.turn_id = turn_id;
        self
    }

    pub fn with_session_id(mut self, session_id: AgentSessionId) -> Self {
        self.session_id = session_id;
        self
    }

    pub fn with_mode(mut self, mode: AgentMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_provider(mut self, provider: ProviderName) -> Self {
        self.provider = provider;
        self
    }

    pub fn with_model(mut self, model: ModelName) -> Self {
        self.model = model;
        self
    }

    pub fn with_status(mut self, status: AgentTurnStatus) -> Self {
        self.status = status;
        self
    }

    pub fn with_stop_reason(mut self, stop_reason: AgentTurnStopReason) -> Self {
        self.stop_reason = stop_reason;
        self
    }

    pub fn with_final_response(mut self, response: Message) -> Self {
        self.final_response = Some(response);
        self
    }

    pub fn push_tool_call(&mut self, result: ToolCallResult) {
        self.tool_calls.push(result);
    }

    pub fn push_diagnostic(&mut self, diagnostic: Diagnostic) {
        self.diagnostics.push(diagnostic);
    }

    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    pub fn with_messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    pub fn turn_id(&self) -> &AgentTurnId {
        &self.turn_id
    }

    pub fn session_id(&self) -> &AgentSessionId {
        &self.session_id
    }

    pub fn mode(&self) -> AgentMode {
        self.mode
    }

    pub fn provider(&self) -> &ProviderName {
        &self.provider
    }

    pub fn model(&self) -> &ModelName {
        &self.model
    }

    pub fn status(&self) -> AgentTurnStatus {
        self.status
    }

    pub fn stop_reason(&self) -> AgentTurnStopReason {
        self.stop_reason
    }

    pub fn final_response(&self) -> Option<&Message> {
        self.final_response.as_ref()
    }

    pub fn tool_calls(&self) -> &[ToolCallResult] {
        &self.tool_calls
    }

    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    pub fn usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Number of tool calls the harness actually performed during the
    /// turn (each model-requested tool call counts once, regardless of
    /// whether it succeeded, was rejected, or failed).
    pub fn tool_steps(&self) -> usize {
        self.tool_calls.len()
    }

    /// `true` when the status is `Completed` (provider returned a final
    /// assistant response with no tool calls).
    pub fn is_completed(&self) -> bool {
        self.status == AgentTurnStatus::Completed
    }

    /// `true` when the status is `Stopped` (a stop condition fired
    /// before the provider produced a final response).
    pub fn is_stopped(&self) -> bool {
        self.status == AgentTurnStatus::Stopped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ToolErrorCode;
    use crate::permissions::{PermissionSet, ToolPermission, ToolRiskLevel};
    use crate::registry::AgentToolRegistry;
    use crate::tool::{AgentTool, ToolSpec};
    use async_trait::async_trait;
    use serde_json::json;

    struct Noop;
    #[async_trait]
    impl AgentTool for Noop {
        fn spec(&self) -> ToolSpec {
            ToolSpec::new(
                ToolName::new("noop"),
                "noop",
                [AgentMode::Agent],
                ToolPermission::new("workflow.read"),
                ToolRiskLevel::Read,
            )
        }
        async fn invoke(
            &self,
            _ctx: &crate::context::ToolContext,
            _input: Value,
        ) -> crate::tool::ToolResult {
            Ok(Value::Null)
        }
    }

    fn dummy_session() -> AgentSession {
        let mut reg = AgentToolRegistry::new();
        reg.register(Noop).unwrap();
        AgentSession::new(
            AgentSessionId::new("sess-1"),
            crate::ids::WorkspaceScope::new("ws-1"),
            AgentMode::Agent,
            ProviderName::new("mock"),
            std::sync::Arc::new(reg),
        )
    }

    #[test]
    fn agent_turn_id_roundtrip() {
        let id: AgentTurnId = "turn-7".into();
        assert_eq!(id.as_str(), "turn-7");
        assert_eq!(format!("{id}"), "turn-7");
        let id2 = AgentTurnId::new(String::from("turn-7"));
        assert_eq!(id, id2);
    }

    #[test]
    fn agent_turn_status_predicates() {
        assert!(AgentTurnStatus::Running.is_running());
        assert!(!AgentTurnStatus::Running.is_completed());
        assert!(AgentTurnStatus::Completed.is_completed());
        assert!(!AgentTurnStatus::Completed.is_running());
        assert!(AgentTurnStatus::Stopped.is_stopped());
        assert!(!AgentTurnStatus::Stopped.is_completed());
        assert!(!AgentTurnStatus::Stopped.is_running());
    }

    #[test]
    fn stop_reason_as_str() {
        assert_eq!(
            AgentTurnStopReason::FinalResponse.as_str(),
            "final_response"
        );
        assert_eq!(
            AgentTurnStopReason::MaxToolStepsExceeded.as_str(),
            "max_tool_steps_exceeded"
        );
        assert_eq!(
            AgentTurnStopReason::ProviderError.as_str(),
            "provider_error"
        );
        assert_eq!(AgentTurnStopReason::Cancelled.as_str(), "cancelled");
    }

    #[test]
    fn tool_call_result_succeeded_carries_output() {
        let r = ToolCallResult::new(ToolCallId::new("c1"), ToolName::new("echo"))
            .with_session(AgentSessionId::new("sess-1"))
            .succeeded(json!({"ok": true}));
        assert_eq!(r.status(), ToolCallStatus::Succeeded);
        assert_eq!(r.output(), Some(&json!({"ok": true})));
        assert!(r.diagnostic().is_none());
        assert_eq!(r.tool_call_id().as_str(), "c1");
        assert_eq!(r.tool_name().as_str(), "echo");
        assert_eq!(r.session_id().as_str(), "sess-1");
    }

    #[test]
    fn tool_call_result_rejected_carries_diagnostic() {
        let err = ToolError::new(ToolErrorCode::PermissionDenied, "no perm");
        let r = ToolCallResult::new(ToolCallId::new("c1"), ToolName::new("x"))
            .with_session(AgentSessionId::new("sess-1"))
            .rejected(err.clone());
        assert_eq!(r.status(), ToolCallStatus::Rejected);
        assert!(r.output().is_none());
        assert_eq!(r.diagnostic(), Some(&err));
    }

    #[test]
    fn tool_call_result_failed_carries_diagnostic() {
        let err = ToolError::new(ToolErrorCode::ExecutionFailed, "boom");
        let r = ToolCallResult::new(ToolCallId::new("c1"), ToolName::new("x"))
            .with_session(AgentSessionId::new("sess-1"))
            .failed(err.clone());
        assert_eq!(r.status(), ToolCallStatus::Failed);
        assert!(r.output().is_none());
        assert_eq!(r.diagnostic(), Some(&err));
    }

    #[test]
    fn tool_call_result_effective_extracted_from_object_output() {
        let r = ToolCallResult::new(ToolCallId::new("c1"), ToolName::new("x"))
            .succeeded(json!({"effective": true, "data": 1}))
            .set_effective_from_output();
        assert_eq!(r.effective(), Some(true));

        let r2 = ToolCallResult::new(ToolCallId::new("c2"), ToolName::new("x"))
            .succeeded(json!({"effective": false}))
            .set_effective_from_output();
        assert_eq!(r2.effective(), Some(false));
    }

    #[test]
    fn tool_call_result_effective_left_none_for_missing_field() {
        let r = ToolCallResult::new(ToolCallId::new("c1"), ToolName::new("x"))
            .succeeded(json!({"data": 1}))
            .set_effective_from_output();
        assert_eq!(r.effective(), None);
    }

    #[test]
    fn tool_call_result_effective_left_none_for_non_object_output() {
        let r1 = ToolCallResult::new(ToolCallId::new("c1"), ToolName::new("x"))
            .succeeded(json!(42))
            .set_effective_from_output();
        assert_eq!(r1.effective(), None);

        let r2 = ToolCallResult::new(ToolCallId::new("c2"), ToolName::new("x"))
            .succeeded(json!("hi"))
            .set_effective_from_output();
        assert_eq!(r2.effective(), None);

        let r3 = ToolCallResult::new(ToolCallId::new("c3"), ToolName::new("x"))
            .succeeded(json!([1, 2, 3]))
            .set_effective_from_output();
        assert_eq!(r3.effective(), None);

        let r4 = ToolCallResult::new(ToolCallId::new("c4"), ToolName::new("x"))
            .succeeded(Value::Null)
            .set_effective_from_output();
        assert_eq!(r4.effective(), None);
    }

    #[test]
    fn tool_call_result_effective_left_none_for_non_boolean_field() {
        let r = ToolCallResult::new(ToolCallId::new("c1"), ToolName::new("x"))
            .succeeded(json!({"effective": "yes"}))
            .set_effective_from_output();
        assert_eq!(r.effective(), None);
    }

    #[test]
    fn tool_call_result_with_effective_override() {
        let r = ToolCallResult::new(ToolCallId::new("c1"), ToolName::new("x"))
            .succeeded(json!({}))
            .with_effective(true);
        assert_eq!(r.effective(), Some(true));
    }

    #[test]
    fn tool_call_status_as_str() {
        assert_eq!(ToolCallStatus::Succeeded.as_str(), "succeeded");
        assert_eq!(ToolCallStatus::Rejected.as_str(), "rejected");
        assert_eq!(ToolCallStatus::Failed.as_str(), "failed");
    }

    #[test]
    fn agent_turn_request_default_max_tool_steps() {
        let req = AgentTurnRequest::new(
            dummy_session(),
            AgentTurnId::new("turn-1"),
            ModelName::new("gpt-test"),
            vec![Message::user("hi")],
        );
        assert_eq!(req.max_tool_steps(), DEFAULT_MAX_TOOL_STEPS);
        assert_eq!(DEFAULT_MAX_TOOL_STEPS, 10);
    }

    #[test]
    fn agent_turn_request_explicit_max_tool_steps() {
        let req = AgentTurnRequest::new(
            dummy_session(),
            AgentTurnId::new("turn-1"),
            ModelName::new("gpt-test"),
            vec![Message::user("hi")],
        )
        .with_max_tool_steps(3);
        assert_eq!(req.max_tool_steps(), 3);
    }

    #[test]
    fn agent_turn_request_zero_falls_back_to_default() {
        let req = AgentTurnRequest::new(
            dummy_session(),
            AgentTurnId::new("turn-1"),
            ModelName::new("gpt-test"),
            vec![Message::user("hi")],
        )
        .with_max_tool_steps(0);
        assert_eq!(req.max_tool_steps(), DEFAULT_MAX_TOOL_STEPS);
    }

    #[test]
    fn agent_turn_request_input_roundtrips() {
        let req = AgentTurnRequest::new(
            dummy_session(),
            AgentTurnId::new("turn-1"),
            ModelName::new("gpt-test"),
            vec![Message::system("sys"), Message::user("hi")],
        );
        assert_eq!(req.turn_id().as_str(), "turn-1");
        assert_eq!(req.input().len(), 2);
        assert_eq!(req.model().as_str(), "gpt-test");
    }

    #[test]
    fn agent_turn_result_default_builds_empty() {
        let r = AgentTurnResult::default();
        assert!(r.final_response().is_none());
        assert!(r.tool_calls().is_empty());
        assert!(r.diagnostics().is_empty());
        assert!(r.usage().is_none());
        assert!(r.messages().is_empty());
        assert_eq!(r.tool_steps(), 0);
    }

    #[test]
    fn agent_turn_result_push_tool_call_appends() {
        let mut r = AgentTurnResult::default();
        r.push_tool_call(
            ToolCallResult::new(ToolCallId::new("a"), ToolName::new("x")).succeeded(json!({})),
        );
        r.push_tool_call(
            ToolCallResult::new(ToolCallId::new("b"), ToolName::new("y")).succeeded(json!({})),
        );
        assert_eq!(r.tool_calls().len(), 2);
        assert_eq!(r.tool_calls()[0].tool_call_id().as_str(), "a");
        assert_eq!(r.tool_calls()[1].tool_call_id().as_str(), "b");
    }

    #[test]
    fn agent_turn_result_is_completed_predicate() {
        let r = AgentTurnResult::default().with_status(AgentTurnStatus::Completed);
        assert!(r.is_completed());
        let r = AgentTurnResult::default().with_status(AgentTurnStatus::Stopped);
        assert!(!r.is_completed());
    }

    #[test]
    fn agent_turn_result_tool_steps_count() {
        let mut r = AgentTurnResult::default();
        for i in 0..3 {
            r.push_tool_call(
                ToolCallResult::new(ToolCallId::new(format!("c{i}")), ToolName::new("x"))
                    .succeeded(json!({})),
            );
        }
        assert_eq!(r.tool_steps(), 3);
    }

    #[test]
    fn agent_turn_result_with_final_response_and_usage() {
        let resp = Message::assistant("all done");
        let r = AgentTurnResult::default()
            .with_final_response(resp.clone())
            .with_usage(Usage::new(Some(7), Some(11)));
        assert_eq!(r.final_response(), Some(&resp));
        assert_eq!(r.usage().unwrap().input_tokens(), Some(7));
    }

    #[test]
    fn permission_set_still_constructable_in_tests() {
        let _ = PermissionSet::from_iter([ToolPermission::new("workflow.read")]);
    }
}
