//! Agent loop harness.
//!
//! `AgentLoop` is the V1 Reimagine-owned turn executor. It owns
//! orchestration: it builds provider requests, advertises registry tool
//! specs, executes requested tools through the registry, feeds tool
//! observations back to the provider, emits agent-local events, and
//! stops when the model produces a final assistant response, the
//! configured max-tool-step guard trips, or the provider returns an
//! error.
//!
//! V1 does not implement streaming turn execution, steering, interruption,
//! subagents, memory, or skills. Cancellation exists as a stop-reason
//! placeholder only; the loop never emits `AgentTurnStopReason::Cancelled`
//! in V1.

use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use crate::context::ToolContext;
use crate::error::{ToolError, ToolErrorCode};
use crate::event::AgentEvent;
use crate::ids::ToolName;
use crate::provider::{AgentProvider, AgentRequest, AgentToolDefinition, Message};
use crate::registry::{AgentToolRegistry, ToolRegistryError};
use crate::turn::{
    AgentTurnRequest, AgentTurnResult, AgentTurnStatus, AgentTurnStopReason, ToolCallResult,
    ToolCallStatus,
};

/// Host-neutral event sink for agent-local events.
///
/// `AgentLoop` forwards `AgentEvent` values through an `AgentEventSink`
/// so callers can wire their own transport (Tauri events, future Axum
/// SSE, a test `Vec`, or a no-op). The trait is intentionally narrow so
/// that custom sinks can be implemented without inheriting a heavy
/// abstraction.
///
/// `handle` takes `&self` (not `&mut self`) so the trait object can sit
/// behind `Arc` without locking the harness on every event. Sinks that
/// need mutable state (`VecAgentEventSink`, channel senders, etc.) are
/// expected to use interior mutability internally.
pub trait AgentEventSink: Send + Sync {
    /// Handle a single `AgentEvent`. Sinks must not block; long work
    /// should be deferred to an internal channel.
    fn handle(&self, event: &AgentEvent);
}

/// `AgentEventSink` implementation that appends every event to a `Vec`.
/// Intended for tests; production hosts should use a Tauri or future
/// Axum adapter that bridges events to the host-facing event stream.
#[derive(Debug, Default)]
pub struct VecAgentEventSink {
    events: Mutex<Vec<AgentEvent>>,
}

impl VecAgentEventSink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn events(&self) -> Vec<AgentEvent> {
        self.events.lock().unwrap().clone()
    }

    pub fn len(&self) -> usize {
        self.events.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.lock().unwrap().is_empty()
    }
}

impl AgentEventSink for VecAgentEventSink {
    fn handle(&self, event: &AgentEvent) {
        self.events.lock().unwrap().push(event.clone());
    }
}

/// V1 Agent loop harness. Cheap to clone (`Arc` fields).
///
/// The harness owns:
/// - an `Arc<dyn AgentProvider>` to issue completion requests;
/// - an `Arc<dyn AgentEventSink>` to surface agent-local events for
///   host observers.
///
/// The tool registry and `ToolContext` metadata (workspace scope,
/// session id, mode, permissions) all come from
/// [`AgentSession`](crate::session::AgentSession) — the session is
/// the authoritative source for the frozen registry, mirroring the
/// Codex-style "capability surface" model in
/// `docs/architecture/modules/agent.md`. The harness must never
/// carry its own registry field, or a host that mutates (or simply
/// forgets to populate) the session's registry would be surprised.
#[derive(Clone)]
pub struct AgentLoop {
    provider: Arc<dyn AgentProvider>,
    sink: Arc<dyn AgentEventSink>,
}

impl std::fmt::Debug for AgentLoop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentLoop")
            .field("provider", &self.provider.name())
            .field("sink", &"Arc<dyn AgentEventSink>")
            .finish()
    }
}

impl AgentLoop {
    /// Build a new harness with the given provider and event sink.
    /// Both are wrapped in `Arc` so the harness is cheap to share.
    /// The tool registry is read from each
    /// [`AgentTurnRequest`](crate::turn::AgentTurnRequest)'s session at
    /// turn time — see [`AgentLoop`]'s type-level docs.
    pub fn new(provider: Arc<dyn AgentProvider>, sink: Arc<dyn AgentEventSink>) -> Self {
        Self { provider, sink }
    }

    /// Reference to the underlying provider. Tests use this to verify
    /// request shape.
    pub fn provider(&self) -> &Arc<dyn AgentProvider> {
        &self.provider
    }

    /// Run a single turn and return the final [`AgentTurnResult`].
    ///
    /// The loop:
    /// 1. Builds an initial message list from the request input.
    /// 2. Builds an `AgentToolDefinition` list from the session's
    ///    registry tool specs that allow the session's mode.
    /// 3. Calls [`AgentProvider::complete`]. On error, stops with
    ///    [`AgentTurnStopReason::ProviderError`] and surfaces a
    ///    diagnostic plus a [`AgentEvent::ProviderError`].
    /// 4. If the assistant message has no tool calls, stops with
    ///    [`AgentTurnStopReason::FinalResponse`].
    /// 5. Otherwise checks the `max_tool_steps` guard; if tripped,
    ///    stops with [`AgentTurnStopReason::MaxToolStepsExceeded`]
    ///    **without** appending the assistant message (the transcript
    ///    ends on the last executed tool observation).
    /// 6. Otherwise executes the requested tool calls sequentially
    ///    through the session's registry, appends `Message::tool_result`
    ///    observations, and loops back to step 3.
    pub async fn run_turn(&self, request: AgentTurnRequest) -> AgentTurnResult {
        let mut result = AgentTurnResult::new()
            .with_turn_id(request.turn_id().clone())
            .with_session_id(request.session().id().clone())
            .with_mode(request.session().mode())
            .with_provider(request.session().provider().clone())
            .with_model(request.model().clone())
            .with_status(AgentTurnStatus::Running);

        let registry = request.session().registry().clone();
        // Seed the provider message list with the session's running
        // conversation history, then append this turn's input on top.
        // `pre_run_len` points just past the prior history (NOT past
        // the input), so `commit_session_history` will append *this
        // turn's input + new assistant + tool observations* to the
        // session history at the end — matching the issue's note
        // about "appending turn input, assistant messages, and tool
        // observation messages to session history."
        let prior_history = request.session().history();
        let pre_run_len = prior_history.len();
        let mut messages: Vec<Message> = prior_history
            .iter()
            .cloned()
            .chain(request.input().iter().cloned())
            .collect();
        let tool_defs = build_tool_definitions(&registry, request.session().mode());

        let max_tool_steps = request.max_tool_steps();
        let mut tool_steps_taken: usize = 0;

        loop {
            let provider_request = AgentRequest::new(request.model().clone(), messages.clone())
                .with_tools(tool_defs.clone());

            let response = match self.provider.complete(provider_request).await {
                Ok(r) => r,
                Err(err) => {
                    let provider_name = request.session().provider().clone();
                    result.push_diagnostic(err.to_diagnostic(None));
                    self.sink.handle(&AgentEvent::ProviderError {
                        session_id: request.session().id().clone(),
                        provider: provider_name,
                        code: err.code().to_string(),
                        message: err.message().to_string(),
                    });
                    result = result
                        .with_stop_reason(AgentTurnStopReason::ProviderError)
                        .with_status(AgentTurnStatus::Stopped)
                        .with_messages(messages.clone());
                    commit_session_history(request.session(), &messages, pre_run_len);
                    return result;
                }
            };

            // Capture usage from this round if the provider reported
            // it. V1 keeps the latest reported usage rather than
            // summing across rounds — providers are expected to
            // report a running total themselves.
            if let Some(usage) = response.usage().cloned() {
                result = result.with_usage(usage);
            }

            let assistant = response.message().clone();
            let has_tool_calls = !assistant.tool_calls().is_empty();

            // No tool calls → final response. Append the assistant
            // message first so the transcript ends on it.
            if !has_tool_calls {
                messages.push(assistant.clone());
                result = result
                    .with_final_response(assistant)
                    .with_stop_reason(AgentTurnStopReason::FinalResponse)
                    .with_status(AgentTurnStatus::Completed)
                    .with_messages(messages.clone());
                commit_session_history(request.session(), &messages, pre_run_len);
                return result;
            }

            // Max-tool-step guard. Counts *individual* tool calls.
            // The check is placed before the round's `for` loop so
            // that the round is *admitted* (and its assistant message
            // appended) only when at least one tool call slot remains.
            // A finer-grained check inside the `for` then stops the
            // turn exactly at the limit, even mid-round.
            if tool_steps_taken >= max_tool_steps {
                result = result
                    .with_stop_reason(AgentTurnStopReason::MaxToolStepsExceeded)
                    .with_status(AgentTurnStatus::Stopped)
                    .with_messages(messages.clone());
                commit_session_history(request.session(), &messages, pre_run_len);
                return result;
            }

            // Append the assistant message only after the guard
            // passes — this way the assistant that requested tools
            // always has corresponding `tool_result` observations.
            messages.push(assistant.clone());

            // Execute the requested tool calls sequentially in the
            // order the provider returned them. All observations are
            // appended before the next provider call so the model sees
            // the full batch. The guard fires *before* each tool call
            // so the limit is exact, not just per-round.
            for tool_call in assistant.tool_calls() {
                if tool_steps_taken >= max_tool_steps {
                    result = result
                        .with_stop_reason(AgentTurnStopReason::MaxToolStepsExceeded)
                        .with_status(AgentTurnStatus::Stopped)
                        .with_messages(messages.clone());
                    commit_session_history(request.session(), &messages, pre_run_len);
                    return result;
                }
                tool_steps_taken += 1;
                let tool_name = ToolName::new(tool_call.name());

                self.sink.handle(&AgentEvent::ToolInvoked {
                    session_id: request.session().id().clone(),
                    tool: tool_name.clone(),
                    id: Some(tool_call.id().clone()),
                });

                let tool_context = ToolContext::new(
                    request.session().workspace_scope().clone(),
                    request.session().id().clone(),
                    request.session().mode(),
                )
                .with_permissions(request.session().permissions().clone());

                let tool_result = self
                    .execute_tool(&registry, &tool_name, tool_call, &tool_context)
                    .await;

                let observation_content = tool_observation_text(&tool_result);
                result.push_tool_call(tool_result);
                messages.push(Message::tool_result(
                    tool_call.id().clone(),
                    observation_content,
                ));
            }
        }
    }

    /// Execute a single tool call through the registry, translating
    /// every registry error into the appropriate `ToolCallResult`
    /// shape and emitting the matching `AgentEvent`.
    async fn execute_tool(
        &self,
        registry: &Arc<AgentToolRegistry>,
        tool_name: &ToolName,
        tool_call: &crate::provider::ToolCall,
        tool_context: &ToolContext,
    ) -> ToolCallResult {
        let session_id = tool_context.agent_session_id().clone();
        let tool_call_id = tool_call.id().clone();
        let tool_name_owned = tool_name.clone();

        match registry
            .invoke(tool_name, tool_context, tool_call.arguments().clone())
            .await
        {
            Ok(output) => {
                self.sink.handle(&AgentEvent::ToolCompleted {
                    session_id: session_id.clone(),
                    tool: tool_name_owned.clone(),
                    id: Some(tool_call_id.clone()),
                });
                ToolCallResult::new(tool_call_id, tool_name_owned)
                    .with_session(session_id)
                    .succeeded(output)
                    .set_effective_from_output()
            }
            Err(ToolRegistryError::UnknownTool(_)) => {
                let err = ToolError::new(
                    ToolErrorCode::UnknownTool,
                    format!("tool `{tool_name_owned}` is not registered"),
                )
                .with_tool(tool_name_owned.clone());
                self.emit_tool_failed(&session_id, &tool_name_owned, &tool_call_id, &err);
                ToolCallResult::new(tool_call_id, tool_name_owned)
                    .with_session(session_id)
                    .rejected(err)
            }
            Err(ToolRegistryError::PolicyDenied(err)) => {
                self.emit_tool_failed(&session_id, &tool_name_owned, &tool_call_id, &err);
                ToolCallResult::new(tool_call_id, tool_name_owned)
                    .with_session(session_id)
                    .rejected(err)
            }
            Err(ToolRegistryError::ToolReturned(err)) => {
                self.emit_tool_failed(&session_id, &tool_name_owned, &tool_call_id, &err);
                ToolCallResult::new(tool_call_id, tool_name_owned)
                    .with_session(session_id)
                    .failed(err)
            }
            Err(other) => {
                // SpecHasNoModes / DuplicateName are registration-time
                // invariants; reaching here means the registry is
                // internally inconsistent. We surface as a tool-level
                // failure rather than panic.
                let err = ToolError::new(
                    ToolErrorCode::ExecutionFailed,
                    format!("registry rejected `{tool_name_owned}`: {other}"),
                )
                .with_tool(tool_name_owned.clone());
                self.emit_tool_failed(&session_id, &tool_name_owned, &tool_call_id, &err);
                ToolCallResult::new(tool_call_id, tool_name_owned)
                    .with_session(session_id)
                    .failed(err)
            }
        }
    }

    fn emit_tool_failed(
        &self,
        session_id: &crate::ids::AgentSessionId,
        tool: &ToolName,
        tool_call_id: &crate::provider::ToolCallId,
        err: &ToolError,
    ) {
        self.sink.handle(&AgentEvent::ToolFailed {
            session_id: session_id.clone(),
            tool: tool.clone(),
            id: Some(tool_call_id.clone()),
            code: err.code(),
            message: err.message().to_string(),
        });
    }
}

/// Append the messages produced during this turn to the session's
/// running conversation history. Only the messages beyond
/// `pre_run_len` are appended — i.e. the messages generated by this
/// turn (input + assistant + tool observations + final assistant),
/// not the prior history the loop prepended at the start of the
/// turn.
fn commit_session_history(
    session: &crate::session::AgentSession,
    messages: &[Message],
    pre_run_len: usize,
) {
    if messages.len() > pre_run_len {
        session.append_history(messages[pre_run_len..].iter().cloned());
    }
}

/// Build `AgentToolDefinition` values from registry specs that allow
/// `mode`. Specs without an `input_schema` are sent with a JSON Schema
/// `{"type": "object"}` placeholder so adapters always see a valid
/// schema.
fn build_tool_definitions(
    registry: &AgentToolRegistry,
    mode: crate::mode::AgentMode,
) -> Vec<AgentToolDefinition> {
    registry
        .list()
        .into_iter()
        .filter(|spec| spec.allows_mode(mode))
        .map(|spec| {
            let schema = spec
                .input_schema()
                .cloned()
                .unwrap_or_else(|| json!({"type": "object"}));
            AgentToolDefinition::new(spec.name().as_str(), spec.description().to_string(), schema)
        })
        .collect()
}

/// Serialize a `ToolCallResult` into the JSON text the model receives
/// as a tool observation. Successful calls carry their raw output
/// JSON; rejected/failed calls carry a structured diagnostic envelope.
fn tool_observation_text(result: &ToolCallResult) -> String {
    match result.status() {
        ToolCallStatus::Succeeded => {
            serde_json::to_string(&result.output().cloned().unwrap_or(Value::Null))
                .unwrap_or_else(|_| "null".to_string())
        }
        ToolCallStatus::Rejected | ToolCallStatus::Failed => {
            let err = result
                .diagnostic()
                .expect("rejected/failed result carries a diagnostic");
            serde_json::to_string(&json!({
                "status": result.status().as_str(),
                "code": err.code().as_str(),
                "message": err.message(),
                "tool": result.tool_name().as_str(),
            }))
            .unwrap_or_else(|_| {
                format!(
                    "{{\"status\":\"{}\",\"code\":\"{}\",\"message\":\"{}\"}}",
                    result.status().as_str(),
                    err.code().as_str(),
                    err.message(),
                )
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ProviderError;
    use crate::ids::{AgentSessionId, ModelName, ProviderName, WorkspaceScope};
    use crate::mode::AgentMode;
    use crate::permissions::{PermissionSet, ToolPermission, ToolRiskLevel};
    use crate::provider::{
        AgentResponse, AgentStream, AgentStreamEvent, ModelInfo, ToolCall, ToolCallId, Usage,
    };
    use crate::session::AgentSession;
    use crate::tool::{AgentTool, ToolResult, ToolSpec};
    use crate::turn::{AgentTurnId, AgentTurnRequest};
    use async_trait::async_trait;
    use serde_json::{Value, json};
    use std::collections::VecDeque;
    use std::sync::Mutex;

    // ----- mock provider -----

    #[derive(Debug)]
    enum ScriptedStep {
        Respond(AgentResponse),
        Fail(ProviderError),
    }

    struct ScriptedProvider {
        name: ProviderName,
        steps: Mutex<VecDeque<ScriptedStep>>,
        call_count: Mutex<usize>,
    }

    impl ScriptedProvider {
        fn new(name: &str, steps: Vec<ScriptedStep>) -> Self {
            Self {
                name: ProviderName::new(name),
                steps: Mutex::new(steps.into()),
                call_count: Mutex::new(0),
            }
        }

        fn call_count(&self) -> usize {
            *self.call_count.lock().unwrap()
        }
    }

    #[async_trait]
    impl AgentProvider for ScriptedProvider {
        fn name(&self) -> ProviderName {
            self.name.clone()
        }

        async fn complete(&self, _request: AgentRequest) -> Result<AgentResponse, ProviderError> {
            let mut count = self.call_count.lock().unwrap();
            *count += 1;
            drop(count);

            let mut steps = self.steps.lock().unwrap();
            match steps.pop_front() {
                Some(ScriptedStep::Respond(response)) => Ok(response),
                Some(ScriptedStep::Fail(err)) => Err(err),
                None => Err(ProviderError::new(
                    "scripted_exhausted",
                    "scripted provider ran out of steps",
                )
                .with_provider(self.name.clone())),
            }
        }

        async fn stream(
            &self,
            _request: AgentRequest,
        ) -> Result<Box<dyn AgentStream>, ProviderError> {
            Err(ProviderError::new(
                "streaming_unsupported",
                "stream() not supported in V1 tests",
            )
            .with_provider(self.name.clone()))
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
            Ok(Vec::new())
        }
    }

    struct UnusedStream;
    #[async_trait]
    impl AgentStream for UnusedStream {
        async fn next_event(&mut self) -> Option<AgentStreamEvent> {
            None
        }
    }

    // ----- mock tools -----

    struct ScriptedTool {
        spec: ToolSpec,
        // Wrap mutable state in `Arc` so `Clone` shares the
        // backing storage — tests can hold an `Arc<ScriptedTool>`
        // outside the session registry and observe its
        // `recorded_inputs` after the turn.
        outputs: Arc<Mutex<VecDeque<Result<Value, ToolError>>>>,
        recorded_inputs: Arc<Mutex<Vec<Value>>>,
    }

    impl Clone for ScriptedTool {
        fn clone(&self) -> Self {
            Self {
                spec: self.spec.clone(),
                outputs: Arc::clone(&self.outputs),
                recorded_inputs: Arc::clone(&self.recorded_inputs),
            }
        }
    }

    impl ScriptedTool {
        fn success(name: &str, outputs: Vec<Value>) -> Self {
            let spec = ToolSpec::new(
                ToolName::new(name),
                "scripted success tool",
                [AgentMode::Agent, AgentMode::Build],
                ToolPermission::new("workflow.read"),
                ToolRiskLevel::Read,
            );
            Self {
                spec,
                outputs: Arc::new(Mutex::new(outputs.into_iter().map(Ok).collect())),
                recorded_inputs: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn failing(name: &str, message: &str) -> Self {
            let spec = ToolSpec::new(
                ToolName::new(name),
                "scripted failing tool",
                [AgentMode::Agent, AgentMode::Build],
                ToolPermission::new("workflow.read"),
                ToolRiskLevel::Read,
            );
            Self {
                spec,
                outputs: Arc::new(Mutex::new(
                    vec![Err(ToolError::new(ToolErrorCode::ExecutionFailed, message))].into(),
                )),
                recorded_inputs: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn recorded_inputs(&self) -> Vec<Value> {
            self.recorded_inputs.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl AgentTool for ScriptedTool {
        fn spec(&self) -> ToolSpec {
            self.spec.clone()
        }

        async fn invoke(&self, _ctx: &ToolContext, input: Value) -> ToolResult {
            self.recorded_inputs.lock().unwrap().push(input.clone());
            let mut outputs = self.outputs.lock().unwrap();
            outputs.pop_front().unwrap_or(Ok(Value::Null))
        }
    }

    // ----- harness builders -----

    fn tool_context_with_read_perm() -> ToolContext {
        ToolContext::new(
            WorkspaceScope::new("ws-1"),
            AgentSessionId::new("sess-1"),
            AgentMode::Agent,
        )
        .with_permissions(PermissionSet::from_iter([ToolPermission::new(
            "workflow.read",
        )]))
    }

    fn make_session(registry: AgentToolRegistry) -> AgentSession {
        AgentSession::new(
            AgentSessionId::new("sess-1"),
            WorkspaceScope::new("ws-1"),
            AgentMode::Agent,
            ProviderName::new("mock"),
            Arc::new(registry),
        )
        .with_permissions(PermissionSet::from_iter([ToolPermission::new(
            "workflow.read",
        )]))
    }

    /// Build a session whose registry is populated by `setup`. Tests
    /// register their scripted tools here; the harness will read the
    /// registry off the session at turn time.
    fn session_with(setup: impl FnOnce(&mut AgentToolRegistry)) -> AgentSession {
        let mut registry = AgentToolRegistry::new();
        setup(&mut registry);
        make_session(registry)
    }

    fn response_with_text(text: &str) -> AgentResponse {
        AgentResponse::new(Message::assistant(text))
    }

    fn response_with_tool_calls(tool_calls: Vec<ToolCall>) -> AgentResponse {
        AgentResponse::new(Message::assistant_with_tool_calls("", tool_calls))
    }

    // ----- tests -----

    #[tokio::test]
    async fn run_turn_final_response_with_no_tool_call() {
        let provider = Arc::new(ScriptedProvider::new(
            "mock",
            vec![ScriptedStep::Respond(response_with_text("hello"))],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider.clone(), sink.clone());

        let req = AgentTurnRequest::new(
            make_session(AgentToolRegistry::new()),
            AgentTurnId::new("turn-1"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        );

        let result = loop_harness.run_turn(req).await;
        assert_eq!(result.status(), AgentTurnStatus::Completed);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::FinalResponse);
        assert_eq!(result.final_response().unwrap().content(), "hello");
        assert_eq!(result.tool_calls().len(), 0);
        assert_eq!(result.tool_steps(), 0);
        assert_eq!(provider.call_count(), 1);
        assert_eq!(result.messages().last().unwrap().content(), "hello");
    }

    #[tokio::test]
    async fn run_turn_single_tool_call_followed_by_final_response() {
        let tool_call = ToolCall::new(ToolCallId::new("c1"), "echo", json!({"x": 1}));

        let provider = Arc::new(ScriptedProvider::new(
            "mock",
            vec![
                ScriptedStep::Respond(response_with_tool_calls(vec![tool_call])),
                ScriptedStep::Respond(response_with_text("done")),
            ],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider.clone(), sink.clone());

        let req = AgentTurnRequest::new(
            session_with(|reg| {
                reg.register(ScriptedTool::success(
                    "echo",
                    vec![json!({"ok": true, "effective": true})],
                ))
                .unwrap();
            }),
            AgentTurnId::new("turn-2"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        );

        let result = loop_harness.run_turn(req).await;
        assert_eq!(result.status(), AgentTurnStatus::Completed);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::FinalResponse);
        assert_eq!(result.tool_calls().len(), 1);
        assert_eq!(result.tool_calls()[0].tool_name().as_str(), "echo");
        assert_eq!(result.tool_calls()[0].status(), ToolCallStatus::Succeeded);
        assert_eq!(result.tool_calls()[0].effective(), Some(true));
        assert_eq!(provider.call_count(), 2);

        let events = sink.events();
        // Invoked + Completed per tool call.
        let invoked = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolInvoked { .. }))
            .count();
        let completed = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolCompleted { .. }))
            .count();
        assert_eq!(invoked, 1);
        assert_eq!(completed, 1);

        // The transcript should contain input + assistant + tool_result + final assistant.
        let messages = result.messages();
        let roles: Vec<&str> = messages.iter().map(|m| m.role()).collect();
        assert_eq!(roles, vec!["user", "assistant", "tool", "assistant"]);
        assert_eq!(messages[2].tool_call_id().unwrap().as_str(), "c1");
    }

    #[tokio::test]
    async fn run_turn_multiple_sequential_tool_calls_in_one_assistant_message() {
        let tool_calls = vec![
            ToolCall::new(ToolCallId::new("c1"), "alpha", json!({})),
            ToolCall::new(ToolCallId::new("c2"), "beta", json!({})),
            ToolCall::new(ToolCallId::new("c3"), "gamma", json!({})),
        ];

        let provider = Arc::new(ScriptedProvider::new(
            "mock",
            vec![
                ScriptedStep::Respond(response_with_tool_calls(tool_calls)),
                ScriptedStep::Respond(response_with_text("all done")),
            ],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink.clone());

        let req = AgentTurnRequest::new(
            session_with(|reg| {
                reg.register(ScriptedTool::success("alpha", vec![json!({"step": 1})]))
                    .unwrap();
                reg.register(ScriptedTool::success("beta", vec![json!({"step": 2})]))
                    .unwrap();
                reg.register(ScriptedTool::success("gamma", vec![json!({"step": 3})]))
                    .unwrap();
            }),
            AgentTurnId::new("turn-3"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        );

        let result = loop_harness.run_turn(req).await;
        assert_eq!(result.status(), AgentTurnStatus::Completed);
        assert_eq!(result.tool_calls().len(), 3);
        // Provider order preserved.
        assert_eq!(result.tool_calls()[0].tool_call_id().as_str(), "c1");
        assert_eq!(result.tool_calls()[1].tool_call_id().as_str(), "c2");
        assert_eq!(result.tool_calls()[2].tool_call_id().as_str(), "c3");

        // All three tool observations must precede the final assistant message.
        let messages = result.messages();
        let tool_indices: Vec<usize> = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role() == "tool")
            .map(|(i, _)| i)
            .collect();
        assert_eq!(tool_indices.len(), 3);
        let final_assistant_index = messages
            .iter()
            .position(|m| m.role() == "assistant" && m.content() == "all done")
            .expect("final assistant message present");
        for i in tool_indices {
            assert!(i < final_assistant_index);
        }
    }

    #[tokio::test]
    async fn run_turn_tool_policy_denial_is_returned_as_observation() {
        let tool_call = ToolCall::new(ToolCallId::new("c1"), "needs-write", json!({}));
        let provider = Arc::new(ScriptedProvider::new(
            "mock",
            vec![
                ScriptedStep::Respond(response_with_tool_calls(vec![tool_call])),
                ScriptedStep::Respond(response_with_text("ok")),
            ],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink);

        // Session has only `workflow.read`; the registered tool
        // requires `workflow.write`, so policy denies.
        let session = session_with(|reg| {
            struct StrictTool(crate::tool::ToolSpec);
            #[async_trait]
            impl crate::tool::AgentTool for StrictTool {
                fn spec(&self) -> crate::tool::ToolSpec {
                    self.0.clone()
                }
                async fn invoke(
                    &self,
                    _ctx: &ToolContext,
                    _input: Value,
                ) -> crate::tool::ToolResult {
                    panic!("policy should have denied this call")
                }
            }
            let spec = ToolSpec::new(
                ToolName::new("needs-write"),
                "needs-write",
                [AgentMode::Agent],
                ToolPermission::new("workflow.write"),
                ToolRiskLevel::Editor,
            );
            reg.register(StrictTool(spec)).unwrap();
        })
        .with_started_at("2026-06-11T00:00:00Z");

        let req = AgentTurnRequest::new(
            session,
            AgentTurnId::new("turn-4"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        )
        .with_max_tool_steps(5);

        let result = loop_harness.run_turn(req).await;
        assert_eq!(result.status(), AgentTurnStatus::Completed);
        assert_eq!(result.tool_calls().len(), 1);
        assert_eq!(result.tool_calls()[0].status(), ToolCallStatus::Rejected);
        let diag = result.tool_calls()[0]
            .diagnostic()
            .expect("rejected carries a diagnostic");
        assert_eq!(diag.code(), ToolErrorCode::PermissionDenied);

        // The observation message should be a JSON envelope containing the
        // diagnostic code so the model can recover.
        let observation = &result.messages()[2];
        assert_eq!(observation.role(), "tool");
        let parsed: Value = serde_json::from_str(observation.content()).unwrap();
        assert_eq!(parsed["status"], "rejected");
        assert_eq!(parsed["code"], "AGENT/TOOL_PERMISSION_DENIED");
    }

    #[tokio::test]
    async fn run_turn_unknown_tool_is_returned_as_observation() {
        let tool_call = ToolCall::new(ToolCallId::new("c1"), "ghost", json!({}));
        let provider = Arc::new(ScriptedProvider::new(
            "mock",
            vec![
                ScriptedStep::Respond(response_with_tool_calls(vec![tool_call])),
                ScriptedStep::Respond(response_with_text("recovered")),
            ],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink.clone());

        let req = AgentTurnRequest::new(
            session_with(|_reg| {
                // Registry stays empty on purpose — the model will
                // request a tool name that isn't registered.
            }),
            AgentTurnId::new("turn-5"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        );

        let result = loop_harness.run_turn(req).await;
        assert_eq!(result.status(), AgentTurnStatus::Completed);
        assert_eq!(result.tool_calls().len(), 1);
        assert_eq!(result.tool_calls()[0].status(), ToolCallStatus::Rejected);
        assert_eq!(
            result.tool_calls()[0].diagnostic().unwrap().code(),
            ToolErrorCode::UnknownTool
        );

        // The model saw the rejection as an observation and recovered.
        let observation = &result.messages()[2];
        let parsed: Value = serde_json::from_str(observation.content()).unwrap();
        assert_eq!(parsed["code"], "AGENT/TOOL_UNKNOWN");
        assert_eq!(sink.events().len(), 2); // ToolInvoked + ToolFailed
    }

    #[tokio::test]
    async fn run_turn_provider_error_stops_turn() {
        let provider = Arc::new(ScriptedProvider::new(
            "mock",
            vec![ScriptedStep::Fail(
                ProviderError::new("RATE_LIMIT", "slow down")
                    .with_provider(ProviderName::new("mock")),
            )],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink.clone());

        let req = AgentTurnRequest::new(
            make_session(AgentToolRegistry::new()),
            AgentTurnId::new("turn-6"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        );

        let result = loop_harness.run_turn(req).await;
        assert_eq!(result.status(), AgentTurnStatus::Stopped);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::ProviderError);
        assert!(result.final_response().is_none());
        assert_eq!(result.diagnostics().len(), 1);
        assert_eq!(
            result.diagnostics()[0].code().as_str(),
            "AGENT/PROVIDER_RATE_LIMIT"
        );

        let provider_event_count = sink
            .events()
            .iter()
            .filter(|e| matches!(e, AgentEvent::ProviderError { .. }))
            .count();
        assert_eq!(provider_event_count, 1);
    }

    #[tokio::test]
    async fn run_turn_max_tool_steps_stops_turn() {
        // Provider keeps asking for tools forever; loop should stop at the guard.
        let scripted_responses: Vec<ScriptedStep> = (0..5)
            .map(|i| {
                ScriptedStep::Respond(response_with_tool_calls(vec![ToolCall::new(
                    ToolCallId::new(format!("c{i}")),
                    "loop",
                    json!({}),
                )]))
            })
            .collect();

        let provider = Arc::new(ScriptedProvider::new("mock", scripted_responses));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink);

        let req = AgentTurnRequest::new(
            session_with(|reg| {
                reg.register(ScriptedTool::success("loop", vec![json!({}); 20]))
                    .unwrap();
            }),
            AgentTurnId::new("turn-7"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        )
        .with_max_tool_steps(2);

        let result = loop_harness.run_turn(req).await;
        assert_eq!(result.status(), AgentTurnStatus::Stopped);
        assert_eq!(
            result.stop_reason(),
            AgentTurnStopReason::MaxToolStepsExceeded
        );
        assert!(result.final_response().is_none());
        // 2 individual tool calls executed (one per round), then the
        // 3rd round hit the guard.
        assert_eq!(result.tool_calls().len(), 2);

        // B1 regression: the transcript must end with the last
        // executed tool observation, NOT an unfulfilled assistant
        // tool-call message.
        let last = result.messages().last().expect("transcript not empty");
        assert_eq!(
            last.role(),
            "tool",
            "transcript should end on tool observation, got role={}",
            last.role()
        );
    }

    #[tokio::test]
    async fn run_turn_max_tool_steps_counts_individual_tool_calls_not_rounds() {
        // Lock the per-tool-call semantics of `max_tool_steps`: a
        // single round that requests 3 tool calls counts as 3 against
        // the guard. With `max_tool_steps=2`, the third tool call in
        // the first round trips the guard mid-round, but the first
        // two tool calls already executed — so we get 2 tool calls
        // and a transcript ending on the second tool observation.
        let tool_calls = vec![
            ToolCall::new(ToolCallId::new("c1"), "loop", json!({"step": 1})),
            ToolCall::new(ToolCallId::new("c2"), "loop", json!({"step": 2})),
            ToolCall::new(ToolCallId::new("c3"), "loop", json!({"step": 3})),
        ];

        let provider = Arc::new(ScriptedProvider::new(
            "mock",
            vec![
                ScriptedStep::Respond(response_with_tool_calls(tool_calls)),
                ScriptedStep::Respond(response_with_text("would never get here")),
            ],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink);

        let req = AgentTurnRequest::new(
            session_with(|reg| {
                reg.register(ScriptedTool::success("loop", vec![json!({}); 20]))
                    .unwrap();
            }),
            AgentTurnId::new("turn-mid-round"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        )
        .with_max_tool_steps(2);

        let result = loop_harness.run_turn(req).await;
        assert_eq!(result.status(), AgentTurnStatus::Stopped);
        assert_eq!(
            result.stop_reason(),
            AgentTurnStopReason::MaxToolStepsExceeded
        );
        // Two tool calls executed before the guard tripped on the third.
        assert_eq!(result.tool_calls().len(), 2);
        assert_eq!(result.tool_calls()[0].tool_call_id().as_str(), "c1");
        assert_eq!(result.tool_calls()[1].tool_call_id().as_str(), "c2");

        // The transcript ends on the last executed tool observation,
        // not on the assistant message that requested the unexecuted
        // third tool call.
        let last = result.messages().last().unwrap();
        assert_eq!(last.role(), "tool");
        assert_eq!(last.tool_call_id().unwrap().as_str(), "c2");
    }

    #[tokio::test]
    async fn run_turn_proposal_style_tool_output_round_trips_as_observation() {
        // A tool that returns `{"effective": false, "proposal_id": "p1"}`
        // should be carried back as a successful observation with the
        // effective flag extracted.
        let tool_call = ToolCall::new(
            ToolCallId::new("c1"),
            "workflow.propose_commands",
            json!({}),
        );

        let provider = Arc::new(ScriptedProvider::new(
            "mock",
            vec![
                ScriptedStep::Respond(response_with_tool_calls(vec![tool_call])),
                ScriptedStep::Respond(response_with_text("proposal ready")),
            ],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink);

        let req = AgentTurnRequest::new(
            session_with(|reg| {
                reg.register(ScriptedTool::success(
                    "workflow.propose_commands",
                    vec![json!({
                        "proposal_id": "p1",
                        "workflow_id": "wf-1",
                        "base_version": 3,
                        "status": "pending",
                        "effective": false
                    })],
                ))
                .unwrap();
            }),
            AgentTurnId::new("turn-8"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        );

        let result = loop_harness.run_turn(req).await;
        assert_eq!(result.status(), AgentTurnStatus::Completed);
        assert_eq!(result.tool_calls().len(), 1);
        let tcr = &result.tool_calls()[0];
        assert_eq!(tcr.status(), ToolCallStatus::Succeeded);
        assert_eq!(tcr.effective(), Some(false));
        assert_eq!(tcr.output().unwrap()["proposal_id"], "p1");

        // Observation carries the raw tool output JSON.
        let observation = &result.messages()[2];
        assert_eq!(observation.role(), "tool");
        let parsed: Value = serde_json::from_str(observation.content()).unwrap();
        assert_eq!(parsed["effective"], false);
        assert_eq!(parsed["proposal_id"], "p1");
    }

    #[tokio::test]
    async fn run_turn_usage_is_aggregated_from_final_response() {
        let provider = Arc::new(ScriptedProvider::new(
            "mock",
            vec![ScriptedStep::Respond(
                response_with_text("done").with_usage(Usage::new(Some(11), Some(22))),
            )],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink);

        let req = AgentTurnRequest::new(
            make_session(AgentToolRegistry::new()),
            AgentTurnId::new("turn-9"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        );

        let result = loop_harness.run_turn(req).await;
        let usage = result.usage().expect("usage recorded");
        assert_eq!(usage.input_tokens(), Some(11));
        assert_eq!(usage.output_tokens(), Some(22));
    }

    #[tokio::test]
    async fn run_turn_tool_execution_error_is_returned_as_failed_observation() {
        let tool_call = ToolCall::new(ToolCallId::new("c1"), "boom", json!({}));
        let provider = Arc::new(ScriptedProvider::new(
            "mock",
            vec![
                ScriptedStep::Respond(response_with_tool_calls(vec![tool_call])),
                ScriptedStep::Respond(response_with_text("ok")),
            ],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink.clone());

        let req = AgentTurnRequest::new(
            session_with(|reg| {
                reg.register(ScriptedTool::failing("boom", "kaboom"))
                    .unwrap();
            }),
            AgentTurnId::new("turn-10"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        );

        let result = loop_harness.run_turn(req).await;
        assert_eq!(result.status(), AgentTurnStatus::Completed);
        assert_eq!(result.tool_calls().len(), 1);
        assert_eq!(result.tool_calls()[0].status(), ToolCallStatus::Failed);
        assert_eq!(
            result.tool_calls()[0].diagnostic().unwrap().code(),
            ToolErrorCode::ExecutionFailed
        );

        let observation = &result.messages()[2];
        let parsed: Value = serde_json::from_str(observation.content()).unwrap();
        assert_eq!(parsed["status"], "failed");
        assert_eq!(parsed["code"], "AGENT/TOOL_EXECUTION_FAILED");
        assert_eq!(parsed["message"], "kaboom");
    }

    #[tokio::test]
    async fn run_turn_advertises_tool_definitions_to_provider() {
        // The provider receives tool definitions derived from the registry.
        // We assert this indirectly by registering two tools, scripting a
        // final response, and verifying the registry's tool list was used.
        struct CapturingProvider {
            name: ProviderName,
            seen_tool_names: Mutex<Vec<String>>,
        }
        #[async_trait]
        impl AgentProvider for CapturingProvider {
            fn name(&self) -> ProviderName {
                self.name.clone()
            }
            async fn complete(
                &self,
                request: AgentRequest,
            ) -> Result<AgentResponse, ProviderError> {
                self.seen_tool_names
                    .lock()
                    .unwrap()
                    .extend(request.tools().iter().map(|d| d.name().to_string()));
                Ok(response_with_text("done"))
            }
            async fn stream(
                &self,
                _request: AgentRequest,
            ) -> Result<Box<dyn AgentStream>, ProviderError> {
                Ok(Box::new(UnusedStream))
            }
            async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
                Ok(Vec::new())
            }
        }

        let mut registry = AgentToolRegistry::new();
        registry
            .register(ScriptedTool::success("alpha", vec![]))
            .unwrap();
        registry
            .register(ScriptedTool::success("beta", vec![]))
            .unwrap();

        let provider = Arc::new(CapturingProvider {
            name: ProviderName::new("capture"),
            seen_tool_names: Mutex::new(Vec::new()),
        });
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider.clone(), sink);

        let req = AgentTurnRequest::new(
            session_with(|reg| {
                reg.register(ScriptedTool::success("alpha", vec![]))
                    .unwrap();
                reg.register(ScriptedTool::success("beta", vec![])).unwrap();
            }),
            AgentTurnId::new("turn-11"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        );

        let _ = loop_harness.run_turn(req).await;
        let mut names = provider.seen_tool_names.lock().unwrap().clone();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[tokio::test]
    async fn run_turn_invokes_registry_for_each_tool_call() {
        // The harness must invoke the registry for every tool call the
        // model requests, in provider order, with the exact arguments.
        // We hold the registered `ScriptedTool` behind an `Arc` so the
        // test can read its `recorded_inputs` after the turn.
        let recorded: Arc<ScriptedTool> = Arc::new(ScriptedTool::success(
            "record",
            vec![json!({"step": 1}), json!({"step": 2})],
        ));

        let tool_calls = vec![
            ToolCall::new(ToolCallId::new("c1"), "record", json!({"step": 1})),
            ToolCall::new(ToolCallId::new("c2"), "record", json!({"step": 2})),
        ];

        let provider = Arc::new(ScriptedProvider::new(
            "mock",
            vec![
                ScriptedStep::Respond(response_with_tool_calls(tool_calls)),
                ScriptedStep::Respond(response_with_text("done")),
            ],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink);

        let recorder_for_session = Arc::clone(&recorded);
        let req = AgentTurnRequest::new(
            session_with(move |reg| {
                // Move a clone into the session registry; the test
                // holds the original `recorded` for assertions.
                reg.register((*recorder_for_session).clone()).unwrap();
            }),
            AgentTurnId::new("turn-12"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        );

        let _ = loop_harness.run_turn(req).await;

        // The harness must have invoked the registry with both inputs,
        // in order. This was previously a no-op assertion; now it
        // actually checks behavior.
        let inputs = recorded.recorded_inputs();
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0], json!({"step": 1}));
        assert_eq!(inputs[1], json!({"step": 2}));
    }

    #[tokio::test]
    async fn vec_event_sink_records_events_in_order() {
        let sink = VecAgentEventSink::new();
        let session_id = AgentSessionId::new("sess-1");
        sink.handle(&AgentEvent::SessionStarted {
            session_id: session_id.clone(),
            provider: ProviderName::new("mock"),
            mode: AgentMode::Agent,
        });
        sink.handle(&AgentEvent::ToolInvoked {
            session_id: session_id.clone(),
            tool: ToolName::new("echo"),
            id: None,
        });
        assert_eq!(sink.len(), 2);
        let events = sink.events();
        assert!(matches!(events[0], AgentEvent::SessionStarted { .. }));
        assert!(matches!(events[1], AgentEvent::ToolInvoked { .. }));
    }

    #[tokio::test]
    async fn build_tool_definitions_omits_tools_not_in_mode() {
        let mut registry = AgentToolRegistry::new();

        let agent_only = ScriptedTool {
            spec: ToolSpec::new(
                ToolName::new("agent-only"),
                "agent-only",
                [AgentMode::Agent],
                ToolPermission::new("workflow.read"),
                ToolRiskLevel::Read,
            ),
            outputs: Arc::new(Mutex::new(VecDeque::new())),
            recorded_inputs: Arc::new(Mutex::new(Vec::new())),
        };
        let build_only = ScriptedTool {
            spec: ToolSpec::new(
                ToolName::new("build-only"),
                "build-only",
                [AgentMode::Build],
                ToolPermission::new("workflow.read"),
                ToolRiskLevel::Read,
            ),
            outputs: Arc::new(Mutex::new(VecDeque::new())),
            recorded_inputs: Arc::new(Mutex::new(Vec::new())),
        };
        registry.register(agent_only).unwrap();
        registry.register(build_only).unwrap();

        let defs = build_tool_definitions(&registry, AgentMode::Agent);
        let names: Vec<&str> = defs.iter().map(|d| d.name()).collect();
        assert_eq!(names, vec!["agent-only"]);
    }

    #[test]
    fn tool_observation_text_for_succeeded_carries_output_json() {
        let result = ToolCallResult::new(ToolCallId::new("c1"), ToolName::new("echo"))
            .succeeded(json!({"a": 1}))
            .set_effective_from_output();
        let text = tool_observation_text(&result);
        assert_eq!(text, json!({"a": 1}).to_string());
    }

    #[test]
    fn tool_observation_text_for_rejected_envelope() {
        let err = ToolError::new(ToolErrorCode::PermissionDenied, "no perm")
            .with_tool(ToolName::new("needs-write"));
        let result =
            ToolCallResult::new(ToolCallId::new("c1"), ToolName::new("needs-write")).rejected(err);
        let text = tool_observation_text(&result);
        let parsed: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["status"], "rejected");
        assert_eq!(parsed["code"], "AGENT/TOOL_PERMISSION_DENIED");
        assert_eq!(parsed["tool"], "needs-write");
    }

    #[test]
    fn tool_observation_text_for_failed_envelope() {
        let err =
            ToolError::new(ToolErrorCode::ExecutionFailed, "boom").with_tool(ToolName::new("boom"));
        let result = ToolCallResult::new(ToolCallId::new("c1"), ToolName::new("boom")).failed(err);
        let text = tool_observation_text(&result);
        let parsed: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["status"], "failed");
        assert_eq!(parsed["code"], "AGENT/TOOL_EXECUTION_FAILED");
    }

    #[test]
    fn make_session_helper() {
        let _ = make_session(AgentToolRegistry::new());
    }

    #[test]
    fn tool_context_with_read_perm_helper() {
        let _ = tool_context_with_read_perm();
    }

    #[tokio::test]
    async fn run_turn_appends_to_session_history() {
        // After a turn completes, the session's running history must
        // contain the input + assistant messages + tool observations
        // from the turn, in order. The next turn (driven from an
        // empty input) must see the prior transcript in its provider
        // request.
        struct CapturingProvider {
            name: ProviderName,
            seen_message_roles: Mutex<Vec<String>>,
        }
        #[async_trait]
        impl AgentProvider for CapturingProvider {
            fn name(&self) -> ProviderName {
                self.name.clone()
            }
            async fn complete(
                &self,
                request: AgentRequest,
            ) -> Result<AgentResponse, ProviderError> {
                self.seen_message_roles
                    .lock()
                    .unwrap()
                    .extend(request.messages().iter().map(|m| m.role().to_string()));
                Ok(response_with_text("done"))
            }
            async fn stream(
                &self,
                _request: AgentRequest,
            ) -> Result<Box<dyn AgentStream>, ProviderError> {
                Ok(Box::new(UnusedStream))
            }
            async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
                Ok(Vec::new())
            }
        }

        let session = make_session(AgentToolRegistry::new());
        assert_eq!(session.history_len(), 0, "history starts empty");

        let provider = Arc::new(CapturingProvider {
            name: ProviderName::new("capture"),
            seen_message_roles: Mutex::new(Vec::new()),
        });
        let loop_harness = AgentLoop::new(provider.clone(), Arc::new(VecAgentEventSink::new()));

        // Turn 1: input is just the user message; session history
        // starts empty.
        let req = AgentTurnRequest::new(
            session.clone(),
            AgentTurnId::new("turn-1"),
            ModelName::new("test-model"),
            vec![Message::user("hello")],
        );
        let _ = loop_harness.run_turn(req).await;

        // Session history should now contain [user, assistant].
        let history = session.history();
        assert_eq!(
            history.len(),
            2,
            "session history grew by exactly 2 messages"
        );
        assert_eq!(history[0].role(), "user");
        assert_eq!(history[0].content(), "hello");
        assert_eq!(history[1].role(), "assistant");
        assert_eq!(history[1].content(), "done");

        // Turn 2: empty input. The loop should still seed the
        // provider request with the prior transcript so the model
        // sees the conversation so far.
        let req = AgentTurnRequest::new(
            session.clone(),
            AgentTurnId::new("turn-2"),
            ModelName::new("test-model"),
            vec![Message::user("and now this")],
        );
        let _ = loop_harness.run_turn(req).await;

        let provider_roles = provider.seen_message_roles.lock().unwrap().clone();
        // Round 1 sent [user]; round 2 sent [user, assistant, user].
        let round_2_start = provider_roles
            .iter()
            .rposition(|r| r == "user")
            .expect("second-turn user message recorded");
        assert_eq!(
            &provider_roles[round_2_start - 2..round_2_start + 1],
            &["user", "assistant", "user"],
            "second turn's provider request must include prior history"
        );

        // Session history now spans both turns.
        let final_history = session.history();
        assert_eq!(final_history.len(), 4);
        assert_eq!(final_history[3].role(), "assistant");
        assert_eq!(final_history[3].content(), "done");
    }

    #[tokio::test]
    async fn run_turn_appends_history_on_provider_error_stop() {
        // Even on a stop condition, the loop must commit whatever
        // transcript it produced so the host sees a non-empty history
        // after the failure.
        let session = make_session(AgentToolRegistry::new());
        let provider = Arc::new(ScriptedProvider::new(
            "mock",
            vec![ScriptedStep::Fail(
                ProviderError::new("BOOM", "hard fail").with_provider(ProviderName::new("mock")),
            )],
        ));
        let loop_harness = AgentLoop::new(provider, Arc::new(VecAgentEventSink::new()));

        let req = AgentTurnRequest::new(
            session.clone(),
            AgentTurnId::new("turn-fail"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        );
        let _ = loop_harness.run_turn(req).await;

        // History includes the input even though the provider never
        // produced an assistant message.
        let history = session.history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role(), "user");
        assert_eq!(history[0].content(), "hi");
    }
}
