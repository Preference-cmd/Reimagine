//! Agent loop harness.
//!
//! `AgentLoop` is the V1 Reimagine-owned turn executor. It owns
//! orchestration: it builds provider requests, advertises registry tool
//! specs, executes requested tools through the registry, feeds tool
//! observations back to the provider, emits agent-local events, and
//! stops when the model produces a final assistant response, the
//! configured max-tool-step guard trips, the provider returns an error,
//! the turn deadline passes, or cancellation is requested.
//!
//! Streaming turn execution (`run_turn_streaming`) is implemented
//! alongside the non-streaming `run_turn`. Steering, subagents, memory,
//! and skills remain future work. Cancellation aborts in-flight work:
//! the loop races the request's `CancellationToken` against the
//! provider call and drops the in-flight future (which cancels the
//! underlying HTTP request) when it fires, then emits
//! `AgentTurnStopReason::Cancelled` (AC-07).
//!
//! Both turn paths run through a single shared skeleton
//! (`run_turn_inner`); only the per-round provider interaction differs
//! (`complete` vs `stream`), so the two paths cannot drift again
//! (AC-08).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::context::ToolContext;
use crate::context_manager::ContextManager;
use crate::error::{ProviderError, ToolError, ToolErrorCode};
use crate::event::AgentEvent;
use crate::ids::{AgentSessionId, ModelName, ToolName};
use crate::permissions::ToolRiskLevel;
use crate::provider::{
    AgentProvider, AgentRequest, AgentStreamEvent, AgentToolDefinition, ContentBlock, Message,
    ToolCall,
};
use crate::registry::{AgentToolRegistry, ToolRegistryError};
use crate::turn::{
    AgentTurnRequest, AgentTurnResult, AgentTurnStatus, AgentTurnStopReason, ToolCallResult,
    ToolCallStatus,
};
use crate::validation::validate_json_value;

/// How many corrective retries a structured-output turn gets before it
/// stops with `STRUCTURED_OUTPUT_INVALID` (AR-30).
const MAX_STRUCTURED_OUTPUT_RETRIES: usize = 1;

/// How long a context-compaction summarizer call may run before it is
/// abandoned (AR-19). A stuck summarizer must not wedge the turn; the
/// abandoned summary is treated as transient so compaction retries next
/// turn.
const COMPACTION_DEADLINE: Duration = Duration::from_secs(60);

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

/// Which provider interaction a turn uses (AC-08). Selects the
/// per-round implementation inside the shared turn skeleton
/// ([`AgentLoop::run_turn_inner`]).
#[derive(Clone, Copy)]
enum TurnRound {
    /// Non-streaming: `provider.complete()`.
    Complete,
    /// Streaming: `provider.stream()` + `next_event()`.
    Streaming,
}

/// Outcome of one provider round inside the shared turn skeleton
/// (AC-08).
enum RoundOutcome {
    /// The turn ended while producing this round (provider error,
    /// empty stream, cancellation, or deadline); carries the final
    /// result.
    Ended(Box<AgentTurnResult>),
    /// The round produced an assistant message. `truncated` marks a
    /// stream the provider cut short (`length`/`max_tokens`), which
    /// downgrades a final response to `Truncated`/`Stopped` (AC-01).
    Assistant { assistant: Message, truncated: bool },
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
    ///
    /// `context` optionally supplies a [`ContextManager`] that owns the
    /// conversation history for this turn. When present, the provider
    /// message list is built by `prepare_messages` (system message +
    /// rolling-windowed history + input) and this turn's new messages
    /// are committed through `commit_turn`; the session history is not
    /// mutated. When `None`, V1 behavior applies: the session history
    /// seeds the provider list and receives the turn's messages.
    pub async fn run_turn(
        &self,
        request: AgentTurnRequest,
        context: Option<&mut ContextManager>,
    ) -> AgentTurnResult {
        let turn_start = Instant::now();
        self.run_turn_inner(request, context, TurnRound::Complete)
            .await
            .with_duration_since(turn_start)
    }

    /// Shared turn skeleton (AC-08): both turn paths run through this
    /// loop. The per-round provider interaction is path-specific
    /// (`complete` vs `stream`; see [`Self::complete_round`] and
    /// [`Self::streaming_round`]) — everything else (message seeding,
    /// the modality gate, cancellation/deadline racing, the
    /// max-tool-step guard, sequential tool execution, and history
    /// commits) is common, so the paths cannot drift again.
    async fn run_turn_inner(
        &self,
        request: AgentTurnRequest,
        mut context: Option<&mut ContextManager>,
        round: TurnRound,
    ) -> AgentTurnResult {
        let mut result = base_result(&request);
        let (mut messages, pre_run_len) = match self.prepare_turn(&request, &mut context).await {
            Ok(seeded) => seeded,
            Err(ended) => return ended,
        };
        let registry = request.session().registry().clone();
        let tool_defs = build_tool_definitions(&registry, request.session().mode());
        let max_tool_steps = request.max_tool_steps();
        let mut tool_steps_taken: usize = 0;
        let mut structured_output_retries: usize = 0;
        let turn_start = std::time::Instant::now();

        loop {
            // Check cancellation before each provider round.
            if request.cancel_token().is_cancelled() {
                return self.stop_cancelled(
                    &request,
                    &mut result,
                    &mut context,
                    &messages,
                    pre_run_len,
                );
            }

            // Produce this round's assistant message. Every round races
            // cancellation and the turn deadline, then either yields an
            // assistant message or ends the turn.
            let outcome = match round {
                TurnRound::Complete => {
                    self.complete_round(
                        &request,
                        &mut result,
                        &mut context,
                        &messages,
                        pre_run_len,
                        turn_start,
                        &tool_defs,
                    )
                    .await
                }
                TurnRound::Streaming => {
                    self.streaming_round(
                        &request,
                        &mut result,
                        &mut context,
                        &messages,
                        pre_run_len,
                        turn_start,
                        &tool_defs,
                    )
                    .await
                }
            };
            let (assistant, truncated) = match outcome {
                RoundOutcome::Ended(ended) => return *ended,
                RoundOutcome::Assistant {
                    assistant,
                    truncated,
                } => (assistant, truncated),
            };

            let has_tool_calls = !assistant.tool_calls().is_empty();

            // No tool calls → final response. Append the assistant
            // message first so the transcript ends on it. A stream the
            // provider cut short (`length`/`max_tokens`) keeps the
            // partial response but is reported as incomplete (AC-01).
            if !has_tool_calls {
                messages.push(assistant.clone());
                if truncated {
                    return self.finish(
                        &request,
                        &mut result,
                        &mut context,
                        &messages,
                        pre_run_len,
                        Some(assistant),
                        AgentTurnStopReason::Truncated,
                        AgentTurnStatus::Stopped,
                    );
                }

                // Structured output gate (AR-30): when the caller
                // requested a schema, the final response must be a JSON
                // value valid against it. The first failure feeds a
                // corrective message back and retries one provider
                // round; a second failure stops with a structured
                // `STRUCTURED_OUTPUT_INVALID` error.
                if let Some(schema) = request.output_schema()
                    && let Err(validation_error) =
                        validate_structured_output(assistant.content(), schema)
                {
                    if structured_output_retries < MAX_STRUCTURED_OUTPUT_RETRIES {
                        structured_output_retries += 1;
                        messages.push(Message::user(format!(
                            "Your final response failed JSON Schema validation: \
                             {validation_error}. Required schema: {schema}. \
                             Return only a corrected JSON value that satisfies the schema."
                        )));
                        continue;
                    }
                    return self.stop_with_error(
                        &request,
                        &mut result,
                        &mut context,
                        &messages,
                        pre_run_len,
                        ProviderError::new("STRUCTURED_OUTPUT_INVALID", validation_error),
                        AgentTurnStopReason::ProviderError,
                    );
                }

                return self.finish(
                    &request,
                    &mut result,
                    &mut context,
                    &messages,
                    pre_run_len,
                    Some(assistant),
                    AgentTurnStopReason::FinalResponse,
                    AgentTurnStatus::Completed,
                );
            }

            // Max-tool-step guard. Counts *individual* tool calls. The
            // check sits before the round's tool loop so the round is
            // *admitted* (and its assistant message appended) only when
            // at least one tool call slot remains; a finer-grained
            // check inside `execute_tool_calls` then stops the turn
            // exactly at the limit, even mid-round.
            if tool_steps_taken >= max_tool_steps {
                return self.stop_max_steps(
                    &request,
                    &mut result,
                    &mut context,
                    &messages,
                    pre_run_len,
                );
            }

            // Append the assistant message only after the guard passes
            // — this way the assistant that requested tools always has
            // corresponding `tool_result` observations.
            messages.push(assistant.clone());

            // Execute the requested tool calls sequentially in the
            // order the provider returned them. All observations are
            // appended before the next provider call so the model sees
            // the full batch.
            if let Some(ended) = self
                .execute_tool_calls(
                    &request,
                    &mut result,
                    &mut context,
                    &mut messages,
                    pre_run_len,
                    &registry,
                    assistant.tool_calls(),
                    max_tool_steps,
                    &mut tool_steps_taken,
                )
                .await
            {
                return ended;
            }
        }
    }

    /// Seed the provider message list from the active history source
    /// and run the image-modality gate (PV-03b). Returns the seeded
    /// messages plus `pre_run_len` — the length of the prior context,
    /// past which this turn's slice begins — or the final stopped
    /// result when the gate rejects the turn.
    ///
    /// With a context manager, `prepare_messages` attaches a system
    /// message and applies the rolling window (V1 has no system
    /// prompt, so the system slot is empty). Without one, the session
    /// history seeds the provider list and receives the turn's
    /// messages.
    async fn prepare_turn(
        &self,
        request: &AgentTurnRequest,
        context: &mut Option<&mut ContextManager>,
    ) -> Result<(Vec<Message>, usize), AgentTurnResult> {
        let (messages, pre_run_len) = if let Some(manager) = context.as_deref_mut() {
            // Compaction driver (CM-V2e): when the stored history
            // crosses the soft budget line and the thrash guard allows,
            // summarize the eviction range before the window is
            // applied. Failures fall back to plain eviction inside
            // `prepare_messages`; the turn still completes.
            if let Some(event) = maybe_compact(
                manager,
                self.provider.as_ref(),
                request.model(),
                request.session().id(),
                request.cancel_token(),
            )
            .await
            {
                self.sink.handle(&event);
            }
            let messages = manager.prepare_messages("", request.input());
            let pre_run_len = messages.len() - request.input().len();
            (messages, pre_run_len)
        } else {
            let prior_history = request.session().history();
            let pre_run_len = prior_history.len();
            let messages = prior_history
                .iter()
                .cloned()
                .chain(request.input().iter().cloned())
                .collect();
            (messages, pre_run_len)
        };

        // Modality gate (PV-03b): image file blocks require a model that
        // advertises image input support. Best-effort — models missing
        // from the provider listing are allowed through.
        if let Err(err) = enforce_image_modality_gate(
            self.provider.as_ref(),
            request.model(),
            &image_media_types(&messages),
        )
        .await
        {
            let mut stopped = base_result(request);
            return Err(self.stop_with_provider_error(
                request,
                &mut stopped,
                context,
                &messages,
                pre_run_len,
                err,
            ));
        }
        Ok((messages, pre_run_len))
    }

    /// Race an in-flight provider future against cancellation and the
    /// turn deadline (AC-02, AC-07). When cancellation or the deadline
    /// wins, the provider future is dropped — which aborts the
    /// underlying HTTP call — and the turn's final result is returned
    /// as the `Err` payload. On `Ok`, the provider output is returned
    /// (the result is left untouched, to be consumed by the round).
    ///
    /// The parameters are the shared turn state threaded through the
    /// skeleton's stop paths; grouping them would hide the flow (AC-08).
    #[allow(clippy::too_many_arguments)]
    async fn race_provider_call<T, F>(
        &self,
        request: &AgentTurnRequest,
        result: &mut AgentTurnResult,
        context: &mut Option<&mut ContextManager>,
        messages: &[Message],
        pre_run_len: usize,
        turn_start: Instant,
        provider_fut: F,
    ) -> Result<T, AgentTurnResult>
    where
        F: std::future::Future<Output = Result<T, ProviderError>>,
    {
        tokio::pin!(provider_fut);
        let deadline = turn_deadline(request.turn_timeout(), turn_start);
        tokio::pin!(deadline);
        tokio::select! {
            _ = request.cancel_token().cancelled() => {
                Err(self.stop_cancelled(request, result, context, messages, pre_run_len))
            }
            _ = &mut deadline => {
                Err(self.stop_with_error(
                    request,
                    result,
                    context,
                    messages,
                    pre_run_len,
                    ProviderError::new(
                        "TIMEOUT",
                        format!(
                            "turn exceeded the {:?} deadline",
                            request.turn_timeout().unwrap_or_default()
                        ),
                    ),
                    AgentTurnStopReason::Timeout,
                ))
            }
            res = &mut provider_fut => match res {
                Ok(output) => Ok(output),
                Err(err) => Err(self.stop_with_provider_error(
                    request,
                    result,
                    context,
                    messages,
                    pre_run_len,
                    err,
                )),
            },
        }
    }

    /// Non-streaming provider round (AC-08): complete the request under
    /// the cancellation/deadline race and surface the assistant
    /// message. Usage reported this round is captured here — V1 keeps
    /// the latest reported usage rather than summing across rounds
    /// (providers are expected to report a running total themselves).
    ///
    /// The parameters are the shared turn state threaded through the
    /// skeleton's stop paths; grouping them would hide the flow (AC-08).
    #[allow(clippy::too_many_arguments)]
    async fn complete_round(
        &self,
        request: &AgentTurnRequest,
        result: &mut AgentTurnResult,
        context: &mut Option<&mut ContextManager>,
        messages: &[Message],
        pre_run_len: usize,
        turn_start: Instant,
        tool_defs: &[AgentToolDefinition],
    ) -> RoundOutcome {
        let provider_request = AgentRequest::new(request.model().clone(), messages.to_vec())
            .with_tools(tool_defs.to_vec());
        let response = match self
            .race_provider_call(
                request,
                result,
                context,
                messages,
                pre_run_len,
                turn_start,
                self.provider.complete(provider_request),
            )
            .await
        {
            Ok(response) => response,
            Err(ended) => return RoundOutcome::Ended(Box::new(ended)),
        };

        if let Some(usage) = response.usage().cloned() {
            // AR-29: accumulate across rounds (not overwrite).
            result.accumulate_usage(usage);
        }

        RoundOutcome::Assistant {
            assistant: response.message().clone(),
            truncated: false,
        }
    }

    /// Streaming provider round (AC-08): open the provider stream under
    /// the cancellation/deadline race, then consume it, forwarding
    /// `ContentDelta`/`ReasoningDelta` to the sink and collecting
    /// content + tool calls. Terminal failure semantics live here: a
    /// zero-event stream is an `EMPTY_STREAM` provider error (AC-06),
    /// an in-stream `Error` is a `STREAM` provider error (AC-05), and a
    /// `Done` carrying a truncation reason (`length`/`max_tokens`)
    /// marks the round truncated (AC-01).
    ///
    /// The parameters are the shared turn state threaded through the
    /// skeleton's stop paths; grouping them would hide the flow (AC-08).
    #[allow(clippy::too_many_arguments)]
    async fn streaming_round(
        &self,
        request: &AgentTurnRequest,
        result: &mut AgentTurnResult,
        context: &mut Option<&mut ContextManager>,
        messages: &[Message],
        pre_run_len: usize,
        turn_start: Instant,
        tool_defs: &[AgentToolDefinition],
    ) -> RoundOutcome {
        let provider_request = AgentRequest::new(request.model().clone(), messages.to_vec())
            .with_tools(tool_defs.to_vec());
        let mut stream = match self
            .race_provider_call(
                request,
                result,
                context,
                messages,
                pre_run_len,
                turn_start,
                self.provider.stream(provider_request),
            )
            .await
        {
            Ok(stream) => stream,
            Err(ended) => return RoundOutcome::Ended(Box::new(ended)),
        };

        // Consume the stream, collecting text and tool calls. Every
        // read races cancellation — when the token fires the stream is
        // dropped (closing the underlying HTTP connection) and the turn
        // stops (AC-07) — and the turn deadline.
        let mut content_text = String::new();
        let mut pending_tool_calls: Vec<ToolCall> = Vec::new();
        let mut saw_any_event = false;
        let mut stream_truncated = false;

        let deadline = turn_deadline(request.turn_timeout(), turn_start);
        tokio::pin!(deadline);
        loop {
            let event = tokio::select! {
                _ = request.cancel_token().cancelled() => {
                    drop(stream);
                    return RoundOutcome::Ended(Box::new(self.stop_cancelled(
                        request,
                        result,
                        context,
                        messages,
                        pre_run_len,
                    )));
                }
                _ = &mut deadline => {
                    drop(stream);
                    return RoundOutcome::Ended(Box::new(self.stop_with_error(
                        request,
                        result,
                        context,
                        messages,
                        pre_run_len,
                        ProviderError::new(
                            "TIMEOUT",
                            format!(
                                "turn exceeded the {:?} deadline",
                                request.turn_timeout().unwrap_or_default()
                            ),
                        ),
                        AgentTurnStopReason::Timeout,
                    )));
                }
                event = stream.next_event() => event,
            };
            let Some(event) = event else { break };
            saw_any_event = true;
            match event {
                AgentStreamEvent::ContentDelta(text) => {
                    content_text.push_str(&text);
                    self.sink.handle(&AgentEvent::ContentDelta {
                        session_id: request.session().id().clone(),
                        text: text.clone(),
                    });
                }
                // Reasoning is display-only: forward to the sink for
                // hosts (UI progress) but never into message history.
                AgentStreamEvent::ReasoningDelta(text) => {
                    self.sink.handle(&AgentEvent::ReasoningDelta {
                        session_id: request.session().id().clone(),
                        text: text.clone(),
                    });
                }
                AgentStreamEvent::ToolCall(tc) => {
                    pending_tool_calls.push(tc);
                }
                AgentStreamEvent::Usage(u) => {
                    // AR-29: accumulate across rounds (not overwrite).
                    result.accumulate_usage(u);
                }
                // Server-side compaction (Responses API, PV-01b
                // reserved channel): informational. The opaque
                // compacted content is never replayed into the
                // message history.
                AgentStreamEvent::Compacted { item_id } => {
                    tracing::info!(item_id, "provider compacted conversation server-side");
                }
                // A terminal transport failure: surface it as a
                // provider error instead of silently treating the
                // stream as finished (AC-05).
                AgentStreamEvent::Error(message) => {
                    return RoundOutcome::Ended(Box::new(self.stop_with_provider_error(
                        request,
                        result,
                        context,
                        messages,
                        pre_run_len,
                        ProviderError::new("STREAM", message),
                    )));
                }
                // A host-visible informational warning (e.g. EOF with
                // incomplete tool-call fragments, D-5). Unlike `Error`
                // this is NOT terminal: log and continue the turn.
                AgentStreamEvent::Warning(message) => {
                    tracing::warn!(message, "provider stream warning");
                }
                AgentStreamEvent::Done { stop_reason } => {
                    if let Some(reason) = stop_reason {
                        // "length"/"max_tokens" means the provider
                        // cut the response short: the final answer
                        // is incomplete (AC-01).
                        stream_truncated = is_truncation_reason(&reason);
                    }
                    break;
                }
                // ToolCallDelta is informational; we only act on ToolCall (complete).
                AgentStreamEvent::ToolCallDelta { .. } => {}
            }
        }

        // A stream that ended before yielding any event is a
        // transport-level failure, not a final response (AC-06).
        if !saw_any_event {
            return RoundOutcome::Ended(Box::new(self.stop_with_provider_error(
                request,
                result,
                context,
                messages,
                pre_run_len,
                ProviderError::new(
                    "EMPTY_STREAM",
                    "provider stream ended before yielding any event",
                ),
            )));
        }

        // Build the assistant message from accumulated content + tool calls.
        let has_tool_calls = !pending_tool_calls.is_empty();
        let assistant = if has_tool_calls {
            Message::assistant_with_tool_calls(&content_text, pending_tool_calls)
        } else {
            Message::assistant(&content_text)
        };

        RoundOutcome::Assistant {
            assistant,
            truncated: stream_truncated,
        }
    }

    /// Run a single turn using streaming. Like [`run_turn`](Self::run_turn)
    /// but calls `provider.stream()` instead of `provider.complete()`,
    /// forwarding `ContentDelta` events through the event sink as the
    /// model generates tokens.
    ///
    /// Tool calls are still executed sequentially after the stream
    /// completes (the model must finish generating all tool calls before
    /// the stream ends). The loop then re-enters the stream for the next
    /// round if tool calls were present.
    ///
    /// `context` behaves exactly as in [`run_turn`](Self::run_turn).
    pub async fn run_turn_streaming(
        &self,
        request: AgentTurnRequest,
        context: Option<&mut ContextManager>,
    ) -> AgentTurnResult {
        let turn_start = Instant::now();
        self.run_turn_inner(request, context, TurnRound::Streaming)
            .await
            .with_duration_since(turn_start)
    }

    /// Stop the turn with a provider-level failure: record the
    /// diagnostic, emit the `ProviderError` event, mark the turn
    /// stopped, and commit the messages seen so far. Shared by the
    /// modality gate and provider call failures in both turn paths.
    fn stop_with_provider_error(
        &self,
        request: &AgentTurnRequest,
        result: &mut AgentTurnResult,
        context: &mut Option<&mut ContextManager>,
        messages: &[Message],
        pre_run_len: usize,
        err: ProviderError,
    ) -> AgentTurnResult {
        self.stop_with_error(
            request,
            result,
            context,
            messages,
            pre_run_len,
            err,
            AgentTurnStopReason::ProviderError,
        )
    }

    /// Like [`Self::stop_with_provider_error`], but with an explicit
    /// stop reason (used for turn timeout, which is not a provider
    /// failure but needs the same diagnostic + event + history-commit
    /// treatment — AC-02).
    ///
    /// The parameters are the shared turn state threaded through the
    /// skeleton's stop paths; grouping them would hide the flow (AC-08).
    #[allow(clippy::too_many_arguments)]
    fn stop_with_error(
        &self,
        request: &AgentTurnRequest,
        result: &mut AgentTurnResult,
        context: &mut Option<&mut ContextManager>,
        messages: &[Message],
        pre_run_len: usize,
        err: ProviderError,
        reason: AgentTurnStopReason,
    ) -> AgentTurnResult {
        let provider_name = request.session().provider().clone();
        result.push_diagnostic(err.to_diagnostic(request.session().id(), None));
        self.sink.handle(&AgentEvent::ProviderError {
            session_id: request.session().id().clone(),
            provider: provider_name,
            code: err.code().to_string(),
            message: err.message().to_string(),
        });
        self.finish(
            request,
            result,
            context,
            messages,
            pre_run_len,
            None,
            reason,
            AgentTurnStatus::Stopped,
        )
    }

    /// Stop the turn on cancellation: mark it stopped with the
    /// `Cancelled` reason and commit the messages seen so far. Shared
    /// by the polling checkpoints and the in-flight abort paths.
    fn stop_cancelled(
        &self,
        request: &AgentTurnRequest,
        result: &mut AgentTurnResult,
        context: &mut Option<&mut ContextManager>,
        messages: &[Message],
        pre_run_len: usize,
    ) -> AgentTurnResult {
        self.finish(
            request,
            result,
            context,
            messages,
            pre_run_len,
            None,
            AgentTurnStopReason::Cancelled,
            AgentTurnStatus::Stopped,
        )
    }

    /// Stop the turn when the max-tool-step guard trips (used by both
    /// the pre-round and the mid-round check).
    fn stop_max_steps(
        &self,
        request: &AgentTurnRequest,
        result: &mut AgentTurnResult,
        context: &mut Option<&mut ContextManager>,
        messages: &[Message],
        pre_run_len: usize,
    ) -> AgentTurnResult {
        self.finish(
            request,
            result,
            context,
            messages,
            pre_run_len,
            None,
            AgentTurnStopReason::MaxToolStepsExceeded,
            AgentTurnStatus::Stopped,
        )
    }

    /// Finish the turn: stamp the result with the stop reason, status,
    /// and transcript (plus the final assistant message when the turn
    /// ended on a model response), commit the turn slice to the active
    /// history store, and hand back the final result (AC-08). Every
    /// stop path funnels through here so result stamping and history
    /// commits cannot drift.
    ///
    /// The parameters are the shared turn state threaded through the
    /// skeleton's stop paths; grouping them would hide the flow (AC-08).
    #[allow(clippy::too_many_arguments)]
    fn finish(
        &self,
        request: &AgentTurnRequest,
        result: &mut AgentTurnResult,
        context: &mut Option<&mut ContextManager>,
        messages: &[Message],
        pre_run_len: usize,
        final_response: Option<Message>,
        stop_reason: AgentTurnStopReason,
        status: AgentTurnStatus,
    ) -> AgentTurnResult {
        let mut result = take_result(result)
            .with_stop_reason(stop_reason)
            .with_status(status)
            .with_messages(messages.to_vec());
        if let Some(response) = final_response {
            result = result.with_final_response(response);
        }
        commit_turn_history(context, request.session(), messages, pre_run_len);
        result
    }

    /// Execute the round's tool calls sequentially through the
    /// registry, appending observations and emitting `ToolInvoked`
    /// events (AC-08). Shared by both turn paths. Returns `Some` with
    /// the final result when a stop condition (max-tool-step guard or
    /// cancellation) tripped mid-round; `None` when all calls executed
    /// and the loop should continue.
    ///
    /// The parameters are the shared turn state threaded through the
    /// skeleton's stop paths; grouping them would hide the flow (AC-08).
    #[allow(clippy::too_many_arguments)]
    async fn execute_tool_calls(
        &self,
        request: &AgentTurnRequest,
        result: &mut AgentTurnResult,
        context: &mut Option<&mut ContextManager>,
        messages: &mut Vec<Message>,
        pre_run_len: usize,
        registry: &Arc<AgentToolRegistry>,
        tool_calls: &[ToolCall],
        max_tool_steps: usize,
        tool_steps_taken: &mut usize,
    ) -> Option<AgentTurnResult> {
        // AR-25: Read tools in a contiguous run execute concurrently;
        // Editor/External tools execute serially. Observations and
        // tool_result messages are appended in provider order so the
        // model sees the full batch deterministically.
        let mut i = 0usize;
        while i < tool_calls.len() {
            // The guard fires *before* each tool slot so the limit is
            // exact, not just per-round.
            if *tool_steps_taken >= max_tool_steps {
                return Some(self.stop_max_steps(request, result, context, messages, pre_run_len));
            }
            // Check cancellation before each execution step.
            if request.cancel_token().is_cancelled() {
                return Some(self.stop_cancelled(request, result, context, messages, pre_run_len));
            }
            let tool_call = &tool_calls[i];
            let tool_name = ToolName::new(tool_call.name());
            let risk = registry
                .spec(&tool_name)
                .map(|s| s.risk())
                .unwrap_or(ToolRiskLevel::External);
            if risk == ToolRiskLevel::Read {
                // Gather the contiguous run of Read calls, bounded by the
                // remaining tool-step budget.
                let start = i;
                let mut end = i;
                while end < tool_calls.len() {
                    if *tool_steps_taken + (end - start) >= max_tool_steps {
                        break;
                    }
                    let n = ToolName::new(tool_calls[end].name());
                    let r = registry
                        .spec(&n)
                        .map(|s| s.risk())
                        .unwrap_or(ToolRiskLevel::External);
                    if r != ToolRiskLevel::Read {
                        break;
                    }
                    end += 1;
                }
                let run = &tool_calls[start..end];
                // Emit ToolInvoked for every call in the run first, then
                // execute concurrently (observations stay in provider order).
                for tc in run {
                    self.sink.handle(&AgentEvent::ToolInvoked {
                        session_id: request.session().id().clone(),
                        tool: ToolName::new(tc.name()),
                        id: Some(tc.id().clone()),
                    });
                }
                let futures = run.iter().map(|tc| {
                    let ctx = ToolContext::new(
                        request.session().workspace_scope().clone(),
                        request.session().id().clone(),
                        request.session().mode(),
                    )
                    .with_permissions(request.session().permissions().clone())
                    .with_project_id_opt(request.session().project_id().cloned());
                    let name = ToolName::new(tc.name());
                    let tc_owned = tc.clone();
                    let cancel = request.cancel_token().clone();
                    let timeout = request.turn_timeout();
                    async move {
                        self.execute_tool(registry, &name, &tc_owned, &ctx, &cancel, timeout)
                            .await
                    }
                });
                let results = futures_util::future::join_all(futures).await;
                *tool_steps_taken += run.len();
                for (tc, tcr) in run.iter().zip(results) {
                    let observation_content = tool_observation_text(&tcr);
                    result.push_tool_call(tcr);
                    messages.push(Message::tool_result(tc.id().clone(), observation_content));
                }
                i = end;
            } else {
                // Editor/External: serial, one at a time.
                *tool_steps_taken += 1;
                self.sink.handle(&AgentEvent::ToolInvoked {
                    session_id: request.session().id().clone(),
                    tool: tool_name.clone(),
                    id: Some(tool_call.id().clone()),
                });
                let ctx = ToolContext::new(
                    request.session().workspace_scope().clone(),
                    request.session().id().clone(),
                    request.session().mode(),
                )
                .with_permissions(request.session().permissions().clone())
                .with_project_id_opt(request.session().project_id().cloned());
                let tool_result = self
                    .execute_tool(
                        registry,
                        &tool_name,
                        tool_call,
                        &ctx,
                        request.cancel_token(),
                        request.turn_timeout(),
                    )
                    .await;
                let observation_content = tool_observation_text(&tool_result);
                result.push_tool_call(tool_result);
                messages.push(Message::tool_result(
                    tool_call.id().clone(),
                    observation_content,
                ));
                i += 1;
            }
        }
        None
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
        cancel_token: &tokio_util::sync::CancellationToken,
        turn_timeout: Option<Duration>,
    ) -> ToolCallResult {
        let session_id = tool_context.agent_session_id().clone();
        let tool_call_id = tool_call.id().clone();
        let tool_name_owned = tool_name.clone();

        let invocation = registry.invoke(tool_name, tool_context, tool_call.arguments().clone());
        let deadline = tokio::time::sleep(turn_timeout.unwrap_or(Duration::from_secs(300)));
        tokio::pin!(deadline);
        let invocation_result = tokio::select! {
            _ = cancel_token.cancelled() => Err(ToolRegistryError::ToolReturned(ToolError::new(ToolErrorCode::ExecutionFailed, "tool execution cancelled"))),
            _ = &mut deadline => Err(ToolRegistryError::ToolReturned(ToolError::new(ToolErrorCode::ExecutionFailed, "tool execution timed out"))),
            result = invocation => result,
        };
        match invocation_result {
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

/// Build the turn result skeleton stamped at turn start: turn/session
/// ids, mode, provider, model, and `Running` status (AC-08).
fn base_result(request: &AgentTurnRequest) -> AgentTurnResult {
    AgentTurnResult::new()
        .with_turn_id(request.turn_id().clone())
        .with_session_id(request.session().id().clone())
        .with_mode(request.session().mode())
        .with_provider(request.session().provider().clone())
        .with_model(request.model().clone())
        .with_status(AgentTurnStatus::Running)
}

/// Consume-and-replace: hand out the current result value (the
/// stamp-with-* builders take `self` by value) and leave a fresh empty
/// result in `slot`, which callers immediately overwrite with the
/// builder-chain output (AC-08).
fn take_result(slot: &mut AgentTurnResult) -> AgentTurnResult {
    std::mem::replace(slot, AgentTurnResult::new())
}

/// Media types of the `image/*` file blocks present in `messages`,
/// deduplicated and sorted. Empty when the turn carries no image
/// blocks.
fn image_media_types(messages: &[Message]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for m in messages {
        for block in m.blocks() {
            if let ContentBlock::File(file) = block
                && file.media_type().starts_with("image/")
            {
                out.push(file.media_type().to_string());
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Modality gate (PV-03b): when a turn's messages carry `image/*` file
/// blocks, the selected model must advertise image input support.
///
/// The provider's model list is the capability source. Best-effort by
/// design: when the model cannot be found in the listing (custom model
/// names, listing failures), the request is allowed through so custom /
/// unlisted models never get blocked.
///
/// An empty `input_modalities` list also counts as "unknown" and is
/// allowed through: the listing translation layer only fills modality
/// data when the upstream API exposes it (Anthropic `/v1/models` does
/// not), so an absent list must never be read as "known unsupported".
async fn enforce_image_modality_gate(
    provider: &dyn AgentProvider,
    model: &ModelName,
    media_types: &[String],
) -> Result<(), ProviderError> {
    if media_types.is_empty() {
        return Ok(());
    }
    let models = match provider.list_models().await {
        Ok(models) => models,
        // Best-effort: a listing failure must never block an image
        // request from reaching a provider that may support it.
        Err(_) => return Ok(()),
    };
    let Some(info) = models.iter().find(|m| m.name() == model) else {
        // Best-effort: custom / unlisted model names are allowed
        // through when the capability cannot be verified.
        return Ok(());
    };
    let supports_image = info
        .input_modalities()
        .iter()
        .any(|m| m == "image" || m.starts_with("image/"));
    if supports_image {
        return Ok(());
    }
    if info.input_modalities().is_empty() {
        // Unknown capability: the listing did not advertise modalities
        // (or the adapter fills only capabilities). Allowing through is
        // the best-effort default; a missing list is not evidence of
        // non-support.
        return Ok(());
    }
    Err(ProviderError::new(
        "MODALITY_UNSUPPORTED",
        format!(
            "model `{model}` does not support image input (modalities: {:?}); \
             image file blocks with media types {media_types:?} cannot be sent",
            info.input_modalities()
        ),
    ))
}

/// Upper bound for the summarization reply (CM-V2c decision: 3k,
/// clamp 2k-4k). The summary is produced by the same provider as the
/// turn, in a dedicated request that never enters the session history.
const SUMMARY_MAX_TOKENS: usize = 3_000;

/// Compaction driver (CM-V2e, decision table M3 loop-driven seam):
/// when the stored history crossed the soft budget line and the
/// thrash guard allows, build the summarizer request from the
/// eviction range, run it through the same provider/model as the
/// turn, and apply the reply as a sticky summary.
///
/// Failure handling:
/// - summarizer request fails -> classified by
///   [`ProviderError::is_transient`] (AC-24): a transient failure
///   (transport/timing/retryable HTTP status) is recorded via
///   `record_transient_compaction_failure` and compaction retries
///   next turn; a permanent failure advances the consecutive-failure
///   streak via `record_compaction_failure`. Either way the caller's
///   `prepare_messages` falls back to plain eviction (drop-oldest);
///   the turn still completes.
/// - the window changed between request and apply
///   (`CompactionStale`) -> the summary is dropped and plain
///   eviction applies.
/// - an empty summary reply (model refusal) counts as a PERMANENT
///   failure (`record_compaction_failure`).
/// - consecutive failures reach [`MAX_CONSECUTIVE_COMPACTION_FAILURES`]
///   -> [`ContextManager::should_attempt_compaction`] stops attempts
///   for the rest of the session (防 thrash).
///
/// Returns the `ContextCompacted` event to emit, if any. The
/// summarizer call is not part of the session history.
async fn maybe_compact(
    manager: &mut ContextManager,
    provider: &dyn AgentProvider,
    model: &ModelName,
    session_id: &AgentSessionId,
    cancel_token: &tokio_util::sync::CancellationToken,
) -> Option<AgentEvent> {
    if !manager.needs_compaction() || !manager.should_attempt_compaction() {
        return None;
    }
    if !manager.has_eviction_pending() {
        // Nothing would be summarized (e.g. a single-turn history the
        // window keeps whole); skip the summarizer call.
        return None;
    }
    let summarize_messages = manager.summarize_request();
    let provider_request = AgentRequest::new(model.clone(), summarize_messages)
        .with_options(serde_json::json!({ "max_tokens": SUMMARY_MAX_TOKENS }));
    // The summarizer call races two stops (AR-19): user cancellation
    // aborts the summary immediately, and a 60s deadline guarantees a
    // stuck summarizer cannot wedge the turn. A cancelled/timed-out
    // summary is recorded as a transient failure so compaction retries
    // next turn instead of advancing the permanent-failure streak.
    let compaction = provider.complete(provider_request);
    let deadline = tokio::time::sleep(COMPACTION_DEADLINE);
    tokio::pin!(deadline);
    let summary_text = match tokio::select! {
        _ = cancel_token.cancelled() => Err(ProviderError::new("CANCELLED", "compaction cancelled")),
        _ = &mut deadline => Err(ProviderError::new("TIMEOUT", "compaction deadline exceeded")),
        result = compaction => result,
    } {
        Ok(response) => response.message().content().to_owned(),
        Err(err) => {
            // AC-24: transient provider hiccups (transport, timeout,
            // retryable HTTP status) must not advance the permanent
            // streak — compaction retries next turn. Permanent
            // failures do (3 disables the attempt for the session).
            if err.is_transient() {
                manager.record_transient_compaction_failure();
            } else {
                manager.record_compaction_failure();
            }
            return None;
        }
    };
    if summary_text.trim().is_empty() {
        // An empty reply (refusal / truncation) must not become a
        // sticky summary: count it as a failure and fall back to
        // plain eviction.
        manager.record_compaction_failure();
        return None;
    }
    match manager.apply_summary(summary_text) {
        Ok(record) => Some(AgentEvent::ContextCompacted {
            session_id: session_id.clone(),
            summary: record.text,
            tokens_before: record.tokens_before,
            tokens_after: record.tokens_after,
        }),
        // CompactionStale: the window moved while summarizing — drop
        // the summary and let plain eviction handle the window.
        Err(_) => None,
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

/// Commit the messages produced during this turn to the active
/// history store.
///
/// When a `ContextManager` is present it replaces the session history
/// as the source of truth: the turn slice is appended through
/// `commit_turn` and the session history is intentionally left
/// untouched, so the two stores never diverge. When it is absent, the
/// V1 path appends the turn slice to the session history via
/// [`commit_session_history`].
fn commit_turn_history(
    context: &mut Option<&mut ContextManager>,
    session: &crate::session::AgentSession,
    messages: &[Message],
    pre_run_len: usize,
) {
    match context {
        Some(manager) => manager.commit_turn(&messages[pre_run_len..]),
        None => commit_session_history(session, messages, pre_run_len),
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

/// Provider stop reasons that mean the response was cut short before a
/// natural end: `max_tokens`/`length` (token budgets), `content_filter`
/// (policy gate), and `refusal` (model declined to answer). Streams
/// ending with these are reported as
/// [`AgentTurnStopReason::Truncated`] (AC-01, R2-05).
fn is_truncation_reason(reason: &str) -> bool {
    matches!(
        reason,
        "length" | "max_tokens" | "content_filter" | "refusal"
    )
}

/// Future that completes when the turn deadline passes; pending forever
/// when no deadline is configured. Races the in-flight provider call so
/// a hung provider cannot outlive the deadline (AC-02).
async fn turn_deadline(timeout: Option<Duration>, turn_start: Instant) {
    if let Some(dur) = timeout {
        tokio::time::sleep(dur.saturating_sub(turn_start.elapsed())).await;
    } else {
        std::future::pending::<()>().await;
    }
}

/// Validate a final assistant response against a structured-output
/// JSON Schema (AR-30). The response must parse as JSON and satisfy
/// the schema; the error string is designed to be fed back to the model
/// for one corrective retry.
fn validate_structured_output(content: &str, schema: &Value) -> Result<(), String> {
    let value: Value = serde_json::from_str(content)
        .map_err(|error| format!("final response is not valid JSON: {error}"))?;
    validate_json_value(schema, &value)
        .map_err(|error| format!("final response does not satisfy the schema: {error}"))
}

/// Maximum serialized bytes of a successful tool observation sent back
/// to the model in a single tool-result message (RF-B4 / AR-20).
///
/// The cap applies when the observation text is built — *before* the
/// next provider request in the same turn — so an oversized
/// `workflow.get` / `model.list` output can never blow up the prompt.
pub const MAX_TOOL_OBSERVATION_BYTES: usize = 32 * 1024;

/// Longest raw-output prefix embedded in a truncated observation
/// envelope. The prefix is stored as an escaped JSON string, so the
/// worst-case envelope stays on the same order as
/// [`MAX_TOOL_OBSERVATION_BYTES`].
const TRUNCATED_OUTPUT_PREFIX_BYTES: usize = MAX_TOOL_OBSERVATION_BYTES / 2;

/// Serialize a `ToolCallResult` into the JSON text the model receives
/// as a tool observation. Successful calls carry their raw output
/// JSON; rejected/failed calls carry a structured diagnostic envelope.
///
/// Successful outputs larger than [`MAX_TOOL_OBSERVATION_BYTES`] are
/// replaced by a truncation envelope that keeps a bounded prefix, the
/// original byte size, and an explicit `output_truncated` marker. The
/// `tool_call_id` pairing lives on the surrounding `Message`, which is
/// unchanged.
fn tool_observation_text(result: &ToolCallResult) -> String {
    match result.status() {
        ToolCallStatus::Succeeded => {
            let serialized =
                serde_json::to_string(&result.output().cloned().unwrap_or(Value::Null))
                    .unwrap_or_else(|_| "null".to_string());
            cap_succeeded_tool_observation(&serialized, result.tool_name().as_str())
        }
        ToolCallStatus::Running => {
            // Defensive arm (AC-20): `execute_tool` always stamps an
            // explicit terminal status before a result is observed, so
            // a `Running` result never reaches the model here. Emit a
            // neutral envelope rather than panicking on a missing
            // diagnostic.
            serde_json::to_string(&json!({
                "status": "running",
                "tool": result.tool_name().as_str(),
            }))
            .unwrap_or_else(|_| "{\"status\":\"running\"}".to_string())
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

/// Cap a serialized successful tool output before it reaches the model.
///
/// Outputs within [`MAX_TOOL_OBSERVATION_BYTES`] pass through
/// unchanged, preserving the existing wire behavior for normal tools.
/// Oversized outputs are replaced by a valid JSON envelope with a
/// bounded `output_prefix`, the original byte count, and an explicit
/// `output_truncated` marker so the model can see that information is
/// missing and ask for a narrower read.
fn cap_succeeded_tool_observation(serialized_output: &str, tool_name: &str) -> String {
    if serialized_output.len() <= MAX_TOOL_OBSERVATION_BYTES {
        return serialized_output.to_owned();
    }

    let mut prefix_end = TRUNCATED_OUTPUT_PREFIX_BYTES.min(serialized_output.len());
    while !serialized_output.is_char_boundary(prefix_end) {
        prefix_end -= 1;
    }
    let output_prefix = &serialized_output[..prefix_end];

    serde_json::to_string(&json!({
        "status": "succeeded",
        "tool": tool_name,
        "output_truncated": true,
        "original_output_bytes": serialized_output.len(),
        "output_prefix": output_prefix,
    }))
    .unwrap_or_else(|_| {
        format!(
            "{{\"status\":\"succeeded\",\"tool\":{tool_name:?},\"output_truncated\":true,\
             \"original_output_bytes\":{}}}",
            serialized_output.len()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context_manager::{ContextConfig, ContextManager};
    use crate::error::ProviderError;
    use crate::ids::{AgentSessionId, ModelName, ProviderName, WorkspaceScope};
    use crate::mode::AgentMode;
    use crate::permissions::{PermissionSet, ToolPermission, ToolRiskLevel};
    use crate::provider::{
        AgentResponse, AgentStream, AgentStreamEvent, FileContentBlock, ModelInfo, ToolCall,
        ToolCallId, Usage,
    };
    use crate::session::AgentSession;
    use crate::tool::{AgentTool, ToolResult, ToolSpec};
    use crate::turn::{AgentTurnId, AgentTurnRequest};
    use async_trait::async_trait;
    use serde_json::{Value, json};
    use std::collections::VecDeque;
    use std::path::PathBuf;
    use std::sync::Mutex;
    use tokio_util::sync::CancellationToken;

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

    /// A tool whose `invoke` cancels the turn's token and returns
    /// normally. Cancellation is only observed at the loop's
    /// checkpoints (the top of the round loop and between tool calls) —
    /// tool execution itself is atomic, so a mid-invoke cancel lands on
    /// the *next* checkpoint, never inside the tool (AC-15).
    #[derive(Clone)]
    struct CancellingTool {
        spec: ToolSpec,
        token: CancellationToken,
        recorded_inputs: Arc<Mutex<Vec<Value>>>,
    }

    impl CancellingTool {
        fn new(token: CancellationToken) -> Self {
            Self {
                spec: ToolSpec::new(
                    ToolName::new("cancel"),
                    "cancels the turn token during invoke",
                    [AgentMode::Agent, AgentMode::Build],
                    ToolPermission::new("workflow.read"),
                    ToolRiskLevel::Read,
                ),
                token,
                recorded_inputs: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn recorded_inputs(&self) -> Vec<Value> {
            self.recorded_inputs.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl AgentTool for CancellingTool {
        fn spec(&self) -> ToolSpec {
            self.spec.clone()
        }

        async fn invoke(&self, _ctx: &ToolContext, input: Value) -> ToolResult {
            self.recorded_inputs.lock().unwrap().push(input.clone());
            self.token.cancel();
            Ok(json!({"cancelled": true}))
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

        let result = loop_harness.run_turn(req, None).await;
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

        let result = loop_harness.run_turn(req, None).await;
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

        let result = loop_harness.run_turn(req, None).await;
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

        let result = loop_harness.run_turn(req, None).await;
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

        let result = loop_harness.run_turn(req, None).await;
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

        let result = loop_harness.run_turn(req, None).await;
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

        let result = loop_harness.run_turn(req, None).await;
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

        let result = loop_harness.run_turn(req, None).await;
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

        let result = loop_harness.run_turn(req, None).await;
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

        let result = loop_harness.run_turn(req, None).await;
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

        let result = loop_harness.run_turn(req, None).await;
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

        let _ = loop_harness.run_turn(req, None).await;
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

        let _ = loop_harness.run_turn(req, None).await;

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
    fn tool_observation_text_for_oversized_success_is_truncated_with_marker() {
        let payload = "x".repeat(MAX_TOOL_OBSERVATION_BYTES + 1_024);
        let result = ToolCallResult::new(ToolCallId::new("c1"), ToolName::new("workflow.get"))
            .succeeded(json!({ "data": payload }));
        let text = tool_observation_text(&result);

        let parsed: Value = serde_json::from_str(&text).expect("truncation envelope is valid JSON");
        assert_eq!(parsed["status"], "succeeded");
        assert_eq!(parsed["tool"], "workflow.get");
        assert_eq!(parsed["output_truncated"], true);
        assert_eq!(
            parsed["original_output_bytes"]
                .as_u64()
                .expect("original byte count"),
            (MAX_TOOL_OBSERVATION_BYTES + 1_024 + 11) as u64 // json!({"data":"..."}) overhead
        );
        let prefix = parsed["output_prefix"]
            .as_str()
            .expect("output prefix is a string");
        assert!(!prefix.is_empty());
        assert!(prefix.len() <= TRUNCATED_OUTPUT_PREFIX_BYTES);
        assert!(prefix.contains("xxx"));
    }

    #[tokio::test]
    async fn run_turn_feeds_truncated_observation_with_matching_tool_call_id() {
        // A turn that executes a large-output tool must feed the next
        // provider round a truncated observation, while the surrounding
        // tool-result message keeps the original tool_call_id pairing.
        struct CapturingProvider {
            name: ProviderName,
            first_call: Mutex<bool>,
            captured: Mutex<Vec<Message>>,
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
                let mut first = self.first_call.lock().unwrap();
                if *first {
                    *first = false;
                    Ok(response_with_tool_calls(vec![ToolCall::new(
                        ToolCallId::new("big-1"),
                        "big",
                        json!({}),
                    )]))
                } else {
                    *self.captured.lock().unwrap() = request.messages().to_vec();
                    Ok(response_with_text("done"))
                }
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

        let provider = Arc::new(CapturingProvider {
            name: ProviderName::new("capture"),
            first_call: Mutex::new(true),
            captured: Mutex::new(Vec::new()),
        });
        let loop_harness = AgentLoop::new(provider.clone(), Arc::new(VecAgentEventSink::new()));

        let req = AgentTurnRequest::new(
            session_with(|reg| {
                reg.register(ScriptedTool::success(
                    "big",
                    vec![json!({ "data": "y".repeat(MAX_TOOL_OBSERVATION_BYTES + 512) })],
                ))
                .unwrap();
            }),
            AgentTurnId::new("truncation-turn"),
            ModelName::new("test-model"),
            vec![Message::user("read the big tool")],
        );

        let result = loop_harness.run_turn(req, None).await;
        assert_eq!(result.status(), AgentTurnStatus::Completed);

        let captured = provider.captured.lock().unwrap().clone();
        let tool_message = captured
            .iter()
            .find(|m| m.tool_call_id() == Some(&ToolCallId::new("big-1")))
            .expect("second provider round carries the tool result");

        // Pairing is preserved at the message level.
        assert_eq!(tool_message.tool_call_id(), Some(&ToolCallId::new("big-1")));

        let parsed: Value =
            serde_json::from_str(tool_message.content()).expect("truncation envelope is JSON");
        assert_eq!(parsed["output_truncated"], true);
        assert_eq!(parsed["tool"], "big");
        assert!(
            parsed["original_output_bytes"]
                .as_u64()
                .is_some_and(|n| n > 32_768)
        );
        assert!(tool_message.content().len() <= 2 * MAX_TOOL_OBSERVATION_BYTES);
    }

    #[tokio::test]
    async fn run_turn_structured_output_accepts_valid_json() {
        let provider = Arc::new(ScriptedProvider::new(
            "mock",
            vec![ScriptedStep::Respond(AgentResponse::new(
                Message::assistant(r#"{"title":"reimagine"}"#),
            ))],
        ));
        let loop_harness = AgentLoop::new(provider.clone(), Arc::new(VecAgentEventSink::new()));

        let schema = json!({
            "type": "object",
            "properties": { "title": { "type": "string" } },
            "required": ["title"]
        });
        let req = AgentTurnRequest::new(
            make_session(AgentToolRegistry::new()),
            AgentTurnId::new("structured-valid"),
            ModelName::new("test-model"),
            vec![Message::user("produce the json")],
        )
        .with_output_schema(schema);

        let result = loop_harness.run_turn(req, None).await;
        assert_eq!(result.status(), AgentTurnStatus::Completed);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::FinalResponse);
        assert_eq!(
            result.final_response().expect("final response").content(),
            r#"{"title":"reimagine"}"#
        );
        assert_eq!(provider.call_count(), 1, "valid response needs no retry");
    }

    #[tokio::test]
    async fn run_turn_structured_output_retries_once_with_corrective_message() {
        struct CapturingProvider {
            name: ProviderName,
            responses: Mutex<VecDeque<AgentResponse>>,
            captured: Mutex<Vec<Vec<Message>>>,
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
                self.captured
                    .lock()
                    .unwrap()
                    .push(request.messages().to_vec());
                self.responses
                    .lock()
                    .unwrap()
                    .pop_front()
                    .ok_or_else(|| ProviderError::new("EXHAUSTED", "no response"))
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

        let provider = Arc::new(CapturingProvider {
            name: ProviderName::new("capture"),
            responses: Mutex::new(VecDeque::from([
                AgentResponse::new(Message::assistant("definitely not json")),
                AgentResponse::new(Message::assistant(r#"{"title":"fixed"}"#)),
            ])),
            captured: Mutex::new(Vec::new()),
        });
        let loop_harness = AgentLoop::new(provider.clone(), Arc::new(VecAgentEventSink::new()));

        let schema = json!({
            "type": "object",
            "properties": { "title": { "type": "string" } },
            "required": ["title"]
        });
        let req = AgentTurnRequest::new(
            make_session(AgentToolRegistry::new()),
            AgentTurnId::new("structured-retry"),
            ModelName::new("test-model"),
            vec![Message::user("produce the json")],
        )
        .with_output_schema(schema);

        let result = loop_harness.run_turn(req, None).await;
        assert_eq!(result.status(), AgentTurnStatus::Completed);
        assert_eq!(
            result.final_response().expect("final response").content(),
            r#"{"title":"fixed"}"#
        );

        let captured = provider.captured.lock().unwrap().clone();
        assert_eq!(captured.len(), 2, "one corrective retry round");
        let corrective = captured[1]
            .iter()
            .find(|m| m.role() == "user" && m.content().contains("JSON Schema validation"))
            .expect("retry round carries the validation error back to the model");
        assert!(corrective.content().contains("not valid JSON"));
    }

    #[tokio::test]
    async fn run_turn_structured_output_fails_with_structured_error_after_retry_budget() {
        let provider = Arc::new(ScriptedProvider::new(
            "mock",
            vec![
                ScriptedStep::Respond(AgentResponse::new(Message::assistant(
                    r#"{"wrong":"shape"}"#,
                ))),
                ScriptedStep::Respond(AgentResponse::new(Message::assistant("still not json"))),
            ],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider.clone(), sink.clone());

        let schema = json!({
            "type": "object",
            "properties": { "title": { "type": "string" } },
            "required": ["title"]
        });
        let req = AgentTurnRequest::new(
            make_session(AgentToolRegistry::new()),
            AgentTurnId::new("structured-fail"),
            ModelName::new("test-model"),
            vec![Message::user("produce the json")],
        )
        .with_output_schema(schema);

        let result = loop_harness.run_turn(req, None).await;
        assert_eq!(result.status(), AgentTurnStatus::Stopped);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::ProviderError);
        assert!(
            result
                .diagnostics()
                .iter()
                .any(|d| d.code().as_str().contains("STRUCTURED_OUTPUT_INVALID")),
            "structured error diagnostic expected, got {:?}",
            result.diagnostics()
        );
        assert_eq!(provider.call_count(), 2, "retry budget is one round");

        let events = sink.events();
        let error_events: Vec<&AgentEvent> = events
            .iter()
            .filter(|event| {
                matches!(event, AgentEvent::ProviderError { code, .. } if code == "STRUCTURED_OUTPUT_INVALID")
            })
            .collect();
        assert_eq!(error_events.len(), 1, "one structured error event");
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
        let _ = loop_harness.run_turn(req, None).await;

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
        let _ = loop_harness.run_turn(req, None).await;

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
        let _ = loop_harness.run_turn(req, None).await;

        // History includes the input even though the provider never
        // produced an assistant message.
        let history = session.history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role(), "user");
        assert_eq!(history[0].content(), "hi");
    }

    // ------------------------------------------------------------------
    // Streaming tests
    // ------------------------------------------------------------------

    /// An in-memory stream that yields pre-scripted events.
    struct MockStream {
        events: Vec<crate::provider::AgentStreamEvent>,
    }

    #[async_trait]
    impl crate::provider::AgentStream for MockStream {
        async fn next_event(&mut self) -> Option<crate::provider::AgentStreamEvent> {
            if self.events.is_empty() {
                None
            } else {
                Some(self.events.remove(0))
            }
        }
    }

    /// A provider that returns scripted streams from `stream()`.
    struct StreamingProvider {
        name: ProviderName,
        streams: Mutex<VecDeque<Vec<crate::provider::AgentStreamEvent>>>,
    }

    impl StreamingProvider {
        fn new(name: &str, streams: Vec<Vec<crate::provider::AgentStreamEvent>>) -> Self {
            Self {
                name: ProviderName::new(name),
                streams: Mutex::new(streams.into()),
            }
        }
    }

    #[async_trait]
    impl AgentProvider for StreamingProvider {
        fn name(&self) -> ProviderName {
            self.name.clone()
        }

        async fn complete(&self, _request: AgentRequest) -> Result<AgentResponse, ProviderError> {
            Err(
                ProviderError::new("streaming_only", "this provider only supports streaming")
                    .with_provider(self.name.clone()),
            )
        }

        async fn stream(
            &self,
            _request: AgentRequest,
        ) -> Result<Box<dyn crate::provider::AgentStream>, ProviderError> {
            let events = self.streams.lock().unwrap().pop_front().unwrap_or_default();
            Ok(Box::new(MockStream { events }))
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
            Ok(Vec::new())
        }
    }

    #[tokio::test]
    async fn run_turn_streaming_final_response() {
        use crate::provider::AgentStreamEvent;

        let provider = Arc::new(StreamingProvider::new(
            "mock",
            vec![vec![
                AgentStreamEvent::ContentDelta("hello".into()),
                AgentStreamEvent::ContentDelta(" world".into()),
                AgentStreamEvent::Done {
                    stop_reason: Some("stop".into()),
                },
            ]],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink.clone());

        let req = AgentTurnRequest::new(
            make_session(AgentToolRegistry::new()),
            AgentTurnId::new("stream-1"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        );

        let result = loop_harness.run_turn_streaming(req, None).await;
        assert_eq!(result.status(), AgentTurnStatus::Completed);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::FinalResponse);
        let final_text = result.final_response().unwrap().content();
        assert_eq!(final_text, "hello world");

        // Verify ContentDelta events were emitted.
        let events = sink.events();
        let deltas: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ContentDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(deltas, vec!["hello", " world"]);
    }

    #[tokio::test]
    async fn run_turn_streaming_with_tool_call() {
        use crate::provider::AgentStreamEvent;

        let tool_call = ToolCall::new(ToolCallId::new("c1"), "echo", json!({"x": 1}));

        let provider = Arc::new(StreamingProvider::new(
            "mock",
            vec![
                // First stream: text + tool call
                vec![
                    AgentStreamEvent::ContentDelta("thinking...".into()),
                    AgentStreamEvent::ToolCall(tool_call),
                    AgentStreamEvent::Done {
                        stop_reason: Some("tool_calls".into()),
                    },
                ],
                // Second stream: final response after tool execution
                vec![
                    AgentStreamEvent::ContentDelta("done".into()),
                    AgentStreamEvent::Done {
                        stop_reason: Some("stop".into()),
                    },
                ],
            ],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink.clone());

        let req = AgentTurnRequest::new(
            session_with(|reg| {
                reg.register(ScriptedTool::success("echo", vec![json!({"ok": true})]))
                    .unwrap();
            }),
            AgentTurnId::new("stream-2"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        );

        let result = loop_harness.run_turn_streaming(req, None).await;
        assert_eq!(result.status(), AgentTurnStatus::Completed);
        assert_eq!(result.tool_calls().len(), 1);
        assert_eq!(result.tool_calls()[0].tool_name().as_str(), "echo");
        assert_eq!(result.tool_calls()[0].status(), ToolCallStatus::Succeeded);

        // Final response should be "done".
        let final_text = result.final_response().unwrap().content();
        assert_eq!(final_text, "done");

        // Should have ContentDelta events for both rounds.
        let events = sink.events();
        let delta_count = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ContentDelta { .. }))
            .count();
        assert_eq!(delta_count, 2); // "thinking..." + "done"
    }

    #[tokio::test]
    async fn run_turn_streaming_empty_stream() {
        let provider = Arc::new(StreamingProvider::new("mock", vec![]));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink.clone());

        let req = AgentTurnRequest::new(
            make_session(AgentToolRegistry::new()),
            AgentTurnId::new("stream-empty"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        );

        let result = loop_harness.run_turn_streaming(req, None).await;
        // A zero-event stream is a transport failure, not a completed
        // empty answer (AC-06).
        assert_eq!(result.status(), AgentTurnStatus::Stopped);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::ProviderError);
        assert!(
            result
                .diagnostics()
                .iter()
                .any(|d| d.code().as_str().ends_with("EMPTY_STREAM")),
            "expected an EMPTY_STREAM diagnostic, got {:?}",
            result.diagnostics()
        );
    }

    #[tokio::test]
    async fn run_turn_streaming_truncated_stream_reports_truncated() {
        // The provider stops the stream with a truncation reason: the
        // partial answer must not be reported as a clean final response
        // (AC-01).
        let provider = Arc::new(StreamingProvider::new(
            "mock",
            vec![vec![
                AgentStreamEvent::ContentDelta("partial answer".into()),
                AgentStreamEvent::Done {
                    stop_reason: Some("length".into()),
                },
            ]],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink);

        let req = AgentTurnRequest::new(
            make_session(AgentToolRegistry::new()),
            AgentTurnId::new("stream-truncated"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        );

        let result = loop_harness.run_turn_streaming(req, None).await;
        assert_eq!(result.status(), AgentTurnStatus::Stopped);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::Truncated);
        // The partial response is still available to hosts.
        assert_eq!(result.final_response().unwrap().content(), "partial answer");
    }

    #[tokio::test]
    async fn run_turn_streaming_non_truncation_stop_reason_is_final() {
        // A normal "stop" reason stays a clean final response (AC-01);
        // only truncation reasons downgrade the turn.
        let provider = Arc::new(StreamingProvider::new(
            "mock",
            vec![vec![
                AgentStreamEvent::ContentDelta("done".into()),
                AgentStreamEvent::Done {
                    stop_reason: Some("stop".into()),
                },
            ]],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink);

        let req = AgentTurnRequest::new(
            make_session(AgentToolRegistry::new()),
            AgentTurnId::new("stream-stop"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        );

        let result = loop_harness.run_turn_streaming(req, None).await;
        assert_eq!(result.status(), AgentTurnStatus::Completed);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::FinalResponse);
        assert_eq!(result.final_response().unwrap().content(), "done");
    }

    #[tokio::test]
    async fn run_turn_streaming_content_filter_is_truncated() {
        // R2-05: a policy gate (`content_filter`) that cuts the stream
        // short must not be reported as a clean final response, just
        // like `length`/`max_tokens`.
        let provider = Arc::new(StreamingProvider::new(
            "mock",
            vec![vec![
                AgentStreamEvent::ContentDelta("partial".into()),
                AgentStreamEvent::Done {
                    stop_reason: Some("content_filter".into()),
                },
            ]],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink);

        let req = AgentTurnRequest::new(
            make_session(AgentToolRegistry::new()),
            AgentTurnId::new("stream-content-filter"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        );

        let result = loop_harness.run_turn_streaming(req, None).await;
        assert_eq!(result.status(), AgentTurnStatus::Stopped);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::Truncated);
        assert_eq!(result.final_response().unwrap().content(), "partial");
    }

    #[tokio::test]
    async fn run_turn_streaming_error_event_stops_with_provider_error() {
        // A terminal Error event from the stream is a provider failure,
        // not a clean end (AC-05).
        let provider = Arc::new(StreamingProvider::new(
            "mock",
            vec![vec![
                AgentStreamEvent::ContentDelta("partial".into()),
                AgentStreamEvent::Error("connection dropped".into()),
            ]],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink.clone());

        let req = AgentTurnRequest::new(
            make_session(AgentToolRegistry::new()),
            AgentTurnId::new("stream-error"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        );

        let result = loop_harness.run_turn_streaming(req, None).await;
        assert_eq!(result.status(), AgentTurnStatus::Stopped);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::ProviderError);
        assert!(
            result
                .diagnostics()
                .iter()
                .any(|d| d.code().as_str().ends_with("STREAM")),
            "expected a STREAM diagnostic, got {:?}",
            result.diagnostics()
        );
        assert!(sink.events().iter().any(|e| matches!(
            e,
            AgentEvent::ProviderError { code, .. } if code == "STREAM"
        )));
    }

    #[tokio::test]
    async fn run_turn_streaming_warning_mid_stream_continues_turn() {
        // A host-visible Warning (e.g. EOF with incomplete tool-call
        // fragments, D-5) is informational: unlike `Error`, it must
        // NOT stop the turn — the stream continues to a normal final
        // response.
        let provider = Arc::new(StreamingProvider::new(
            "mock",
            vec![vec![
                AgentStreamEvent::Warning("stream ended with incomplete tool call(s)".into()),
                AgentStreamEvent::ContentDelta("final".into()),
                AgentStreamEvent::Done {
                    stop_reason: Some("stop".into()),
                },
            ]],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink.clone());

        let req = AgentTurnRequest::new(
            make_session(AgentToolRegistry::new()),
            AgentTurnId::new("stream-warning"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        );

        let result = loop_harness.run_turn_streaming(req, None).await;
        assert_eq!(result.status(), AgentTurnStatus::Completed);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::FinalResponse);
        assert_eq!(result.final_response().unwrap().content(), "final");
        // No provider-error diagnostic or event: the warning was not
        // treated as a terminal failure.
        assert!(
            result
                .diagnostics()
                .iter()
                .all(|d| !d.code().as_str().ends_with("STREAM")),
            "the Warning must not surface a provider error diagnostic"
        );
        assert!(
            !sink
                .events()
                .iter()
                .any(|e| matches!(e, AgentEvent::ProviderError { .. }))
        );
        // The deltas after the warning still reach the sink.
        let events = sink.events();
        let deltas: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ContentDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(deltas, vec!["final"]);
    }

    // ------------------------------------------------------------------
    // Cancellation tests
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn run_turn_cancelled_before_first_provider_call() {
        use tokio_util::sync::CancellationToken;

        let provider = Arc::new(ScriptedProvider::new(
            "mock",
            vec![ScriptedStep::Respond(response_with_text("too late"))],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider.clone(), sink);

        let cancel_token = CancellationToken::new();
        cancel_token.cancel(); // Cancel immediately.

        let req = AgentTurnRequest::new(
            make_session(AgentToolRegistry::new()),
            AgentTurnId::new("cancel-1"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        )
        .with_cancel_token(cancel_token);

        let result = loop_harness.run_turn(req, None).await;
        assert_eq!(result.status(), AgentTurnStatus::Stopped);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::Cancelled);
        // Provider should NOT have been called.
        assert_eq!(provider.call_count(), 0);
    }

    #[tokio::test]
    async fn run_turn_streaming_cancelled_before_first_provider_call() {
        use tokio_util::sync::CancellationToken;

        let provider = Arc::new(StreamingProvider::new(
            "mock",
            vec![vec![AgentStreamEvent::ContentDelta("too late".into())]],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink);

        let cancel_token = CancellationToken::new();
        cancel_token.cancel();

        let req = AgentTurnRequest::new(
            make_session(AgentToolRegistry::new()),
            AgentTurnId::new("cancel-2"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        )
        .with_cancel_token(cancel_token);

        let result = loop_harness.run_turn_streaming(req, None).await;
        assert_eq!(result.status(), AgentTurnStatus::Stopped);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::Cancelled);
    }

    // ----- hanging mocks (in-flight cancellation / deadline) -----

    /// A provider whose calls never complete: only cancellation or the
    /// turn deadline can end a turn against it (AC-02, AC-07).
    struct HangingProvider;
    #[async_trait]
    impl AgentProvider for HangingProvider {
        fn name(&self) -> ProviderName {
            ProviderName::new("hang")
        }
        async fn complete(&self, _request: AgentRequest) -> Result<AgentResponse, ProviderError> {
            std::future::pending().await
        }
        async fn stream(
            &self,
            _request: AgentRequest,
        ) -> Result<Box<dyn AgentStream>, ProviderError> {
            Ok(Box::new(HangingStream))
        }
        async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
            Ok(Vec::new())
        }
    }

    /// A stream whose reads never complete.
    struct HangingStream;
    #[async_trait]
    impl AgentStream for HangingStream {
        async fn next_event(&mut self) -> Option<AgentStreamEvent> {
            std::future::pending().await
        }
    }

    #[tokio::test]
    async fn run_turn_timeout_stops_with_timeout_and_visible_diagnostics() {
        // A provider that never answers: only the deadline can end the
        // turn, and the timeout must be visible to hosts (AC-02).
        let provider = Arc::new(HangingProvider);
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink.clone());

        let req = AgentTurnRequest::new(
            make_session(AgentToolRegistry::new()),
            AgentTurnId::new("timeout-1"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        )
        .with_turn_timeout(Duration::from_millis(50));

        let result = loop_harness.run_turn(req, None).await;
        assert_eq!(result.status(), AgentTurnStatus::Stopped);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::Timeout);
        // The timeout is observable: a diagnostic plus a ProviderError
        // event with code TIMEOUT.
        assert!(
            result
                .diagnostics()
                .iter()
                .any(|d| d.code().as_str().ends_with("TIMEOUT")),
            "expected a TIMEOUT diagnostic, got {:?}",
            result.diagnostics()
        );
        assert!(sink.events().iter().any(|e| matches!(
            e,
            AgentEvent::ProviderError { code, .. } if code == "TIMEOUT"
        )));
    }

    #[tokio::test]
    async fn run_turn_streaming_timeout_stops_with_timeout() {
        let provider = Arc::new(HangingProvider);
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink);

        let req = AgentTurnRequest::new(
            make_session(AgentToolRegistry::new()),
            AgentTurnId::new("timeout-2"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        )
        .with_turn_timeout(Duration::from_millis(50));

        let result = loop_harness.run_turn_streaming(req, None).await;
        assert_eq!(result.status(), AgentTurnStatus::Stopped);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::Timeout);
    }

    #[tokio::test]
    async fn run_turn_cancel_aborts_in_flight_provider_call() {
        // The provider never returns; cancellation must end the turn
        // promptly by dropping the in-flight call (AC-07).
        let provider = Arc::new(HangingProvider);
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink);

        let cancel_token = CancellationToken::new();
        let req = AgentTurnRequest::new(
            make_session(AgentToolRegistry::new()),
            AgentTurnId::new("cancel-inflight-1"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        )
        .with_cancel_token(cancel_token.clone());

        let handle = tokio::spawn(async move { loop_harness.run_turn(req, None).await });
        // Give the provider call time to start, then cancel mid-flight.
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel_token.cancel();
        let result = handle.await.expect("turn finishes promptly after cancel");
        assert_eq!(result.status(), AgentTurnStatus::Stopped);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::Cancelled);
    }

    #[tokio::test]
    async fn run_turn_streaming_cancel_aborts_in_flight_stream() {
        let provider = Arc::new(HangingProvider);
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink);

        let cancel_token = CancellationToken::new();
        let req = AgentTurnRequest::new(
            make_session(AgentToolRegistry::new()),
            AgentTurnId::new("cancel-inflight-2"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        )
        .with_cancel_token(cancel_token.clone());

        let handle = tokio::spawn(async move { loop_harness.run_turn_streaming(req, None).await });
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel_token.cancel();
        let result = handle.await.expect("turn finishes promptly after cancel");
        assert_eq!(result.status(), AgentTurnStatus::Stopped);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::Cancelled);
    }

    // ----- execute_tool cancel / deadline (AR-19) -----

    /// A tool whose invocation never completes: only the execute_tool
    /// cancel/deadline race can end it (AR-19).
    struct HangingTool {
        spec: ToolSpec,
    }

    #[async_trait]
    impl AgentTool for HangingTool {
        fn spec(&self) -> ToolSpec {
            self.spec.clone()
        }

        async fn invoke(&self, _ctx: &ToolContext, _input: Value) -> ToolResult {
            std::future::pending().await
        }
    }

    fn session_with_hanging_tool(name: &str) -> AgentSession {
        session_with(|reg| {
            let spec = ToolSpec::new(
                ToolName::new(name),
                "hangs forever",
                [AgentMode::Agent, AgentMode::Build],
                ToolPermission::new("workflow.read"),
                ToolRiskLevel::Read,
            );
            reg.register(HangingTool { spec }).unwrap();
        })
    }

    #[tokio::test]
    async fn run_turn_cancel_aborts_hanging_tool_execution() {
        // A tool whose invoke never resolves: cancellation must end the
        // turn promptly via the execute_tool select! race (AR-19), not
        // wedge on the in-flight tool future.
        let provider = Arc::new(ScriptedProvider::new(
            "mock",
            vec![ScriptedStep::Respond(response_with_tool_calls(vec![
                ToolCall::new(ToolCallId::new("hc1"), "hang", json!({})),
            ]))],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink.clone());

        let cancel_token = CancellationToken::new();
        let req = AgentTurnRequest::new(
            session_with_hanging_tool("hang"),
            AgentTurnId::new("cancel-tool-1"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        )
        .with_cancel_token(cancel_token.clone());

        let handle = tokio::spawn(async move { loop_harness.run_turn(req, None).await });
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel_token.cancel();
        let result = handle
            .await
            .expect("turn finishes promptly after tool cancel");
        assert_eq!(result.status(), AgentTurnStatus::Stopped);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::Cancelled);
        // The tool failure is observable as a tool_failed event.
        assert!(sink.events().iter().any(
            |e| matches!(e, AgentEvent::ToolFailed { id: Some(id), .. } if id.as_str() == "hc1")
        ));
    }

    #[tokio::test]
    async fn run_turn_tool_execution_deadline_stops_turn() {
        // A hanging tool plus a short turn timeout: the execute_tool
        // deadline fires, marking the tool failed, then the loop's next
        // round observes no more provider steps and completes. AR-19's
        // tool deadline must not hang the turn past the timeout.
        let provider = Arc::new(ScriptedProvider::new(
            "mock",
            vec![ScriptedStep::Respond(response_with_tool_calls(vec![
                ToolCall::new(ToolCallId::new("hd1"), "hang", json!({})),
            ]))],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink.clone());

        let req = AgentTurnRequest::new(
            session_with_hanging_tool("hang"),
            AgentTurnId::new("deadline-tool-1"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        )
        .with_turn_timeout(Duration::from_millis(80));

        let result = tokio::time::timeout(Duration::from_secs(5), loop_harness.run_turn(req, None))
            .await
            .expect("turn must not hang past the tool deadline");
        // Scripted provider is exhausted after the tool round, so the
        // turn ends (Stopped, FinalResponse-free) — the key assertion is
        // that it did not hang.
        // Scripted provider is exhausted after the tool round, so the
        // turn ends with ProviderError — the key assertion is that it
        // did not hang.
        assert_eq!(result.stop_reason(), AgentTurnStopReason::ProviderError);
        assert!(sink.events().iter().any(
            |e| matches!(e, AgentEvent::ToolFailed { id: Some(id), .. } if id.as_str() == "hd1")
        ));
    }

    // ----- mid-turn cancellation checkpoints (AC-15) -----

    #[tokio::test]
    async fn run_turn_cancel_between_rounds_stops_at_next_checkpoint() {
        // The first round requests a tool call whose `invoke` cancels
        // the token mid-execution. The tool still runs to completion
        // (tool execution is atomic) and its observation is recorded;
        // the loop's next checkpoint — the top of the round loop —
        // sees the cancelled token and stops the turn. The second
        // scripted provider response is never consumed.
        let cancel_token = CancellationToken::new();
        let canceller = Arc::new(CancellingTool::new(cancel_token.clone()));

        let provider = Arc::new(ScriptedProvider::new(
            "mock",
            vec![
                ScriptedStep::Respond(response_with_tool_calls(vec![ToolCall::new(
                    ToolCallId::new("c1"),
                    "cancel",
                    json!({"round": 1}),
                )])),
                ScriptedStep::Respond(response_with_text("never reached")),
            ],
        ));
        let loop_harness = AgentLoop::new(provider.clone(), Arc::new(VecAgentEventSink::new()));
        let session = session_with(|reg| {
            reg.register((*canceller).clone()).unwrap();
        });

        let req = AgentTurnRequest::new(
            session.clone(),
            AgentTurnId::new("cancel-rounds"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        )
        .with_cancel_token(cancel_token.clone());

        let result = loop_harness.run_turn(req, None).await;
        assert_eq!(result.status(), AgentTurnStatus::Stopped);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::Cancelled);
        // Only the first round ran: the second scripted response is
        // still queued and was never requested.
        assert_eq!(provider.call_count(), 1);
        // The tool executed and its observation was recorded into the
        // transcript before the checkpoint stopped the turn.
        assert_eq!(canceller.recorded_inputs().len(), 1);
        assert_eq!(result.tool_calls().len(), 1);
        assert_eq!(result.tool_calls()[0].status(), ToolCallStatus::Succeeded);
        let messages = result.messages();
        assert_eq!(messages.len(), 3); // user + assistant tool call + tool observation
        assert_eq!(messages[2].role(), "tool");
        assert_eq!(messages[2].tool_call_id().unwrap().as_str(), "c1");
        // The cancelled turn still committed the messages seen so far.
        assert_eq!(session.history_len(), 3);
    }

    #[tokio::test]
    async fn run_turn_cancel_between_tool_calls_skips_remaining_calls() {
        // One round requests TWO tool calls; the first tool cancels the
        // token during its invoke. Cancellation is only checked
        // *between* tool calls in `execute_tool_calls` — tool execution
        // is atomic — so the second tool call is never executed and the
        // turn stops at the mid-round checkpoint with `Cancelled`.
        let cancel_token = CancellationToken::new();
        let canceller = Arc::new(CancellingTool::new(cancel_token.clone()));
        let second = Arc::new(ScriptedTool::success("second", vec![json!({"ran": true})]));

        let provider = Arc::new(ScriptedProvider::new(
            "mock",
            vec![ScriptedStep::Respond(response_with_tool_calls(vec![
                ToolCall::new(ToolCallId::new("c1"), "cancel", json!({"step": 1})),
                ToolCall::new(ToolCallId::new("c2"), "second", json!({"step": 2})),
            ]))],
        ));
        let loop_harness = AgentLoop::new(provider.clone(), Arc::new(VecAgentEventSink::new()));
        let session = session_with(|reg| {
            reg.register((*canceller).clone()).unwrap();
            reg.register((*second).clone()).unwrap();
        });

        let req = AgentTurnRequest::new(
            session.clone(),
            AgentTurnId::new("cancel-mid-round"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        )
        .with_cancel_token(cancel_token.clone());

        let result = loop_harness.run_turn(req, None).await;
        assert_eq!(result.status(), AgentTurnStatus::Stopped);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::Cancelled);
        // AR-25: Read tools in one round run as a concurrent batch. The
        // canceller's sibling may have started concurrently (both calls
        // are atomic); cancellation stops the loop at the batch boundary
        // and must never produce a phantom success beyond it.
        assert_eq!(canceller.recorded_inputs().len(), 1);
        // The sibling may or may not have started before the boundary;
        // it must never be re-invoked.
        assert!(second.recorded_inputs().len() <= 1, "no re-invocation");
        assert_eq!(provider.call_count(), 1);
        assert!(result.tool_calls().len() <= 2, "no tools beyond the batch");
        // The transcript ends on the observed batch: assistant + at least
        // the first tool observation.
        let messages = result.messages();
        assert!(messages.len() >= 3);
        assert_eq!(messages[2].role(), "tool");
    }

    #[tokio::test]
    async fn run_turn_read_tools_execute_in_parallel_preserving_order() {
        // AR-25: two Read tools in one round run concurrently, but their
        // observation messages are appended in provider order.
        let alpha = Arc::new(ScriptedTool::success(
            "alpha",
            vec![json!({"a": 1}), json!({"a": 2})],
        ));
        let beta = Arc::new(ScriptedTool::success(
            "beta",
            vec![json!({"b": 1}), json!({"b": 2})],
        ));

        let provider = Arc::new(ScriptedProvider::new(
            "mock",
            vec![ScriptedStep::Respond(response_with_tool_calls(vec![
                ToolCall::new(ToolCallId::new("c1"), "alpha", json!({})),
                ToolCall::new(ToolCallId::new("c2"), "beta", json!({})),
            ]))],
        ));
        let loop_harness = AgentLoop::new(provider.clone(), Arc::new(VecAgentEventSink::new()));
        let session = session_with(|reg| {
            reg.register((*alpha).clone()).unwrap();
            reg.register((*beta).clone()).unwrap();
        });

        let req = AgentTurnRequest::new(
            session,
            AgentTurnId::new("parallel-read"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        );
        let result = loop_harness.run_turn(req, None).await;

        // Both tools executed exactly once each.
        assert_eq!(alpha.recorded_inputs().len(), 1);
        assert_eq!(beta.recorded_inputs().len(), 1);
        // Both observations present and ordered c1 then c2.
        let tool_msgs: Vec<_> = result
            .messages()
            .iter()
            .filter(|m| m.role() == "tool")
            .collect();
        assert_eq!(tool_msgs.len(), 2);
        assert_eq!(tool_msgs[0].tool_call_id().unwrap().as_str(), "c1");
        assert_eq!(tool_msgs[1].tool_call_id().unwrap().as_str(), "c2");
    }

    #[tokio::test]
    async fn run_turn_read_failure_isolates_siblings() {
        // AR-25: a failing Read tool does not cancel sibling Read tools;
        // both still produce observations and the loop continues.
        let ok = Arc::new(ScriptedTool::success("ok", vec![json!({"x": 1})]));
        let boom = Arc::new(ScriptedTool::failing("boom", "kaboom"));

        let provider = Arc::new(ScriptedProvider::new(
            "mock",
            vec![
                ScriptedStep::Respond(response_with_tool_calls(vec![
                    ToolCall::new(ToolCallId::new("c1"), "ok", json!({})),
                    ToolCall::new(ToolCallId::new("c2"), "boom", json!({})),
                ])),
                ScriptedStep::Respond(response_with_text("done")),
            ],
        ));
        let loop_harness = AgentLoop::new(provider.clone(), Arc::new(VecAgentEventSink::new()));
        let session = session_with(|reg| {
            reg.register((*ok).clone()).unwrap();
            reg.register((*boom).clone()).unwrap();
        });

        let req = AgentTurnRequest::new(
            session,
            AgentTurnId::new("isolation"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        );
        let result = loop_harness.run_turn(req, None).await;

        // Both ran; boom failed but did not cancel ok.
        assert_eq!(ok.recorded_inputs().len(), 1, "sibling must still run");
        assert_eq!(boom.recorded_inputs().len(), 1);
        // One success and one failure observation, then the final answer.
        assert_eq!(result.tool_calls().len(), 2);
        let ok_call = result
            .tool_calls()
            .iter()
            .find(|tc| tc.tool_name().as_str() == "ok")
            .expect("ok");
        assert!(
            ok_call.output().is_some(),
            "ok tool produced an observation"
        );
        assert_eq!(result.stop_reason(), AgentTurnStopReason::FinalResponse);
    }

    // ------------------------------------------------------------------
    // Context manager integration
    // ------------------------------------------------------------------

    fn temp_session_dir(prefix: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("reimagine-agent-context-{prefix}-{nonce}"))
    }

    #[tokio::test]
    async fn run_turn_with_context_manager_prepares_and_commits_history() {
        struct CapturingProvider {
            name: ProviderName,
            seen_message_contents: Mutex<Vec<Vec<String>>>,
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
                self.seen_message_contents.lock().unwrap().push(
                    request
                        .messages()
                        .iter()
                        .map(|m| m.content().to_string())
                        .collect(),
                );
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

        let dir = temp_session_dir("ctx-loop");
        // reserved 0 puts the soft trigger at the hard limit (10k):
        // the tiny test history never compacts, so this test exercises
        // prepare/commit wiring only (compaction has dedicated tests).
        let config = ContextConfig {
            reserved_tokens: 0,
            ..ContextConfig::new(10_000, dir.clone())
        };
        let mut context = ContextManager::new(config);
        let session = make_session(AgentToolRegistry::new());
        let provider = Arc::new(CapturingProvider {
            name: ProviderName::new("capture"),
            seen_message_contents: Mutex::new(Vec::new()),
        });
        let loop_harness = AgentLoop::new(provider.clone(), Arc::new(VecAgentEventSink::new()));

        // Five turns push the stored history past the window (the
        // verbatim tail of `tail_turns = 2` complete turns); each turn
        // commits its own [user, assistant] pair.
        for i in 1..=5 {
            let req = AgentTurnRequest::new(
                session.clone(),
                AgentTurnId::new(format!("ctx-{i}")),
                ModelName::new("test-model"),
                vec![Message::user(format!("u{i}"))],
            );
            let result = loop_harness.run_turn(req, Some(&mut context)).await;
            assert_eq!(result.status(), AgentTurnStatus::Completed);
        }

        // The provider received prepare_messages output: an (empty,
        // V1 has no system prompt) system message, the windowed
        // history, then this turn's input. Within the (10k) budget the
        // window keeps everything.
        let seen = provider.seen_message_contents.lock().unwrap().clone();
        assert_eq!(seen[0], vec!["", "u1"]);
        assert_eq!(seen[1], vec!["", "u1", "done", "u2"]);
        assert_eq!(seen[2], vec!["", "u1", "done", "u2", "done", "u3"]);
        assert_eq!(
            seen[3],
            vec!["", "u1", "done", "u2", "done", "u3", "done", "u4"]
        );
        assert_eq!(
            seen[4],
            vec![
                "", "u1", "done", "u2", "done", "u3", "done", "u4", "done", "u5"
            ]
        );

        // The context manager owns the history; the session transcript
        // must stay untouched.
        assert_eq!(session.history_len(), 0);

        // persist/load round trip after a turn: the loaded manager
        // serves the same windowed history as the live one.
        context.persist("sess-ctx").expect("persist failed");
        let mut loaded =
            ContextManager::load("sess-ctx", ContextConfig::new(10_000, dir)).expect("load failed");
        assert_eq!(loaded.token_count(), context.token_count());
        let prepared = loaded.prepare_messages("", &[Message::user("u6")]);
        let contents: Vec<String> = prepared.iter().map(|m| m.content().to_string()).collect();
        assert_eq!(contents, vec!["", "u4", "done", "u5", "done", "u6"]);
    }

    // ----- compaction driver (CM-V2e) -----

    /// Provider that answers summarizer prompts (a lone user message
    /// starting with the schema prompt) with a fixed summary and every
    /// other request with "done".
    struct SummarizingProvider {
        name: ProviderName,
        seen: Mutex<Vec<Vec<String>>>,
        summarize_calls: Mutex<usize>,
        fail_summarize: Mutex<bool>,
        /// When set, the summarizer returns this exact error (used to
        /// script transient vs permanent failures, AC-24).
        fail_with: Mutex<Option<ProviderError>>,
        /// When set, the summarizer replies with an empty message
        /// (refusal / truncation) instead of the fixed summary.
        empty_summary: Mutex<bool>,
    }

    impl SummarizingProvider {
        fn new(name: &str) -> Self {
            Self {
                name: ProviderName::new(name),
                seen: Mutex::new(Vec::new()),
                summarize_calls: Mutex::new(0),
                fail_summarize: Mutex::new(false),
                fail_with: Mutex::new(None),
                empty_summary: Mutex::new(false),
            }
        }

        fn is_summarize_request(request: &AgentRequest) -> bool {
            request.messages().len() == 1
                && request.messages()[0]
                    .content()
                    .starts_with("Summarize the conversation")
        }
    }

    #[async_trait]
    impl AgentProvider for SummarizingProvider {
        fn name(&self) -> ProviderName {
            self.name.clone()
        }

        async fn complete(&self, request: AgentRequest) -> Result<AgentResponse, ProviderError> {
            if Self::is_summarize_request(&request) {
                *self.summarize_calls.lock().unwrap() += 1;
                if let Some(err) = self.fail_with.lock().unwrap().clone() {
                    return Err(err);
                }
                if *self.fail_summarize.lock().unwrap() {
                    return Err(ProviderError::new("SUMMARIZE_FAIL", "summarizer down"));
                }
                if *self.empty_summary.lock().unwrap() {
                    return Ok(response_with_text(""));
                }
                return Ok(response_with_text("summary-text"));
            }
            self.seen.lock().unwrap().push(
                request
                    .messages()
                    .iter()
                    .map(|m| m.content().to_string())
                    .collect(),
            );
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

    /// Provider for the streaming compaction path: `stream()` returns
    /// scripted streams for turn rounds while `complete()` answers
    /// summarizer prompts — the compaction driver calls `complete` for
    /// the summarize request even when the turn itself streams.
    struct StreamingSummarizeProvider {
        name: ProviderName,
        streams: Mutex<VecDeque<Vec<AgentStreamEvent>>>,
        summarize_calls: Mutex<usize>,
        summary_text: Mutex<String>,
    }

    impl StreamingSummarizeProvider {
        fn new(name: &str, streams: Vec<Vec<AgentStreamEvent>>) -> Self {
            Self {
                name: ProviderName::new(name),
                streams: Mutex::new(streams.into()),
                summarize_calls: Mutex::new(0),
                summary_text: Mutex::new("summary-text".to_string()),
            }
        }
    }

    #[async_trait]
    impl AgentProvider for StreamingSummarizeProvider {
        fn name(&self) -> ProviderName {
            self.name.clone()
        }

        async fn complete(&self, request: AgentRequest) -> Result<AgentResponse, ProviderError> {
            if SummarizingProvider::is_summarize_request(&request) {
                *self.summarize_calls.lock().unwrap() += 1;
                return Ok(response_with_text(
                    &self.summary_text.lock().unwrap().clone(),
                ));
            }
            Ok(response_with_text("done"))
        }

        async fn stream(
            &self,
            _request: AgentRequest,
        ) -> Result<Box<dyn AgentStream>, ProviderError> {
            let events = self.streams.lock().unwrap().pop_front().unwrap_or_default();
            Ok(Box::new(MockStream { events }))
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
            Ok(Vec::new())
        }
    }

    fn big_turn_input(i: usize) -> Vec<Message> {
        // One turn (~1.3k tokens) already crosses the 1k soft line.
        vec![Message::user(format!("u{i}:{}", "a".repeat(5_000)))]
    }

    #[tokio::test]
    async fn run_turn_compacts_before_window_when_soft_triggered() {
        let dir = temp_session_dir("ctx-compact");
        let config = ContextConfig {
            reserved_tokens: 9_000, // soft line at 1k
            tail_turns: 1,
            ..ContextConfig::new(10_000, dir)
        };
        let mut context = ContextManager::new(config);
        let session = make_session(AgentToolRegistry::new());
        let provider = Arc::new(SummarizingProvider::new("summarize"));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider.clone(), sink.clone());

        // Two turns cross the soft line; with tail_turns = 1 the first
        // turn is pending eviction.
        for i in 1..=2 {
            let req = AgentTurnRequest::new(
                session.clone(),
                AgentTurnId::new(format!("fill-{i}")),
                ModelName::new("test-model"),
                big_turn_input(i),
            );
            let result = loop_harness.run_turn(req, Some(&mut context)).await;
            assert_eq!(result.status(), AgentTurnStatus::Completed);
        }
        assert!(context.needs_compaction());
        assert!(context.has_eviction_pending());
        let seen_before = provider.seen.lock().unwrap().len();

        // The next turn trips the soft trigger: the summarizer runs
        // first, then the windowed request starts with the sticky
        // summary.
        let req = AgentTurnRequest::new(
            session.clone(),
            AgentTurnId::new("compact-1"),
            ModelName::new("test-model"),
            big_turn_input(3),
        );
        let result = loop_harness.run_turn(req, Some(&mut context)).await;
        assert_eq!(result.status(), AgentTurnStatus::Completed);
        assert_eq!(*provider.summarize_calls.lock().unwrap(), 1);
        assert_eq!(context.sticky_count(), 1);

        let seen = provider.seen.lock().unwrap().clone();
        assert_eq!(seen.len(), seen_before + 1);
        assert_eq!(seen[seen_before][1], "summary-text");

        // The event carries the record details.
        let events = sink.events();
        let compacted: Vec<&AgentEvent> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ContextCompacted { .. }))
            .collect();
        assert_eq!(compacted.len(), 1);
        if let AgentEvent::ContextCompacted {
            summary,
            tokens_before,
            tokens_after,
            ..
        } = compacted[0]
        {
            assert_eq!(summary, "summary-text");
            assert!(*tokens_before > 0);
            assert!(*tokens_after > 0);
        }
    }

    #[tokio::test]
    async fn run_turn_compaction_failure_falls_back_to_eviction() {
        let dir = temp_session_dir("ctx-compact-fail");
        let config = ContextConfig {
            reserved_tokens: 9_000,
            tail_turns: 1,
            ..ContextConfig::new(10_000, dir)
        };
        let mut context = ContextManager::new(config);
        let session = make_session(AgentToolRegistry::new());
        let provider = Arc::new(SummarizingProvider::new("summarize-fail"));
        *provider.fail_summarize.lock().unwrap() = true;
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider.clone(), sink.clone());

        for i in 1..=2 {
            let req = AgentTurnRequest::new(
                session.clone(),
                AgentTurnId::new(format!("fill-{i}")),
                ModelName::new("test-model"),
                big_turn_input(i),
            );
            let result = loop_harness.run_turn(req, Some(&mut context)).await;
            assert_eq!(result.status(), AgentTurnStatus::Completed);
        }

        let req = AgentTurnRequest::new(
            session.clone(),
            AgentTurnId::new("compact-fail"),
            ModelName::new("test-model"),
            big_turn_input(3),
        );
        let result = loop_harness.run_turn(req, Some(&mut context)).await;
        assert_eq!(result.status(), AgentTurnStatus::Completed);
        // The summarizer failure is registered; no event is emitted;
        // the turn completed via plain eviction.
        assert!(context.consecutive_compaction_failures() >= 1);
        assert!(
            !sink
                .events()
                .iter()
                .any(|e| matches!(e, AgentEvent::ContextCompacted { .. }))
        );
        // The window still bounds the history.
        assert!(context.token_count() <= 10_000);
    }

    #[tokio::test]
    async fn run_turn_compaction_thrash_stops_attempts() {
        let dir = temp_session_dir("ctx-thrash");
        let config = ContextConfig {
            reserved_tokens: 9_000,
            tail_turns: 1,
            ..ContextConfig::new(10_000, dir)
        };
        let mut context = ContextManager::new(config);
        let session = make_session(AgentToolRegistry::new());
        let provider = Arc::new(SummarizingProvider::new("summarize-thrash"));
        *provider.fail_summarize.lock().unwrap() = true;
        let loop_harness = AgentLoop::new(provider.clone(), Arc::new(VecAgentEventSink::new()));

        for i in 1..=2 {
            let req = AgentTurnRequest::new(
                session.clone(),
                AgentTurnId::new(format!("fill-{i}")),
                ModelName::new("test-model"),
                big_turn_input(i),
            );
            let result = loop_harness.run_turn(req, Some(&mut context)).await;
            assert_eq!(result.status(), AgentTurnStatus::Completed);
        }

        // Three failing compaction attempts...
        for i in 0..3 {
            let req = AgentTurnRequest::new(
                session.clone(),
                AgentTurnId::new(format!("thrash-{i}")),
                ModelName::new("test-model"),
                big_turn_input(100 + i),
            );
            let result = loop_harness.run_turn(req, Some(&mut context)).await;
            assert_eq!(result.status(), AgentTurnStatus::Completed);
        }
        assert_eq!(*provider.summarize_calls.lock().unwrap(), 3);
        assert_eq!(context.consecutive_compaction_failures(), 3);

        // ...and the guard holds: no further attempts, turns still
        // complete.
        for i in 0..3 {
            let req = AgentTurnRequest::new(
                session.clone(),
                AgentTurnId::new(format!("after-{i}")),
                ModelName::new("test-model"),
                big_turn_input(200 + i),
            );
            let result = loop_harness.run_turn(req, Some(&mut context)).await;
            assert_eq!(result.status(), AgentTurnStatus::Completed);
        }
        assert_eq!(*provider.summarize_calls.lock().unwrap(), 3);
        assert!(!context.should_attempt_compaction());
    }

    #[tokio::test]
    async fn run_turn_transient_compaction_failure_retries_next_turn() {
        // AC-24: a transient summarizer failure (transport code) must
        // NOT advance the permanent failure streak: compaction is
        // retried on the next turn and is not disabled, unlike the
        // permanent-failure thrash path above.
        let dir = temp_session_dir("ctx-transient");
        let config = ContextConfig {
            reserved_tokens: 9_000,
            tail_turns: 1,
            ..ContextConfig::new(10_000, dir)
        };
        let mut context = ContextManager::new(config);
        let session = make_session(AgentToolRegistry::new());
        let provider = Arc::new(SummarizingProvider::new("summarize-transient"));
        *provider.fail_with.lock().unwrap() = Some(ProviderError::new(
            "TRANSPORT",
            "connection reset mid-summarize",
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider.clone(), sink.clone());

        for i in 1..=2 {
            let req = AgentTurnRequest::new(
                session.clone(),
                AgentTurnId::new(format!("fill-{i}")),
                ModelName::new("test-model"),
                big_turn_input(i),
            );
            let result = loop_harness.run_turn(req, Some(&mut context)).await;
            assert_eq!(result.status(), AgentTurnStatus::Completed);
        }

        // Three turns trip the driver while the summarizer fails
        // transiently. Each turn retries the attempt; the permanent
        // streak stays at 0, so the thrash guard never trips.
        for i in 0..3 {
            let req = AgentTurnRequest::new(
                session.clone(),
                AgentTurnId::new(format!("transient-{i}")),
                ModelName::new("test-model"),
                big_turn_input(100 + i),
            );
            let result = loop_harness.run_turn(req, Some(&mut context)).await;
            assert_eq!(result.status(), AgentTurnStatus::Completed);
        }
        assert_eq!(*provider.summarize_calls.lock().unwrap(), 3);
        assert_eq!(
            context.consecutive_compaction_failures(),
            0,
            "transient failures must not advance the permanent streak"
        );
        assert!(
            context.should_attempt_compaction(),
            "compaction stays enabled after transient failures"
        );
        assert!(
            !sink
                .events()
                .iter()
                .any(|e| matches!(e, AgentEvent::ContextCompacted { .. }))
        );

        // Once the summarizer recovers, the very next turn compacts —
        // no session-wide disable happened.
        *provider.fail_with.lock().unwrap() = None;
        let req = AgentTurnRequest::new(
            session.clone(),
            AgentTurnId::new("transient-recovered"),
            ModelName::new("test-model"),
            big_turn_input(200),
        );
        let result = loop_harness.run_turn(req, Some(&mut context)).await;
        assert_eq!(result.status(), AgentTurnStatus::Completed);
        assert_eq!(*provider.summarize_calls.lock().unwrap(), 4);
        assert_eq!(context.sticky_count(), 1);
        assert!(context.compaction_summary().is_some());
    }

    #[tokio::test]
    async fn run_turn_empty_summary_reply_falls_back_to_plain_eviction() {
        // The summarizer answers the compaction request with an empty
        // message (refusal / truncation). The loop driver must reject
        // it: no sticky summary, no ContextCompacted event, and the
        // turn still completes via plain eviction. The empty-reply
        // guard lives in the loop driver (`maybe_compact`), not in
        // `ContextManager::apply_summary`.
        let dir = temp_session_dir("ctx-empty-summary");
        let config = ContextConfig {
            reserved_tokens: 9_000,
            tail_turns: 1,
            ..ContextConfig::new(10_000, dir)
        };
        let mut context = ContextManager::new(config);
        let session = make_session(AgentToolRegistry::new());
        let provider = Arc::new(SummarizingProvider::new("summarize-empty"));
        *provider.empty_summary.lock().unwrap() = true;
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider.clone(), sink.clone());

        // Two turns cross the soft line (same shape as the happy-path
        // compaction test), so the third turn trips the driver.
        for i in 1..=2 {
            let req = AgentTurnRequest::new(
                session.clone(),
                AgentTurnId::new(format!("fill-{i}")),
                ModelName::new("test-model"),
                big_turn_input(i),
            );
            let result = loop_harness.run_turn(req, Some(&mut context)).await;
            assert_eq!(result.status(), AgentTurnStatus::Completed);
        }
        assert!(context.needs_compaction());
        assert!(context.has_eviction_pending());

        let req = AgentTurnRequest::new(
            session.clone(),
            AgentTurnId::new("empty-summary"),
            ModelName::new("test-model"),
            big_turn_input(3),
        );
        let result = loop_harness.run_turn(req, Some(&mut context)).await;
        assert_eq!(result.status(), AgentTurnStatus::Completed);
        // The summarizer ran and its empty reply counted as a failure.
        assert_eq!(*provider.summarize_calls.lock().unwrap(), 1);
        assert!(context.consecutive_compaction_failures() >= 1);
        // No summary was applied and no compaction event was emitted.
        assert!(context.compaction_summary().is_none());
        assert_eq!(context.sticky_count(), 0);
        assert!(
            !sink
                .events()
                .iter()
                .any(|e| matches!(e, AgentEvent::ContextCompacted { .. }))
        );
        // Plain eviction still bounds the window.
        assert!(context.token_count() <= 10_000);
    }

    #[tokio::test]
    async fn run_turn_streaming_triggers_compaction_driver() {
        // The compaction driver runs inside `prepare_turn`, which both
        // turn paths share: on the streaming path the summarizer call
        // still goes through `provider.complete` (never `stream`), and
        // a successful summary emits `ContextCompacted` with
        // tokens_before > tokens_after.
        let dir = temp_session_dir("ctx-stream-compact");
        let config = ContextConfig {
            reserved_tokens: 9_000,
            tail_turns: 1,
            ..ContextConfig::new(10_000, dir)
        };
        let mut context = ContextManager::new(config);
        let session = make_session(AgentToolRegistry::new());
        let provider = Arc::new(StreamingSummarizeProvider::new(
            "stream-compact",
            vec![
                // Two fill turns cross the soft line...
                vec![
                    AgentStreamEvent::ContentDelta("fill one".into()),
                    AgentStreamEvent::Done {
                        stop_reason: Some("stop".into()),
                    },
                ],
                vec![
                    AgentStreamEvent::ContentDelta("fill two".into()),
                    AgentStreamEvent::Done {
                        stop_reason: Some("stop".into()),
                    },
                ],
                // ...and the third turn trips the compaction driver
                // before its round streams.
                vec![
                    AgentStreamEvent::ContentDelta("final".into()),
                    AgentStreamEvent::Done {
                        stop_reason: Some("stop".into()),
                    },
                ],
            ],
        ));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider.clone(), sink.clone());

        for i in 1..=2 {
            let req = AgentTurnRequest::new(
                session.clone(),
                AgentTurnId::new(format!("fill-{i}")),
                ModelName::new("test-model"),
                big_turn_input(i),
            );
            let result = loop_harness
                .run_turn_streaming(req, Some(&mut context))
                .await;
            assert_eq!(result.status(), AgentTurnStatus::Completed);
        }
        assert!(context.needs_compaction());
        assert!(context.has_eviction_pending());

        let req = AgentTurnRequest::new(
            session.clone(),
            AgentTurnId::new("stream-compact"),
            ModelName::new("test-model"),
            big_turn_input(3),
        );
        let result = loop_harness
            .run_turn_streaming(req, Some(&mut context))
            .await;
        assert_eq!(result.status(), AgentTurnStatus::Completed);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::FinalResponse);
        assert_eq!(result.final_response().unwrap().content(), "final");

        // The summarizer was called once via `complete` and the sticky
        // summary was applied.
        assert_eq!(*provider.summarize_calls.lock().unwrap(), 1);
        assert_eq!(context.sticky_count(), 1);
        assert_eq!(
            context
                .compaction_summary()
                .map(|record| record.text.as_str()),
            Some("summary-text")
        );

        let events = sink.events();
        let compacted: Vec<&AgentEvent> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ContextCompacted { .. }))
            .collect();
        assert_eq!(compacted.len(), 1);
        if let AgentEvent::ContextCompacted {
            summary,
            tokens_before,
            tokens_after,
            ..
        } = compacted[0]
        {
            assert_eq!(summary, "summary-text");
            assert!(
                *tokens_before > *tokens_after,
                "compaction must shrink the stored history (before={tokens_before}, after={tokens_after})"
            );
        }
    }

    #[tokio::test]
    async fn run_turn_streaming_consumes_server_compaction() {
        // A server-side compaction notification is informational: the
        // turn completes with only the real content.
        let provider = Arc::new(StreamingProvider::new(
            "server-compact",
            vec![vec![
                crate::provider::AgentStreamEvent::Compacted {
                    item_id: "fc_compacted_1".to_string(),
                },
                crate::provider::AgentStreamEvent::ContentDelta("final".to_string()),
                crate::provider::AgentStreamEvent::Done { stop_reason: None },
            ]],
        ));
        let loop_harness = AgentLoop::new(provider.clone(), Arc::new(VecAgentEventSink::new()));
        let session = make_session(AgentToolRegistry::new());
        let req = AgentTurnRequest::new(
            session.clone(),
            AgentTurnId::new("stream-compacted"),
            ModelName::new("test-model"),
            vec![Message::user("hi")],
        );
        let result = loop_harness.run_turn_streaming(req, None).await;
        assert_eq!(result.status(), AgentTurnStatus::Completed);
        let messages = result.messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].content(), "final");
    }

    // ----- image modality gate (PV-03b) -----

    struct ModelsProvider {
        name: ProviderName,
        models: Mutex<Vec<ModelInfo>>,
        list_calls: Mutex<usize>,
        fail_listing: Mutex<bool>,
    }

    impl ModelsProvider {
        fn with_models(models: Vec<ModelInfo>) -> Self {
            Self {
                name: ProviderName::new("models"),
                models: Mutex::new(models),
                list_calls: Mutex::new(0),
                fail_listing: Mutex::new(false),
            }
        }
    }

    #[async_trait]
    impl AgentProvider for ModelsProvider {
        fn name(&self) -> ProviderName {
            self.name.clone()
        }

        async fn complete(&self, _request: AgentRequest) -> Result<AgentResponse, ProviderError> {
            unreachable!("gate tests never complete")
        }

        async fn stream(
            &self,
            _request: AgentRequest,
        ) -> Result<Box<dyn AgentStream>, ProviderError> {
            unreachable!("gate tests never stream")
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
            *self.list_calls.lock().unwrap() += 1;
            if *self.fail_listing.lock().unwrap() {
                return Err(ProviderError::new(
                    "list_failed",
                    "models endpoint unreachable",
                ));
            }
            Ok(self.models.lock().unwrap().clone())
        }
    }

    fn image_input_messages() -> Vec<Message> {
        vec![Message::user_with_blocks(vec![
            ContentBlock::Text("describe".into()),
            ContentBlock::File(FileContentBlock::data("image/png", "AAAA")),
        ])]
    }

    #[test]
    fn image_media_types_collects_deduped_media_types() {
        let messages = vec![
            Message::user_with_blocks(vec![
                ContentBlock::Text("a".into()),
                ContentBlock::File(FileContentBlock::data("image/png", "AAAA")),
            ]),
            Message::user_with_blocks(vec![
                ContentBlock::File(FileContentBlock::data("image/png", "BBBB")),
                ContentBlock::File(FileContentBlock::data("image/jpeg", "CCCC")),
                ContentBlock::File(FileContentBlock::data("audio/mpeg", "DDDD")),
            ]),
        ];
        assert_eq!(
            image_media_types(&messages),
            vec!["image/jpeg".to_string(), "image/png".to_string()]
        );
        assert!(image_media_types(&[Message::user("plain")]).is_empty());
    }

    #[tokio::test]
    async fn image_modality_gate_allows_models_with_image_input() {
        let provider = ModelsProvider::with_models(vec![
            ModelInfo::new(ModelName::new("vision-1"))
                .with_input_modalities(["text".to_string(), "image".to_string()]),
        ]);
        let result = enforce_image_modality_gate(
            &provider,
            &ModelName::new("vision-1"),
            &["image/png".to_string()],
        )
        .await;
        assert!(result.is_ok());
        assert_eq!(*provider.list_calls.lock().unwrap(), 1);
    }

    #[tokio::test]
    async fn image_modality_gate_allows_image_prefix_modalities() {
        let provider = ModelsProvider::with_models(vec![
            ModelInfo::new(ModelName::new("m"))
                .with_input_modalities(["text".to_string(), "image/*".to_string()]),
        ]);
        let result = enforce_image_modality_gate(
            &provider,
            &ModelName::new("m"),
            &["image/png".to_string()],
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn image_modality_gate_rejects_models_without_image_input() {
        let provider = ModelsProvider::with_models(vec![
            ModelInfo::new(ModelName::new("text-1")).with_input_modalities(["text".to_string()]),
        ]);
        let err = enforce_image_modality_gate(
            &provider,
            &ModelName::new("text-1"),
            &["image/png".to_string()],
        )
        .await
        .expect_err("must reject");
        assert_eq!(err.code(), "MODALITY_UNSUPPORTED");
        assert!(err.message().contains("text-1"), "{}", err.message());
        assert!(err.message().contains("image/png"), "{}", err.message());
    }

    #[tokio::test]
    async fn image_modality_gate_allows_listed_model_with_unknown_modalities() {
        // The listing translation layer only fills `input_modalities`
        // when the upstream exposes them (Anthropic /v1/models does
        // not); an empty list means "unknown", never "known
        // unsupported". Regression for the production path: a listed
        // model with empty modalities must be allowed through.
        let provider = ModelsProvider::with_models(vec![ModelInfo::new(ModelName::new("m"))]);
        let result = enforce_image_modality_gate(
            &provider,
            &ModelName::new("m"),
            &["image/png".to_string()],
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn image_modality_gate_allows_unlisted_models() {
        let provider = ModelsProvider::with_models(vec![ModelInfo::new(ModelName::new("other"))]);
        let result = enforce_image_modality_gate(
            &provider,
            &ModelName::new("custom-model"),
            &["image/png".to_string()],
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn image_modality_gate_allows_when_listing_fails() {
        let provider = ModelsProvider::with_models(Vec::new());
        *provider.fail_listing.lock().unwrap() = true;
        let result = enforce_image_modality_gate(
            &provider,
            &ModelName::new("x"),
            &["image/png".to_string()],
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn image_modality_gate_skips_listing_without_image_blocks() {
        let provider = ModelsProvider::with_models(Vec::new());
        let result = enforce_image_modality_gate(&provider, &ModelName::new("x"), &[]).await;
        assert!(result.is_ok());
        assert_eq!(*provider.list_calls.lock().unwrap(), 0);
    }

    #[tokio::test]
    async fn run_turn_stops_with_provider_error_when_model_lacks_image_support() {
        let provider = Arc::new(ModelsProvider::with_models(vec![
            ModelInfo::new(ModelName::new("text-model"))
                .with_input_modalities(["text".to_string()]),
        ]));
        let sink = Arc::new(VecAgentEventSink::new());
        let loop_harness = AgentLoop::new(provider, sink.clone());

        let req = AgentTurnRequest::new(
            make_session(AgentToolRegistry::new()),
            AgentTurnId::new("turn-img"),
            ModelName::new("text-model"),
            image_input_messages(),
        );

        let result = loop_harness.run_turn(req, None).await;
        assert_eq!(result.status(), AgentTurnStatus::Stopped);
        assert_eq!(result.stop_reason(), AgentTurnStopReason::ProviderError);
        assert_eq!(result.diagnostics().len(), 1);
        assert_eq!(sink.events().len(), 1);
        assert!(matches!(sink.events()[0], AgentEvent::ProviderError { .. }));
    }
}
