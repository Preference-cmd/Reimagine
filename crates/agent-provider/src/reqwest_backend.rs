//! Reqwest-backed completion backend.
//!
//! `ReqwestBackend` is the production `CompletionBackend` used by the
//! two concrete adapters. It makes direct HTTP calls via `reqwest::Client`
//! to the OpenAI-compatible and Anthropic APIs. We deliberately do NOT
//! use any agent loop or tool execution framework — Reimagine owns the
//! agent loop, tool execution, and tool policy.
//!
//! The backend is not used in V1 unit tests (which always substitute
//! `FakeCompletionBackend`). It is provided so app-host can construct
//! the production provider with `build_provider(config)` and the
//! adapters will route to a working real backend.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use reimagine_agent::{
    AgentRequest, AgentResponse, AgentStream, AgentStreamEvent, ModelInfo, ProviderName,
};
use serde_json::Value;

use crate::backend::CompletionBackend;
use crate::config::{AnthropicConfig, OpenAiCompatibleConfig};
use crate::error::ProviderAdapterError;
use crate::translation;
use crate::translation::sse_parser::SseParser;

/// Maximum number of retries for transient HTTP errors.
const MAX_RETRIES: u32 = 3;
/// Base delay for exponential backoff (doubled each retry).
const RETRY_BASE_DELAY: Duration = Duration::from_millis(500);

/// Production backend. `complete` and `list_models` route through
/// direct reqwest HTTP calls. `stream` remains `streaming_unsupported`
/// — V2 work.
#[derive(Clone)]
pub struct ReqwestBackend {
    name: ProviderName,
    kind: BackendKind,
    http: reqwest::Client,
}

impl std::fmt::Debug for ReqwestBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind_repr = match &self.kind {
            BackendKind::OpenAiCompatible(_) => "OpenAiCompatible(<redacted>)",
            BackendKind::Anthropic(_) => "Anthropic(<redacted>)",
        };
        f.debug_struct("ReqwestBackend")
            .field("name", &self.name)
            .field("kind", &kind_repr)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
pub enum BackendKind {
    OpenAiCompatible(OpenAiCompatibleConfig),
    Anthropic(AnthropicConfig),
}

impl ReqwestBackend {
    /// Construct an OpenAI-compatible backend with a default
    /// `reqwest::Client`.
    pub fn openai_compatible(name: ProviderName, cfg: OpenAiCompatibleConfig) -> Self {
        Self::openai_compatible_with_http_client(name, cfg, reqwest::Client::new())
    }

    /// Construct an OpenAI-compatible backend with an explicit
    /// `reqwest::Client` (used by tests).
    pub fn openai_compatible_with_http_client(
        name: ProviderName,
        cfg: OpenAiCompatibleConfig,
        http: reqwest::Client,
    ) -> Self {
        Self {
            name,
            kind: BackendKind::OpenAiCompatible(cfg),
            http,
        }
    }

    /// Construct an Anthropic backend with a default `reqwest::Client`.
    pub fn anthropic(name: ProviderName, cfg: AnthropicConfig) -> Self {
        Self::anthropic_with_http_client(name, cfg, reqwest::Client::new())
    }

    /// Construct an Anthropic backend with an explicit `reqwest::Client`
    /// (used by tests).
    pub fn anthropic_with_http_client(
        name: ProviderName,
        cfg: AnthropicConfig,
        http: reqwest::Client,
    ) -> Self {
        Self {
            name,
            kind: BackendKind::Anthropic(cfg),
            http,
        }
    }

    fn openai_config(&self) -> Result<&OpenAiCompatibleConfig, ProviderAdapterError> {
        match &self.kind {
            BackendKind::OpenAiCompatible(cfg) => Ok(cfg),
            BackendKind::Anthropic(_) => Err(ProviderAdapterError::configuration(
                "expected OpenAI-compatible backend, got Anthropic",
            )),
        }
    }

    fn anthropic_config(&self) -> Result<&AnthropicConfig, ProviderAdapterError> {
        match &self.kind {
            BackendKind::Anthropic(cfg) => Ok(cfg),
            BackendKind::OpenAiCompatible(_) => Err(ProviderAdapterError::configuration(
                "expected Anthropic backend, got OpenAI-compatible",
            )),
        }
    }

    /// OpenAI-compatible completion. Builds the request body from
    /// the V1 translation functions, POSTs it via reqwest, and parses
    /// the response via `translation::response`.
    async fn run_openai_complete(
        &self,
        request: &AgentRequest,
    ) -> Result<AgentResponse, ProviderAdapterError> {
        let cfg = self.openai_config()?;
        let url = format!("{}/chat/completions", cfg.base_url().trim_end_matches('/'));

        let messages = translation::request::to_openai_messages(request.messages());
        let tools = translation::tools::to_openai_tools(request.tools());
        let body = serde_json::json!({
            "model": request.model().as_str(),
            "messages": messages,
            "tools": tools,
        });

        let resp = self
            .http
            .post(&url)
            .bearer_auth(cfg.api_key())
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;

        let value = response_json_or_error(resp).await?;
        translation::response::from_openai_response(&value)
    }

    /// Anthropic completion. Mirrors `run_openai_complete` but
    /// splits `system` out of the messages array and sets a
    /// `max_tokens` default of 4096 (overridable via
    /// `request.options().get("max_tokens")`).
    async fn run_anthropic_complete(
        &self,
        request: &AgentRequest,
    ) -> Result<AgentResponse, ProviderAdapterError> {
        let cfg = self.anthropic_config()?;
        let base = cfg.base_url().unwrap_or("https://api.anthropic.com");
        let url = format!("{}/v1/messages", base.trim_end_matches('/'));

        let (system, messages) = translation::request::to_anthropic_messages(request.messages());
        let tools = translation::tools::to_anthropic_tools(request.tools());
        let max_tokens = request
            .options()
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .filter(|n| *n > 0)
            .map(|n| n as u32)
            .unwrap_or(4096);
        let mut body = serde_json::json!({
            "model": request.model().as_str(),
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
        });
        if let Some(sys) = system {
            body["system"] = serde_json::json!(sys);
        }

        let resp = self
            .http
            .post(&url)
            .header("x-api-key", cfg.api_key())
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;

        let value = response_json_or_error(resp).await?;
        translation::response::from_anthropic_response(&value)
    }

    /// OpenAI-compatible model listing. Hits `/models` relative to the
    /// configured base URL and stamps the configured provider name on
    /// every entry.
    async fn run_openai_list_models(&self) -> Result<Vec<ModelInfo>, ProviderAdapterError> {
        let cfg = self.openai_config()?;
        let url = format!("{}/models", cfg.base_url().trim_end_matches('/'));

        let resp = self
            .http
            .get(&url)
            .bearer_auth(cfg.api_key())
            .send()
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;

        let value = response_json_or_error(resp).await?;
        let models = translation::listing::from_openai_models(&value)?;
        Ok(models
            .into_iter()
            .map(|m| m.with_provider(self.name.clone()))
            .collect())
    }

    /// Anthropic model listing. Hits `/v1/models` and stamps the
    /// configured provider name on every entry.
    async fn run_anthropic_list_models(&self) -> Result<Vec<ModelInfo>, ProviderAdapterError> {
        let cfg = self.anthropic_config()?;
        let base = cfg.base_url().unwrap_or("https://api.anthropic.com");
        let url = format!("{}/v1/models", base.trim_end_matches('/'));

        let resp = self
            .http
            .get(&url)
            .header("x-api-key", cfg.api_key())
            .header("anthropic-version", "2023-06-01")
            .send()
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;

        let value = response_json_or_error(resp).await?;
        let models = translation::listing::from_anthropic_models(&value)?;
        Ok(models
            .into_iter()
            .map(|m| m.with_provider(self.name.clone()))
            .collect())
    }

    /// OpenAI-compatible streaming. Sends `stream: true` and returns a
    /// `ReqwestSseStream` that yields `AgentStreamEvent` values.
    async fn run_openai_stream(
        &self,
        request: &AgentRequest,
    ) -> Result<Box<dyn AgentStream>, ProviderAdapterError> {
        let cfg = self.openai_config()?;
        let url = format!("{}/chat/completions", cfg.base_url().trim_end_matches('/'));

        let messages = translation::request::to_openai_messages(request.messages());
        let tools = translation::tools::to_openai_tools(request.tools());
        let body = serde_json::json!({
            "model": request.model().as_str(),
            "messages": messages,
            "tools": tools,
            "stream": true,
        });

        let resp = self
            .http
            .post(&url)
            .bearer_auth(cfg.api_key())
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;

        let resp = check_stream_status(resp).await?;
        Ok(Box::new(ReqwestSseStream::new(resp, StreamKind::OpenAi)))
    }

    /// Anthropic streaming. Sends `stream: true` and returns a
    /// `ReqwestSseStream` that yields `AgentStreamEvent` values.
    async fn run_anthropic_stream(
        &self,
        request: &AgentRequest,
    ) -> Result<Box<dyn AgentStream>, ProviderAdapterError> {
        let cfg = self.anthropic_config()?;
        let base = cfg.base_url().unwrap_or("https://api.anthropic.com");
        let url = format!("{}/v1/messages", base.trim_end_matches('/'));

        let (system, messages) = translation::request::to_anthropic_messages(request.messages());
        let tools = translation::tools::to_anthropic_tools(request.tools());
        let max_tokens = request
            .options()
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .filter(|n| *n > 0)
            .map(|n| n as u32)
            .unwrap_or(4096);
        let mut body = serde_json::json!({
            "model": request.model().as_str(),
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
            "stream": true,
        });
        if let Some(sys) = system {
            body["system"] = serde_json::json!(sys);
        }

        let resp = self
            .http
            .post(&url)
            .header("x-api-key", cfg.api_key())
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;

        let resp = check_stream_status(resp).await?;
        Ok(Box::new(ReqwestSseStream::new(
            resp,
            StreamKind::Anthropic,
        )))
    }
}

/// Parse a reqwest response into a `serde_json::Value`, mapping non-2xx
/// status codes to `ProviderAdapterError::Api`.
async fn response_json_or_error(resp: reqwest::Response) -> Result<Value, ProviderAdapterError> {
    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;

    if status.is_success() {
        serde_json::from_str(&text)
            .map_err(|e| ProviderAdapterError::serialization(format!("response json: {e}")))
    } else {
        Err(ProviderAdapterError::api(status.as_u16().to_string(), text))
    }
}

/// Check a streaming response for non-2xx status. Consumes the response
/// on error so the caller gets the error body.
async fn check_stream_status(
    resp: reqwest::Response,
) -> Result<reqwest::Response, ProviderAdapterError> {
    let status = resp.status();
    if status.is_success() {
        Ok(resp)
    } else {
        let text = resp
            .text()
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;
        Err(ProviderAdapterError::api(status.as_u16().to_string(), text))
    }
}

// ------------------------------------------------------------------
// Streaming adapter
// ------------------------------------------------------------------

/// Which provider format the SSE stream carries.
#[derive(Debug, Clone, Copy)]
enum StreamKind {
    OpenAi,
    Anthropic,
}

/// An `AgentStream` backed by a reqwest SSE response.
///
/// Reads chunks from the HTTP response, feeds them through the SSE
/// parser, and yields `AgentStreamEvent` values produced by the
/// provider-specific accumulator.
struct ReqwestSseStream {
    response: reqwest::Response,
    parser: SseParser,
    kind: StreamKind,
    /// Events produced by the current chunk but not yet yielded.
    pending: std::collections::VecDeque<AgentStreamEvent>,
    /// Accumulated text for the OpenAI path.
    openai_text: String,
    /// Accumulated tool calls for the OpenAI path.
    openai_tool_calls: Vec<OpenAiPartialToolCall>,
    /// Accumulated text for the Anthropic path.
    anthropic_text: String,
    /// Accumulated tool calls for the Anthropic path.
    anthropic_tool_calls: Vec<AnthropicPartialToolCall>,
    /// Whether the stream is done.
    done: bool,
}

#[derive(Debug, Default, Clone)]
struct OpenAiPartialToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

#[derive(Debug, Default, Clone)]
struct AnthropicPartialToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl ReqwestSseStream {
    fn new(response: reqwest::Response, kind: StreamKind) -> Self {
        Self {
            response,
            parser: SseParser::new(),
            kind,
            pending: std::collections::VecDeque::new(),
            openai_text: String::new(),
            openai_tool_calls: Vec::new(),
            anthropic_text: String::new(),
            anthropic_tool_calls: Vec::new(),
            done: false,
        }
    }

    /// Process one SSE event, pushing any resulting `AgentStreamEvent`
    /// values into `self.pending`.
    fn process_event(&mut self, event: translation::sse_parser::SseEvent) {
        // OpenAI sends "data: [DONE]" as the terminal event.
        if event.data == "[DONE]" {
            self.done = true;
            return;
        }

        let value: Value = match serde_json::from_str(&event.data) {
            Ok(v) => v,
            Err(_) => return,
        };

        match self.kind {
            StreamKind::OpenAi => self.process_openai_chunk(&value),
            StreamKind::Anthropic => self.process_anthropic_event(&event.event, &value),
        }
    }

    fn process_openai_chunk(&mut self, chunk: &Value) {
        // Extract content delta.
        if let Some(delta_content) = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("delta"))
            .and_then(|d| d.get("content"))
            .and_then(|v| v.as_str())
            && !delta_content.is_empty()
        {
            self.openai_text.push_str(delta_content);
            self.pending
                .push_back(AgentStreamEvent::ContentDelta(delta_content.to_string()));
        }

        // Extract tool call deltas.
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
                while self.openai_tool_calls.len() <= index {
                    self.openai_tool_calls.push(OpenAiPartialToolCall::default());
                }
                let entry = &mut self.openai_tool_calls[index];
                if let Some(id) = call.get("id").and_then(|v| v.as_str()) {
                    entry.id = Some(id.to_string());
                }
                if let Some(name) = call
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                {
                    entry.name = Some(name.to_string());
                }
                if let Some(args) = call
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|v| v.as_str())
                {
                    entry.arguments.push_str(args);
                }
            }
        }

        // Extract finish reason — flush complete tool calls.
        if let Some(finish_reason) = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("finish_reason"))
            .and_then(|v| v.as_str())
        {
            if finish_reason == "tool_calls" {
                self.flush_openai_tool_calls();
            }
        }

        // Extract usage.
        if let Some(usage) = chunk.get("usage") {
            let input = usage.get("prompt_tokens").and_then(|v| v.as_u64());
            let output = usage.get("completion_tokens").and_then(|v| v.as_u64());
            self.pending
                .push_back(AgentStreamEvent::Usage(reimagine_agent::Usage::new(
                    input, output,
                )));
        }
    }

    fn flush_openai_tool_calls(&mut self) {
        for partial in self.openai_tool_calls.drain(..) {
            if let (Some(id), Some(name)) = (partial.id, partial.name) {
                let arguments = if partial.arguments.is_empty() {
                    Value::Null
                } else {
                    serde_json::from_str(&partial.arguments).unwrap_or(Value::Null)
                };
                self.pending
                    .push_back(AgentStreamEvent::ToolCall(reimagine_agent::ToolCall::new(
                        reimagine_agent::ToolCallId::new(id),
                        name,
                        arguments,
                    )));
            }
        }
    }

    fn process_anthropic_event(&mut self, event_type: &Option<String>, data: &Value) {
        let event_type = match event_type.as_deref() {
            Some(t) => t,
            None => return,
        };

        match event_type {
            "content_block_start" => {
                if let Some(index) = data.get("index").and_then(|v| v.as_u64()) {
                    let index = index as usize;
                    while self.anthropic_tool_calls.len() <= index {
                        self.anthropic_tool_calls
                            .push(AnthropicPartialToolCall::default());
                    }
                    if let Some(block) = data.get("content_block") {
                        if block.get("type").and_then(|v| v.as_str()) == Some("tool_use") {
                            if let Some(id) = block.get("id").and_then(|v| v.as_str()) {
                                self.anthropic_tool_calls[index].id = Some(id.to_string());
                            }
                            if let Some(name) = block.get("name").and_then(|v| v.as_str()) {
                                self.anthropic_tool_calls[index].name = Some(name.to_string());
                            }
                        }
                    }
                }
            }
            "content_block_delta" => {
                if let Some(index) = data.get("index").and_then(|v| v.as_u64()) {
                    let index = index as usize;
                    if let Some(delta) = data.get("delta") {
                        match delta.get("type").and_then(|v| v.as_str()) {
                            Some("text_delta") => {
                                if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                                    self.anthropic_text.push_str(text);
                                    self.pending
                                        .push_back(AgentStreamEvent::ContentDelta(text.to_string()));
                                }
                            }
                            Some("input_json_delta") => {
                                if let Some(partial) =
                                    delta.get("partial_json").and_then(|v| v.as_str())
                                {
                                    while self.anthropic_tool_calls.len() <= index {
                                        self.anthropic_tool_calls
                                            .push(AnthropicPartialToolCall::default());
                                    }
                                    self.anthropic_tool_calls[index]
                                        .arguments
                                        .push_str(partial);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            "content_block_stop" => {
                if let Some(index) = data.get("index").and_then(|v| v.as_u64()) {
                    let index = index as usize;
                    if let Some(partial) = self.anthropic_tool_calls.get_mut(index) {
                        if let (Some(id), Some(name)) =
                            (partial.id.clone(), partial.name.clone())
                        {
                            let arguments = if partial.arguments.is_empty() {
                                Value::Null
                            } else {
                                serde_json::from_str(&partial.arguments).unwrap_or(Value::Null)
                            };
                            *partial = AnthropicPartialToolCall::default();
                            self.pending
                                .push_back(AgentStreamEvent::ToolCall(
                                    reimagine_agent::ToolCall::new(
                                        reimagine_agent::ToolCallId::new(id),
                                        name,
                                        arguments,
                                    ),
                                ));
                        }
                    }
                }
            }
            "message_delta" => {
                if let Some(delta) = data.get("delta") {
                    if let Some(reason) = delta.get("stop_reason").and_then(|v| v.as_str()) {
                        // Store stop_reason; will be emitted on message_stop.
                        let _ = reason;
                    }
                }
                if let Some(usage) = data.get("usage") {
                    let input = usage.get("input_tokens").and_then(|v| v.as_u64());
                    let output = usage.get("output_tokens").and_then(|v| v.as_u64());
                    self.pending
                        .push_back(AgentStreamEvent::Usage(reimagine_agent::Usage::new(
                            input, output,
                        )));
                }
            }
            "message_stop" => {
                self.done = true;
                self.pending.push_back(AgentStreamEvent::Done {
                    stop_reason: None,
                });
            }
            _ => {}
        }
    }
}

#[async_trait]
impl AgentStream for ReqwestSseStream {
    async fn next_event(&mut self) -> Option<AgentStreamEvent> {
        loop {
            // Return any pending events from the last chunk.
            if let Some(event) = self.pending.pop_front() {
                return Some(event);
            }

            if self.done {
                return None;
            }

            // Read next chunk from the HTTP response.
            let chunk = match self.response.chunk().await {
                Ok(Some(c)) => c,
                Ok(None) => {
                    // Stream ended. Flush any remaining SSE data.
                    if let Some(final_event) = self.parser.flush() {
                        self.process_event(final_event);
                        if let Some(event) = self.pending.pop_front() {
                            return Some(event);
                        }
                    }
                    self.done = true;
                    return None;
                }
                Err(e) => {
                    eprintln!("agent stream read error: {e}");
                    self.done = true;
                    return None;
                }
            };

            // Convert bytes to string and feed to SSE parser.
            let text = String::from_utf8_lossy(&chunk);
            let events = self.parser.feed(&text);

            for event in events {
                self.process_event(event);
            }
        }
    }
}

pub fn arc_real_backend(
    name: ProviderName,
    cfg: OpenAiCompatibleConfig,
) -> Arc<dyn CompletionBackend> {
    Arc::new(ReqwestBackend::openai_compatible(name, cfg))
}

pub fn arc_real_backend_with_http_client(
    name: ProviderName,
    cfg: OpenAiCompatibleConfig,
    http: reqwest::Client,
) -> Arc<dyn CompletionBackend> {
    Arc::new(ReqwestBackend::openai_compatible_with_http_client(
        name, cfg, http,
    ))
}

pub fn arc_real_anthropic_backend(
    name: ProviderName,
    cfg: AnthropicConfig,
) -> Arc<dyn CompletionBackend> {
    Arc::new(ReqwestBackend::anthropic(name, cfg))
}

pub fn arc_real_anthropic_backend_with_http_client(
    name: ProviderName,
    cfg: AnthropicConfig,
    http: reqwest::Client,
) -> Arc<dyn CompletionBackend> {
    Arc::new(ReqwestBackend::anthropic_with_http_client(name, cfg, http))
}

#[async_trait]
impl CompletionBackend for ReqwestBackend {
    async fn complete(&self, request: AgentRequest) -> Result<AgentResponse, ProviderAdapterError> {
        let mut last_err: Option<ProviderAdapterError> = None;

        for attempt in 0..=MAX_RETRIES {
            if attempt > 0 {
                let delay = RETRY_BASE_DELAY * 2u32.pow(attempt - 1);
                tokio::time::sleep(delay).await;
            }

            let result = match &self.kind {
                BackendKind::OpenAiCompatible(_) => self.run_openai_complete(&request).await,
                BackendKind::Anthropic(_) => self.run_anthropic_complete(&request).await,
            };

            match result {
                Ok(resp) => return Ok(resp),
                Err(err) => {
                    if err.is_retryable() && attempt < MAX_RETRIES {
                        last_err = Some(err);
                        continue;
                    }
                    return Err(err);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| {
            ProviderAdapterError::transport("retry exhausted".to_string())
        }))
    }

    async fn stream(
        &self,
        request: AgentRequest,
    ) -> Result<Box<dyn AgentStream>, ProviderAdapterError> {
        // Streaming does not retry — the caller owns the stream lifecycle.
        // Retry is the caller's responsibility for streaming.
        match &self.kind {
            BackendKind::OpenAiCompatible(_) => self.run_openai_stream(&request).await,
            BackendKind::Anthropic(_) => self.run_anthropic_stream(&request).await,
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderAdapterError> {
        let mut last_err: Option<ProviderAdapterError> = None;

        for attempt in 0..=MAX_RETRIES {
            if attempt > 0 {
                let delay = RETRY_BASE_DELAY * 2u32.pow(attempt - 1);
                tokio::time::sleep(delay).await;
            }

            let result = match &self.kind {
                BackendKind::OpenAiCompatible(_) => self.run_openai_list_models().await,
                BackendKind::Anthropic(_) => self.run_anthropic_list_models().await,
            };

            match result {
                Ok(models) => return Ok(models),
                Err(err) => {
                    if err.is_retryable() && attempt < MAX_RETRIES {
                        last_err = Some(err);
                        continue;
                    }
                    return Err(err);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| {
            ProviderAdapterError::transport("retry exhausted".to_string())
        }))
    }
}
