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

use async_trait::async_trait;
use reimagine_agent::{AgentRequest, AgentResponse, AgentStream, ModelInfo, ProviderName};
use serde_json::Value;

use crate::backend::CompletionBackend;
use crate::config::{AnthropicConfig, OpenAiCompatibleConfig};
use crate::error::ProviderAdapterError;
use crate::translation;

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
        match &self.kind {
            BackendKind::OpenAiCompatible(_) => self.run_openai_complete(&request).await,
            BackendKind::Anthropic(_) => self.run_anthropic_complete(&request).await,
        }
    }

    async fn stream(
        &self,
        _request: AgentRequest,
    ) -> Result<Box<dyn AgentStream>, ProviderAdapterError> {
        Err(ProviderAdapterError::streaming_unsupported())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderAdapterError> {
        match &self.kind {
            BackendKind::OpenAiCompatible(_) => self.run_openai_list_models().await,
            BackendKind::Anthropic(_) => self.run_anthropic_list_models().await,
        }
    }
}
