//! Real Rig-backed completion backend.
//!
//! `RealRigBackend` is the production `CompletionBackend` used by the
//! two concrete adapters. It calls into the Rig OpenAI-compatible and
//! Anthropic clients. We deliberately do NOT use Rig's `Agent` loop —
//! Reimagine owns the agent loop, tool execution, and tool policy.
//!
//! The backend is not used in V1 unit tests (which always substitute
//! `FakeCompletionBackend`). It is provided so app-host can construct
//! the production provider with `build_provider(config)` and the
//! adapters will route to a working real backend.

use std::sync::Arc;

use async_trait::async_trait;
use reimagine_agent::{
    AgentRequest, AgentResponse, AgentStream, ModelInfo, ProviderName,
};
use rig::http_client::HttpClientExt;

use crate::backend::CompletionBackend;
use crate::config::{AnthropicConfig, OpenAiCompatibleConfig};
use crate::error::ProviderAdapterError;
use crate::translation;

/// Production backend. `complete` and `list_models` route through
/// the lower-level `rig::Client` HTTP seam (not the
/// `CoreCompletionModel` agent-loop layer). `stream` remains
/// `streaming_unsupported` — V2 work.
#[derive(Debug, Clone)]
pub struct RealRigBackend {
    name: ProviderName,
    kind: RealBackendKind,
    http: rig::http_client::ReqwestClient,
}

#[derive(Debug, Clone)]
pub enum RealBackendKind {
    OpenAiCompatible(OpenAiCompatibleConfig),
    Anthropic(AnthropicConfig),
}

impl RealRigBackend {
    /// Construct an OpenAI-compatible backend with a default
    /// `rig::http_client::ReqwestClient`.
    pub fn openai_compatible(name: ProviderName, cfg: OpenAiCompatibleConfig) -> Self {
        Self::openai_compatible_with_http_client(name, cfg, rig::http_client::ReqwestClient::new())
    }

    /// Construct an OpenAI-compatible backend with an explicit
    /// `rig::http_client::ReqwestClient` (used by tests).
    pub fn openai_compatible_with_http_client(
        name: ProviderName,
        cfg: OpenAiCompatibleConfig,
        http: rig::http_client::ReqwestClient,
    ) -> Self {
        Self {
            name,
            kind: RealBackendKind::OpenAiCompatible(cfg),
            http,
        }
    }

    /// Construct an Anthropic backend with a default
    /// `rig::http_client::ReqwestClient`.
    pub fn anthropic(name: ProviderName, cfg: AnthropicConfig) -> Self {
        Self::anthropic_with_http_client(name, cfg, rig::http_client::ReqwestClient::new())
    }

    /// Construct an Anthropic backend with an explicit
    /// `rig::http_client::ReqwestClient` (used by tests).
    pub fn anthropic_with_http_client(
        name: ProviderName,
        cfg: AnthropicConfig,
        http: rig::http_client::ReqwestClient,
    ) -> Self {
        Self {
            name,
            kind: RealBackendKind::Anthropic(cfg),
            http,
        }
    }

    fn openai_config(&self) -> &OpenAiCompatibleConfig {
        match &self.kind {
            RealBackendKind::OpenAiCompatible(cfg) => cfg,
            RealBackendKind::Anthropic(_) => {
                panic!("RealRigBackend::openai_config called on Anthropic backend")
            }
        }
    }

    fn anthropic_config(&self) -> &AnthropicConfig {
        match &self.kind {
            RealBackendKind::Anthropic(cfg) => cfg,
            RealBackendKind::OpenAiCompatible(_) => {
                panic!("RealRigBackend::anthropic_config called on OpenAI backend")
            }
        }
    }

    /// OpenAI-compatible completion. Builds the request body from
    /// the V1 translation functions, POSTs it via `rig::Client`,
    /// and parses the response via `translation::response`.
    async fn run_openai_complete(
        &self,
        request: &AgentRequest,
    ) -> Result<Result<AgentResponse, ProviderAdapterError>, ProviderAdapterError> {
        let cfg = self.openai_config();
        let rig_client = rig::providers::openai::Client::<rig::http_client::ReqwestClient>::builder()
            .api_key(cfg.api_key())
            .base_url(cfg.base_url())
            .http_client(self.http.clone())
            .build()
            .map_err(|e| ProviderAdapterError::configuration(format!("openai client: {e}")))?;

        let messages = translation::request::to_openai_messages(request.messages());
        let tools = translation::tools::to_openai_tools(request.tools());
        let body = serde_json::json!({
            "model": request.model().as_str(),
            "messages": messages,
            "tools": tools,
        });
        let bytes = serde_json::to_vec(&body)
            .map_err(|e| ProviderAdapterError::serialization(format!("body encode: {e}")))?;

        let req = rig_client
            .post("/chat/completions")
            .map_err(|e| ProviderAdapterError::configuration(format!("openai post: {e}")))?
            .body(bytes)
            .map_err(|e| ProviderAdapterError::configuration(format!("openai body: {e}")))?;
        let resp = rig_client
            .send(req)
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16().to_string();
            let text = rig::http_client::text(resp)
                .await
                .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;
            return Ok(Err(ProviderAdapterError::api(status, text)));
        }

        let text = rig::http_client::text(resp)
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;
        let value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| ProviderAdapterError::serialization(format!("response json: {e}")))?;
        Ok(translation::response::from_openai_response(&value))
    }

    /// Anthropic completion. Mirrors `run_openai_complete` but
    /// splits `system` out of the messages array and sets a
    /// `max_tokens` default of 4096 (overridable via
    /// `request.options().get("max_tokens")`).
    async fn run_anthropic_complete(
        &self,
        request: &AgentRequest,
    ) -> Result<Result<AgentResponse, ProviderAdapterError>, ProviderAdapterError> {
        let cfg = self.anthropic_config();
        let rig_client = rig::providers::anthropic::Client::<rig::http_client::ReqwestClient>::builder()
            .api_key(cfg.api_key())
            .http_client(self.http.clone())
            .build()
            .map_err(|e| ProviderAdapterError::configuration(format!("anthropic client: {e}")))?;

        let (system, messages) =
            translation::request::to_anthropic_messages(request.messages());
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
        let bytes = serde_json::to_vec(&body)
            .map_err(|e| ProviderAdapterError::serialization(format!("body encode: {e}")))?;

        let req = rig_client
            .post("/v1/messages")
            .map_err(|e| ProviderAdapterError::configuration(format!("anthropic post: {e}")))?
            .body(bytes)
            .map_err(|e| ProviderAdapterError::configuration(format!("anthropic body: {e}")))?;
        let resp = rig_client
            .send(req)
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16().to_string();
            let text = rig::http_client::text(resp)
                .await
                .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;
            return Ok(Err(ProviderAdapterError::api(status, text)));
        }

        let text = rig::http_client::text(resp)
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;
        let value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| ProviderAdapterError::serialization(format!("response json: {e}")))?;
        Ok(translation::response::from_anthropic_response(&value))
    }

    /// OpenAI-compatible model listing. Hits `/v1/models` and
    /// stamps the configured provider name on every entry.
    async fn run_openai_list_models(
        &self,
    ) -> Result<Vec<ModelInfo>, ProviderAdapterError> {
        let cfg = self.openai_config();
        let rig_client = rig::providers::openai::Client::<rig::http_client::ReqwestClient>::builder()
            .api_key(cfg.api_key())
            .base_url(cfg.base_url())
            .http_client(self.http.clone())
            .build()
            .map_err(|e| ProviderAdapterError::configuration(format!("openai client: {e}")))?;

        let req = rig_client
            .get("/v1/models")
            .map_err(|e| ProviderAdapterError::configuration(format!("openai get: {e}")))?
            .body(rig::http_client::NoBody)
            .map_err(|e| ProviderAdapterError::configuration(format!("openai body: {e}")))?;
        let resp = rig_client
            .send(req)
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16().to_string();
            let text = rig::http_client::text(resp)
                .await
                .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;
            return Err(ProviderAdapterError::api(status, text));
        }

        let text = rig::http_client::text(resp)
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;
        let value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| ProviderAdapterError::serialization(format!("response json: {e}")))?;
        let models = translation::listing::from_openai_models(&value)?;
        Ok(models
            .into_iter()
            .map(|m| m.with_provider(self.name.clone()))
            .collect())
    }

    /// Anthropic model listing. Hits `/v1/models` and stamps the
    /// configured provider name on every entry.
    async fn run_anthropic_list_models(
        &self,
    ) -> Result<Vec<ModelInfo>, ProviderAdapterError> {
        let cfg = self.anthropic_config();
        let rig_client = rig::providers::anthropic::Client::<rig::http_client::ReqwestClient>::builder()
            .api_key(cfg.api_key())
            .http_client(self.http.clone())
            .build()
            .map_err(|e| ProviderAdapterError::configuration(format!("anthropic client: {e}")))?;

        let req = rig_client
            .get("/v1/models")
            .map_err(|e| ProviderAdapterError::configuration(format!("anthropic get: {e}")))?
            .body(rig::http_client::NoBody)
            .map_err(|e| ProviderAdapterError::configuration(format!("anthropic body: {e}")))?;
        let resp = rig_client
            .send(req)
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16().to_string();
            let text = rig::http_client::text(resp)
                .await
                .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;
            return Err(ProviderAdapterError::api(status, text));
        }

        let text = rig::http_client::text(resp)
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;
        let value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| ProviderAdapterError::serialization(format!("response json: {e}")))?;
        let models = translation::listing::from_anthropic_models(&value)?;
        Ok(models
            .into_iter()
            .map(|m| m.with_provider(self.name.clone()))
            .collect())
    }
}

pub fn arc_real_backend(
    name: ProviderName,
    cfg: OpenAiCompatibleConfig,
) -> Arc<dyn CompletionBackend> {
    Arc::new(RealRigBackend::openai_compatible(name, cfg))
}

pub fn arc_real_backend_with_http_client(
    name: ProviderName,
    cfg: OpenAiCompatibleConfig,
    http: rig::http_client::ReqwestClient,
) -> Arc<dyn CompletionBackend> {
    Arc::new(RealRigBackend::openai_compatible_with_http_client(name, cfg, http))
}

pub fn arc_real_anthropic_backend(
    name: ProviderName,
    cfg: AnthropicConfig,
) -> Arc<dyn CompletionBackend> {
    Arc::new(RealRigBackend::anthropic(name, cfg))
}

pub fn arc_real_anthropic_backend_with_http_client(
    name: ProviderName,
    cfg: AnthropicConfig,
    http: rig::http_client::ReqwestClient,
) -> Arc<dyn CompletionBackend> {
    Arc::new(RealRigBackend::anthropic_with_http_client(name, cfg, http))
}

#[async_trait]
impl CompletionBackend for RealRigBackend {
    async fn complete(
        &self,
        request: AgentRequest,
    ) -> Result<Result<AgentResponse, ProviderAdapterError>, ProviderAdapterError> {
        match &self.kind {
            RealBackendKind::OpenAiCompatible(_) => self.run_openai_complete(&request).await,
            RealBackendKind::Anthropic(_) => self.run_anthropic_complete(&request).await,
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
            RealBackendKind::OpenAiCompatible(_) => self.run_openai_list_models().await,
            RealBackendKind::Anthropic(_) => self.run_anthropic_list_models().await,
        }
    }
}
