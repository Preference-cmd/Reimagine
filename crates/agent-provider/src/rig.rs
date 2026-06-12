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
    AgentRequest, AgentResponse, AgentStream, Message, ModelInfo, ModelName, ProviderName,
};

use crate::backend::CompletionBackend;
use crate::config::{AnthropicConfig, OpenAiCompatibleConfig};
use crate::error::ProviderAdapterError;

/// Production backend. `complete` and `list_models` route through
/// the lower-level `rig::Client` HTTP seam (not the
/// `CoreCompletionModel` agent-loop layer). `stream` remains
/// `streaming_unsupported` — V2 work.
#[derive(Debug, Clone)]
pub struct RealRigBackend {
    name: ProviderName,
    kind: RealBackendKind,
    http: reqwest::Client,
}

#[derive(Debug, Clone)]
pub enum RealBackendKind {
    OpenAiCompatible(OpenAiCompatibleConfig),
    Anthropic(AnthropicConfig),
}

impl RealRigBackend {
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
            kind: RealBackendKind::OpenAiCompatible(cfg),
            http,
        }
    }

    /// Construct an Anthropic backend with a default
    /// `reqwest::Client`.
    pub fn anthropic(name: ProviderName, cfg: AnthropicConfig) -> Self {
        Self::anthropic_with_http_client(name, cfg, reqwest::Client::new())
    }

    /// Construct an Anthropic backend with an explicit
    /// `reqwest::Client` (used by tests).
    pub fn anthropic_with_http_client(
        name: ProviderName,
        cfg: AnthropicConfig,
        http: reqwest::Client,
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
    http: reqwest::Client,
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
    http: reqwest::Client,
) -> Arc<dyn CompletionBackend> {
    Arc::new(RealRigBackend::anthropic_with_http_client(name, cfg, http))
}

#[async_trait]
impl CompletionBackend for RealRigBackend {
    async fn complete(
        &self,
        _request: AgentRequest,
    ) -> Result<Result<AgentResponse, ProviderAdapterError>, ProviderAdapterError> {
        Err(ProviderAdapterError::configuration(
            "RealRigBackend::complete is wired in Task 4; use FakeCompletionBackend in tests",
        ))
    }

    async fn stream(
        &self,
        _request: AgentRequest,
    ) -> Result<Box<dyn AgentStream>, ProviderAdapterError> {
        Err(ProviderAdapterError::streaming_unsupported())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderAdapterError> {
        Err(ProviderAdapterError::configuration(
            "RealRigBackend::list_models is wired in Task 4; use FakeCompletionBackend in tests",
        ))
    }
}
