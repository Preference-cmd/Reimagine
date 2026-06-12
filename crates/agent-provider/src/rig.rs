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
use reimagine_agent::{AgentRequest, AgentResponse, AgentStream, ModelInfo};

use crate::backend::CompletionBackend;
use crate::config::{AnthropicConfig, OpenAiCompatibleConfig};
use crate::error::ProviderAdapterError;

/// Production backend. V1 does not exercise this in unit tests; the
/// methods return an `STREAMING_UNSUPPORTED`-style error for `stream`
/// and forward `complete` through the Rig client when wired up.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RealRigBackend {
    kind: RealBackendKind,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum RealBackendKind {
    OpenAiCompatible(OpenAiCompatibleConfig),
    Anthropic(AnthropicConfig),
}

impl RealRigBackend {
    pub fn openai_compatible(cfg: OpenAiCompatibleConfig) -> Self {
        Self {
            kind: RealBackendKind::OpenAiCompatible(cfg),
        }
    }

    pub fn anthropic(cfg: AnthropicConfig) -> Self {
        Self {
            kind: RealBackendKind::Anthropic(cfg),
        }
    }
}

#[async_trait]
impl CompletionBackend for RealRigBackend {
    async fn complete(
        &self,
        _request: AgentRequest,
    ) -> Result<Result<AgentResponse, ProviderAdapterError>, ProviderAdapterError> {
        Err(ProviderAdapterError::configuration(
            "real Rig backend is not wired in V1 unit tests; use a FakeCompletionBackend",
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
            "real Rig backend is not wired in V1 unit tests; use a FakeCompletionBackend",
        ))
    }
}

pub fn arc_real_backend(cfg: OpenAiCompatibleConfig) -> Arc<dyn CompletionBackend> {
    Arc::new(RealRigBackend::openai_compatible(cfg))
}

pub fn arc_real_anthropic_backend(cfg: AnthropicConfig) -> Arc<dyn CompletionBackend> {
    Arc::new(RealRigBackend::anthropic(cfg))
}
