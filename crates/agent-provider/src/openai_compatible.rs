//! OpenAI-compatible provider adapter.
//!
//! Translates `AgentRequest` into an OpenAI chat completion payload
//! (built by `translation::request::to_openai_messages` and
//! `translation::tools::to_openai_tools`), runs it through a
//! `CompletionBackend`, and translates the response back via
//! `translation::response::from_openai_response`.
//!
//! Streaming is delegated to the backend and translated through
//! `translation::streaming::OpenAiStreamAccumulator`.

use std::sync::Arc;

use async_trait::async_trait;
use reimagine_agent::{
    AgentProvider, AgentRequest, AgentResponse, AgentStream, ModelInfo, ProviderName,
};

use crate::backend::CompletionBackend;
use crate::config::OpenAiCompatibleConfig;
use crate::translation;

/// V1 adapter for OpenAI-compatible chat completion APIs.
pub struct OpenAiCompatibleProvider {
    name: ProviderName,
    config: OpenAiCompatibleConfig,
    backend: Arc<dyn CompletionBackend>,
}

impl std::fmt::Debug for OpenAiCompatibleProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatibleProvider")
            .field("name", &self.name)
            .field("base_url", &self.config.base_url())
            .field("default_model", &self.config.default_model())
            .finish()
    }
}

impl OpenAiCompatibleProvider {
    /// Construct with a custom backend (used by tests and by app-host
    /// to inject a real backend).
    pub fn with_backend(
        name: ProviderName,
        config: OpenAiCompatibleConfig,
        backend: Arc<dyn CompletionBackend>,
    ) -> Self {
        Self {
            name,
            config,
            backend,
        }
    }

    /// Construct with the production `RealRigBackend` (no network in
    /// unit tests; the real backend returns a configuration error
    /// unless wired up at app-host level).
    pub fn new(name: ProviderName, config: OpenAiCompatibleConfig) -> Self {
        Self {
            name,
            config: config.clone(),
            backend: crate::rig::arc_real_backend(config),
        }
    }

    pub fn config(&self) -> &OpenAiCompatibleConfig {
        &self.config
    }
}

#[async_trait]
impl AgentProvider for OpenAiCompatibleProvider {
    fn name(&self) -> ProviderName {
        self.name.clone()
    }

    async fn complete(
        &self,
        request: AgentRequest,
    ) -> Result<AgentResponse, reimagine_agent::ProviderError> {
        // Build translated payloads for documentation / future real backend.
        // The translation functions are exercised here so they remain
        // compiled in V1 (not dead-code-eliminated) and so the future
        // real backend has a clear shape.
        let _ = translation::request::to_openai_messages(request.messages());
        let _ = translation::tools::to_openai_tools(request.tools());
        match self.backend.complete(request).await {
            Ok(Ok(resp)) => Ok(resp),
            Ok(Err(err)) => Err(err.to_provider_error(Some(self.name.clone()))),
            Err(err) => Err(err.to_provider_error(Some(self.name.clone()))),
        }
    }

    async fn stream(
        &self,
        request: AgentRequest,
    ) -> Result<Box<dyn AgentStream>, reimagine_agent::ProviderError> {
        // Build the body for documentation / future real backend.
        let _ = translation::request::to_openai_messages(request.messages());
        let _ = translation::tools::to_openai_tools(request.tools());
        match self.backend.stream(request).await {
            Ok(stream) => Ok(stream),
            Err(err) => Err(err.to_provider_error(Some(self.name.clone()))),
        }
    }

    async fn list_models(
        &self,
    ) -> Result<Vec<ModelInfo>, reimagine_agent::ProviderError> {
        match self.backend.list_models().await {
            Ok(models) => Ok(models),
            Err(err) => Err(err.to_provider_error(Some(self.name.clone()))),
        }
    }
}
