//! Concrete provider adapters for `reimagine_agent::AgentProvider`.
//!
//! V1 supports reqwest-backed OpenAI-compatible and Anthropic providers
//! behind a small `CompletionBackend` seam so tests avoid live network
//! calls.
//!
//! See `docs/architecture/modules/agent-provider.md` for the design source.

#![deny(unsafe_code)]

mod anthropic;
mod backend;
mod config;
mod error;
mod openai_compatible;
mod openai_responses;
pub mod reqwest_backend;
pub mod translation;

pub use anthropic::AnthropicMessagesProvider;
pub use backend::{CompletionBackend, FakeCompletionBackend, ScriptedBackendStep};
pub use config::{
    AgentProviderConfigDocument, AnthropicMessagesConfig, OpenAiChatCompletionsConfig,
    OpenAiResponsesConfig, Protocol, ProviderConfig,
};
pub use error::ProviderAdapterError;
pub use openai_compatible::OpenAiChatCompletionsProvider;
pub use openai_responses::OpenAiResponsesProvider;
pub use reqwest_backend::{
    ReqwestBackend, arc_real_anthropic_messages_backend,
    arc_real_anthropic_messages_backend_with_http_client, arc_real_openai_chat_completions_backend,
    arc_real_openai_chat_completions_backend_with_http_client, arc_real_openai_responses_backend,
    arc_real_openai_responses_backend_with_http_client,
};
pub use translation::sse_parser::{SseEvent, SseParser};

use std::sync::Arc;

use reimagine_agent::{AgentProvider, ProviderName};

/// Build an `Arc<dyn AgentProvider>` from a `ProviderConfig`. The protocol
/// determines which concrete adapter is constructed. Missing inner
/// config is rejected with `ProviderAdapterError::MissingConfig`.
pub fn build_provider(
    config: ProviderConfig,
) -> Result<Arc<dyn AgentProvider>, ProviderAdapterError> {
    match config.protocol() {
        Protocol::OpenAiChatCompletions => {
            let cfg = config.openai_chat_completions().ok_or_else(|| {
                ProviderAdapterError::MissingConfig {
                    provider: config.name().to_string(),
                    protocol: Protocol::OpenAiChatCompletions,
                }
            })?;
            let provider = OpenAiChatCompletionsProvider::new(
                ProviderName::new(config.name().to_string()),
                cfg.clone(),
            );
            Ok(Arc::new(provider))
        }
        Protocol::AnthropicMessages => {
            let cfg =
                config
                    .anthropic_messages()
                    .ok_or_else(|| ProviderAdapterError::MissingConfig {
                        provider: config.name().to_string(),
                        protocol: Protocol::AnthropicMessages,
                    })?;
            let provider = AnthropicMessagesProvider::new(
                ProviderName::new(config.name().to_string()),
                cfg.clone(),
            );
            Ok(Arc::new(provider))
        }
        Protocol::OpenAiResponses => {
            let cfg =
                config
                    .openai_responses()
                    .ok_or_else(|| ProviderAdapterError::MissingConfig {
                        provider: config.name().to_string(),
                        protocol: Protocol::OpenAiResponses,
                    })?;
            let provider = OpenAiResponsesProvider::new(
                ProviderName::new(config.name().to_string()),
                cfg.clone(),
            );
            Ok(Arc::new(provider))
        }
    }
}

/// Helper to extract a `ProviderName` from a config so the caller can
/// register it under its identity in `AgentProviderCatalog`.
pub fn provider_name(config: &ProviderConfig) -> ProviderName {
    ProviderName::new(config.name().to_string())
}
