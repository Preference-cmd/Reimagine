//! Concrete provider adapters for `reimagine_agent::AgentProvider`.
//!
//! V1 supports Rig-backed OpenAI-compatible and Anthropic providers behind
//! a small `CompletionBackend` seam so tests avoid live network calls.
//!
//! See `docs/architecture/modules/agent-provider.md` for the design source.

#![deny(unsafe_code)]

mod anthropic;
mod backend;
mod config;
mod error;
mod openai_compatible;
mod rig;
pub mod translation;

pub use anthropic::AnthropicProvider;
pub use backend::{CompletionBackend, FakeCompletionBackend, ScriptedBackendStep};
pub use config::{
    AgentProviderConfigDocument, AnthropicConfig, OpenAiCompatibleConfig, ProviderConfig,
    ProviderKind,
};
pub use error::ProviderAdapterError;
pub use openai_compatible::OpenAiCompatibleProvider;
pub use rig::{
    RealRigBackend, arc_real_anthropic_backend, arc_real_anthropic_backend_with_http_client,
    arc_real_backend, arc_real_backend_with_http_client,
};

use std::sync::Arc;

use reimagine_agent::{AgentProvider, ProviderName};

/// Build an `Arc<dyn AgentProvider>` from a `ProviderConfig`. The kind
/// determines which concrete adapter is constructed. Missing inner
/// config is rejected with `ProviderAdapterError::MissingConfig`.
pub fn build_provider(
    config: ProviderConfig,
) -> Result<Arc<dyn AgentProvider>, ProviderAdapterError> {
    match config.kind() {
        ProviderKind::OpenAiCompatible => {
            let cfg =
                config
                    .openai_compatible()
                    .ok_or_else(|| ProviderAdapterError::MissingConfig {
                        provider: config.name().to_string(),
                        kind: ProviderKind::OpenAiCompatible,
                    })?;
            let provider = OpenAiCompatibleProvider::new(
                ProviderName::new(config.name().to_string()),
                cfg.clone(),
            );
            Ok(Arc::new(provider))
        }
        ProviderKind::Anthropic => {
            let cfg = config
                .anthropic()
                .ok_or_else(|| ProviderAdapterError::MissingConfig {
                    provider: config.name().to_string(),
                    kind: ProviderKind::Anthropic,
                })?;
            let provider =
                AnthropicProvider::new(ProviderName::new(config.name().to_string()), cfg.clone());
            Ok(Arc::new(provider))
        }
    }
}

/// Helper to extract a `ProviderName` from a config so the caller can
/// register it under its identity in `AgentProviderCatalog`.
pub fn provider_name(config: &ProviderConfig) -> ProviderName {
    ProviderName::new(config.name().to_string())
}
