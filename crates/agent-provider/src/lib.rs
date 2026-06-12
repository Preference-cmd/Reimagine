//! Concrete provider adapters for `reimagine_agent::AgentProvider`.
//!
//! V1 supports Rig-backed OpenAI-compatible and Anthropic providers behind
//! a small `CompletionBackend` seam so tests avoid live network calls.
//!
//! See `docs/architecture/modules/agent-provider.md` for the design source.

#![deny(unsafe_code)]

mod backend;
mod config;
mod error;
mod openai_compatible;
mod anthropic;
mod rig;
mod translation;

pub use backend::{CompletionBackend, FakeCompletionBackend, ScriptedBackendStep};
pub use config::{
    AgentProviderConfigDocument, AnthropicConfig, OpenAiCompatibleConfig, ProviderConfig,
    ProviderKind,
};
pub use error::ProviderAdapterError;
