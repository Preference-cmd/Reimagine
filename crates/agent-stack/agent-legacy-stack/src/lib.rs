//! reimagine-agent-legacy-stack — the frozen harness-shaped provider stack.
//!
//! This crate is the relocated home of the former frozen faces of
//! `reimagine-ai-protocol` (harness-shaped DTO translation: `translation/
//! {request, response, streaming, params, tools, listing, files}`,
//! `backend`, `error`, `adapter_config`) and `reimagine-agent-provider`
//! (`BackendProvider`, `ReqwestBackend`). It serves the frozen
//! `reimagine-agent-harness` stack — `AgentProvider` ← `BackendProvider`
//! ← `CompletionBackend` ← the harness-shaped translation — which is
//! still the production provider wiring in `app-host`'s
//! `AgentProviderCatalog`.
//!
//! # Charter
//!
//! - **Frozen**: no new production semantics; every resident carries the
//!   `⚠️ FROZEN` discipline from its former home.
//! - **Dies with Slice 9** (harness dissolution): when app-host migrates
//!   to the kernel face (`reimagine-agent-provider`'s kernel gateways +
//!   `reimagine-context-kernel::ModelGateway`), this crate is deleted.
//! - **Not in the OSS publish set**: the OSS crates
//!   (`reimagine-ai-protocol`, `reimagine-agent-provider`) carry only the
//!   kernel-native face; this crate stays Reimagine-side.
//! - Shared kernel-face vocabulary (`Protocol`, `translation::usage`,
//!   `translation::sse_parser`) is re-exported from
//!   `reimagine-ai-protocol`, not duplicated.

#![deny(unsafe_code)]

pub mod adapter_config;
pub mod backend;
pub mod error;
pub mod translation;

mod backend_provider;
pub mod reqwest_backend;

pub use adapter_config::{
    AnthropicMessagesConfig, OpenAiChatCompletionsConfig, OpenAiResponsesConfig,
};
pub use backend::{CompletionBackend, FakeCompletionBackend, ScriptedBackendStep};
pub use backend_provider::{BackendProvider, ProviderConfig};
pub use error::ProviderAdapterError;
pub use reimagine_ai_protocol::{Protocol, SseEvent, SseParser};
pub use reqwest_backend::ReqwestBackend;

/// V1 adapter for OpenAI-compatible chat completion APIs
/// (delegation over [`BackendProvider`], AC-10).
pub type OpenAiChatCompletionsProvider = BackendProvider<OpenAiChatCompletionsConfig>;
/// V1 adapter for the Anthropic Messages API
/// (delegation over [`BackendProvider`], AC-10).
pub type AnthropicMessagesProvider = BackendProvider<AnthropicMessagesConfig>;
/// V1 adapter for the OpenAI Responses API
/// (delegation over [`BackendProvider`], AC-10).
pub type OpenAiResponsesProvider = BackendProvider<OpenAiResponsesConfig>;
