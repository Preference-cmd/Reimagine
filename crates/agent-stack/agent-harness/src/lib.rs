//! Reimagine-owned agent harness domain.
//!
//! This crate defines the workspace-scoped Agent session, the host-neutral
//! tool abstraction, the tool policy and registry, the Reimagine-owned
//! provider boundary, the LLM model catalog service, and the agent event
//! model. It must not depend on Tauri, Axum, app-host, runtime,
//! model-manager, inference-backends, Rig, or Cersei.
//!
//! See `docs/architecture/modules/agent.md` for the architecture source of
//! truth.

#![deny(unsafe_code)]

mod context;
mod context_manager;
mod error;
mod event;
mod ids;
mod mode;
mod model_catalog;
mod permissions;
mod policy;
mod provider;
mod registry;
mod session;
mod tool;
mod turn;
mod validation;

mod event_adapter;
mod r#loop;

pub use context::{Actor, ToolContext};
pub use context_manager::{
    BudgetSnapshot, CompactionRecord, ContextConfig, ContextManager, HeuristicEstimator,
    TokenEstimator,
};
pub use error::{ProviderError, ToolError, ToolErrorCode};
pub use event::AgentEvent;
/// Domain-event projection adapter for `AgentEvent`.
///
/// No host consumer yet; planned wiring via the core domain-event
/// stream (agent-stack cleanup roadmap AC-18). Tests keep the
/// projection contract honest.
#[doc(hidden)]
pub use event_adapter::AgentDomainEventAdapter;
pub use ids::{AgentSessionId, ModelName, ProviderName, ToolName, WorkspaceScope};
pub use r#loop::{AgentEventSink, AgentLoop, VecAgentEventSink};
pub use mode::AgentMode;
pub use model_catalog::{LlmCatalogError, LlmModelCatalog, ProviderCatalogEntry};
pub use permissions::{PermissionSet, ToolPermission, ToolRiskLevel};
pub use policy::{PolicyDecision, PolicyDenialReason, ToolPolicy};
pub use provider::{
    AgentProvider, AgentRequest, AgentResponse, AgentStream, AgentStreamEvent, AgentToolDefinition,
    ContentBlock, FileContentBlock, FileSource, Message, ModelCapability, ModelCost, ModelInfo,
    ToolCall, ToolCallId, Usage,
};
pub use registry::{AgentToolRegistry, ToolRegistryError};
pub use session::AgentSession;
pub use tool::{AgentTool, ToolInput, ToolOutput, ToolResult, ToolSpec};
pub use turn::{
    AgentTurnId, AgentTurnRequest, AgentTurnResult, AgentTurnStatus, AgentTurnStopReason,
    DEFAULT_MAX_TOOL_STEPS, ToolCallResult, ToolCallStatus,
};
pub use validation::{
    MAX_TOOL_OUTPUT_SIZE, validate_json_value, validate_tool_input, validate_tool_output_size,
};
