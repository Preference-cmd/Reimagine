//! Reimagine-owned agent runtime domain.
//!
//! This crate defines the workspace-scoped Agent session, the host-neutral
//! tool abstraction, the tool policy and registry, the Reimagine-owned
//! provider boundary, and the agent event model. It must not depend on
//! Tauri, Axum, app-host, runtime, model-manager, candle-integration, Rig,
//! or Cersei.
//!
//! See `docs/architecture/modules/agent.md` for the architecture source of
//! truth.

#![deny(unsafe_code)]

mod context;
mod error;
mod event;
mod ids;
mod mode;
mod permissions;
mod policy;
mod provider;
mod registry;
mod report;
mod session;
mod tool;

mod event_adapter;

pub use context::{Actor, ToolContext};
pub use error::{AgentError, ProviderError, ToolError, ToolErrorCode};
pub use event::AgentEvent;
pub use event_adapter::AgentDomainEventAdapter;
pub use ids::{AgentSessionId, ModelName, ProviderName, ToolName, WorkspaceScope};
pub use mode::AgentMode;
pub use permissions::{PermissionSet, ToolPermission, ToolRiskLevel};
pub use policy::{PolicyDecision, PolicyDenialReason, ToolPolicy};
pub use provider::{
    AgentProvider, AgentRequest, AgentResponse, AgentStreamEvent, Message, ModelCapability,
    ModelInfo, ToolCall, ToolCallId, Usage,
};
pub use registry::{AgentToolRegistry, ToolRegistryError};
pub use report::{AgentReport, ToolInvocationReport};
pub use session::AgentSession;
pub use tool::{AgentTool, ToolInput, ToolOutput, ToolResult, ToolSpec};
