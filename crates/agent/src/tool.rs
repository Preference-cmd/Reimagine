//! Host-neutral `AgentTool` trait and tool input/output shapes.
//!
//! The registry boundary uses `serde_json::Value` for `ToolInput` and
//! `ToolOutput` so the same surface can carry OpenAI-compatible,
//! Anthropic, and future Rig-backed tool schemas. Concrete app-host tools
//! deserialize the JSON value into their own strongly typed input/output
//! structs.

use async_trait::async_trait;
use serde_json::Value;

use crate::context::ToolContext;
use crate::error::ToolError;
use crate::ids::ToolName;
use crate::mode::AgentMode;
use crate::permissions::{ToolPermission, ToolRiskLevel};

/// JSON-typed tool input carried across the registry boundary.
pub type ToolInput = Value;

/// JSON-typed tool output carried across the registry boundary.
pub type ToolOutput = Value;

/// Result alias for tool invocations. A `ToolResult` is the successful
/// output of a tool; failures are returned as `ToolError`.
pub type ToolResult = Result<ToolOutput, ToolError>;

/// Static description of a tool, returned by `AgentTool::spec`.
///
/// Specs are used by:
/// - the tool registry for listing, deduplication, and policy checks;
/// - provider adapters that translate the tool surface into
///   OpenAI-compatible or Anthropic tool/function-call schemas.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolSpec {
    name: ToolName,
    description: String,
    modes: Vec<AgentMode>,
    permission: ToolPermission,
    risk: ToolRiskLevel,
    /// Optional JSON-Schema for the tool's input shape. Stored as a
    /// `serde_json::Value` so the agent crate does not depend on a
    /// schema-generation crate. Concrete tools may emit any valid JSON
    /// Schema object.
    input_schema: Option<Value>,
    /// Optional JSON-Schema for the tool's output shape.
    output_schema: Option<Value>,
}

impl ToolSpec {
    /// Create a tool spec. `modes` must be non-empty; the registry will
    /// reject specs that advertise no mode (every tool must be invokable
    /// in at least one mode).
    pub fn new(
        name: ToolName,
        description: impl Into<String>,
        modes: impl IntoIterator<Item = AgentMode>,
        permission: ToolPermission,
        risk: ToolRiskLevel,
    ) -> Self {
        Self {
            name,
            description: description.into(),
            modes: modes.into_iter().collect(),
            permission,
            risk,
            input_schema: None,
            output_schema: None,
        }
    }

    /// Attach a JSON-Schema for the input shape.
    pub fn with_input_schema(mut self, schema: Value) -> Self {
        self.input_schema = Some(schema);
        self
    }

    /// Attach a JSON-Schema for the output shape.
    pub fn with_output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    pub fn name(&self) -> &ToolName {
        &self.name
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn modes(&self) -> &[AgentMode] {
        &self.modes
    }

    pub fn permission(&self) -> &ToolPermission {
        &self.permission
    }

    pub fn risk(&self) -> ToolRiskLevel {
        self.risk
    }

    pub fn input_schema(&self) -> Option<&Value> {
        self.input_schema.as_ref()
    }

    pub fn output_schema(&self) -> Option<&Value> {
        self.output_schema.as_ref()
    }

    /// `true` if this spec advertises `mode` as an allowed mode.
    pub fn allows_mode(&self, mode: AgentMode) -> bool {
        self.modes.contains(&mode)
    }
}

/// Host-neutral tool interface.
///
/// The trait is async because tool execution commonly involves
/// asynchronous app-host facades. The trait lives behind a single
/// explicit method, `invoke`, so the registry can mediate policy before
/// execution and so providers can translate the tool surface without
/// relying on procedural macros.
#[async_trait]
pub trait AgentTool: Send + Sync {
    /// Static description of the tool. Returned once at registration
    /// time and used for listing and policy evaluation.
    fn spec(&self) -> ToolSpec;

    /// Execute the tool. The registry has already verified that the
    /// `ToolContext` passes policy; concrete tool implementations still
    /// re-verify the `workspace_scope` matches the captured workspace
    /// (this check is part of the app-host tool boundary).
    async fn invoke(&self, ctx: &ToolContext, input: ToolInput) -> ToolResult;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_allows_mode() {
        let spec = ToolSpec::new(
            ToolName::new("workflow.preview_commands"),
            "Preview",
            [AgentMode::Agent, AgentMode::Build],
            ToolPermission::new("workflow.read"),
            ToolRiskLevel::Read,
        );
        assert!(spec.allows_mode(AgentMode::Agent));
        assert!(spec.allows_mode(AgentMode::Build));
        assert_eq!(spec.name().as_str(), "workflow.preview_commands");
        assert_eq!(spec.modes().len(), 2);
    }
}
