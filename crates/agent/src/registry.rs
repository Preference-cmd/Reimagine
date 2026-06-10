//! Explicit, deterministic tool registry.
//!
//! The registry stores tools in a `BTreeMap` keyed by `ToolName` so
//! listing is always deterministic. Duplicate names are rejected at
//! registration time. Tool execution is gated by `ToolPolicy`; concrete
//! tools may not be invoked through a path that bypasses the policy.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::context::ToolContext;
use crate::error::{ToolError, ToolErrorCode};
use crate::ids::ToolName;
use crate::policy::{PolicyDecision, PolicyDenialReason, ToolPolicy};
use crate::tool::{AgentTool, ToolInput, ToolSpec};

/// Errors that the registry can return to its caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolRegistryError {
    /// A tool with this name is already registered.
    DuplicateName(ToolName),
    /// The registry was asked to look up a tool name that is not
    /// registered.
    UnknownTool(ToolName),
    /// The spec advertised no mode. A tool must be invokable in at least
    /// one mode.
    SpecHasNoModes(ToolName),
    /// Policy denied the invocation. The carried `ToolError` is suitable
    /// for surfacing through the diagnostic bridge.
    PolicyDenied(ToolError),
    /// The concrete tool returned a `ToolError` (typically an
    /// `ExecutionFailed`).
    ToolReturned(ToolError),
}

impl std::fmt::Display for ToolRegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateName(n) => write!(f, "tool `{n}` is already registered"),
            Self::UnknownTool(n) => write!(f, "tool `{n}` is not registered"),
            Self::SpecHasNoModes(n) => {
                write!(f, "tool `{n}` spec does not declare any allowed mode")
            }
            Self::PolicyDenied(e) => write!(f, "policy denied: {e}"),
            Self::ToolReturned(e) => write!(f, "tool error: {e}"),
        }
    }
}

impl std::error::Error for ToolRegistryError {}

/// Tool registry. Holds `Arc<dyn AgentTool>` instances keyed by name and
/// mediates every invocation through the configured `ToolPolicy`.
#[derive(Clone)]
pub struct AgentToolRegistry {
    tools: BTreeMap<ToolName, Arc<dyn AgentTool>>,
    policy: Arc<ToolPolicy>,
}

impl std::fmt::Debug for AgentToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentToolRegistry")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .field("policy", &"ToolPolicy")
            .finish()
    }
}

impl Default for AgentToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentToolRegistry {
    /// Create a new, empty registry using the default `ToolPolicy`.
    pub fn new() -> Self {
        Self {
            tools: BTreeMap::new(),
            policy: Arc::new(ToolPolicy::new()),
        }
    }

    /// Create a registry with a custom `ToolPolicy`. Tests use this to
    /// swap in stricter or more permissive policies.
    pub fn with_policy(policy: ToolPolicy) -> Self {
        Self {
            tools: BTreeMap::new(),
            policy: Arc::new(policy),
        }
    }

    /// Register a tool. Returns the spec's name on success.
    ///
    /// Registration is explicit (no inventory / linkme magic). Duplicate
    /// names are rejected. Specs that advertise no mode are rejected.
    pub fn register<T>(&mut self, tool: T) -> Result<ToolName, ToolRegistryError>
    where
        T: AgentTool + 'static,
    {
        let spec = tool.spec();
        if spec.modes().is_empty() {
            return Err(ToolRegistryError::SpecHasNoModes(spec.name().clone()));
        }
        let name = spec.name().clone();
        if self.tools.contains_key(&name) {
            return Err(ToolRegistryError::DuplicateName(name));
        }
        self.tools.insert(name.clone(), Arc::new(tool));
        Ok(name)
    }

    /// Register a boxed `dyn AgentTool`. This entry point is used by
    /// `app-host` which constructs tool closures that capture
    /// `Arc<WorkspaceHost>`.
    pub fn register_arc(
        &mut self,
        tool: Arc<dyn AgentTool>,
    ) -> Result<ToolName, ToolRegistryError> {
        let spec = tool.spec();
        if spec.modes().is_empty() {
            return Err(ToolRegistryError::SpecHasNoModes(spec.name().clone()));
        }
        let name = spec.name().clone();
        if self.tools.contains_key(&name) {
            return Err(ToolRegistryError::DuplicateName(name));
        }
        self.tools.insert(name.clone(), tool);
        Ok(name)
    }

    /// Returns `true` if a tool with this name is registered.
    pub fn contains(&self, name: &ToolName) -> bool {
        self.tools.contains_key(name)
    }

    /// Returns the registered tool's spec, if any.
    pub fn spec(&self, name: &ToolName) -> Option<ToolSpec> {
        self.tools.get(name).map(|t| t.spec())
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// `true` if no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Return all tool names in deterministic (sorted) order.
    pub fn tool_names(&self) -> Vec<ToolName> {
        self.tools.keys().cloned().collect()
    }

    /// Return all tool specs in deterministic (sorted) order. This is
    /// the listing used by the provider adapters to translate the tool
    /// surface into a model-call schema.
    pub fn list(&self) -> Vec<ToolSpec> {
        self.tools.values().map(|t| t.spec()).collect()
    }

    /// Invoke a tool through the policy gate.
    ///
    /// This is the only public way to execute a tool. The registry
    /// evaluates `ToolPolicy::evaluate`; on deny it surfaces the policy's
    /// `ToolError`. On allow it calls the tool and returns its result
    /// (mapping execution errors into `ToolRegistryError::ToolReturned`).
    pub async fn invoke(
        &self,
        name: &ToolName,
        ctx: &ToolContext,
        input: ToolInput,
    ) -> Result<crate::tool::ToolOutput, ToolRegistryError> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| ToolRegistryError::UnknownTool(name.clone()))?;

        match self.policy.evaluate(name, tool.as_ref(), ctx) {
            PolicyDecision::Allow => {}
            PolicyDecision::Deny(reason) => {
                let err = self.policy.denial_to_error(name, reason);
                return Err(match reason {
                    PolicyDenialReason::ToolNameMissing => {
                        ToolRegistryError::UnknownTool(name.clone())
                    }
                    _ => ToolRegistryError::PolicyDenied(err),
                });
            }
        }

        tool.invoke(ctx, input)
            .await
            .map_err(ToolRegistryError::ToolReturned)
    }

    /// Apply policy to a tool spec without invoking it. Useful for hosts
    /// that want to ask "can I invoke this in this context?" before
    /// constructing the full input payload.
    pub fn policy_check(
        &self,
        name: &ToolName,
        ctx: &ToolContext,
    ) -> Result<PolicyDecision, ToolRegistryError> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| ToolRegistryError::UnknownTool(name.clone()))?;
        Ok(self.policy.evaluate(name, tool.as_ref(), ctx))
    }

    /// Validate a tool spec, returning a `ToolError` for invalid specs.
    /// Concrete tools are expected to also re-validate their workspace
    /// scope on the captured app-host workspace.
    pub fn validate_spec(spec: &ToolSpec) -> Result<(), ToolError> {
        if spec.modes().is_empty() {
            return Err(ToolError::new(
                ToolErrorCode::InvalidInput,
                format!(
                    "tool `{}` spec must declare at least one allowed mode",
                    spec.name().as_str()
                ),
            )
            .with_tool(spec.name().clone()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::{Value, json};

    use crate::mode::AgentMode;
    use crate::permissions::{PermissionSet, ToolPermission, ToolRiskLevel};
    use crate::tool::ToolResult;

    fn ctx(mode: AgentMode, perms: &[&str]) -> ToolContext {
        ToolContext::new(
            crate::ids::WorkspaceScope::new("ws-1"),
            crate::ids::AgentSessionId::new("sess-1"),
            mode,
        )
        .with_permissions(PermissionSet::from_iter(
            perms.into_iter().map(|p| ToolPermission::new(*p)),
        ))
    }

    struct EchoTool {
        spec: ToolSpec,
    }

    #[async_trait]
    impl AgentTool for EchoTool {
        fn spec(&self) -> ToolSpec {
            self.spec.clone()
        }
        async fn invoke(&self, _ctx: &ToolContext, input: Value) -> ToolResult {
            Ok(input)
        }
    }

    fn read_tool(name: &str) -> EchoTool {
        EchoTool {
            spec: ToolSpec::new(
                ToolName::new(name),
                "echo",
                [AgentMode::Agent, AgentMode::Build],
                ToolPermission::new("workflow.read"),
                ToolRiskLevel::Read,
            ),
        }
    }

    #[test]
    fn registration_succeeds_for_unique_names() {
        let mut reg = AgentToolRegistry::new();
        assert_eq!(reg.register(read_tool("a")).unwrap().as_str(), "a");
        assert_eq!(reg.register(read_tool("b")).unwrap().as_str(), "b");
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn duplicate_registration_is_rejected() {
        let mut reg = AgentToolRegistry::new();
        reg.register(read_tool("a")).unwrap();
        let err = reg.register(read_tool("a")).unwrap_err();
        assert!(matches!(err, ToolRegistryError::DuplicateName(ref n) if n.as_str() == "a"));
    }

    #[test]
    fn spec_with_no_modes_is_rejected() {
        let mut reg = AgentToolRegistry::new();
        let tool = EchoTool {
            spec: ToolSpec::new(
                ToolName::new("empty"),
                "empty",
                std::iter::empty::<AgentMode>(),
                ToolPermission::new("workflow.read"),
                ToolRiskLevel::Read,
            ),
        };
        let err = reg.register(tool).unwrap_err();
        assert!(matches!(err, ToolRegistryError::SpecHasNoModes(ref n) if n.as_str() == "empty"));
    }

    #[test]
    fn list_returns_deterministic_sorted_names() {
        let mut reg = AgentToolRegistry::new();
        reg.register(read_tool("zeta")).unwrap();
        reg.register(read_tool("alpha")).unwrap();
        reg.register(read_tool("mid")).unwrap();
        let tool_names = reg.tool_names();
        let names: Vec<&str> = tool_names.iter().map(|n| n.as_str()).collect();
        assert_eq!(names, vec!["alpha", "mid", "zeta"]);
    }

    #[test]
    fn list_returns_specs_in_sorted_order() {
        let mut reg = AgentToolRegistry::new();
        reg.register(read_tool("zeta")).unwrap();
        reg.register(read_tool("alpha")).unwrap();
        let specs = reg.list();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].name().as_str(), "alpha");
        assert_eq!(specs[1].name().as_str(), "zeta");
    }

    #[tokio::test]
    async fn invoke_succeeds_through_policy_allow() {
        let mut reg = AgentToolRegistry::new();
        reg.register(read_tool("echo")).unwrap();
        let out = reg
            .invoke(
                &ToolName::new("echo"),
                &ctx(AgentMode::Agent, &["workflow.read"]),
                json!({"hello": "world"}),
            )
            .await
            .unwrap();
        assert_eq!(out, json!({"hello": "world"}));
    }

    #[tokio::test]
    async fn invoke_denies_when_permission_missing() {
        let mut reg = AgentToolRegistry::new();
        reg.register(read_tool("echo")).unwrap();
        let err = reg
            .invoke(
                &ToolName::new("echo"),
                &ctx(AgentMode::Agent, &[]),
                json!({}),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolRegistryError::PolicyDenied(_)));
    }

    #[tokio::test]
    async fn invoke_denies_when_mode_not_allowed() {
        let mut reg = AgentToolRegistry::new();
        let tool = EchoTool {
            spec: ToolSpec::new(
                ToolName::new("build-only"),
                "build-only",
                [AgentMode::Build],
                ToolPermission::new("workflow.read"),
                ToolRiskLevel::Read,
            ),
        };
        reg.register(tool).unwrap();
        let err = reg
            .invoke(
                &ToolName::new("build-only"),
                &ctx(AgentMode::Agent, &["workflow.read"]),
                json!({}),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolRegistryError::PolicyDenied(_)));
    }

    #[tokio::test]
    async fn invoke_unknown_tool_returns_unknown() {
        let reg = AgentToolRegistry::new();
        let err = reg
            .invoke(
                &ToolName::new("nope"),
                &ctx(AgentMode::Agent, &[]),
                json!({}),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolRegistryError::UnknownTool(_)));
    }

    #[tokio::test]
    async fn tool_returning_execution_error_is_surfaced() {
        struct Boom;
        #[async_trait]
        impl AgentTool for Boom {
            fn spec(&self) -> ToolSpec {
                ToolSpec::new(
                    ToolName::new("boom"),
                    "boom",
                    [AgentMode::Agent],
                    ToolPermission::new("workflow.read"),
                    ToolRiskLevel::Read,
                )
            }
            async fn invoke(&self, _ctx: &ToolContext, _input: Value) -> ToolResult {
                Err(ToolError::new(ToolErrorCode::ExecutionFailed, "kaboom"))
            }
        }
        let mut reg = AgentToolRegistry::new();
        reg.register(Boom).unwrap();
        let err = reg
            .invoke(
                &ToolName::new("boom"),
                &ctx(AgentMode::Agent, &["workflow.read"]),
                json!({}),
            )
            .await
            .unwrap_err();
        match err {
            ToolRegistryError::ToolReturned(e) => {
                assert_eq!(e.code(), ToolErrorCode::ExecutionFailed);
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn policy_check_returns_decision_without_invoking() {
        let mut reg = AgentToolRegistry::new();
        reg.register(read_tool("echo")).unwrap();
        let decision = reg
            .policy_check(&ToolName::new("echo"), &ctx(AgentMode::Agent, &[]))
            .unwrap();
        assert!(matches!(
            decision,
            PolicyDecision::Deny(PolicyDenialReason::PermissionMissing)
        ));
    }

    #[test]
    fn validate_spec_rejects_no_modes() {
        let spec = ToolSpec::new(
            ToolName::new("x"),
            "x",
            std::iter::empty::<AgentMode>(),
            ToolPermission::new("p"),
            ToolRiskLevel::Read,
        );
        assert!(AgentToolRegistry::validate_spec(&spec).is_err());
    }

    #[test]
    fn registration_via_arc_works_and_rejects_duplicates() {
        let mut reg = AgentToolRegistry::new();
        let tool: Arc<dyn AgentTool> = Arc::new(read_tool("a"));
        reg.register_arc(tool).unwrap();
        let tool: Arc<dyn AgentTool> = Arc::new(read_tool("a"));
        let err = reg.register_arc(tool).unwrap_err();
        assert!(matches!(err, ToolRegistryError::DuplicateName(_)));
    }
}
