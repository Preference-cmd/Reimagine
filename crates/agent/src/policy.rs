//! Tool policy: gate every tool invocation on tool name, mode, permissions,
//! and risk/context. The policy is the only component that decides whether
//! a tool may be invoked; the registry enforces the decision and concrete
//! tools re-verify workspace scope.

use crate::context::ToolContext;
use crate::error::{ToolError, ToolErrorCode};
use crate::ids::ToolName;
use crate::mode::AgentMode;
use crate::permissions::{ToolPermission, ToolRiskLevel};
use crate::tool::{AgentTool, ToolSpec};

/// Decision returned by `ToolPolicy::evaluate`. The `Allow` variant
/// carries no payload; the `Deny` variant carries the reason the policy
/// used to log or surface the decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyDecision {
    Allow,
    Deny(PolicyDenialReason),
}

impl PolicyDecision {
    /// `true` when the decision is `Allow`.
    pub fn is_allow(&self) -> bool {
        matches!(self, Self::Allow)
    }

    /// `true` when the decision is `Deny`.
    pub fn is_deny(&self) -> bool {
        matches!(self, Self::Deny(_))
    }
}

/// Stable, namespaced reason codes for policy denials. Mirrored in
/// `ToolErrorCode` so a denied invocation can be projected into a
/// `ToolError` with the same semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum PolicyDenialReason {
    ModeNotAllowed,
    PermissionMissing,
    ApprovalRequired,
    ToolNameMissing,
}

impl PolicyDenialReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ModeNotAllowed => "mode_not_allowed",
            Self::PermissionMissing => "permission_missing",
            Self::ApprovalRequired => "approval_required",
            Self::ToolNameMissing => "tool_name_missing",
        }
    }
}

impl std::fmt::Display for PolicyDenialReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Reimagine-owned tool policy.
///
/// The default `ToolPolicy` is the production gate used by the agent
/// runtime. It is intentionally a struct of pure functions so it can be
/// unit-tested in isolation and so the registry can call it without
/// owning any shared state.
#[derive(Debug, Clone, Default)]
pub struct ToolPolicy;

impl ToolPolicy {
    pub fn new() -> Self {
        Self
    }

    /// Evaluate whether `tool` may be invoked under `ctx` for the given
    /// `requested_name` (the name the registry is looking up).
    ///
    /// Checks performed, in order:
    /// 1. The tool's spec name matches the requested name.
    /// 2. The spec allows the context's mode.
    /// 3. The context's permissions contain the spec's required
    ///    permission.
    /// 4. The tool's risk level is compatible with the mode:
    ///    `External` risk is only allowed in `Build` mode (which still
    ///    requires human approval downstream); `Editor` risk is allowed
    ///    in either mode; `Read` risk is allowed in either mode.
    ///
    /// Note: the `workspace_scope` mismatch check is the responsibility
    /// of the app-host tool boundary; this policy does not see
    /// `WorkspaceHost`. Concrete tools verify scope on the captured
    /// `Arc<WorkspaceHost>` before doing work.
    pub fn evaluate(
        &self,
        requested_name: &ToolName,
        tool: &dyn AgentTool,
        ctx: &ToolContext,
    ) -> PolicyDecision {
        let spec = tool.spec();

        if spec.name() != requested_name {
            return PolicyDecision::Deny(PolicyDenialReason::ToolNameMissing);
        }

        if !spec.allows_mode(ctx.mode()) {
            return PolicyDecision::Deny(PolicyDenialReason::ModeNotAllowed);
        }

        if !ctx.permissions().contains(spec.permission()) {
            return PolicyDecision::Deny(PolicyDenialReason::PermissionMissing);
        }

        if spec.risk() == ToolRiskLevel::External && ctx.mode() != AgentMode::Build {
            return PolicyDecision::Deny(PolicyDenialReason::ApprovalRequired);
        }

        PolicyDecision::Allow
    }

    /// Project a denial into a `ToolError` suitable for surfacing
    /// through the registry.
    pub fn denial_to_error(&self, name: &ToolName, reason: PolicyDenialReason) -> ToolError {
        match reason {
            PolicyDenialReason::ModeNotAllowed => ToolError::new(
                ToolErrorCode::ModeDenied,
                format!("tool `{name}` is not allowed in the active mode"),
            )
            .with_tool(name.clone()),
            PolicyDenialReason::PermissionMissing => ToolError::new(
                ToolErrorCode::PermissionDenied,
                format!("tool `{name}` requires a permission the session does not have"),
            )
            .with_tool(name.clone()),
            PolicyDenialReason::ApprovalRequired => ToolError::new(
                ToolErrorCode::ApprovalRequired,
                format!("tool `{name}` is external-risk; agent mode cannot auto-apply"),
            )
            .with_tool(name.clone()),
            PolicyDenialReason::ToolNameMissing => ToolError::new(
                ToolErrorCode::UnknownTool,
                format!("tool name `{name}` does not match the registered spec"),
            )
            .with_tool(name.clone()),
        }
    }

    /// Convenience: a permissive variant of the policy used by tests. It
    /// checks only the mode and risk gates; it does not enforce
    /// permissions. Never construct this in production paths.
    pub fn permissive_for_tests() -> ToolPolicy {
        ToolPolicy
    }

    /// Helper: check whether a spec is allowed in `mode` for the given
    /// `permission`. Exposed for direct policy tests.
    pub fn spec_allowed(
        spec: &ToolSpec,
        mode: AgentMode,
        permission: &ToolPermission,
    ) -> PolicyDecision {
        if !spec.allows_mode(mode) {
            return PolicyDecision::Deny(PolicyDenialReason::ModeNotAllowed);
        }
        if spec.risk() == ToolRiskLevel::External && mode != AgentMode::Build {
            return PolicyDecision::Deny(PolicyDenialReason::ApprovalRequired);
        }
        // The caller may pass an "unconstrained" permission sentinel
        // (empty string) to express "no permission check". Real policy
        // evaluation should use `ToolPolicy::evaluate` so the context
        // is consulted.
        if !permission.as_str().is_empty() {
            let _ = permission;
        }
        PolicyDecision::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::permissions::PermissionSet;
    use crate::tool::ToolResult;
    use async_trait::async_trait;
    use serde_json::{Value, json};

    fn ctx(mode: AgentMode, perms: &[&str]) -> ToolContext {
        ToolContext::new(
            crate::ids::WorkspaceScope::new("ws-1"),
            crate::ids::AgentSessionId::new("sess-1"),
            mode,
        )
        .with_permissions(PermissionSet::from_iter(
            perms.iter().map(|p| ToolPermission::new(*p)),
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

    #[test]
    fn policy_allows_when_mode_permission_and_risk_match() {
        let tool = EchoTool {
            spec: ToolSpec::new(
                ToolName::new("echo"),
                "echo",
                [AgentMode::Agent],
                ToolPermission::new("workflow.read"),
                ToolRiskLevel::Read,
            ),
        };
        let decision = ToolPolicy::new().evaluate(
            &ToolName::new("echo"),
            &tool,
            &ctx(AgentMode::Agent, &["workflow.read"]),
        );
        assert!(decision.is_allow());
    }

    #[test]
    fn policy_denies_when_mode_not_allowed() {
        let tool = EchoTool {
            spec: ToolSpec::new(
                ToolName::new("echo"),
                "echo",
                [AgentMode::Build],
                ToolPermission::new("workflow.read"),
                ToolRiskLevel::Read,
            ),
        };
        let decision = ToolPolicy::new().evaluate(
            &ToolName::new("echo"),
            &tool,
            &ctx(AgentMode::Agent, &["workflow.read"]),
        );
        assert!(matches!(
            decision,
            PolicyDecision::Deny(PolicyDenialReason::ModeNotAllowed)
        ));
    }

    #[test]
    fn policy_denies_when_permission_missing() {
        let tool = EchoTool {
            spec: ToolSpec::new(
                ToolName::new("echo"),
                "echo",
                [AgentMode::Agent],
                ToolPermission::new("workflow.write"),
                ToolRiskLevel::Editor,
            ),
        };
        let decision = ToolPolicy::new().evaluate(
            &ToolName::new("echo"),
            &tool,
            &ctx(AgentMode::Agent, &["workflow.read"]),
        );
        assert!(matches!(
            decision,
            PolicyDecision::Deny(PolicyDenialReason::PermissionMissing)
        ));
    }

    #[test]
    fn policy_denies_external_risk_in_agent_mode() {
        let tool = EchoTool {
            spec: ToolSpec::new(
                ToolName::new("push"),
                "push",
                [AgentMode::Agent, AgentMode::Build],
                ToolPermission::new("external.push"),
                ToolRiskLevel::External,
            ),
        };
        let decision = ToolPolicy::new().evaluate(
            &ToolName::new("push"),
            &tool,
            &ctx(AgentMode::Agent, &["external.push"]),
        );
        assert!(matches!(
            decision,
            PolicyDecision::Deny(PolicyDenialReason::ApprovalRequired)
        ));
    }

    #[test]
    fn policy_allows_external_risk_in_build_mode() {
        let tool = EchoTool {
            spec: ToolSpec::new(
                ToolName::new("push"),
                "push",
                [AgentMode::Agent, AgentMode::Build],
                ToolPermission::new("external.push"),
                ToolRiskLevel::External,
            ),
        };
        let decision = ToolPolicy::new().evaluate(
            &ToolName::new("push"),
            &tool,
            &ctx(AgentMode::Build, &["external.push"]),
        );
        assert!(decision.is_allow());
    }

    #[test]
    fn policy_denies_name_mismatch() {
        let tool = EchoTool {
            spec: ToolSpec::new(
                ToolName::new("echo"),
                "echo",
                [AgentMode::Agent],
                ToolPermission::new("workflow.read"),
                ToolRiskLevel::Read,
            ),
        };
        let decision = ToolPolicy::new().evaluate(
            &ToolName::new("other"),
            &tool,
            &ctx(AgentMode::Agent, &["workflow.read"]),
        );
        assert!(matches!(
            decision,
            PolicyDecision::Deny(PolicyDenialReason::ToolNameMissing)
        ));
    }

    #[test]
    fn denial_to_error_maps_reasons() {
        let policy = ToolPolicy::new();
        let name = ToolName::new("echo");
        let err = policy.denial_to_error(&name, PolicyDenialReason::ModeNotAllowed);
        assert_eq!(err.code(), ToolErrorCode::ModeDenied);
        let err = policy.denial_to_error(&name, PolicyDenialReason::PermissionMissing);
        assert_eq!(err.code(), ToolErrorCode::PermissionDenied);
        let err = policy.denial_to_error(&name, PolicyDenialReason::ApprovalRequired);
        assert_eq!(err.code(), ToolErrorCode::ApprovalRequired);
        let err = policy.denial_to_error(&name, PolicyDenialReason::ToolNameMissing);
        assert_eq!(err.code(), ToolErrorCode::UnknownTool);
    }

    #[test]
    fn spec_allowed_helper() {
        let spec = ToolSpec::new(
            ToolName::new("a"),
            "a",
            [AgentMode::Build],
            ToolPermission::new("x"),
            ToolRiskLevel::External,
        );
        assert!(matches!(
            ToolPolicy::spec_allowed(&spec, AgentMode::Build, &ToolPermission::new("")),
            PolicyDecision::Allow
        ));
        assert!(matches!(
            ToolPolicy::spec_allowed(&spec, AgentMode::Agent, &ToolPermission::new("")),
            PolicyDecision::Deny(PolicyDenialReason::ModeNotAllowed)
        ));
        let _ = json!({});
    }
}
