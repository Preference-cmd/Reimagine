use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use reimagine_agent_harness::{
    AgentMode, AgentTool, AgentToolRegistry, ToolContext, ToolError, ToolErrorCode, ToolInput,
    ToolName, ToolPermission, ToolResult, ToolRiskLevel, ToolSpec,
};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::services::WorkspaceServices;

#[derive(Debug, Clone, Copy)]
pub(super) struct WorkspaceToolSpec {
    pub(super) name: &'static str,
    pub(super) description: &'static str,
    pub(super) modes: &'static [AgentMode],
    pub(super) permission: &'static str,
    pub(super) risk: ToolRiskLevel,
    /// JSON Schema (draft-07 subset) for the tool input, as text so it
    /// stays Copy. Defaults to a transparent `object`; tools advertise
    /// field-accurate schemas via `with_schemas` (AR-38).
    pub(super) input_schema: &'static str,
    /// JSON Schema (draft-07 subset) for the tool output.
    pub(super) output_schema: &'static str,
}

pub(super) const OBJECT_SCHEMA: &str = r#"{"type":"object"}"#;

impl WorkspaceToolSpec {
    pub(super) const fn new(
        name: &'static str,
        description: &'static str,
        modes: &'static [AgentMode],
        permission: &'static str,
        risk: ToolRiskLevel,
    ) -> Self {
        Self {
            name,
            description,
            modes,
            permission,
            risk,
            input_schema: OBJECT_SCHEMA,
            output_schema: OBJECT_SCHEMA,
        }
    }

    /// Attach field-accurate input/output JSON Schemas (AR-38). Schemas
    /// are raw JSON text kept const so the spec stays Copy.
    pub(super) const fn with_schemas(
        mut self,
        input_schema: &'static str,
        output_schema: &'static str,
    ) -> Self {
        self.input_schema = input_schema;
        self.output_schema = output_schema;
        self
    }
}

#[async_trait]
pub(super) trait WorkspaceToolHandler<I, O>: Send + Sync + 'static {
    async fn call(
        &self,
        services: Arc<WorkspaceServices>,
        ctx: ToolContext,
        input: I,
    ) -> ToolResult<O>;
}

#[async_trait]
impl<I, O, F, Fut> WorkspaceToolHandler<I, O> for F
where
    I: Send + 'static,
    O: Send + 'static,
    F: Fn(Arc<WorkspaceServices>, ToolContext, I) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ToolResult<O>> + Send + 'static,
{
    async fn call(
        &self,
        services: Arc<WorkspaceServices>,
        ctx: ToolContext,
        input: I,
    ) -> ToolResult<O> {
        (self)(services, ctx, input).await
    }
}

pub(super) struct WorkspaceTool<I, O, H> {
    services: Arc<WorkspaceServices>,
    spec: WorkspaceToolSpec,
    handler: H,
    /// Field-accurate input/output JSON Schemas (AR-38), parsed once at
    /// registration time so a malformed schema surfaces there instead of
    /// degrading at listing time (AR-40).
    input_schema: serde_json::Value,
    output_schema: serde_json::Value,
    _marker: PhantomData<fn(I) -> O>,
}

impl<I, O, H> WorkspaceTool<I, O, H> {
    fn new(
        services: Arc<WorkspaceServices>,
        spec: WorkspaceToolSpec,
        handler: H,
        input_schema: serde_json::Value,
        output_schema: serde_json::Value,
    ) -> Self {
        Self {
            services,
            spec,
            handler,
            input_schema,
            output_schema,
            _marker: PhantomData,
        }
    }

    fn verify_workspace_scope(&self, ctx: &ToolContext) -> ToolResult<()> {
        if self.services.workspace_scope() != ctx.workspace_scope() {
            return Err(ToolError::new(
                ToolErrorCode::WorkspaceMismatch,
                format!(
                    "tool `{}` was invoked with workspace `{}` but is bound to `{}`",
                    self.spec.name,
                    ctx.workspace_scope().as_str(),
                    self.services.workspace_scope().as_str(),
                ),
            )
            .with_tool(ToolName::new(self.spec.name)));
        }
        // AR-14: project ownership. A thread bound to a project may only
        // drive tools whose target service is scoped to that same
        // project; unbound sessions (ctx has no project) keep legacy
        // single-project behaviour.
        if let Some(ctx_project) = ctx.project_id() {
            let service_project = self.services.workflow_service().project_id();
            if ctx_project != service_project {
                return Err(ToolError::new(
                    ToolErrorCode::WorkspaceMismatch,
                    format!(
                        "tool `{}` was invoked for project `{}` but the workspace is scoped to `{}`",
                        self.spec.name,
                        ctx_project.as_str(),
                        service_project.as_str(),
                    ),
                )
                .with_tool(ToolName::new(self.spec.name)));
            }
        }
        Ok(())
    }
}

#[async_trait]
impl<I, O, H> AgentTool for WorkspaceTool<I, O, H>
where
    I: DeserializeOwned + Send + 'static,
    O: Serialize + Send + 'static,
    H: WorkspaceToolHandler<I, O>,
{
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            ToolName::new(self.spec.name),
            self.spec.description,
            self.spec.modes.iter().copied(),
            ToolPermission::new(self.spec.permission),
            self.spec.risk,
        )
        .with_input_schema(self.input_schema.clone())
        .with_output_schema(self.output_schema.clone())
    }

    async fn invoke(&self, ctx: &ToolContext, input: ToolInput) -> ToolResult {
        self.verify_workspace_scope(ctx)?;
        let typed_input: I = serde_json::from_value(input).map_err(|e| {
            ToolError::new(ToolErrorCode::InvalidInput, format!("invalid input: {e}"))
                .with_tool(ToolName::new(self.spec.name))
        })?;
        let output = self
            .handler
            .call(Arc::clone(&self.services), ctx.clone(), typed_input)
            .await?;
        serde_json::to_value(output).map_err(|e| {
            ToolError::new(
                ToolErrorCode::ExecutionFailed,
                format!("serialization failed: {e}"),
            )
            .with_tool(ToolName::new(self.spec.name))
        })
    }
}

pub(super) fn register_workspace_tool<I, O, H>(
    registry: &mut AgentToolRegistry,
    services: Arc<WorkspaceServices>,
    spec: WorkspaceToolSpec,
    handler: H,
) where
    I: DeserializeOwned + Send + 'static,
    O: Serialize + Send + 'static,
    H: WorkspaceToolHandler<I, O>,
{
    // AR-40: fail closed on a malformed schema. A broken schema is a
    // registration-time defect — diagnose and skip the tool rather than
    // silently advertising a transparent object the model cannot use.
    let input_schema = match parse_schema(spec.input_schema) {
        Ok(schema) => schema,
        Err(error) => {
            tracing::error!(
                tool = spec.name,
                %error,
                "invalid input schema text; skipping tool",
            );
            return;
        }
    };
    let output_schema = match parse_schema(spec.output_schema) {
        Ok(schema) => schema,
        Err(error) => {
            tracing::error!(
                tool = spec.name,
                %error,
                "invalid output schema text; skipping tool",
            );
            return;
        }
    };
    if let Err(error) = registry.register_arc(Arc::new(WorkspaceTool::<I, O, H>::new(
        services,
        spec,
        handler,
        input_schema,
        output_schema,
    ))) {
        tracing::error!(%error, "failed to register workspace tool; skipping it");
    }
}

/// Parse a JSON Schema from its const text.
///
/// AR-40 (fail-closed): a malformed schema text is returned as `Err` —
/// registration skips the tool with a diagnostic instead of silently
/// degrading to `{type: object}`. The "tool listing must not panic"
/// guarantee from AR-38 is preserved because parsing happens once at
/// registration, never inside the listing path.
///
/// Supported validator keyword subset (V1, enforced by the harness's
/// `validate_json_value`): `type`, `required`, `properties`
/// (recursively, including nested schemas), and `enum`. Any other
/// draft-07 keyword is intentionally not interpreted here — an
/// unsupported keyword is carried as a model-visible annotation, not
/// enforced. Workspace tools must stay within this subset.
fn parse_schema(text: &str) -> Result<serde_json::Value, serde_json::Error> {
    serde_json::from_str(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_agent_harness::{
        AgentSessionId, PermissionSet, ToolRegistryError, WorkspaceScope,
    };

    #[test]
    fn parse_schema_accepts_valid_json() {
        let schema =
            parse_schema(r#"{"type":"object","properties":{}}"#).expect("valid schema parses");
        assert_eq!(schema["type"], "object");
    }

    #[test]
    fn parse_schema_rejects_malformed_text() {
        // AR-40 fail-closed: a typo'd schema text must surface as an
        // error here — registration skips the tool with a diagnostic;
        // it is never silently downgraded to a transparent object.
        assert!(parse_schema("not json").is_err());
        assert!(parse_schema("{\"type\": \"object\"").is_err());
    }

    #[tokio::test]
    async fn malformed_schema_is_skipped_and_invalid_input_never_calls_handler() {
        use serde::{Deserialize, Serialize};
        use std::sync::atomic::{AtomicUsize, Ordering};

        #[derive(Deserialize)]
        struct Input {
            // Intentionally unread: its existence + `required` entry in the
            // advertised schema is the whole point - invalid input must be
            // rejected by schema validation *before* the handler runs.
            #[allow(dead_code)]
            required: String,
        }
        #[derive(Serialize)]
        struct Output {
            ok: bool,
        }

        let base = std::env::temp_dir().join(format!(
            "reimagine-ar40-workspace-tool-{}",
            std::process::id(),
        ));
        let host = crate::WorkspaceHost::with_defaults(WorkspaceScope::new("ws-ar40-unit"), &base);
        let services = Arc::clone(host.services());

        let mut malformed_registry = AgentToolRegistry::new();
        let malformed_spec = WorkspaceToolSpec::new(
            "test.malformed",
            "test malformed schema",
            &[AgentMode::Agent],
            "test.read",
            ToolRiskLevel::Read,
        )
        .with_schemas("not-json", OBJECT_SCHEMA);
        register_workspace_tool(
            &mut malformed_registry,
            Arc::clone(&services),
            malformed_spec,
            |_services, _ctx, _input: Input| async { Ok(Output { ok: true }) },
        );
        assert!(
            malformed_registry.is_empty(),
            "malformed schema must skip registration",
        );

        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_handler = Arc::clone(&calls);
        let mut registry = AgentToolRegistry::new();
        let valid_spec = WorkspaceToolSpec::new(
            "test.valid",
            "test valid schema",
            &[AgentMode::Agent],
            "test.read",
            ToolRiskLevel::Read,
        )
        .with_schemas(
            r#"{"type":"object","properties":{"required":{"type":"string"}},"required":["required"]}"#,
            OBJECT_SCHEMA,
        );
        register_workspace_tool(
            &mut registry,
            services,
            valid_spec,
            move |_services, _ctx, _input: Input| {
                calls_for_handler.fetch_add(1, Ordering::SeqCst);
                async { Ok(Output { ok: true }) }
            },
        );

        let ctx = ToolContext::new(
            WorkspaceScope::new("ws-ar40-unit"),
            AgentSessionId::new("schema-test"),
            AgentMode::Agent,
        )
        .with_permissions(PermissionSet::from_iter([ToolPermission::new("test.read")]));
        let error = registry
            .invoke(&ToolName::new("test.valid"), &ctx, serde_json::json!({}))
            .await
            .expect_err("missing required input must fail before the handler");
        match error {
            ToolRegistryError::ToolReturned(error) => {
                assert_eq!(error.code(), ToolErrorCode::InvalidInput);
            }
            other => panic!("expected ToolReturned, got {other:?}"),
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "schema validation must run before the handler",
        );
        let _ = std::fs::remove_dir_all(base);
    }
}
