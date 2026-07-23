use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use reimagine_agent::{
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
}

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
        }
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
    _marker: PhantomData<fn(I) -> O>,
}

impl<I, O, H> WorkspaceTool<I, O, H> {
    fn new(services: Arc<WorkspaceServices>, spec: WorkspaceToolSpec, handler: H) -> Self {
        Self {
            services,
            spec,
            handler,
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
        .with_input_schema(object_schema())
        .with_output_schema(object_schema())
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
    registry
        .register_arc(Arc::new(WorkspaceTool::<I, O, H>::new(
            services, spec, handler,
        )))
        .expect("duplicate tool registration in app-host built-ins");
}

fn object_schema() -> serde_json::Value {
    serde_json::json!({"type": "object"})
}
