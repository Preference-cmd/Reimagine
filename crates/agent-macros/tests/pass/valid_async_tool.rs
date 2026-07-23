use reimagine_agent::{
    AgentMode, AgentSessionId, AgentToolRegistry, PermissionSet, ToolContext, ToolError,
    ToolPermission, ToolResult, WorkspaceScope,
};
use reimagine_agent_macros::agent_tool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Deserialize, JsonSchema)]
struct EchoInput {
    value: String,
}

#[derive(Debug, Serialize, JsonSchema)]
struct EchoOutput {
    echoed: String,
}

#[agent_tool(
    name = "workflow.echo",
    description = "Echo a value",
    modes = ["agent", "build"],
    permission = "workflow.read",
    risk = "read"
)]
async fn echo_tool(_ctx: ToolContext, input: EchoInput) -> ToolResult<EchoOutput> {
    Ok(EchoOutput {
        echoed: input.value,
    })
}

fn main() {
    let mut registry = AgentToolRegistry::new();
    registry.register(echo_tool_agent_tool()).unwrap();

    let spec = registry.spec(&"workflow.echo".into()).unwrap();
    assert_eq!(spec.name().as_str(), "workflow.echo");
    assert!(spec.input_schema().is_some());
    assert!(spec.output_schema().is_some());

    let ctx = ToolContext::new(
        WorkspaceScope::new("workspace"),
        AgentSessionId::new("session"),
        AgentMode::Agent,
    )
    .with_permissions(PermissionSet::from_iter([ToolPermission::new(
        "workflow.read",
    )]));

    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let output = rt
        .block_on(registry.invoke(&"workflow.echo".into(), &ctx, json!({"value": "ok"})))
        .unwrap();
    assert_eq!(output, json!({"echoed": "ok"}));

    let denied_ctx = ToolContext::new(
        WorkspaceScope::new("workspace"),
        AgentSessionId::new("session"),
        AgentMode::Agent,
    );
    assert!(rt
        .block_on(registry.invoke(&"workflow.echo".into(), &denied_ctx, json!({"value": "no"})))
        .is_err());

    let _ = ToolError::new(
        reimagine_agent::ToolErrorCode::ExecutionFailed,
        "keeps type visible",
    );
}
