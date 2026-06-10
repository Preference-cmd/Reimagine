use reimagine_agent::{ToolContext, ToolResult};
use reimagine_agent_macros::agent_tool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, JsonSchema)]
struct Input {
    value: String,
}

#[derive(Serialize, JsonSchema)]
struct Output {
    value: String,
}

#[agent_tool(
    name = "workflow.invalid_signature",
    description = "Invalid signature",
    modes = ["agent"],
    permission = "workflow.read"
)]
async fn invalid_signature(_ctx: ToolContext) -> ToolResult<Output> {
    Ok(Output {
        value: "invalid".to_owned(),
    })
}

fn main() {}
