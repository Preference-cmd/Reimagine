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
    name = "workflow.missing",
    modes = ["agent"],
    permission = "workflow.read"
)]
async fn missing_description(_ctx: ToolContext, input: Input) -> ToolResult<Output> {
    Ok(Output { value: input.value })
}

fn main() {}
