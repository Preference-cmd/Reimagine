//! Attribute macros for Reimagine Agent tools.
//!
//! `#[agent_tool]` is declaration ergonomics only. It generates an explicit
//! wrapper that implements `reimagine_agent::AgentTool`; callers still register
//! that wrapper with `AgentToolRegistry`, and all execution still goes through
//! registry policy checks.

#![deny(unsafe_code)]

use proc_macro::TokenStream;

mod agent_tool;

#[proc_macro_attribute]
pub fn agent_tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    agent_tool::expand(attr.into(), item.into()).into()
}
