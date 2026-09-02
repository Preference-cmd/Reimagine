//! Frozen harness-shaped translation (relocated from `reimagine-ai-protocol`).
//!
//! Maps `reimagine_agent_harness` DTOs ↔ provider wire payloads:
//! `request` / `response` / `streaming` / `params` / `tools` / `listing` /
//! `files`. Each submodule keeps its `⚠️ FROZEN` discipline: no new
//! production semantics; the face dies with Slice 9 harness dissolution.
//!
//! Shared kernel-face vocabulary is re-exported from
//! `reimagine-ai-protocol` rather than duplicated: `usage` (the provider
//! usage-JSON helpers) and `sse_parser` (the SSE frame parser used by
//! `streaming`).

pub mod files;
pub mod listing;
pub mod params;
pub mod request;
pub mod response;
pub mod streaming;
pub mod tools;

pub use reimagine_ai_protocol::translation::{sse_parser, usage};
