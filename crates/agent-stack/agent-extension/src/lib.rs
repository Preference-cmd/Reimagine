//! reimagine-agent-extension — the staging crate for implemented
//! agent-stack entities whose layer assignment is not yet decided.
//!
//! # Charter (Slice 12)
//!
//! - **Purpose**: the single temporary home for entities that are already
//!   implemented but whose crate-level placement is not yet clear. Not a
//!   junk drawer: every resident registers its open question below, and
//!   graduation follows the rules below.
//! - **Dependency discipline**: `reimagine-context-kernel` and
//!   `reimagine-ai-protocol` NEVER depend on this crate. This crate may
//!   depend on any agent-stack crate. `agent-runtime` / `agent-provider`
//!   default to not depending on it; hosts may.
//! - **Graduation rules**: when a real consumer appears or the layer
//!   judgment becomes clear, the resident moves out (→ `agent-runtime`,
//!   `agent-provider`, or the host crate) and its registration line is
//!   deleted. Nothing ships publicly from this crate in the v0.1 publish
//!   set (Slice 11).
//!
//! # Residents
//!
//! | Entity | Module | Open question |
//! |---|---|---|
//! | `FsConversationStore` | `store` | Infra implementation of the kernel `ConversationStore` port — should hosts bring their own (FS / Sqlite / …) so this never graduates, or is a reference store worth `agent-runtime`'s public API? Decides at first host adoption (Slice 4B). |
//!
//! # Adjacent, not resident
//!
//! `agent-macros` stays a separate crate (proc-macros cannot be merged
//! into a normal crate), but its DP-4 decision (adopt the `#[agent_tool]`
//! macro for real or delete it) is registered here as an extension-scope
//! item, to be resolved after the v0.1 publish (Slice 11).
//!
//! Future candidates (tool search / progressive disclosure, OTel
//! exporters) enter here first if their layer is unclear at birth.
#![deny(unsafe_code)]

pub mod store;

pub use store::FsConversationStore;
