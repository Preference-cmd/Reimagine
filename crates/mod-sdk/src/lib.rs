//! The mod interface: what a domain mod declares, what it implements, and
//! how the kernel registers and calls it.
//!
//! - [`ModManifest`]: plain data naming the mod's operations and events.
//! - [`Mod`]: the async lifecycle and the single `invoke` entry point.
//! - [`ModContext`]: what the kernel hands a started mod.
//! - [`ModRegistry`]: validates manifests, starts mods and dispatches calls.
//! - [`ProcessMod`]: run a mod out of process over stdio JSON-RPC.
//! - [`sdk`]: serve a mod in a child process.
//! - [`testing`]: conformance testing kit and sample mod (behind the `testing` feature).

#![deny(unsafe_code)]
// A diagnostic is the error of every mod call. It is large but only built on
// the failure path, so boxing it would only add noise to every signature.
#![allow(clippy::result_large_err)]

mod context;
pub mod diagnostic;
mod interface;
mod manifest;
mod process;
mod protocol;
mod registry;
pub mod schema;
pub mod sdk;

#[cfg(feature = "testing")]
pub mod testing;

pub use context::ModContext;
pub use interface::{Invocation, Mod};
pub use manifest::{API_VERSION, ModEvent, ModManifest, OperationAnnotations, OperationSpec};
pub use process::{ProcessMod, ProcessModSpec};
pub use protocol::PROTOCOL_VERSION;
pub use registry::{ModEntry, ModRegistry, RESERVED_NAMESPACES};
pub use schema::{
    AssetField, InlineTextField, check_asset_marker_locations, find_asset_fields,
    find_inline_text_fields,
};

// Re-exported so a mod can write `mod_sdk::async_trait` and
// `mod_sdk::serde_json` without its own dependency on either crate.
pub use async_trait::async_trait;
pub use serde_json;

// Re-exported because they appear in the signatures a mod uses to drive its
// own conformance tests (`Invocation`, `Mod::start`, `ModRegistry::start`):
// a mod must not need a direct kernel dependency to build an invocation.
pub use reimagine_core::event::EventBus;
pub use reimagine_core::task::CancelToken;
