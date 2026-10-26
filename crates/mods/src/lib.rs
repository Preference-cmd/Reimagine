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
mod schema;
pub mod sdk;

#[cfg(feature = "testing")]
pub mod testing;

pub use context::ModContext;
pub use interface::Mod;
pub use manifest::{API_VERSION, ModEvent, ModManifest, ModOperation, OperationAnnotations};
pub use process::{ProcessMod, ProcessModSpec};
pub use protocol::PROTOCOL_VERSION;
pub use registry::{ModEntry, ModRegistry, RESERVED_NAMESPACES};
