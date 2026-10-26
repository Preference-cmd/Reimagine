//! The mod interface: what a domain mod declares, what it implements, and
//! how the kernel registers and calls it.
//!
//! - [`ModManifest`]: plain data naming the mod's operations and events.
//! - [`Mod`]: the async lifecycle and the single `invoke` entry point.
//! - [`ModContext`]: what the kernel hands a started mod.
//! - [`ModRegistry`]: validates manifests, starts mods and dispatches calls.
//! - [`testing`]: conformance testing kit and sample mod (behind the `testing` feature).

#![deny(unsafe_code)]
// A diagnostic is the error of every mod call. It is large but only built on
// the failure path, so boxing it would only add noise to every signature.
#![allow(clippy::result_large_err)]

mod context;
pub mod diagnostic;
mod interface;
mod manifest;
mod registry;
mod schema;

#[cfg(feature = "testing")]
pub mod testing;

pub use context::ModContext;
pub use interface::Mod;
pub use manifest::{API_VERSION, ModEvent, ModManifest, ModOperation, OperationAnnotations};
pub use registry::{ModEntry, ModRegistry, RESERVED_NAMESPACES};
