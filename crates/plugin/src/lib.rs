//! Plugin metadata contract and static extension identity.
//!
//! `reimagine-plugin` defines the shared vocabulary that host code uses to
//! describe plugin packages and the extensions they contribute. It is the
//! metadata surface only; it does not own dynamic plugin loading, factory
//! traits, or any host-specific construction logic.
//!
//! V1 registration is static: built-in packages such as the Candle inference
//! backend are described as plugins at compile time. Runtime loading of
//! third-party plugins is deferred.
//!
//! See `docs/architecture/modules/plugin.md` for the full module architecture.

#![deny(unsafe_code)]

mod descriptor;
mod error;
mod extension;
mod ids;
mod package;

pub use descriptor::{PluginDescriptor, PluginOrigin};
pub use error::PluginError;
pub use extension::{Extension, HostSurface, PluginExtension};
pub use ids::{Plugin, PluginApiVersion, PluginVersion};
pub use package::PluginPackage;
