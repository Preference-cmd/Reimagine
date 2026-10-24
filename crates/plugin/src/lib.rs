//! Plugin metadata contract and static extension identity.
//!
//! `reimagine-plugin` defines the shared vocabulary that host code uses to
//! describe plugin packages and the extensions they contribute. It is the
//! metadata surface only; it does not own dynamic plugin loading, factory
//! traits, or any host-specific construction logic.
//!
//! V1 registration is static: built-in packages are described as plugins at compile time. Runtime loading of
//! third-party plugins is deferred.
//!
//! Packages implement [`PluginPackage`] to declare a [`PluginDescriptor`] and
//! enumerate [`PluginExtension`] records targeted at host extension slots
//! ([`HostSurface`]), keeping extension identity decoupled from concrete host
//! execution logic.

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
