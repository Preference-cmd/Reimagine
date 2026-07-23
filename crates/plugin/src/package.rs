//! Base `PluginPackage` trait.

use crate::descriptor::PluginDescriptor;
use crate::extension::PluginExtension;

/// An object that exposes plugin metadata for a host to collect.
///
/// Implementations may wrap built-in or statically linked packages. This
/// trait is metadata only; constructing concrete extension instances is the
/// responsibility of the owning domain crate.
pub trait PluginPackage: Send + Sync + 'static {
    /// Return the package descriptor.
    fn descriptor(&self) -> PluginDescriptor;

    /// Return the extensions this package contributes.
    fn extensions(&self) -> Vec<PluginExtension>;
}
