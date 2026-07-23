//! Plugin package descriptor and origin metadata.

use serde::{Deserialize, Serialize};

use crate::ids::{Plugin, PluginApiVersion, PluginVersion};

/// Where a plugin package comes from. Metadata only; the crate does not
/// define any loading semantics for `External` packages in V1.
///
/// `#[non_exhaustive]` allows adding a new origin kind without breaking
/// downstream match arms.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PluginOrigin {
    /// Compiled into the host binary.
    Builtin,
    /// Provided by an external package. The `source` string is opaque
    /// metadata; the host does not interpret it in V1. Empty strings are
    /// permitted (treated as "no source info") and validation is deferred
    /// to the host that ultimately consumes this descriptor.
    External {
        /// Opaque source identifier (e.g. a crate name, a path, a URL).
        source: String,
    },
}

impl PluginOrigin {
    /// Short diagnostic label.
    pub fn as_str(&self) -> &str {
        match self {
            PluginOrigin::Builtin => "builtin",
            PluginOrigin::External { .. } => "external",
        }
    }
}

/// Stable metadata describing a plugin package.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PluginDescriptor {
    /// Stable identity of the package.
    pub plugin: Plugin,
    /// Human-readable display name.
    pub name: String,
    /// Plugin-reported package version.
    pub version: PluginVersion,
    /// Plugin API version this package targets.
    pub api_version: PluginApiVersion,
    /// Where the package originates from.
    pub origin: PluginOrigin,
}
