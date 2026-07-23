//! Error type for plugin metadata construction.

use std::fmt;

/// Failure when constructing a plugin metadata newtype.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginError {
    /// An identity newtype was constructed from an empty string.
    EmptyIdentity {
        /// The newtype whose value was empty (e.g. `"plugin"`, `"plugin version"`).
        kind: &'static str,
    },
}

impl fmt::Display for PluginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PluginError::EmptyIdentity { kind } => {
                write!(f, "{kind} identity must not be empty")
            }
        }
    }
}

impl std::error::Error for PluginError {}
