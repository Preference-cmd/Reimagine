//! Plugin extensions and the host surfaces they extend.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};

use crate::error::PluginError;
use crate::ids::validate_non_empty;

/// Stable identity of a plugin extension.
///
/// Examples: `"backend.candle"`, `"tool.propose_node"`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct Extension(String);

impl Extension {
    /// Construct an `Extension` from a non-empty string.
    pub fn new(value: impl Into<String>) -> Result<Self, PluginError> {
        let value = value.into();
        validate_non_empty(&value, "extension")?;
        Ok(Self(value))
    }

    /// Borrow the underlying identity string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Extension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for Extension {
    type Error = PluginError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<&str> for Extension {
    type Error = PluginError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for Extension {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// Host-owned surface that a [`PluginExtension`] may extend.
///
/// `#[non_exhaustive]` allows adding a new host surface without breaking
/// downstream match arms. Adding a variant is still a SemVer-breaking change
/// for the data model; this attribute only preserves source compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum HostSurface {
    /// An inference backend implementation.
    InferenceBackend,
    /// A node catalog contributor.
    NodeCatalog,
    /// A node executor implementation.
    NodeExecutor,
    /// A workflow import/export adapter.
    WorkflowAdapter,
    /// An Agent tool implementation.
    AgentTool,
    /// An Agent provider implementation.
    AgentProvider,
}

impl HostSurface {
    /// Stable diagnostic label. Distinct from the wire format produced by
    /// `Serialize`; the `serde` representation is the Rust variant name.
    pub fn as_str(&self) -> &'static str {
        match self {
            HostSurface::InferenceBackend => "inference_backend",
            HostSurface::NodeCatalog => "node_catalog",
            HostSurface::NodeExecutor => "node_executor",
            HostSurface::WorkflowAdapter => "workflow_adapter",
            HostSurface::AgentTool => "agent_tool",
            HostSurface::AgentProvider => "agent_provider",
        }
    }
}

impl fmt::Display for HostSurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One capability a plugin contributes to a host surface.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PluginExtension {
    /// Stable identity of this extension.
    pub extension: Extension,
    /// Host surface that this extension extends.
    pub extends: HostSurface,
    /// Human-readable display name.
    pub name: String,
}
