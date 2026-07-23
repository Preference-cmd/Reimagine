//! Typed newtypes for stable plugin identities.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};

use crate::error::PluginError;

/// Stable identity of a plugin package.
///
/// Examples: `"builtin.candle"`, `"external.foo"`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct Plugin(String);

impl Plugin {
    /// Construct a `Plugin` from a non-empty string.
    pub fn new(value: impl Into<String>) -> Result<Self, PluginError> {
        let value = value.into();
        validate_non_empty(&value, "plugin")?;
        Ok(Self(value))
    }

    /// Borrow the underlying identity string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Plugin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for Plugin {
    type Error = PluginError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<&str> for Plugin {
    type Error = PluginError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for Plugin {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// Version of a plugin package. Format is owned by the plugin author.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct PluginVersion(String);

impl PluginVersion {
    /// Construct a `PluginVersion` from a non-empty string.
    pub fn new(value: impl Into<String>) -> Result<Self, PluginError> {
        let value = value.into();
        validate_non_empty(&value, "plugin version")?;
        Ok(Self(value))
    }

    /// Borrow the underlying version string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PluginVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for PluginVersion {
    type Error = PluginError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<&str> for PluginVersion {
    type Error = PluginError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for PluginVersion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// Plugin-facing API version. Plugins built against one API version are
/// expected to be compatible with host surfaces that declare the same version.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct PluginApiVersion(String);

impl PluginApiVersion {
    /// Construct a `PluginApiVersion` from a non-empty string.
    pub fn new(value: impl Into<String>) -> Result<Self, PluginError> {
        let value = value.into();
        validate_non_empty(&value, "plugin api version")?;
        Ok(Self(value))
    }

    /// Borrow the underlying API version string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PluginApiVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for PluginApiVersion {
    type Error = PluginError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<&str> for PluginApiVersion {
    type Error = PluginError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for PluginApiVersion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

pub(crate) fn validate_non_empty(value: &str, kind: &'static str) -> Result<(), PluginError> {
    if value.is_empty() {
        return Err(PluginError::EmptyIdentity { kind });
    }
    Ok(())
}
