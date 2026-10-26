//! Typed identities for backend plugins and extensions.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};

/// Error when constructing a [`Plugin`] or [`Extension`] identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginIdError {
    EmptyIdentity { kind: &'static str },
}

impl fmt::Display for PluginIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentity { kind } => write!(f, "identity must not be empty ({kind})"),
        }
    }
}

impl std::error::Error for PluginIdError {}

/// Stable identity of an inference backend plugin.
///
/// Examples: `"builtin.burn"`, `"builtin.fake"`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct Plugin(String);

impl Plugin {
    /// Construct a `Plugin` from a non-empty string.
    pub fn new(value: impl Into<String>) -> Result<Self, PluginIdError> {
        let value = value.into();
        if value.is_empty() {
            return Err(PluginIdError::EmptyIdentity { kind: "plugin" });
        }
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
    type Error = PluginIdError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<&str> for Plugin {
    type Error = PluginIdError;

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

/// Stable identity of an inference backend extension.
///
/// Examples: `"backend.burn"`, `"backend.fake"`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct Extension(String);

impl Extension {
    /// Construct an `Extension` from a non-empty string.
    pub fn new(value: impl Into<String>) -> Result<Self, PluginIdError> {
        let value = value.into();
        if value.is_empty() {
            return Err(PluginIdError::EmptyIdentity { kind: "extension" });
        }
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
    type Error = PluginIdError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<&str> for Extension {
    type Error = PluginIdError;

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
