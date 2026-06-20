//! Backend identifier and payload key types used on execution values.
//!
//! These are the stable cross-crate identifiers that runtime,
//! inference, inference-core, and concrete backends use to address a
//! backend's stored payload. They do not own backend-local data and
//! are not the same as a backend's `InferenceBackend` impl.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BackendKind(String);

impl BackendKind {
    pub fn new(kind: impl Into<String>) -> Self {
        Self(kind.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for BackendKind {
    fn from(kind: String) -> Self {
        Self(kind)
    }
}

impl From<&str> for BackendKind {
    fn from(kind: &str) -> Self {
        Self(kind.to_owned())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BackendPayloadKey(String);

impl BackendPayloadKey {
    pub fn new(key: impl Into<String>) -> Self {
        Self(key.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for BackendPayloadKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for BackendPayloadKey {
    fn from(key: String) -> Self {
        Self(key)
    }
}

impl From<&str> for BackendPayloadKey {
    fn from(key: &str) -> Self {
        Self(key.to_owned())
    }
}
