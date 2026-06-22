//! Payload key type used on execution values.
//!
//! The open backend label [`Backend`] lives in
//! [`super::backend_selection`] and is re-exported through
//! [`crate::Backend`] so it can ride on the backend-affine handles
//! defined alongside this module.

use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
