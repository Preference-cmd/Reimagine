use std::path::{Component, Path};

use serde::{Deserialize, Serialize};

use crate::{ConfigError, ConfigResult};

/// Relative path of a config document under `<base_path>/config/`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConfigKey(String);

impl ConfigKey {
    pub fn new(key: impl Into<String>) -> ConfigResult<Self> {
        let key = key.into();
        validate_key(&key)?;
        Ok(Self(key))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ConfigKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<&str> for ConfigKey {
    type Error = ConfigError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

fn validate_key(key: &str) -> ConfigResult<()> {
    if key.is_empty() {
        return Err(invalid(key, "key cannot be empty"));
    }

    let path = Path::new(key);
    if path.is_absolute() {
        return Err(invalid(key, "key must be relative"));
    }

    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            Component::ParentDir => return Err(invalid(key, "key cannot escape config dir")),
            Component::CurDir => return Err(invalid(key, "key cannot contain `.` components")),
            Component::RootDir | Component::Prefix(_) => {
                return Err(invalid(key, "key must be relative"));
            }
        }
    }

    Ok(())
}

fn invalid(key: &str, reason: impl Into<String>) -> ConfigError {
    ConfigError::PathInvalid {
        key: key.to_owned(),
        reason: reason.into(),
    }
}
