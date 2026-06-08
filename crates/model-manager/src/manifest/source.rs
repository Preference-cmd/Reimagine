use serde::{Deserialize, Serialize};

use super::ModelRootId;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelSource {
    LocalFileRelative { root_id: ModelRootId, path: String },
    LocalFileAbsolute { path: String },
}

impl ModelSource {
    pub fn relative(root_id: ModelRootId, path: impl Into<String>) -> Self {
        Self::LocalFileRelative {
            root_id,
            path: path.into(),
        }
    }

    pub fn absolute(path: impl Into<String>) -> Self {
        Self::LocalFileAbsolute { path: path.into() }
    }
}
