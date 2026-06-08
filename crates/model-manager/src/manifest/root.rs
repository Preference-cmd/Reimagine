use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelRootId(String);

impl ModelRootId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ModelRootId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for ModelRootId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl From<&str> for ModelRootId {
    fn from(id: &str) -> Self {
        Self(id.to_owned())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelRootKind {
    #[serde(rename = "base_path_models")]
    BasePathModels,
    #[serde(rename = "user_selected")]
    UserSelected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRoot {
    id: ModelRootId,
    path: String,
    kind: ModelRootKind,
}

impl ModelRoot {
    pub fn new(id: ModelRootId, path: impl Into<String>, kind: ModelRootKind) -> Self {
        Self {
            id,
            path: path.into(),
            kind,
        }
    }

    pub fn base_models() -> Self {
        Self::new(ModelRootId::new("base"), ".", ModelRootKind::BasePathModels)
    }

    pub fn id(&self) -> &ModelRootId {
        &self.id
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn kind(&self) -> ModelRootKind {
        self.kind
    }
}
