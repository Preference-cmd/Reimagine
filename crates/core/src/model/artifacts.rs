#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ArtifactRef(String);

impl ArtifactRef {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ArtifactRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for ArtifactRef {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl From<&str> for ArtifactRef {
    fn from(id: &str) -> Self {
        Self(id.to_owned())
    }
}
