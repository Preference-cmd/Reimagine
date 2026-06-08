/// Extensible newtype for the kind of a domain event (e.g. "model.loaded",
/// "config.changed").
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct DomainEventKind(String);

impl DomainEventKind {
    pub fn new(kind: impl Into<String>) -> Self {
        Self(kind.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for DomainEventKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for DomainEventKind {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for DomainEventKind {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_from_str() {
        let k = DomainEventKind::new("model.loaded");
        assert_eq!(k.as_str(), "model.loaded");
        assert_eq!(format!("{k}"), "model.loaded");
    }
}
