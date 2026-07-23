/// Extensible newtype for the subsystem that emitted a domain event
/// (e.g. "config", "model-manager", "runtime").
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct DomainEventSource(String);

impl DomainEventSource {
    pub fn new(source: impl Into<String>) -> Self {
        Self(source.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for DomainEventSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for DomainEventSource {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for DomainEventSource {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_from_str() {
        let s = DomainEventSource::new("model-manager");
        assert_eq!(s.as_str(), "model-manager");
    }
}
