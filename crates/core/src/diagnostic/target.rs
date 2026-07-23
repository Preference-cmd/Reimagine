/// Extensible newtype for the domain a diagnostic target belongs to
/// (e.g. "config", "model", "workflow", "runtime").
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct DiagnosticTargetDomain(String);

impl DiagnosticTargetDomain {
    pub fn new(domain: impl Into<String>) -> Self {
        Self(domain.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for DiagnosticTargetDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for DiagnosticTargetDomain {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for DiagnosticTargetDomain {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

/// Identifies the entity a diagnostic is about.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DiagnosticTarget {
    domain: DiagnosticTargetDomain,
    id: Option<String>,
    path: Option<String>,
}

impl DiagnosticTarget {
    /// Create a target with only a domain.
    pub fn new(domain: DiagnosticTargetDomain) -> Self {
        Self {
            domain,
            id: None,
            path: None,
        }
    }

    /// Attach an entity id.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Attach a filesystem or logical path.
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    pub fn domain(&self) -> &DiagnosticTargetDomain {
        &self.domain
    }

    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    pub fn path(&self) -> Option<&str> {
        self.path.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_domain_only() {
        let t = DiagnosticTarget::new(DiagnosticTargetDomain::new("config"));
        assert_eq!(t.domain().as_str(), "config");
        assert_eq!(t.id(), None);
        assert_eq!(t.path(), None);
    }

    #[test]
    fn target_with_id_and_path() {
        let t = DiagnosticTarget::new(DiagnosticTargetDomain::new("model"))
            .with_id("sd-1.5")
            .with_path("/models/sd-1.5.safetensors");
        assert_eq!(t.id(), Some("sd-1.5"));
        assert_eq!(t.path(), Some("/models/sd-1.5.safetensors"));
    }
}
