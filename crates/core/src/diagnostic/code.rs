/// Extensible newtype for diagnostic codes (e.g. "CONFIG/MISSING_FILE").
///
/// Callers define their own code namespaces; core only carries the shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct DiagnosticCode(String);

impl DiagnosticCode {
    pub fn new(code: impl Into<String>) -> Self {
        Self(code.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for DiagnosticCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for DiagnosticCode {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for DiagnosticCode {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_from_str() {
        let code = DiagnosticCode::new("CONFIG/MISSING_FILE");
        assert_eq!(code.as_str(), "CONFIG/MISSING_FILE");
    }

    #[test]
    fn display_roundtrip() {
        let code = DiagnosticCode::from("MODEL/LOAD_FAIL");
        assert_eq!(format!("{code}"), "MODEL/LOAD_FAIL");
    }
}
