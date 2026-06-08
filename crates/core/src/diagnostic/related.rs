use super::target::DiagnosticTarget;

/// A secondary target referenced by a diagnostic, with an explanatory message.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DiagnosticRelated {
    target: DiagnosticTarget,
    message: String,
}

impl DiagnosticRelated {
    pub fn new(target: DiagnosticTarget, message: impl Into<String>) -> Self {
        Self {
            target,
            message: message.into(),
        }
    }

    pub fn target(&self) -> &DiagnosticTarget {
        &self.target
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostic::target::DiagnosticTargetDomain;

    #[test]
    fn construct_related() {
        let r = DiagnosticRelated::new(
            DiagnosticTarget::new(DiagnosticTargetDomain::new("model")).with_id("clip"),
            "used by this workflow node",
        );
        assert_eq!(r.target().id(), Some("clip"));
        assert_eq!(r.message(), "used by this workflow node");
    }
}
