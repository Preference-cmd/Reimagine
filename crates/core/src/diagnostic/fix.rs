/// A hint for an automated or suggested fix.
///
/// This slice intentionally omits workflow command previews; a later slice
/// may add them after `CommandBatch` is stable.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DiagnosticFixHint {
    label: String,
    description: Option<String>,
    requires_confirmation: bool,
}

impl DiagnosticFixHint {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            description: None,
            requires_confirmation: false,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_requires_confirmation(mut self, requires: bool) -> Self {
        self.requires_confirmation = requires;
        self
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    pub fn requires_confirmation(&self) -> bool {
        self.requires_confirmation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fix_hint_defaults() {
        let h = DiagnosticFixHint::new("download model");
        assert_eq!(h.label(), "download model");
        assert_eq!(h.description(), None);
        assert!(!h.requires_confirmation());
    }

    #[test]
    fn fix_hint_with_description_and_confirmation() {
        let h = DiagnosticFixHint::new("remove stale cache")
            .with_description("Deletes cached weights for this model")
            .with_requires_confirmation(true);
        assert_eq!(
            h.description(),
            Some("Deletes cached weights for this model")
        );
        assert!(h.requires_confirmation());
    }
}
