use std::path::{Path, PathBuf};

use reimagine_core::diagnostic::Diagnostic;

use crate::ConfigKey;

/// Infrastructure report for config IO and document validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigReport {
    key: ConfigKey,
    path: PathBuf,
    diagnostics: Vec<Diagnostic>,
}

impl ConfigReport {
    pub fn new(key: ConfigKey, path: impl Into<PathBuf>, diagnostics: Vec<Diagnostic>) -> Self {
        Self {
            key,
            path: path.into(),
            diagnostics,
        }
    }

    pub fn key(&self) -> &ConfigKey {
        &self.key
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    pub fn into_diagnostics(self) -> Vec<Diagnostic> {
        self.diagnostics
    }
}
