//! Traits and supporting structures for generic documents.

use std::fmt::Debug;

use crate::diagnostic::{CorrelationId, Diagnostic};
use crate::model::{DocumentVersion, HistoryEntryId};

/// Forward and inverse changes produced by applying a single command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandApplication<Ch> {
    pub forward: Vec<Ch>,
    pub inverse: Vec<Ch>,
}

impl<Ch> CommandApplication<Ch> {
    pub fn new(forward: Vec<Ch>, inverse: Vec<Ch>) -> Self {
        Self { forward, inverse }
    }
}

/// A document that can be previewed, edited, and validated via commands in a `DocumentSession`.
pub trait Document: Clone + PartialEq + Debug + 'static {
    type Command: Clone + PartialEq + Debug + 'static;
    type Change: Clone + PartialEq + Debug + 'static;
    type Context: ?Sized;

    /// Current document version.
    fn version(&self) -> DocumentVersion;

    /// Set the document version.
    fn set_version(&mut self, version: DocumentVersion);

    /// Apply a single command to this working copy of the document.
    fn apply_command(
        &mut self,
        command: &Self::Command,
        context: &Self::Context,
        correlation_id: Option<&CorrelationId>,
    ) -> Result<CommandApplication<Self::Change>, Vec<Diagnostic>>;

    /// Post-command document validation.
    fn validate(&self, context: &Self::Context) -> Vec<Diagnostic> {
        let _ = context;
        Vec::new()
    }

    /// Construct a domain-specific version conflict diagnostic.
    fn version_conflict_diagnostic(
        &self,
        current_version: DocumentVersion,
        base_version: DocumentVersion,
        correlation_id: Option<&CorrelationId>,
    ) -> Diagnostic;

    /// Generate a history entry id for an applied batch.
    fn history_entry_id(&self, batch_id: &str) -> HistoryEntryId {
        HistoryEntryId::new(format!("history:{batch_id}"))
    }
}
