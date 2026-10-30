//! Generic document session managing commands, preview, apply, undo, and redo.

use crate::command::{CommandBatch, CommandResult, CommandStatus};
use crate::diagnostic::Diagnostic;
use crate::model::DocumentVersion;

use super::history::{History, HistoryEntry};
use super::traits::Document;

/// An editing session over a generic document `D`.
pub struct DocumentSession<D: Document> {
    document: D,
    history: History<D>,
}

impl<D: Document> DocumentSession<D> {
    /// Create a new session wrapping `document`.
    pub fn new(document: D) -> Self {
        Self {
            document,
            history: History::new(),
        }
    }

    /// Access the current document snapshot.
    pub fn document(&self) -> &D {
        &self.document
    }

    /// Current document version.
    pub fn version(&self) -> DocumentVersion {
        self.document.version()
    }

    /// Access the undo/redo history.
    pub fn history(&self) -> &History<D> {
        &self.history
    }

    /// Preview a command batch without mutating document or history.
    pub fn preview_batch(
        &self,
        context: &D::Context,
        batch: CommandBatch<D::Command>,
    ) -> CommandResult<D::Change> {
        match self.evaluate_batch(context, &batch) {
            BatchEvaluation::Rejected { diagnostics } => CommandResult::new(
                CommandStatus::Rejected,
                self.version(),
                Vec::new(),
                diagnostics,
                None,
            ),
            BatchEvaluation::NoOp => CommandResult::new(
                CommandStatus::NoOp,
                self.version(),
                Vec::new(),
                Vec::new(),
                None,
            ),
            BatchEvaluation::Applied {
                projected_version,
                forward_changes,
                ..
            } => CommandResult::new(
                CommandStatus::Applied,
                projected_version,
                forward_changes,
                Vec::new(),
                None,
            ),
        }
    }

    /// Apply a command batch to the document.
    pub fn apply_batch(
        &mut self,
        context: &D::Context,
        batch: CommandBatch<D::Command>,
    ) -> CommandResult<D::Change> {
        match self.evaluate_batch(context, &batch) {
            BatchEvaluation::Rejected { diagnostics } => CommandResult::new(
                CommandStatus::Rejected,
                self.version(),
                Vec::new(),
                diagnostics,
                None,
            ),
            BatchEvaluation::NoOp => CommandResult::new(
                CommandStatus::NoOp,
                self.version(),
                Vec::new(),
                Vec::new(),
                None,
            ),
            BatchEvaluation::Applied {
                mut document,
                projected_version,
                forward_changes,
                inverse_changes,
            } => {
                let before = self.document.clone();
                document.set_version(projected_version);
                let history_entry_id = self.document.history_entry_id(batch.id().as_str());
                let history_entry = HistoryEntry::new(
                    history_entry_id.clone(),
                    batch.clone(),
                    before,
                    document.clone(),
                    forward_changes.clone(),
                    inverse_changes,
                );

                self.document = document;
                self.history.truncate_to_cursor();
                self.history.push(history_entry);

                CommandResult::new(
                    CommandStatus::Applied,
                    projected_version,
                    forward_changes,
                    Vec::new(),
                    Some(history_entry_id),
                )
            }
        }
    }

    /// Undo the last applied command batch.
    pub fn undo(&mut self) -> Option<CommandResult<D::Change>> {
        let entry = self.history.entry_to_undo()?.clone();
        let current_version = self.version();
        let next_version = increment_version(current_version);
        let mut restored = entry.before().clone();
        restored.set_version(next_version);

        self.document = restored;
        self.history.move_cursor_back();

        let changes = entry.inverse_changes().to_vec();

        Some(CommandResult::new(
            CommandStatus::Applied,
            next_version,
            changes,
            Vec::new(),
            Some(entry.id().clone()),
        ))
    }

    /// Redo the last undone command batch.
    pub fn redo(&mut self) -> Option<CommandResult<D::Change>> {
        let entry = self.history.entry_to_redo()?.clone();
        let current_version = self.version();
        let next_version = increment_version(current_version);
        let mut restored = entry.after().clone();
        restored.set_version(next_version);

        self.document = restored;
        self.history.move_cursor_forward();

        let changes = entry.forward_changes().to_vec();

        Some(CommandResult::new(
            CommandStatus::Applied,
            next_version,
            changes,
            Vec::new(),
            Some(entry.id().clone()),
        ))
    }

    /// Restore the session to a pre-apply snapshot (e.g. after persistence failure).
    pub fn restore(&mut self, document: D) {
        if self.history.cursor() > 0 {
            self.history.move_cursor_back();
            self.history.truncate_to_cursor();
        }
        self.document = document;
    }

    fn evaluate_batch(
        &self,
        context: &D::Context,
        batch: &CommandBatch<D::Command>,
    ) -> BatchEvaluation<D> {
        let current_version = self.version();
        if batch.base_version() != current_version {
            return BatchEvaluation::Rejected {
                diagnostics: vec![self.document.version_conflict_diagnostic(
                    current_version,
                    batch.base_version(),
                    batch.context().correlation_id(),
                )],
            };
        }

        let mut working = self.document.clone();
        let mut forward_changes = Vec::new();
        let mut inverse_steps = Vec::<Vec<D::Change>>::new();
        let mut diagnostics = Vec::new();

        for command in batch.commands() {
            match working.apply_command(command, context, batch.context().correlation_id()) {
                Ok(app) => {
                    forward_changes.extend(app.forward);
                    if !app.inverse.is_empty() {
                        inverse_steps.push(app.inverse);
                    }
                }
                Err(diags) => {
                    diagnostics.extend(diags);
                }
            }
        }

        if diagnostics.is_empty() {
            diagnostics.extend(working.validate(context));
        }

        if !diagnostics.is_empty() {
            return BatchEvaluation::Rejected { diagnostics };
        }

        if forward_changes.is_empty() {
            return BatchEvaluation::NoOp;
        }

        let projected_version = increment_version(current_version);
        let inverse_changes = inverse_steps.into_iter().rev().flatten().collect();

        BatchEvaluation::Applied {
            document: working,
            projected_version,
            forward_changes,
            inverse_changes,
        }
    }
}

enum BatchEvaluation<D: Document> {
    Rejected {
        diagnostics: Vec<Diagnostic>,
    },
    NoOp,
    Applied {
        document: D,
        projected_version: DocumentVersion,
        forward_changes: Vec<D::Change>,
        inverse_changes: Vec<D::Change>,
    },
}

fn increment_version(version: DocumentVersion) -> DocumentVersion {
    DocumentVersion::new(version.get() + 1)
}
