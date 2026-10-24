//! Host-side bridge between async model-manager resolution and the
//! synchronous [`ExternalReadinessProvider`] expected by core readiness.
//!
//! Core readiness drives a synchronous callback per node input. Model
//! resolution through `model-manager` is async because it can touch the
//! filesystem. App-host bridges this gap by collecting the required
//! [`ExternalReadinessSubject::ModelRef`] values from a workflow snapshot
//! up-front, resolving them asynchronously, and pre-populating a
//! [`SnapshotExternalReadinessProvider`] that core can query
//! synchronously.

use std::collections::HashMap;

use reimagine_core::diagnostic::Diagnostic;
use reimagine_core::readiness::{
    ExternalReadinessContext, ExternalReadinessProvider, ExternalReadinessSubject,
};

/// Synchronous, pre-populated bridge provider for
/// [`ExternalReadinessProvider`].
///
/// Core readiness calls [`Self::diagnostics_for`] for every
/// [`ExternalReadinessSubject`] it encounters. The bridge is pre-loaded
/// asynchronously by [`crate::ModelService::build_readiness_snapshot`]
/// and then handed to core so readiness can stay synchronous.
#[derive(Debug, Default, Clone)]
pub struct SnapshotExternalReadinessProvider {
    diagnostics: HashMap<ExternalReadinessSubject, Vec<Diagnostic>>,
}

impl SnapshotExternalReadinessProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the diagnostics produced for a single subject.
    pub fn insert(&mut self, subject: ExternalReadinessSubject, diagnostics: Vec<Diagnostic>) {
        self.diagnostics.insert(subject, diagnostics);
    }

    /// Record an "ok" verdict for a subject by storing an empty diagnostic
    /// list. An empty list lets core know the subject is ready and is
    /// distinguishable from "no entry" which signals a missing subject.
    pub fn record_ok(&mut self, subject: ExternalReadinessSubject) {
        self.diagnostics.entry(subject).or_default();
    }

    /// Number of distinct subjects in the snapshot.
    pub fn len(&self) -> usize {
        self.diagnostics.len()
    }

    /// Returns `true` when the snapshot has no recorded subjects.
    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }
}

impl ExternalReadinessProvider for SnapshotExternalReadinessProvider {
    fn diagnostics_for(
        &self,
        _context: &ExternalReadinessContext,
        subject: &ExternalReadinessSubject,
    ) -> Option<Vec<Diagnostic>> {
        self.diagnostics.get(subject).cloned()
    }
}
