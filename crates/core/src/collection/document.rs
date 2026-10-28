//! Collection document: an ordered sequence of slots pointing to assets.

use crate::collection::command::{CollectionChange, CollectionCommand};
use crate::collection::slot::Slot;
use crate::diagnostic::{
    CorrelationId, Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName,
    DiagnosticTarget, DiagnosticTargetDomain,
};
use crate::document::{CommandApplication, Document};
use crate::model::{
    CollectionId, DiagnosticId, DocumentVersion, HistoryEntryId, ProjectId, SlotId,
};

/// A collection document holding ordered slots of assets.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CollectionDocument {
    pub id: CollectionId,
    pub project_id: ProjectId,
    pub name: String,
    pub version: DocumentVersion,
    pub slots: Vec<Slot>,
}

impl CollectionDocument {
    /// Create a new collection document.
    pub fn new(
        id: impl Into<CollectionId>,
        project_id: impl Into<ProjectId>,
        name: impl Into<String>,
        version: DocumentVersion,
        slots: Vec<Slot>,
    ) -> Self {
        Self {
            id: id.into(),
            project_id: project_id.into(),
            name: name.into(),
            version,
            slots,
        }
    }

    /// Access the collection id.
    pub fn id(&self) -> &CollectionId {
        &self.id
    }

    /// Access the project id.
    pub fn project_id(&self) -> &ProjectId {
        &self.project_id
    }

    /// Access the collection name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Access the document version.
    pub fn version(&self) -> DocumentVersion {
        self.version
    }

    /// Access the slots slice.
    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }

    /// Look up a slot by its id.
    pub fn slot(&self, slot_id: &SlotId) -> Option<&Slot> {
        self.slots.iter().find(|s| &s.id == slot_id)
    }
}

impl Document for CollectionDocument {
    type Command = CollectionCommand;
    type Change = CollectionChange;
    type Context = ();

    fn version(&self) -> DocumentVersion {
        self.version
    }

    fn set_version(&mut self, version: DocumentVersion) {
        self.version = version;
    }

    fn apply_command(
        &mut self,
        command: &Self::Command,
        _context: &Self::Context,
        correlation_id: Option<&CorrelationId>,
    ) -> Result<CommandApplication<Self::Change>, Vec<Diagnostic>> {
        match command {
            CollectionCommand::SetSlotAsset { slot_id, asset } => {
                let Some(slot) = self.slots.iter_mut().find(|s| s.id == *slot_id) else {
                    let mut diagnostic = Diagnostic::new(
                        DiagnosticId::new(format!(
                            "collection-unknown-slot-{}-{}",
                            self.id.as_str(),
                            slot_id.as_str()
                        )),
                        DiagnosticCode::new("COLLECTION/CALLER_UNKNOWN_SLOT"),
                        DiagnosticSeverity::Error,
                        DiagnosticSourceName::new("core"),
                        format!(
                            "slot `{}` not found in collection `{}`",
                            slot_id.as_str(),
                            self.id.as_str()
                        ),
                        DiagnosticTarget::new(DiagnosticTargetDomain::new("collection"))
                            .with_id(slot_id.as_str()),
                    );
                    if let Some(correlation_id) = correlation_id.cloned() {
                        diagnostic = diagnostic.with_correlation_id(correlation_id);
                    }
                    return Err(vec![diagnostic]);
                };

                if slot.asset == *asset {
                    return Ok(CommandApplication::new(Vec::new(), Vec::new()));
                }

                let before = slot.asset.clone();
                let after = asset.clone();
                slot.asset = asset.clone();

                let forward = vec![CollectionChange::SlotAssetChanged {
                    slot_id: slot_id.clone(),
                    before: before.clone(),
                    after: after.clone(),
                }];
                let inverse = vec![CollectionChange::SlotAssetChanged {
                    slot_id: slot_id.clone(),
                    before: after,
                    after: before,
                }];

                Ok(CommandApplication::new(forward, inverse))
            }
        }
    }

    fn version_conflict_diagnostic(
        &self,
        current_version: DocumentVersion,
        base_version: DocumentVersion,
        correlation_id: Option<&CorrelationId>,
    ) -> Diagnostic {
        let mut diagnostic = Diagnostic::new(
            DiagnosticId::new(format!(
                "collection-version-conflict-{}-{}",
                self.id.as_str(),
                base_version.get()
            )),
            DiagnosticCode::new("COLLECTION/VERSION_CONFLICT"),
            DiagnosticSeverity::Error,
            DiagnosticSourceName::new("core"),
            format!(
                "collection version conflict: expected {}, got {}",
                current_version, base_version
            ),
            DiagnosticTarget::new(DiagnosticTargetDomain::new("collection"))
                .with_id(self.id.as_str()),
        );
        if let Some(correlation_id) = correlation_id.cloned() {
            diagnostic = diagnostic.with_correlation_id(correlation_id);
        }
        diagnostic
    }

    fn history_entry_id(&self, batch_id: &str) -> HistoryEntryId {
        HistoryEntryId::new(format!("collection-history:{batch_id}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::{
        CommandActor, CommandActorKind, CommandBatch, CommandProvenance, CommandStatus,
        OperationContext,
    };
    use crate::document::DocumentSession;
    use crate::event::Timestamp;
    use crate::model::{AssetId, CommandBatchId};

    fn test_collection() -> CollectionDocument {
        CollectionDocument::new(
            CollectionId::new("col-test"),
            ProjectId::new("proj-test"),
            "Test Collection",
            DocumentVersion::new(0),
            vec![
                Slot::new("slot-1", "asset-1"),
                Slot::new("slot-2", "asset-2"),
            ],
        )
    }

    fn test_batch(
        version: DocumentVersion,
        commands: Vec<CollectionCommand>,
    ) -> CommandBatch<CollectionCommand> {
        CommandBatch::new(
            CommandBatchId::new("batch-1"),
            OperationContext::new(
                CommandActor::new(CommandActorKind::Human),
                CommandProvenance::Direct,
            ),
            version,
            Timestamp::new("2026-01-01T00:00:00Z"),
            commands,
        )
    }

    #[test]
    fn set_slot_applies() {
        let mut doc = test_collection();
        let cmd = CollectionCommand::SetSlotAsset {
            slot_id: SlotId::new("slot-1"),
            asset: AssetId::new("asset-updated"),
        };

        let app = doc.apply_command(&cmd, &(), None).expect("apply succeeded");
        assert_eq!(
            doc.slot(&SlotId::new("slot-1")).unwrap().asset.as_str(),
            "asset-updated"
        );
        assert_eq!(
            app.forward,
            vec![CollectionChange::SlotAssetChanged {
                slot_id: SlotId::new("slot-1"),
                before: AssetId::new("asset-1"),
                after: AssetId::new("asset-updated"),
            }]
        );
        assert_eq!(
            app.inverse,
            vec![CollectionChange::SlotAssetChanged {
                slot_id: SlotId::new("slot-1"),
                before: AssetId::new("asset-updated"),
                after: AssetId::new("asset-1"),
            }]
        );
    }

    #[test]
    fn unknown_slot_rejected() {
        let mut doc = test_collection();
        let cmd = CollectionCommand::SetSlotAsset {
            slot_id: SlotId::new("slot-nonexistent"),
            asset: AssetId::new("asset-updated"),
        };

        let err = doc.apply_command(&cmd, &(), None).expect_err("should fail");
        assert_eq!(err.len(), 1);
        assert_eq!(err[0].code().as_str(), "COLLECTION/CALLER_UNKNOWN_SLOT");
    }

    #[test]
    fn same_asset_is_no_op() {
        let mut doc = test_collection();
        let cmd = CollectionCommand::SetSlotAsset {
            slot_id: SlotId::new("slot-1"),
            asset: AssetId::new("asset-1"),
        };

        let app = doc.apply_command(&cmd, &(), None).expect("no-op succeeds");
        assert!(app.forward.is_empty());
        assert!(app.inverse.is_empty());
        assert_eq!(
            doc.slot(&SlotId::new("slot-1")).unwrap().asset.as_str(),
            "asset-1"
        );
    }

    #[test]
    fn undo_redo_restores_document() {
        let doc = test_collection();
        let mut session = DocumentSession::new(doc);

        let batch = test_batch(
            DocumentVersion::new(0),
            vec![CollectionCommand::SetSlotAsset {
                slot_id: SlotId::new("slot-1"),
                asset: AssetId::new("asset-new"),
            }],
        );

        let res = session.apply_batch(&(), batch);
        assert_eq!(res.status(), CommandStatus::Applied);
        assert_eq!(session.version().get(), 1);
        assert_eq!(
            session
                .document()
                .slot(&SlotId::new("slot-1"))
                .unwrap()
                .asset
                .as_str(),
            "asset-new"
        );

        // Undo
        let undo_res = session.undo().expect("undo exists");
        assert_eq!(undo_res.status(), CommandStatus::Applied);
        assert_eq!(session.version().get(), 2);
        assert_eq!(
            session
                .document()
                .slot(&SlotId::new("slot-1"))
                .unwrap()
                .asset
                .as_str(),
            "asset-1"
        );

        // Redo
        let redo_res = session.redo().expect("redo exists");
        assert_eq!(redo_res.status(), CommandStatus::Applied);
        assert_eq!(session.version().get(), 3);
        assert_eq!(
            session
                .document()
                .slot(&SlotId::new("slot-1"))
                .unwrap()
                .asset
                .as_str(),
            "asset-new"
        );
    }
}
