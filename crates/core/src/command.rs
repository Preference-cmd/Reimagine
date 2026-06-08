use std::collections::BTreeMap;

use crate::diagnostic::{CorrelationId, Diagnostic};
use crate::event::Timestamp;
use crate::model::{
    CommandBatchId, EdgeId, HistoryEntryId, NodeId, NodeTypeId, ParamValue, ProposalId, SlotId,
    WorkflowVersion,
};
use crate::workflow::{
    Endpoint, Position, WorkflowEdge, WorkflowLayout, WorkflowMetadata, WorkflowNode,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum CommandActorKind {
    Human,
    Agent,
    Importer,
    System,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CommandActor {
    kind: CommandActorKind,
    id: Option<String>,
    label: Option<String>,
}

impl CommandActor {
    pub fn new(kind: CommandActorKind) -> Self {
        Self {
            kind,
            id: None,
            label: None,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn kind(&self) -> CommandActorKind {
        self.kind
    }

    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CommandProvenance {
    Direct,
    AgentProposal {
        proposal_id: ProposalId,
        approved_by: Option<CommandActor>,
    },
    Import {
        format: String,
        source: Option<String>,
    },
    Migration {
        from_schema_version: String,
    },
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkflowCommand {
    AddNode {
        node_id: NodeId,
        type_id: NodeTypeId,
        label: Option<String>,
        params: BTreeMap<SlotId, ParamValue>,
        position: Option<Position>,
    },
    RemoveNode {
        node_id: NodeId,
    },
    Connect {
        edge_id: EdgeId,
        from: Endpoint,
        to: Endpoint,
    },
    Disconnect {
        edge_id: EdgeId,
    },
    SetParam {
        node_id: NodeId,
        slot_id: SlotId,
        value: ParamValue,
    },
    RemoveParam {
        node_id: NodeId,
        slot_id: SlotId,
    },
    MoveNode {
        node_id: NodeId,
        position: Position,
    },
    ApplyLayout {
        layout: WorkflowLayout,
    },
    SetNodeLabel {
        node_id: NodeId,
        label: Option<String>,
    },
    SetWorkflowMetadata {
        metadata: WorkflowMetadata,
    },
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CommandBatch {
    id: CommandBatchId,
    actor: CommandActor,
    base_version: WorkflowVersion,
    provenance: CommandProvenance,
    created_at: Timestamp,
    correlation_id: Option<CorrelationId>,
    commands: Vec<WorkflowCommand>,
}

impl CommandBatch {
    pub fn new(
        id: CommandBatchId,
        actor: CommandActor,
        base_version: WorkflowVersion,
        provenance: CommandProvenance,
        created_at: Timestamp,
        commands: Vec<WorkflowCommand>,
    ) -> Self {
        Self {
            id,
            actor,
            base_version,
            provenance,
            created_at,
            correlation_id: None,
            commands,
        }
    }

    pub fn with_correlation_id(mut self, correlation_id: CorrelationId) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    pub fn id(&self) -> &CommandBatchId {
        &self.id
    }

    pub fn actor(&self) -> &CommandActor {
        &self.actor
    }

    pub fn base_version(&self) -> WorkflowVersion {
        self.base_version
    }

    pub fn provenance(&self) -> &CommandProvenance {
        &self.provenance
    }

    pub fn created_at(&self) -> &Timestamp {
        &self.created_at
    }

    pub fn correlation_id(&self) -> Option<&CorrelationId> {
        self.correlation_id.as_ref()
    }

    pub fn commands(&self) -> &[WorkflowCommand] {
        &self.commands
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum CommandResultStatus {
    Applied,
    Rejected,
    NoOp,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum WorkflowChange {
    NodeAdded {
        node: WorkflowNode,
    },
    NodeRemoved {
        node: WorkflowNode,
        removed_edges: Vec<WorkflowEdge>,
        removed_layout: Option<Position>,
    },
    EdgeAdded {
        edge: WorkflowEdge,
    },
    EdgeRemoved {
        edge: WorkflowEdge,
    },
    ParamSet {
        node_id: NodeId,
        slot_id: SlotId,
        before: Option<ParamValue>,
        after: ParamValue,
    },
    ParamRemoved {
        node_id: NodeId,
        slot_id: SlotId,
        before: ParamValue,
    },
    NodeMoved {
        node_id: NodeId,
        before: Option<Position>,
        after: Option<Position>,
    },
    LayoutApplied {
        before: WorkflowLayout,
        after: WorkflowLayout,
    },
    NodeLabelSet {
        node_id: NodeId,
        before: Option<String>,
        after: Option<String>,
    },
    WorkflowMetadataSet {
        before: WorkflowMetadata,
        after: WorkflowMetadata,
    },
    VersionAdvanced {
        before: WorkflowVersion,
        after: WorkflowVersion,
    },
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CommandResult {
    status: CommandResultStatus,
    workflow_version: WorkflowVersion,
    changes: Vec<WorkflowChange>,
    diagnostics: Vec<Diagnostic>,
    history_entry_id: Option<HistoryEntryId>,
}

impl CommandResult {
    pub fn new(
        status: CommandResultStatus,
        workflow_version: WorkflowVersion,
        changes: Vec<WorkflowChange>,
        diagnostics: Vec<Diagnostic>,
        history_entry_id: Option<HistoryEntryId>,
    ) -> Self {
        Self {
            status,
            workflow_version,
            changes,
            diagnostics,
            history_entry_id,
        }
    }

    pub fn status(&self) -> CommandResultStatus {
        self.status.clone()
    }

    pub fn workflow_version(&self) -> WorkflowVersion {
        self.workflow_version
    }

    pub fn changes(&self) -> &[WorkflowChange] {
        &self.changes
    }

    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    pub fn history_entry_id(&self) -> Option<&HistoryEntryId> {
        self.history_entry_id.as_ref()
    }
}
