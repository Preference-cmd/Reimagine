//! Batch run documents: targets, items, states and the persisted snapshot.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::command::OperationContext;
use crate::diagnostic::Diagnostic;
use crate::dynamic_workflow::workflow::WorkflowSpec;
use crate::event::Timestamp;
use crate::model::{CollectionId, DynamicWorkflowId, ProjectId, RunId, SlotId};

/// The set of items a run covers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RunTarget {
    /// Slots of an existing collection; `slots` omitted means all of them.
    Collection {
        collection_id: CollectionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        slots: Option<Vec<SlotId>>,
    },
    /// One new collection built from one parameter group per item.
    Groups { groups: Vec<Value>, name: String },
}

impl RunTarget {
    pub fn collection_id(&self) -> Option<&CollectionId> {
        match self {
            Self::Collection { collection_id, .. } => Some(collection_id),
            Self::Groups { .. } => None,
        }
    }
}

/// Overall state of a batch run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunState {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
    Interrupted,
    NeedsConfirmation,
}

impl RunState {
    /// States whose owning task may still be alive.
    pub fn is_active(self) -> bool {
        matches!(self, Self::Queued | Self::Running)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Interrupted => "interrupted",
            Self::NeedsConfirmation => "needs_confirmation",
        }
    }
}

impl std::fmt::Display for RunState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// State of one run item (one collection slot or one parameter group).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotState {
    Queued,
    Running,
    Succeeded,
    Failed,
    /// Reserved for replay: a downstream result no longer matches its inputs.
    Stale,
    Cancelled,
    NeedsConfirmation,
}

impl SlotState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Stale => "stale",
            Self::Cancelled => "cancelled",
            Self::NeedsConfirmation => "needs_confirmation",
        }
    }
}

impl std::fmt::Display for SlotState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One item of a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunItem {
    pub item_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slot_id: Option<SlotId>,
    pub state: SlotState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostic: Option<Diagnostic>,
}

impl RunItem {
    pub fn new(item_id: impl Into<String>, slot_id: Option<SlotId>) -> Self {
        Self {
            item_id: item_id.into(),
            slot_id,
            state: SlotState::Queued,
            diagnostic: None,
        }
    }
}

/// Persisted state of one batch run, rewritten atomically on every change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunDocument {
    pub id: RunId,
    pub project_id: ProjectId,
    pub dynamic_workflow_id: DynamicWorkflowId,
    /// Workflow snapshot taken when the run was created; resuming uses it
    /// rather than re-reading a workflow that may have changed.
    pub workflow: WorkflowSpec,
    pub target: RunTarget,
    /// Parameters for a collection target; absent for parameter-group runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
    pub context: OperationContext,
    pub state: RunState,
    pub version: u64,
    pub items: Vec<RunItem>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection_id: Option<CollectionId>,
    pub created_at: Timestamp,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<Timestamp>,
}

impl RunDocument {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: RunId,
        project_id: ProjectId,
        dynamic_workflow_id: DynamicWorkflowId,
        workflow: WorkflowSpec,
        target: RunTarget,
        params: Option<Value>,
        context: OperationContext,
        items: Vec<RunItem>,
        created_at: Timestamp,
    ) -> Self {
        Self {
            id,
            project_id,
            dynamic_workflow_id,
            workflow,
            target,
            params,
            context,
            state: RunState::Queued,
            version: 1,
            items,
            collection_id: None,
            created_at,
            finished_at: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn collection_target_roundtrip_omits_absent_slots() {
        let target: RunTarget = serde_json::from_value(json!({
            "collection_id": "col-1"
        }))
        .expect("collection target");
        assert_eq!(
            target,
            RunTarget::Collection {
                collection_id: CollectionId::new("col-1"),
                slots: None
            }
        );
        assert_eq!(
            serde_json::to_value(&target).unwrap(),
            json!({ "collection_id": "col-1" })
        );

        let target: RunTarget = serde_json::from_value(json!({
            "collection_id": "col-1",
            "slots": ["slot-2", "slot-1"]
        }))
        .expect("collection target with slots");
        assert_eq!(
            target,
            RunTarget::Collection {
                collection_id: CollectionId::new("col-1"),
                slots: Some(vec![SlotId::new("slot-2"), SlotId::new("slot-1")])
            }
        );
    }

    #[test]
    fn groups_target_roundtrip() {
        let target: RunTarget = serde_json::from_value(json!({
            "groups": [{ "text": "a" }, { "text": "b" }],
            "name": "Batch"
        }))
        .expect("groups target");
        assert_eq!(
            target,
            RunTarget::Groups {
                groups: vec![json!({ "text": "a" }), json!({ "text": "b" })],
                name: "Batch".to_owned()
            }
        );
        assert_eq!(target.collection_id(), None);
    }

    #[test]
    fn item_defaults_to_queued_and_states_serialize_snake_case() {
        let item = RunItem::new("slot-1", Some(SlotId::new("slot-1")));
        assert_eq!(item.state, SlotState::Queued);
        assert_eq!(
            serde_json::to_value(&item).unwrap(),
            json!({ "item_id": "slot-1", "slot_id": "slot-1", "state": "queued" })
        );
        assert_eq!(
            serde_json::to_value(RunState::NeedsConfirmation).unwrap(),
            json!("needs_confirmation")
        );
        assert_eq!(
            serde_json::to_value(SlotState::Stale).unwrap(),
            json!("stale")
        );
    }
}
