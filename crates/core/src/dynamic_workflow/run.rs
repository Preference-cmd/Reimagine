//! Batch run documents: targets, items, states and the persisted snapshot.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::command::OperationContext;
use crate::diagnostic::Diagnostic;
use crate::dynamic_workflow::workflow::WorkflowSpec;
use crate::event::Timestamp;
use crate::model::{AssetId, CollectionId, DynamicWorkflowId, ProjectId, RunId, SlotId};

/// The set of items a run covers.
///
/// Deserialization is explicit rather than untagged: the two shapes are told
/// apart by their keys, and an unknown key (a misspelled `slots`, say) is
/// rejected instead of silently falling back to "all slots".
#[derive(Debug, Clone, PartialEq, Serialize)]
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

impl<'de> Deserialize<'de> for RunTarget {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let Value::Object(mut fields) = value else {
            return Err(D::Error::custom("run target must be a JSON object"));
        };

        if let Some(raw) = fields.remove("collection_id") {
            let collection_id = serde_json::from_value::<CollectionId>(raw)
                .map_err(|error| D::Error::custom(format!("invalid 'collection_id': {error}")))?;
            let slots = match fields.remove("slots") {
                Some(raw) => Some(
                    serde_json::from_value::<Vec<SlotId>>(raw)
                        .map_err(|error| D::Error::custom(format!("invalid 'slots': {error}")))?,
                ),
                None => None,
            };
            reject_unknown_fields::<D::Error>(&fields)?;
            return Ok(Self::Collection {
                collection_id,
                slots,
            });
        }

        if let Some(raw) = fields.remove("groups") {
            let groups = serde_json::from_value::<Vec<Value>>(raw)
                .map_err(|error| D::Error::custom(format!("invalid 'groups': {error}")))?;
            let name = match fields.remove("name") {
                Some(Value::String(name)) => name,
                Some(_) => {
                    return Err(D::Error::custom(
                        "'name' of a parameter-group run target must be a string",
                    ));
                }
                None => {
                    return Err(D::Error::custom(
                        "a parameter-group run target requires 'name'",
                    ));
                }
            };
            reject_unknown_fields::<D::Error>(&fields)?;
            return Ok(Self::Groups { groups, name });
        }

        Err(D::Error::custom(
            "run target must contain 'collection_id' or 'groups'",
        ))
    }
}

fn reject_unknown_fields<E: serde::de::Error>(
    fields: &serde_json::Map<String, Value>,
) -> Result<(), E> {
    if fields.is_empty() {
        return Ok(());
    }
    let mut names: Vec<&str> = fields.keys().map(String::as_str).collect();
    names.sort_unstable();
    Err(E::custom(format!(
        "unknown run target field(s): {}",
        names.join(", ")
    )))
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
    /// The asset this item read when the run was created. Resuming uses this
    /// pinned value instead of re-reading a slot that may have changed while
    /// the run was interrupted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_asset: Option<AssetId>,
    pub state: SlotState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostic: Option<Diagnostic>,
}

impl RunItem {
    pub fn new(item_id: impl Into<String>, slot_id: Option<SlotId>) -> Self {
        Self {
            item_id: item_id.into(),
            slot_id,
            input_asset: None,
            state: SlotState::Queued,
            diagnostic: None,
        }
    }

    /// Pin the input asset resolved when the run was created.
    pub fn with_input_asset(mut self, asset: AssetId) -> Self {
        self.input_asset = Some(asset);
        self
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
    fn unknown_target_fields_are_rejected() {
        let error = serde_json::from_value::<RunTarget>(json!({
            "collection_id": "col-1",
            "slot": ["slot-1"]
        }))
        .expect_err("a misspelled 'slots' must not mean 'all slots'");
        assert!(
            error
                .to_string()
                .contains("unknown run target field(s): slot"),
            "unexpected error: {error}"
        );

        let error = serde_json::from_value::<RunTarget>(json!({
            "groups": [{ "text": "a" }],
            "name": "Batch",
            "collection_id": "col-1"
        }))
        .expect_err("both shapes at once");
        assert!(
            error.to_string().contains("unknown run target field(s):"),
            "unexpected error: {error}"
        );

        let error =
            serde_json::from_value::<RunTarget>(json!({ "name": "Batch" })).expect_err("no shape");
        assert!(
            error.to_string().contains("'collection_id' or 'groups'"),
            "unexpected error: {error}"
        );

        let error = serde_json::from_value::<RunTarget>(json!({
            "groups": [],
            "name": "Batch",
            "extra": true
        }))
        .expect_err("unknown group target field");
        assert!(
            error
                .to_string()
                .contains("unknown run target field(s): extra"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn run_item_carries_its_input_asset() {
        let item = RunItem::new("slot-1", Some(SlotId::new("slot-1")))
            .with_input_asset(AssetId::new("asset-1"));
        let value = serde_json::to_value(&item).unwrap();
        assert_eq!(
            value,
            json!({
                "item_id": "slot-1",
                "slot_id": "slot-1",
                "input_asset": "asset-1",
                "state": "queued"
            })
        );

        // Run documents written before `input_asset` existed still load.
        let legacy: RunItem = serde_json::from_value(json!({
            "item_id": "slot-1",
            "slot_id": "slot-1",
            "state": "queued"
        }))
        .expect("legacy item");
        assert_eq!(legacy.input_asset, None);
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
