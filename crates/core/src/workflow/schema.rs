use std::collections::BTreeMap;

use crate::model::{EdgeId, NodeId, NodeTypeId, ParamValue, SlotId, WorkflowId, WorkflowVersion};

use super::{Endpoint, WorkflowInterface, WorkflowLayout, WorkflowMetadata};

pub const WORKFLOW_SCHEMA_VERSION: &str = "reimagine.workflow.v1";

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct WorkflowSchemaVersion(String);

impl WorkflowSchemaVersion {
    pub fn new(version: impl Into<String>) -> Self {
        Self(version.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for WorkflowSchemaVersion {
    fn default() -> Self {
        Self(WORKFLOW_SCHEMA_VERSION.to_owned())
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Workflow {
    schema_version: WorkflowSchemaVersion,
    id: WorkflowId,
    version: WorkflowVersion,
    metadata: WorkflowMetadata,
    interface: WorkflowInterface,
    nodes: Vec<WorkflowNode>,
    edges: Vec<WorkflowEdge>,
    layout: WorkflowLayout,
}

impl Workflow {
    pub fn schema_version(&self) -> &WorkflowSchemaVersion {
        &self.schema_version
    }

    pub fn id(&self) -> &WorkflowId {
        &self.id
    }

    pub fn version(&self) -> WorkflowVersion {
        self.version
    }

    pub fn metadata(&self) -> &WorkflowMetadata {
        &self.metadata
    }

    pub fn interface(&self) -> &WorkflowInterface {
        &self.interface
    }

    pub fn nodes(&self) -> &[WorkflowNode] {
        &self.nodes
    }

    pub fn edges(&self) -> &[WorkflowEdge] {
        &self.edges
    }

    pub fn layout(&self) -> &WorkflowLayout {
        &self.layout
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorkflowNode {
    id: NodeId,
    type_id: NodeTypeId,
    #[serde(skip_serializing_if = "Option::is_none")]
    label: Option<String>,
    #[serde(default)]
    params: BTreeMap<SlotId, ParamValue>,
}

impl WorkflowNode {
    pub fn id(&self) -> &NodeId {
        &self.id
    }

    pub fn type_id(&self) -> &NodeTypeId {
        &self.type_id
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    pub fn params(&self) -> &BTreeMap<SlotId, ParamValue> {
        &self.params
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WorkflowEdge {
    id: EdgeId,
    from: Endpoint,
    to: Endpoint,
}

impl WorkflowEdge {
    pub fn id(&self) -> &EdgeId {
        &self.id
    }

    pub fn from(&self) -> &Endpoint {
        &self.from
    }

    pub fn to(&self) -> &Endpoint {
        &self.to
    }
}
