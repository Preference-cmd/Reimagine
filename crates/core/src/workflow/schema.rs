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
    pub fn new(id: impl Into<WorkflowId>, version: WorkflowVersion) -> Self {
        Self {
            schema_version: WorkflowSchemaVersion::default(),
            id: id.into(),
            version,
            metadata: WorkflowMetadata::new(),
            interface: WorkflowInterface::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
            layout: WorkflowLayout::new(),
        }
    }

    pub fn with_metadata(mut self, metadata: WorkflowMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_interface(mut self, interface: WorkflowInterface) -> Self {
        self.interface = interface;
        self
    }

    pub fn with_node(mut self, node: WorkflowNode) -> Self {
        self.nodes.push(node);
        self
    }

    pub fn with_edge(mut self, edge: WorkflowEdge) -> Self {
        self.edges.push(edge);
        self
    }

    pub fn with_layout(mut self, layout: WorkflowLayout) -> Self {
        self.layout = layout;
        self
    }

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

    pub(crate) fn set_version(&mut self, version: WorkflowVersion) {
        self.version = version;
    }

    pub(crate) fn set_metadata(&mut self, metadata: WorkflowMetadata) {
        self.metadata = metadata;
    }

    pub(crate) fn set_layout(&mut self, layout: WorkflowLayout) {
        self.layout = layout;
    }

    pub(crate) fn nodes_mut(&mut self) -> &mut Vec<WorkflowNode> {
        &mut self.nodes
    }

    pub(crate) fn edges_mut(&mut self) -> &mut Vec<WorkflowEdge> {
        &mut self.edges
    }

    pub(crate) fn layout_mut(&mut self) -> &mut WorkflowLayout {
        &mut self.layout
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
    pub fn new(id: impl Into<NodeId>, type_id: impl Into<NodeTypeId>) -> Self {
        Self {
            id: id.into(),
            type_id: type_id.into(),
            label: None,
            params: BTreeMap::new(),
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_param(mut self, slot_id: impl Into<SlotId>, value: ParamValue) -> Self {
        self.params.insert(slot_id.into(), value);
        self
    }

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

    pub(crate) fn set_label(&mut self, label: Option<String>) {
        self.label = label;
    }

    pub(crate) fn params_mut(&mut self) -> &mut BTreeMap<SlotId, ParamValue> {
        &mut self.params
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WorkflowEdge {
    id: EdgeId,
    from: Endpoint,
    to: Endpoint,
}

impl WorkflowEdge {
    pub fn new(id: impl Into<EdgeId>, from: Endpoint, to: Endpoint) -> Self {
        Self {
            id: id.into(),
            from,
            to,
        }
    }

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
