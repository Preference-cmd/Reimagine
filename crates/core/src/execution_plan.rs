use crate::event::OperationReport;
use crate::model::{
    EdgeId, NodeId, NodeTypeId, SlotId, WorkflowId, WorkflowInputId, WorkflowOutputId,
    WorkflowVersion,
};

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExecutionPlanResult {
    plan: Option<ExecutionPlan>,
    report: OperationReport,
}

impl ExecutionPlanResult {
    pub fn new(plan: Option<ExecutionPlan>, report: OperationReport) -> Self {
        Self { plan, report }
    }

    pub fn plan(&self) -> Option<&ExecutionPlan> {
        self.plan.as_ref()
    }

    pub fn report(&self) -> &OperationReport {
        &self.report
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExecutionPlan {
    workflow_id: WorkflowId,
    workflow_version: WorkflowVersion,
    target_selection: RunTargetSelection,
    targets: Vec<RunTarget>,
    nodes: Vec<ExecutionNode>,
    edges: Vec<ExecutionEdge>,
    workflow_outputs: Vec<ExecutionWorkflowOutput>,
    stages: Vec<ExecutionStage>,
}

impl ExecutionPlan {
    pub fn new(
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        target_selection: RunTargetSelection,
        targets: Vec<RunTarget>,
        nodes: Vec<ExecutionNode>,
        edges: Vec<ExecutionEdge>,
        workflow_outputs: Vec<ExecutionWorkflowOutput>,
        stages: Vec<ExecutionStage>,
    ) -> Self {
        Self {
            workflow_id,
            workflow_version,
            target_selection,
            targets,
            nodes,
            edges,
            workflow_outputs,
            stages,
        }
    }

    pub fn workflow_id(&self) -> &WorkflowId {
        &self.workflow_id
    }

    pub fn workflow_version(&self) -> WorkflowVersion {
        self.workflow_version
    }

    pub fn target_selection(&self) -> &RunTargetSelection {
        &self.target_selection
    }

    pub fn targets(&self) -> &[RunTarget] {
        &self.targets
    }

    pub fn nodes(&self) -> &[ExecutionNode] {
        &self.nodes
    }

    pub fn edges(&self) -> &[ExecutionEdge] {
        &self.edges
    }

    pub fn workflow_outputs(&self) -> &[ExecutionWorkflowOutput] {
        &self.workflow_outputs
    }

    pub fn stages(&self) -> &[ExecutionStage] {
        &self.stages
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExecutionWorkflowOutput {
    output_id: WorkflowOutputId,
    source: ExecutionWorkflowOutputSource,
}

impl ExecutionWorkflowOutput {
    pub fn new(output_id: WorkflowOutputId, source: ExecutionWorkflowOutputSource) -> Self {
        Self { output_id, source }
    }

    pub fn output_id(&self) -> &WorkflowOutputId {
        &self.output_id
    }

    pub fn source(&self) -> &ExecutionWorkflowOutputSource {
        &self.source
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutionWorkflowOutputSource {
    NodeOutput { node_id: NodeId, slot_id: SlotId },
    WorkflowInput { workflow_input_id: WorkflowInputId },
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExecutionNode {
    node_id: NodeId,
    type_id: NodeTypeId,
    input_bindings: Vec<ExecutionInputBinding>,
    output_slots: Vec<SlotId>,
}

impl ExecutionNode {
    pub fn new(
        node_id: NodeId,
        type_id: NodeTypeId,
        input_bindings: Vec<ExecutionInputBinding>,
        output_slots: Vec<SlotId>,
    ) -> Self {
        Self {
            node_id,
            type_id,
            input_bindings,
            output_slots,
        }
    }

    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    pub fn type_id(&self) -> &NodeTypeId {
        &self.type_id
    }

    pub fn input_bindings(&self) -> &[ExecutionInputBinding] {
        &self.input_bindings
    }

    pub fn output_slots(&self) -> &[SlotId] {
        &self.output_slots
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExecutionInputBinding {
    slot_id: SlotId,
    source: ExecutionInputSource,
}

impl ExecutionInputBinding {
    pub fn new(slot_id: SlotId, source: ExecutionInputSource) -> Self {
        Self { slot_id, source }
    }

    pub fn slot_id(&self) -> &SlotId {
        &self.slot_id
    }

    pub fn source(&self) -> &ExecutionInputSource {
        &self.source
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutionInputSource {
    Edge {
        edge_id: EdgeId,
        from_node_id: NodeId,
        from_slot_id: SlotId,
    },
    WorkflowInput {
        edge_id: EdgeId,
        workflow_input_id: WorkflowInputId,
    },
    Param {
        slot_id: SlotId,
    },
    Default {
        slot_id: SlotId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExecutionEdge {
    edge_id: EdgeId,
    from_node_id: NodeId,
    from_slot_id: SlotId,
    to_node_id: NodeId,
    to_slot_id: SlotId,
}

impl ExecutionEdge {
    pub fn new(
        edge_id: EdgeId,
        from_node_id: NodeId,
        from_slot_id: SlotId,
        to_node_id: NodeId,
        to_slot_id: SlotId,
    ) -> Self {
        Self {
            edge_id,
            from_node_id,
            from_slot_id,
            to_node_id,
            to_slot_id,
        }
    }

    pub fn edge_id(&self) -> &EdgeId {
        &self.edge_id
    }

    pub fn from_node_id(&self) -> &NodeId {
        &self.from_node_id
    }

    pub fn from_slot_id(&self) -> &SlotId {
        &self.from_slot_id
    }

    pub fn to_node_id(&self) -> &NodeId {
        &self.to_node_id
    }

    pub fn to_slot_id(&self) -> &SlotId {
        &self.to_slot_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExecutionStage {
    index: usize,
    node_ids: Vec<NodeId>,
}

impl ExecutionStage {
    pub fn new(index: usize, node_ids: Vec<NodeId>) -> Self {
        Self { index, node_ids }
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn node_ids(&self) -> &[NodeId] {
        &self.node_ids
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", content = "targets", rename_all = "snake_case")]
pub enum RunTargetSelection {
    AllDefaultTargets,
    ExplicitTargets(Vec<RunTarget>),
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunTarget {
    Node { node_id: NodeId },
    NodeOutput { node_id: NodeId, slot_id: SlotId },
    WorkflowOutput { output_id: WorkflowOutputId },
}
