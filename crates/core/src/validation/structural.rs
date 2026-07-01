use std::collections::HashSet;

use crate::event::OperationReport;
use crate::model::{EdgeId, NodeCatalog, NodeDef, NodeId, SlotId, SlotKind};
use crate::workflow::{Endpoint, Workflow, WorkflowSchemaVersion};

mod diagnostics;
mod kinds;

use diagnostics::{edge_diagnostic, node_diagnostic, workflow_diagnostic};
use kinds::param_value_matches_slot;

pub fn validate_structure(workflow: &Workflow, node_catalog: &impl NodeCatalog) -> OperationReport {
    let mut validator = StructuralValidator::new(workflow, node_catalog);
    validator.validate();
    validator.report
}

struct StructuralValidator<'a, C> {
    workflow: &'a Workflow,
    node_catalog: &'a C,
    report: OperationReport,
}

impl<'a, C: NodeCatalog> StructuralValidator<'a, C> {
    fn new(workflow: &'a Workflow, node_catalog: &'a C) -> Self {
        Self {
            workflow,
            node_catalog,
            report: OperationReport::new(),
        }
    }

    fn validate(&mut self) {
        self.validate_schema_version();
        self.validate_node_ids_and_types();
        self.validate_node_params();
        self.validate_edges();
        self.validate_layout();
    }

    fn validate_schema_version(&mut self) {
        if self.workflow.schema_version() != &WorkflowSchemaVersion::default() {
            self.push_workflow_diagnostic(
                "schema_unsupported",
                "CORE/WORKFLOW_SCHEMA_UNSUPPORTED",
                "workflow schema version is unsupported",
                Some("schema_version"),
            );
        }
    }

    fn validate_node_ids_and_types(&mut self) {
        let mut seen = HashSet::new();
        for node in self.workflow.nodes() {
            if !seen.insert(node.id().clone()) {
                self.push_node_diagnostic(
                    "node_id_duplicate",
                    node.id(),
                    "CORE/WORKFLOW_NODE_ID_DUPLICATE",
                    "workflow node id is duplicated",
                    Some("id"),
                );
            }

            if self.node_catalog.get(node.type_id()).is_none() {
                self.push_node_diagnostic(
                    "node_type_unknown",
                    node.id(),
                    "CORE/WORKFLOW_NODE_TYPE_UNKNOWN",
                    "workflow node type is not available in the node catalog",
                    Some("type_id"),
                );
            }
        }
    }

    fn validate_node_params(&mut self) {
        for node in self.workflow.nodes() {
            let Some(node_def) = self.node_catalog.get(node.type_id()) else {
                continue;
            };

            for (slot_id, value) in node.params() {
                let Some(input_slot) = node_def.input_slot(slot_id) else {
                    self.push_node_diagnostic(
                        "param_slot_missing",
                        node.id(),
                        "CORE/WORKFLOW_PARAM_SLOT_MISSING",
                        "workflow node param references a missing input slot",
                        Some(format!("params.{}", slot_id.as_str())),
                    );
                    continue;
                };

                if input_slot.is_dynamic() {
                    self.push_node_diagnostic(
                        "param_on_dynamic_slot",
                        node.id(),
                        "CORE/WORKFLOW_PARAM_ON_DYNAMIC_SLOT",
                        "dynamic input slots cannot store static params",
                        Some(format!("params.{}", slot_id.as_str())),
                    );
                }

                if !param_value_matches_slot(value, input_slot.kind()) {
                    self.push_node_diagnostic(
                        "param_kind_mismatch",
                        node.id(),
                        "CORE/WORKFLOW_PARAM_KIND_MISMATCH",
                        "workflow node param value kind does not match the input slot kind",
                        Some(format!("params.{}", slot_id.as_str())),
                    );
                }
            }
        }
    }

    fn validate_edges(&mut self) {
        let mut seen_edge_ids = HashSet::new();
        let mut incoming_inputs = HashSet::<(NodeId, SlotId)>::new();

        for edge in self.workflow.edges() {
            if !seen_edge_ids.insert(edge.id().clone()) {
                self.push_edge_diagnostic(
                    "edge_id_duplicate",
                    edge.id(),
                    "CORE/WORKFLOW_EDGE_ID_DUPLICATE",
                    "workflow edge id is duplicated",
                    Some("id"),
                );
            }

            let from_kind = self.validate_from_endpoint(edge.id(), edge.from());
            let to_kind = self.validate_to_endpoint(edge.id(), edge.to());

            if let (Some(from_kind), Some(to_kind)) = (from_kind, to_kind)
                && from_kind != to_kind
            {
                self.push_edge_diagnostic(
                    "slot_kind_mismatch",
                    edge.id(),
                    "CORE/WORKFLOW_SLOT_KIND_MISMATCH",
                    "edge connects incompatible slot kinds",
                    Some("to.slot"),
                );
            }

            if let Endpoint::NodeSlot { node, slot } = edge.to() {
                let key = (node.clone(), slot.clone());
                if !incoming_inputs.insert(key) {
                    self.push_edge_diagnostic(
                        "input_edge_duplicate",
                        edge.id(),
                        "CORE/WORKFLOW_INPUT_EDGE_DUPLICATE",
                        "input slot has more than one incoming edge",
                        Some("to"),
                    );
                }
            }
        }
    }

    fn validate_from_endpoint(
        &mut self,
        edge_id: &EdgeId,
        endpoint: &Endpoint,
    ) -> Option<SlotKind> {
        match endpoint {
            Endpoint::NodeSlot { node, slot } => {
                let node_def = self.node_def_for_endpoint(edge_id, node, "from.node")?;
                let Some(output_slot) = node_def.output_slot(slot) else {
                    self.push_edge_diagnostic(
                        "endpoint_slot_missing",
                        edge_id,
                        "CORE/WORKFLOW_ENDPOINT_SLOT_MISSING",
                        "edge source slot does not exist on the source node type",
                        Some("from.slot"),
                    );
                    return None;
                };
                Some(output_slot.kind())
            }
            Endpoint::WorkflowInput { workflow_input } => {
                let Some(input_def) = self.workflow.interface().input(workflow_input) else {
                    self.push_edge_diagnostic(
                        "interface_input_missing",
                        edge_id,
                        "CORE/WORKFLOW_INTERFACE_INPUT_MISSING",
                        "edge source workflow input does not exist in the workflow interface",
                        Some("from.workflow_input"),
                    );
                    return None;
                };
                Some(input_def.kind())
            }
            Endpoint::WorkflowOutput { .. } => {
                self.push_edge_diagnostic(
                    "endpoint_direction_invalid",
                    edge_id,
                    "CORE/WORKFLOW_ENDPOINT_DIRECTION_INVALID",
                    "workflow output cannot be used as an edge source",
                    Some("from"),
                );
                None
            }
        }
    }

    fn validate_to_endpoint(&mut self, edge_id: &EdgeId, endpoint: &Endpoint) -> Option<SlotKind> {
        match endpoint {
            Endpoint::NodeSlot { node, slot } => {
                let node_def = self.node_def_for_endpoint(edge_id, node, "to.node")?;
                let Some(input_slot) = node_def.input_slot(slot) else {
                    self.push_edge_diagnostic(
                        "endpoint_slot_missing",
                        edge_id,
                        "CORE/WORKFLOW_ENDPOINT_SLOT_MISSING",
                        "edge target slot does not exist on the target node type",
                        Some("to.slot"),
                    );
                    return None;
                };
                Some(input_slot.kind())
            }
            Endpoint::WorkflowOutput { workflow_output } => {
                let Some(output_def) = self.workflow.interface().output(workflow_output) else {
                    self.push_edge_diagnostic(
                        "interface_output_missing",
                        edge_id,
                        "CORE/WORKFLOW_INTERFACE_OUTPUT_MISSING",
                        "edge target workflow output does not exist in the workflow interface",
                        Some("to.workflow_output"),
                    );
                    return None;
                };
                Some(output_def.kind())
            }
            Endpoint::WorkflowInput { .. } => {
                self.push_edge_diagnostic(
                    "endpoint_direction_invalid",
                    edge_id,
                    "CORE/WORKFLOW_ENDPOINT_DIRECTION_INVALID",
                    "workflow input cannot be used as an edge target",
                    Some("to"),
                );
                None
            }
        }
    }

    fn node_def_for_endpoint(
        &mut self,
        edge_id: &EdgeId,
        node_id: &NodeId,
        path: &'static str,
    ) -> Option<&NodeDef> {
        let Some(node) = self
            .workflow
            .nodes()
            .iter()
            .find(|node| node.id() == node_id)
        else {
            self.push_edge_diagnostic(
                "endpoint_node_missing",
                edge_id,
                "CORE/WORKFLOW_ENDPOINT_NODE_MISSING",
                "edge endpoint references a missing node",
                Some(path),
            );
            return None;
        };

        self.node_catalog.get(node.type_id())
    }

    fn validate_layout(&mut self) {
        let node_ids: HashSet<NodeId> = self
            .workflow
            .nodes()
            .iter()
            .map(|node| node.id().clone())
            .collect();

        for node_id in self.workflow.layout().nodes().keys() {
            if !node_ids.contains(node_id) {
                self.push_workflow_diagnostic(
                    "layout_node_missing",
                    "CORE/WORKFLOW_LAYOUT_NODE_MISSING",
                    "workflow layout references a missing node",
                    Some(format!("layout.nodes.{}", node_id.as_str())),
                );
            }
        }
    }

    fn push_workflow_diagnostic(
        &mut self,
        suffix: &str,
        code: &str,
        message: &str,
        path: Option<impl Into<String>>,
    ) {
        self.report.push_diagnostic(workflow_diagnostic(
            suffix,
            self.workflow.id(),
            code,
            message,
            path.map(Into::into),
        ));
    }

    fn push_node_diagnostic(
        &mut self,
        suffix: &str,
        node_id: &NodeId,
        code: &str,
        message: &str,
        path: Option<impl Into<String>>,
    ) {
        self.report.push_diagnostic(node_diagnostic(
            suffix,
            node_id,
            code,
            message,
            path.map(Into::into),
        ));
    }

    fn push_edge_diagnostic(
        &mut self,
        suffix: &str,
        edge_id: &EdgeId,
        code: &str,
        message: &str,
        path: Option<impl Into<String>>,
    ) {
        self.report.push_diagnostic(edge_diagnostic(
            suffix,
            edge_id,
            code,
            message,
            path.map(Into::into),
        ));
    }
}
