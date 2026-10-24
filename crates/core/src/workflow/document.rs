//! Document trait implementation for Workflow.

use crate::command::{WorkflowChange, WorkflowCommand};
use crate::diagnostic::{
    CorrelationId, Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName,
    DiagnosticTarget, DiagnosticTargetDomain,
};
use crate::document::{CommandApplication, Document};
use crate::model::{
    DiagnosticId, DocumentVersion, EdgeId, NodeCatalog, NodeId, WorkflowInputId, WorkflowOutputId,
};
use crate::validation::validate_structure;
use crate::workflow::{
    Endpoint, Workflow, WorkflowEdge, WorkflowInputDef, WorkflowNode, WorkflowOutputDef,
};

impl Document for Workflow {
    type Command = WorkflowCommand;
    type Change = WorkflowChange;
    type Context = dyn NodeCatalog;

    fn version(&self) -> DocumentVersion {
        self.version()
    }

    fn set_version(&mut self, version: DocumentVersion) {
        self.set_version(version);
    }

    fn apply_command(
        &mut self,
        command: &Self::Command,
        _context: &Self::Context,
        correlation_id: Option<&CorrelationId>,
    ) -> Result<CommandApplication<Self::Change>, Vec<Diagnostic>> {
        let mut forward_changes = Vec::new();
        let mut inverse_steps = Vec::new();
        let mut diagnostics = Vec::new();

        apply_workflow_command(
            self,
            command,
            &mut forward_changes,
            &mut inverse_steps,
            &mut diagnostics,
            correlation_id,
        );

        if !diagnostics.is_empty() {
            Err(diagnostics)
        } else {
            let inverse = inverse_steps.into_iter().flatten().collect();
            Ok(CommandApplication::new(forward_changes, inverse))
        }
    }

    fn validate(&self, context: &Self::Context) -> Vec<Diagnostic> {
        validate_structure(self, context).diagnostics().to_vec()
    }

    fn version_conflict_diagnostic(
        &self,
        current_version: DocumentVersion,
        base_version: DocumentVersion,
        correlation_id: Option<&CorrelationId>,
    ) -> Diagnostic {
        let mut diagnostic = Diagnostic::new(
            DiagnosticId::new(format!(
                "workflow-version-conflict-{}-{}",
                self.id().as_str(),
                base_version.get()
            )),
            DiagnosticCode::new("CORE/WORKFLOW_VERSION_CONFLICT"),
            DiagnosticSeverity::Error,
            DiagnosticSourceName::new("core"),
            format!(
                "workflow version conflict: expected {}, got {}",
                current_version, base_version
            ),
            DiagnosticTarget::new(DiagnosticTargetDomain::new("workflow"))
                .with_id(self.id().as_str()),
        );
        if let Some(correlation_id) = correlation_id.cloned() {
            diagnostic = diagnostic.with_correlation_id(correlation_id);
        }
        diagnostic
    }
}

fn apply_workflow_command(
    workflow: &mut Workflow,
    command: &WorkflowCommand,
    forward_changes: &mut Vec<WorkflowChange>,
    inverse_steps: &mut Vec<Vec<WorkflowChange>>,
    diagnostics: &mut Vec<Diagnostic>,
    correlation_id: Option<&CorrelationId>,
) {
    match command {
        WorkflowCommand::AddNode {
            node_id,
            type_id,
            label,
            params,
            position,
        } => {
            if find_node_index(workflow, node_id).is_some() {
                diagnostics.push(node_duplicate_diagnostic(
                    node_id.as_str(),
                    correlation_id,
                    "workflow node id already exists",
                ));
                return;
            }
            let mut node = WorkflowNode::new(node_id.clone(), type_id.clone());
            if let Some(label) = label {
                node = node.with_label(label.clone());
            }
            for (slot_id, value) in params {
                node = node.with_param(slot_id.clone(), value.clone());
            }
            workflow.nodes_mut().push(node.clone());
            if let Some(position) = position {
                workflow
                    .layout_mut()
                    .set_node_position(node_id.clone(), position.clone());
            }
            forward_changes.push(WorkflowChange::NodeAdded { node: node.clone() });
            let mut inverse = Vec::new();
            if let Some(position) = position.clone() {
                forward_changes.push(WorkflowChange::NodeMoved {
                    node_id: node_id.clone(),
                    before: None,
                    after: Some(position.clone()),
                });
                inverse.push(WorkflowChange::NodeMoved {
                    node_id: node_id.clone(),
                    before: Some(position),
                    after: None,
                });
            }
            inverse.push(WorkflowChange::NodeRemoved {
                node,
                removed_edges: Vec::new(),
                removed_layout: position.clone(),
            });
            inverse_steps.push(inverse);
        }
        WorkflowCommand::RemoveNode { node_id } => {
            let Some(index) = find_node_index(workflow, node_id) else {
                diagnostics.push(node_missing_diagnostic(
                    node_id.as_str(),
                    correlation_id,
                    "workflow node does not exist",
                ));
                return;
            };

            let node = workflow.nodes_mut().remove(index);
            let removed_edges = remove_edges_for_node(workflow, node_id);
            let removed_layout = workflow.layout_mut().remove_node(node_id);

            forward_changes.push(WorkflowChange::NodeRemoved {
                node: node.clone(),
                removed_edges: removed_edges.clone(),
                removed_layout: removed_layout.clone(),
            });

            let mut inverse = vec![WorkflowChange::NodeAdded { node }];
            if let Some(position) = removed_layout.clone() {
                inverse.push(WorkflowChange::NodeMoved {
                    node_id: node_id.clone(),
                    before: None,
                    after: Some(position),
                });
            }
            inverse.extend(
                removed_edges
                    .iter()
                    .cloned()
                    .map(|edge| WorkflowChange::EdgeAdded { edge }),
            );
            inverse_steps.push(inverse);
        }
        WorkflowCommand::Connect { edge_id, from, to } => {
            if find_edge_index(workflow, edge_id).is_some() {
                diagnostics.push(edge_duplicate_diagnostic(
                    edge_id.as_str(),
                    correlation_id,
                    "workflow edge id already exists",
                ));
                return;
            }
            let edge = WorkflowEdge::new(edge_id.clone(), from.clone(), to.clone());
            workflow.edges_mut().push(edge.clone());
            forward_changes.push(WorkflowChange::EdgeAdded { edge: edge.clone() });
            inverse_steps.push(vec![WorkflowChange::EdgeRemoved { edge }]);
        }
        WorkflowCommand::Disconnect { edge_id } => {
            let Some(index) = find_edge_index(workflow, edge_id) else {
                diagnostics.push(edge_missing_diagnostic(
                    edge_id.as_str(),
                    correlation_id,
                    "workflow edge does not exist",
                ));
                return;
            };
            let edge = workflow.edges_mut().remove(index);
            forward_changes.push(WorkflowChange::EdgeRemoved { edge: edge.clone() });
            inverse_steps.push(vec![WorkflowChange::EdgeAdded { edge }]);
        }
        WorkflowCommand::SetParam {
            node_id,
            slot_id,
            value,
        } => {
            let Some(node) = find_node_mut(workflow, node_id) else {
                diagnostics.push(node_missing_diagnostic(
                    node_id.as_str(),
                    correlation_id,
                    "workflow node does not exist",
                ));
                return;
            };
            let before = node.params_mut().insert(slot_id.clone(), value.clone());
            if before.as_ref() == Some(value) {
                return;
            }
            forward_changes.push(WorkflowChange::ParamSet {
                node_id: node_id.clone(),
                slot_id: slot_id.clone(),
                before: before.clone(),
                after: value.clone(),
            });
            inverse_steps.push(vec![match before {
                Some(previous) => WorkflowChange::ParamSet {
                    node_id: node_id.clone(),
                    slot_id: slot_id.clone(),
                    before: Some(value.clone()),
                    after: previous,
                },
                None => WorkflowChange::ParamRemoved {
                    node_id: node_id.clone(),
                    slot_id: slot_id.clone(),
                    before: value.clone(),
                },
            }]);
        }
        WorkflowCommand::RemoveParam { node_id, slot_id } => {
            let Some(node) = find_node_mut(workflow, node_id) else {
                diagnostics.push(node_missing_diagnostic(
                    node_id.as_str(),
                    correlation_id,
                    "workflow node does not exist",
                ));
                return;
            };
            let Some(before) = node.params_mut().remove(slot_id) else {
                return;
            };
            forward_changes.push(WorkflowChange::ParamRemoved {
                node_id: node_id.clone(),
                slot_id: slot_id.clone(),
                before: before.clone(),
            });
            inverse_steps.push(vec![WorkflowChange::ParamSet {
                node_id: node_id.clone(),
                slot_id: slot_id.clone(),
                before: None,
                after: before,
            }]);
        }
        WorkflowCommand::MoveNode { node_id, position } => {
            if find_node_index(workflow, node_id).is_none() {
                diagnostics.push(node_missing_diagnostic(
                    node_id.as_str(),
                    correlation_id,
                    "workflow node does not exist",
                ));
                return;
            }
            let before = workflow.layout().nodes().get(node_id).cloned();
            if before.as_ref() == Some(position) {
                return;
            }
            workflow
                .layout_mut()
                .set_node_position(node_id.clone(), position.clone());
            forward_changes.push(WorkflowChange::NodeMoved {
                node_id: node_id.clone(),
                before: before.clone(),
                after: Some(position.clone()),
            });
            inverse_steps.push(vec![WorkflowChange::NodeMoved {
                node_id: node_id.clone(),
                before: Some(position.clone()),
                after: before,
            }]);
        }
        WorkflowCommand::ApplyLayout { layout } => {
            let before = workflow.layout().clone();
            if before == *layout {
                return;
            }
            workflow.set_layout(layout.clone());
            forward_changes.push(WorkflowChange::LayoutApplied {
                before: before.clone(),
                after: layout.clone(),
            });
            inverse_steps.push(vec![WorkflowChange::LayoutApplied {
                before: layout.clone(),
                after: before,
            }]);
        }
        WorkflowCommand::SetNodeLabel { node_id, label } => {
            let Some(node) = find_node_mut(workflow, node_id) else {
                diagnostics.push(node_missing_diagnostic(
                    node_id.as_str(),
                    correlation_id,
                    "workflow node does not exist",
                ));
                return;
            };
            let before = node.label().map(str::to_owned);
            if before == *label {
                return;
            }
            node.set_label(label.clone());
            forward_changes.push(WorkflowChange::NodeLabelSet {
                node_id: node_id.clone(),
                before: before.clone(),
                after: label.clone(),
            });
            inverse_steps.push(vec![WorkflowChange::NodeLabelSet {
                node_id: node_id.clone(),
                before: label.clone(),
                after: before,
            }]);
        }
        WorkflowCommand::SetWorkflowMetadata { metadata } => {
            let before = workflow.metadata().clone();
            if before == *metadata {
                return;
            }
            workflow.set_metadata(metadata.clone());
            forward_changes.push(WorkflowChange::WorkflowMetadataSet {
                before: before.clone(),
                after: metadata.clone(),
            });
            inverse_steps.push(vec![WorkflowChange::WorkflowMetadataSet {
                before: metadata.clone(),
                after: before,
            }]);
        }
        WorkflowCommand::AddWorkflowInput {
            input_id,
            slot,
            kind,
        } => {
            if workflow.interface().input(input_id).is_some() {
                diagnostics.push(interface_diagnostic(
                    format!("workflow-input-duplicate-{input_id}"),
                    "CORE/WORKFLOW_INPUT_DUPLICATE",
                    "workflow input id already exists",
                    input_id.as_str(),
                    correlation_id,
                ));
                return;
            }
            if workflow
                .interface()
                .inputs()
                .iter()
                .any(|input| input.slot() == slot)
            {
                diagnostics.push(interface_diagnostic(
                    format!("workflow-input-slot-taken-{slot}"),
                    "CORE/WORKFLOW_INPUT_SLOT_TAKEN",
                    "workflow input slot is already used",
                    slot.as_str(),
                    correlation_id,
                ));
                return;
            }
            let input = WorkflowInputDef::new(input_id.clone(), slot.clone(), *kind);
            workflow.interface_mut().add_input(input.clone());
            forward_changes.push(WorkflowChange::WorkflowInputAdded {
                input: input.clone(),
            });
            inverse_steps.push(vec![WorkflowChange::WorkflowInputRemoved {
                input,
                removed_edges: Vec::new(),
            }]);
        }
        WorkflowCommand::RemoveWorkflowInput { input_id } => {
            let Some(input) = workflow.interface_mut().remove_input(input_id) else {
                diagnostics.push(interface_diagnostic(
                    format!("workflow-input-missing-{input_id}"),
                    "CORE/WORKFLOW_INPUT_MISSING",
                    "workflow input does not exist",
                    input_id.as_str(),
                    correlation_id,
                ));
                return;
            };
            let removed_edges = remove_edges_for_workflow_input(workflow, input_id);
            forward_changes.push(WorkflowChange::WorkflowInputRemoved {
                input: input.clone(),
                removed_edges: removed_edges.clone(),
            });
            let mut inverse = vec![WorkflowChange::WorkflowInputAdded { input }];
            inverse.extend(
                removed_edges
                    .iter()
                    .cloned()
                    .map(|edge| WorkflowChange::EdgeAdded { edge }),
            );
            inverse_steps.push(inverse);
        }
        WorkflowCommand::AddWorkflowOutput {
            output_id,
            slot,
            kind,
        } => {
            if workflow.interface().output(output_id).is_some() {
                diagnostics.push(interface_diagnostic(
                    format!("workflow-output-duplicate-{output_id}"),
                    "CORE/WORKFLOW_OUTPUT_DUPLICATE",
                    "workflow output id already exists",
                    output_id.as_str(),
                    correlation_id,
                ));
                return;
            }
            if workflow
                .interface()
                .outputs()
                .iter()
                .any(|output| output.slot() == slot)
            {
                diagnostics.push(interface_diagnostic(
                    format!("workflow-output-slot-taken-{slot}"),
                    "CORE/WORKFLOW_OUTPUT_SLOT_TAKEN",
                    "workflow output slot is already used",
                    slot.as_str(),
                    correlation_id,
                ));
                return;
            }
            let output = WorkflowOutputDef::new(output_id.clone(), slot.clone(), *kind);
            workflow.interface_mut().add_output(output.clone());
            forward_changes.push(WorkflowChange::WorkflowOutputAdded {
                output: output.clone(),
            });
            inverse_steps.push(vec![WorkflowChange::WorkflowOutputRemoved {
                output,
                removed_edges: Vec::new(),
            }]);
        }
        WorkflowCommand::RemoveWorkflowOutput { output_id } => {
            let Some(output) = workflow.interface_mut().remove_output(output_id) else {
                diagnostics.push(interface_diagnostic(
                    format!("workflow-output-missing-{output_id}"),
                    "CORE/WORKFLOW_OUTPUT_MISSING",
                    "workflow output does not exist",
                    output_id.as_str(),
                    correlation_id,
                ));
                return;
            };
            let removed_edges = remove_edges_for_workflow_output(workflow, output_id);
            forward_changes.push(WorkflowChange::WorkflowOutputRemoved {
                output: output.clone(),
                removed_edges: removed_edges.clone(),
            });
            let mut inverse = vec![WorkflowChange::WorkflowOutputAdded { output }];
            inverse.extend(
                removed_edges
                    .iter()
                    .cloned()
                    .map(|edge| WorkflowChange::EdgeAdded { edge }),
            );
            inverse_steps.push(inverse);
        }
    }
}

fn find_node_index(workflow: &Workflow, node_id: &NodeId) -> Option<usize> {
    workflow
        .nodes()
        .iter()
        .position(|node| node.id() == node_id)
}

fn find_node_mut<'a>(workflow: &'a mut Workflow, node_id: &NodeId) -> Option<&'a mut WorkflowNode> {
    workflow
        .nodes_mut()
        .iter_mut()
        .find(|node| node.id() == node_id)
}

fn find_edge_index(workflow: &Workflow, edge_id: &EdgeId) -> Option<usize> {
    workflow
        .edges()
        .iter()
        .position(|edge| edge.id() == edge_id)
}

fn remove_edges_for_node(workflow: &mut Workflow, node_id: &NodeId) -> Vec<WorkflowEdge> {
    let mut removed = Vec::new();
    let mut kept = Vec::new();

    for edge in workflow.edges().iter().cloned() {
        if endpoint_references_node(edge.from(), node_id)
            || endpoint_references_node(edge.to(), node_id)
        {
            removed.push(edge);
        } else {
            kept.push(edge);
        }
    }

    *workflow.edges_mut() = kept;
    removed
}

fn remove_edges_for_workflow_input(
    workflow: &mut Workflow,
    input_id: &WorkflowInputId,
) -> Vec<WorkflowEdge> {
    let mut removed = Vec::new();
    let mut kept = Vec::new();

    for edge in workflow.edges().iter().cloned() {
        if matches!(
            edge.from(),
            Endpoint::WorkflowInput { workflow_input }
                if *workflow_input == *input_id
        ) {
            removed.push(edge);
        } else {
            kept.push(edge);
        }
    }

    *workflow.edges_mut() = kept;
    removed
}

fn remove_edges_for_workflow_output(
    workflow: &mut Workflow,
    output_id: &WorkflowOutputId,
) -> Vec<WorkflowEdge> {
    let mut removed = Vec::new();
    let mut kept = Vec::new();

    for edge in workflow.edges().iter().cloned() {
        if matches!(
            edge.to(),
            Endpoint::WorkflowOutput { workflow_output }
                if *workflow_output == *output_id
        ) {
            removed.push(edge);
        } else {
            kept.push(edge);
        }
    }

    *workflow.edges_mut() = kept;
    removed
}

fn endpoint_references_node(endpoint: &Endpoint, node_id: &NodeId) -> bool {
    matches!(endpoint, Endpoint::NodeSlot { node, .. } if node == node_id)
}

fn node_missing_diagnostic(
    node_id: &str,
    correlation_id: Option<&CorrelationId>,
    message: &str,
) -> Diagnostic {
    simple_diagnostic(
        format!("workflow-node-missing-{node_id}"),
        "CORE/WORKFLOW_NODE_MISSING",
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new("workflow")).with_id(node_id),
        correlation_id,
    )
}

fn edge_missing_diagnostic(
    edge_id: &str,
    correlation_id: Option<&CorrelationId>,
    message: &str,
) -> Diagnostic {
    simple_diagnostic(
        format!("workflow-edge-missing-{edge_id}"),
        "CORE/WORKFLOW_EDGE_MISSING",
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new("workflow")).with_id(edge_id),
        correlation_id,
    )
}

fn node_duplicate_diagnostic(
    node_id: &str,
    correlation_id: Option<&CorrelationId>,
    message: &str,
) -> Diagnostic {
    simple_diagnostic(
        format!("workflow-node-duplicate-{node_id}"),
        "CORE/WORKFLOW_NODE_DUPLICATE",
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new("workflow")).with_id(node_id),
        correlation_id,
    )
}

fn edge_duplicate_diagnostic(
    edge_id: &str,
    correlation_id: Option<&CorrelationId>,
    message: &str,
) -> Diagnostic {
    simple_diagnostic(
        format!("workflow-edge-duplicate-{edge_id}"),
        "CORE/WORKFLOW_EDGE_DUPLICATE",
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new("workflow")).with_id(edge_id),
        correlation_id,
    )
}

fn interface_diagnostic(
    id: String,
    code: &str,
    message: &str,
    target_id: &str,
    correlation_id: Option<&CorrelationId>,
) -> Diagnostic {
    simple_diagnostic(
        id,
        code,
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new("workflow")).with_id(target_id),
        correlation_id,
    )
}

fn simple_diagnostic(
    id: String,
    code: &str,
    message: &str,
    primary: DiagnosticTarget,
    correlation_id: Option<&CorrelationId>,
) -> Diagnostic {
    let mut diagnostic = Diagnostic::new(
        DiagnosticId::new(id),
        DiagnosticCode::new(code),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("core"),
        message,
        primary,
    );
    if let Some(correlation_id) = correlation_id.cloned() {
        diagnostic = diagnostic.with_correlation_id(correlation_id);
    }
    diagnostic
}
