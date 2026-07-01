use crate::command::{
    CommandBatch, CommandResult, CommandResultStatus, WorkflowChange, WorkflowCommand,
};
use crate::diagnostic::{
    CorrelationId, Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName,
    DiagnosticTarget, DiagnosticTargetDomain,
};
use crate::history::{HistoryEntry, WorkflowHistory};
use crate::model::{DiagnosticId, EdgeId, HistoryEntryId, NodeCatalog, NodeId, WorkflowVersion};
use crate::validation::validate_structure;
use crate::workflow::{Endpoint, Workflow, WorkflowEdge, WorkflowNode};

pub struct WorkflowSession {
    workflow: Workflow,
    history: WorkflowHistory,
}

impl WorkflowSession {
    pub fn new(workflow: Workflow) -> Self {
        Self {
            workflow,
            history: WorkflowHistory::new(),
        }
    }

    pub fn workflow(&self) -> &Workflow {
        &self.workflow
    }

    pub fn version(&self) -> WorkflowVersion {
        self.workflow.version()
    }

    pub fn history(&self) -> &WorkflowHistory {
        &self.history
    }

    pub fn preview_batch(
        &self,
        node_catalog: &impl NodeCatalog,
        batch: CommandBatch,
    ) -> CommandResult {
        match self.evaluate_batch(node_catalog, &batch) {
            BatchEvaluation::Rejected { diagnostics } => CommandResult::new(
                CommandResultStatus::Rejected,
                self.version(),
                Vec::new(),
                diagnostics,
                None,
            ),
            BatchEvaluation::NoOp => CommandResult::new(
                CommandResultStatus::NoOp,
                self.version(),
                Vec::new(),
                Vec::new(),
                None,
            ),
            BatchEvaluation::Applied {
                projected_version,
                forward_changes,
                ..
            } => CommandResult::new(
                CommandResultStatus::Applied,
                projected_version,
                forward_changes,
                Vec::new(),
                None,
            ),
        }
    }

    pub fn apply_batch(
        &mut self,
        node_catalog: &impl NodeCatalog,
        batch: CommandBatch,
    ) -> CommandResult {
        match self.evaluate_batch(node_catalog, &batch) {
            BatchEvaluation::Rejected { diagnostics } => CommandResult::new(
                CommandResultStatus::Rejected,
                self.version(),
                Vec::new(),
                diagnostics,
                None,
            ),
            BatchEvaluation::NoOp => CommandResult::new(
                CommandResultStatus::NoOp,
                self.version(),
                Vec::new(),
                Vec::new(),
                None,
            ),
            BatchEvaluation::Applied {
                workflow,
                projected_version,
                forward_changes,
                inverse_changes,
            } => {
                let mut workflow = *workflow;
                let before = self.workflow.clone();
                workflow.set_version(projected_version);
                let history_entry_id = history_entry_id(batch.id().as_str());
                let history_entry = HistoryEntry::new(
                    history_entry_id.clone(),
                    batch.clone(),
                    before,
                    workflow.clone(),
                    forward_changes.clone(),
                    inverse_changes,
                    batch.created_at().clone(),
                );

                self.workflow = workflow;
                self.history.truncate_to_cursor();
                self.history.push(history_entry);

                CommandResult::new(
                    CommandResultStatus::Applied,
                    projected_version,
                    forward_changes,
                    Vec::new(),
                    Some(history_entry_id),
                )
            }
        }
    }

    pub fn undo(&mut self) -> Option<CommandResult> {
        let entry = self.history.entry_to_undo()?.clone();
        let current_version = self.version();
        let next_version = increment_version(current_version);
        let mut restored = entry.before().clone();
        restored.set_version(next_version);

        self.workflow = restored;
        self.history.move_cursor_back();

        let mut changes = entry.inverse_changes().to_vec();
        changes.push(WorkflowChange::VersionAdvanced {
            before: current_version,
            after: next_version,
        });

        Some(CommandResult::new(
            CommandResultStatus::Applied,
            next_version,
            changes,
            Vec::new(),
            None,
        ))
    }

    pub fn redo(&mut self) -> Option<CommandResult> {
        let entry = self.history.entry_to_redo()?.clone();
        let current_version = self.version();
        let next_version = increment_version(current_version);
        let mut restored = entry.after().clone();
        restored.set_version(next_version);

        self.workflow = restored;
        self.history.move_cursor_forward();

        let mut changes = without_version_change(entry.forward_changes());
        changes.push(WorkflowChange::VersionAdvanced {
            before: current_version,
            after: next_version,
        });

        Some(CommandResult::new(
            CommandResultStatus::Applied,
            next_version,
            changes,
            Vec::new(),
            None,
        ))
    }

    fn evaluate_batch(
        &self,
        node_catalog: &impl NodeCatalog,
        batch: &CommandBatch,
    ) -> BatchEvaluation {
        let current_version = self.version();
        if batch.base_version() != current_version {
            return BatchEvaluation::Rejected {
                diagnostics: vec![version_conflict_diagnostic(
                    self.workflow.id().as_str(),
                    current_version,
                    batch.base_version(),
                    batch.correlation_id(),
                )],
            };
        }

        let mut working = self.workflow.clone();
        let mut forward_changes = Vec::new();
        let mut inverse_steps = Vec::<Vec<WorkflowChange>>::new();
        let mut diagnostics = Vec::new();

        for command in batch.commands() {
            apply_command(
                &mut working,
                command,
                &mut forward_changes,
                &mut inverse_steps,
                &mut diagnostics,
                batch.correlation_id(),
            );
        }

        if diagnostics.is_empty() {
            let report = validate_structure(&working, node_catalog);
            diagnostics.extend(report.diagnostics().iter().cloned());
        }

        if !diagnostics.is_empty() {
            return BatchEvaluation::Rejected { diagnostics };
        }

        if forward_changes.is_empty() {
            return BatchEvaluation::NoOp;
        }

        let projected_version = increment_version(current_version);
        forward_changes.push(WorkflowChange::VersionAdvanced {
            before: current_version,
            after: projected_version,
        });

        let inverse_changes = inverse_steps.into_iter().rev().flatten().collect();

        BatchEvaluation::Applied {
            workflow: Box::new(working),
            projected_version,
            forward_changes,
            inverse_changes,
        }
    }
}

enum BatchEvaluation {
    Rejected {
        diagnostics: Vec<Diagnostic>,
    },
    NoOp,
    Applied {
        workflow: Box<Workflow>,
        projected_version: WorkflowVersion,
        forward_changes: Vec<WorkflowChange>,
        inverse_changes: Vec<WorkflowChange>,
    },
}

fn apply_command(
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
    }
}

fn without_version_change(changes: &[WorkflowChange]) -> Vec<WorkflowChange> {
    changes
        .iter()
        .filter(|change| !matches!(change, WorkflowChange::VersionAdvanced { .. }))
        .cloned()
        .collect()
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

fn endpoint_references_node(endpoint: &Endpoint, node_id: &NodeId) -> bool {
    matches!(endpoint, Endpoint::NodeSlot { node, .. } if node == node_id)
}

fn increment_version(version: WorkflowVersion) -> WorkflowVersion {
    WorkflowVersion::new(version.get() + 1)
}

fn history_entry_id(batch_id: &str) -> HistoryEntryId {
    HistoryEntryId::new(format!("history:{batch_id}"))
}

fn version_conflict_diagnostic(
    workflow_id: &str,
    current_version: WorkflowVersion,
    batch_version: WorkflowVersion,
    correlation_id: Option<&CorrelationId>,
) -> Diagnostic {
    let mut diagnostic = Diagnostic::new(
        DiagnosticId::new(format!(
            "workflow-version-conflict-{}-{}",
            workflow_id,
            batch_version.get()
        )),
        DiagnosticCode::new("CORE/WORKFLOW_VERSION_CONFLICT"),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("core"),
        format!(
            "workflow version conflict: expected {}, got {}",
            current_version, batch_version
        ),
        DiagnosticTarget::new(DiagnosticTargetDomain::new("workflow")).with_id(workflow_id),
    );
    if let Some(correlation_id) = correlation_id.cloned() {
        diagnostic = diagnostic.with_correlation_id(correlation_id);
    }
    diagnostic
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
