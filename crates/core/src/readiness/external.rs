use crate::diagnostic::{Diagnostic, project_diagnostic};
use crate::model::{ModelRef, NodeId, SlotId, WorkflowId, WorkflowVersion};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalReadinessContext {
    workflow_id: WorkflowId,
    workflow_version: WorkflowVersion,
    node_id: Option<NodeId>,
    slot_id: Option<SlotId>,
    workflow_input_id: Option<String>,
    path: String,
}

impl ExternalReadinessContext {
    pub fn new(
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        path: impl Into<String>,
    ) -> Self {
        Self {
            workflow_id,
            workflow_version,
            node_id: None,
            slot_id: None,
            workflow_input_id: None,
            path: path.into(),
        }
    }

    pub fn with_node(mut self, node_id: NodeId) -> Self {
        self.node_id = Some(node_id);
        self
    }

    pub fn with_slot(mut self, slot_id: SlotId) -> Self {
        self.slot_id = Some(slot_id);
        self
    }

    pub fn workflow_id(&self) -> &WorkflowId {
        &self.workflow_id
    }

    pub fn workflow_version(&self) -> WorkflowVersion {
        self.workflow_version
    }

    pub fn node_id(&self) -> Option<&NodeId> {
        self.node_id.as_ref()
    }

    pub fn slot_id(&self) -> Option<&SlotId> {
        self.slot_id.as_ref()
    }

    pub fn workflow_input_id(&self) -> Option<&str> {
        self.workflow_input_id.as_deref()
    }

    pub fn path(&self) -> &str {
        &self.path
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExternalReadinessSubject {
    ModelRef(ModelRef),
}

pub trait ExternalReadinessProvider {
    fn diagnostics_for(
        &self,
        context: &ExternalReadinessContext,
        subject: &ExternalReadinessSubject,
    ) -> Option<Vec<Diagnostic>>;
}

pub fn check_external_readiness(
    provider: &dyn ExternalReadinessProvider,
    context: &ExternalReadinessContext,
    subject: &ExternalReadinessSubject,
) -> Option<Vec<Diagnostic>> {
    provider.diagnostics_for(context, subject)
}

pub fn project_external_diagnostic(
    diagnostic: &Diagnostic,
    projected_primary: crate::diagnostic::DiagnosticTarget,
) -> Diagnostic {
    project_diagnostic(diagnostic, projected_primary)
}
