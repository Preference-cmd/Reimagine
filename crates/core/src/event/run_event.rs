use crate::diagnostic::{CorrelationId, Diagnostic};
use crate::model::{ArtifactId, NodeId, RunId, WorkflowId, WorkflowVersion};

use super::Timestamp;

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RunEventId(String);

impl RunEventId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for RunEventId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for RunEventId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum RunEventKind {
    RunQueued,
    RunStarted,
    RunCompleted,
    RunFailed,
    RunCancelled,
    NodeQueued,
    NodeStarted,
    NodeCompleted,
    NodeFailed,
    NodeSkipped,
    NodeCancelled,
    ArtifactCreated,
    PreviewUpdated,
    DiagnosticEmitted,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RunEvent {
    id: RunEventId,
    run_id: RunId,
    workflow_id: WorkflowId,
    workflow_version: WorkflowVersion,
    kind: RunEventKind,
    node_id: Option<NodeId>,
    artifact: Option<ArtifactId>,
    diagnostics: Vec<Diagnostic>,
    created_at: Timestamp,
    correlation_id: Option<CorrelationId>,
}

impl RunEvent {
    pub fn new(
        id: impl Into<RunEventId>,
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        kind: RunEventKind,
        created_at: Timestamp,
    ) -> Self {
        Self {
            id: id.into(),
            run_id,
            workflow_id,
            workflow_version,
            kind,
            node_id: None,
            artifact: None,
            diagnostics: Vec::new(),
            created_at,
            correlation_id: None,
        }
    }

    pub fn with_node_id(mut self, node_id: impl Into<NodeId>) -> Self {
        self.node_id = Some(node_id.into());
        self
    }

    pub fn with_artifact(mut self, artifact: ArtifactId) -> Self {
        self.artifact = Some(artifact);
        self
    }

    pub fn with_diagnostic(mut self, diagnostic: Diagnostic) -> Self {
        self.diagnostics.push(diagnostic);
        self
    }

    pub fn with_correlation_id(mut self, correlation_id: CorrelationId) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    pub fn id(&self) -> &RunEventId {
        &self.id
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn workflow_id(&self) -> &WorkflowId {
        &self.workflow_id
    }

    pub fn workflow_version(&self) -> WorkflowVersion {
        self.workflow_version
    }

    pub fn kind(&self) -> RunEventKind {
        self.kind
    }

    pub fn node_id(&self) -> Option<&NodeId> {
        self.node_id.as_ref()
    }

    pub fn artifact(&self) -> Option<&ArtifactId> {
        self.artifact.as_ref()
    }

    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    pub fn created_at(&self) -> &Timestamp {
        &self.created_at
    }

    pub fn correlation_id(&self) -> Option<&CorrelationId> {
        self.correlation_id.as_ref()
    }
}
