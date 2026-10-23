use reimagine_core::command::{CommandActor, CommandBatch, CommandResult};
use reimagine_core::model::{ProposalId, WorkflowId, WorkflowVersion};

/// Status of a workflow proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalStatus {
    Pending,
    Approved,
    Rejected,
    Superseded,
    Conflicted,
}

/// Receipt for a stored workflow proposal.
///
/// Carries enough metadata for the proposing operator to reference the
/// proposal later, and an explicit `effective` flag set to `false` because
/// proposals do not mutate the workflow directly.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProposalReceipt {
    proposal_id: ProposalId,
    workflow_id: WorkflowId,
    base_version: WorkflowVersion,
    preview_result: CommandResult,
    status: ProposalStatus,
    effective: bool,
}

impl ProposalReceipt {
    pub fn new(
        proposal_id: ProposalId,
        workflow_id: WorkflowId,
        base_version: WorkflowVersion,
        preview_result: CommandResult,
    ) -> Self {
        Self {
            proposal_id,
            workflow_id,
            base_version,
            preview_result,
            status: ProposalStatus::Pending,
            effective: false,
        }
    }

    pub fn proposal_id(&self) -> &ProposalId {
        &self.proposal_id
    }

    pub fn workflow_id(&self) -> &WorkflowId {
        &self.workflow_id
    }

    pub fn base_version(&self) -> WorkflowVersion {
        self.base_version
    }

    pub fn preview_result(&self) -> &CommandResult {
        &self.preview_result
    }

    pub fn status(&self) -> ProposalStatus {
        self.status
    }

    pub fn effective(&self) -> bool {
        self.effective
    }

    pub fn with_status(mut self, status: ProposalStatus) -> Self {
        self.status = status;
        self
    }
}

/// Internal V1 workflow proposal stored in `WorkflowService`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorkflowProposal {
    proposal_id: ProposalId,
    workflow_id: WorkflowId,
    base_version: WorkflowVersion,
    proposed_by: CommandActor,
    command_batch: CommandBatch,
    preview_result: CommandResult,
    created_at: String,
    status: ProposalStatus,
}

impl WorkflowProposal {
    pub fn new(
        proposal_id: ProposalId,
        workflow_id: WorkflowId,
        base_version: WorkflowVersion,
        proposed_by: CommandActor,
        command_batch: CommandBatch,
        preview_result: CommandResult,
        created_at: impl Into<String>,
    ) -> Self {
        Self {
            proposal_id,
            workflow_id,
            base_version,
            proposed_by,
            command_batch,
            preview_result,
            created_at: created_at.into(),
            status: ProposalStatus::Pending,
        }
    }

    pub fn proposal_id(&self) -> &ProposalId {
        &self.proposal_id
    }

    pub fn workflow_id(&self) -> &WorkflowId {
        &self.workflow_id
    }

    pub fn base_version(&self) -> WorkflowVersion {
        self.base_version
    }

    /// The operator that proposed the batch; it becomes the batch actor when
    /// the proposal is applied.
    pub fn proposed_by(&self) -> &CommandActor {
        &self.proposed_by
    }

    pub fn command_batch(&self) -> &CommandBatch {
        &self.command_batch
    }

    pub fn preview_result(&self) -> &CommandResult {
        &self.preview_result
    }

    pub fn created_at(&self) -> &str {
        &self.created_at
    }

    pub fn status(&self) -> ProposalStatus {
        self.status
    }

    pub fn set_status(&mut self, status: ProposalStatus) {
        self.status = status;
    }
}
