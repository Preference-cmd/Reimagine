//! Shared domain data model for workflows and nodes.

mod artifacts;
mod ids;
mod models;
mod nodes;
mod slots;
mod values;

pub use artifacts::ArtifactRef;
pub use ids::{
    ArtifactId, CommandBatchId, DiagnosticId, EdgeId, HistoryEntryId, ModelId, NodeId, NodeTypeId,
    ProjectId, ProposalId, RunId, SlotId, WorkflowId, WorkflowInputId, WorkflowOutputId,
    WorkflowVersion,
};
pub use models::{ModelFormat, ModelRef, ModelRole, ModelSeries, ModelVariant};
pub use nodes::{
    BackendCapability, ComponentRole, ModelFamily, NodeCatalog, NodeDef, NodeEffect,
    NodeResourceRequirements,
};
pub use slots::{
    InputSlotDef, OutputSlotDef, SlotConstraint, SlotEditConstraint, SlotKind, SlotType, SlotUi,
};
pub use values::{NodeValue, ParamValue, TensorDType, TensorData, TensorShape};
