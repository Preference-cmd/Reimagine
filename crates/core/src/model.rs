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
    ProposalId, RunId, SlotId, WorkflowId, WorkflowInputId, WorkflowOutputId, WorkflowVersion,
};
pub use models::{ModelRef, ModelRole, ModelSeries, ModelVariant};
pub use nodes::{NodeCatalog, NodeDef, NodeEffect};
pub use slots::{InputSlotDef, OutputSlotDef, SlotConstraint, SlotKind, SlotUi};
pub use values::{NodeValue, ParamValue, TensorDType, TensorData, TensorShape};
