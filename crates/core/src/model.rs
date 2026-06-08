//! Shared domain data model for workflows and nodes.

mod artifacts;
mod ids;
mod models;
mod nodes;
mod params;
mod sockets;
mod values;

pub use artifacts::ArtifactRef;
pub use ids::{
    ArtifactId, CommandBatchId, DiagnosticId, EdgeId, HistoryEntryId, ModelId, NodeId, ProposalId,
    RunId, WorkflowId,
};
pub use models::{ModelRef, ModelRole, ModelSeries, ModelVariant};
pub use nodes::NodeDef;
pub use params::{ParamDef, ParamKind};
pub use sockets::{SocketDef, SocketKind};
pub use values::{NodeValue, ParamValue, TensorDType, TensorData, TensorShape};
