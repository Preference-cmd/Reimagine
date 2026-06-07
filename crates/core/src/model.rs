//! Shared domain data model for workflows and nodes.

#[path = "model/artifacts.rs"]
mod artifacts;
#[path = "model/ids.rs"]
mod ids;
#[path = "model/models.rs"]
mod models;
#[path = "model/nodes.rs"]
mod nodes;
#[path = "model/params.rs"]
mod params;
#[path = "model/sockets.rs"]
mod sockets;
#[path = "model/values.rs"]
mod values;

pub use artifacts::ArtifactRef;
pub use ids::{
    ArtifactId, CommandBatchId, DiagnosticId, EdgeId, HistoryEntryId, ModelId, NodeId,
    ProposalId, RunId, WorkflowId,
};
pub use models::{ModelRef, ModelRole, ModelSeries, ModelVariant};
pub use nodes::NodeDef;
pub use params::{ParamDef, ParamKind};
pub use sockets::{SocketDef, SocketKind};
pub use values::{NodeValue, ParamValue, TensorDType, TensorData, TensorShape};
