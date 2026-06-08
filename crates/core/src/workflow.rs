//! Canonical workflow schema.

mod endpoint;
mod layout;
mod metadata;
mod schema;

pub use endpoint::Endpoint;
pub use layout::{Position, Viewport, WorkflowLayout};
pub use metadata::{WorkflowInputDef, WorkflowInterface, WorkflowMetadata, WorkflowOutputDef};
pub use schema::{Workflow, WorkflowEdge, WorkflowNode, WorkflowSchemaVersion};
