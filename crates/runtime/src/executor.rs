//! Node executor trait and registry.

use std::collections::HashMap;
use std::sync::Arc;

use reimagine_core::model::NodeTypeId;

use crate::node_context::NodeExecutionContext;
use crate::value::RuntimeValue;

/// Result of executing one node.
///
/// V1 returns a `Vec<(SlotId, Arc<RuntimeValue>)>` of outputs. The runner
/// task is responsible for inserting these into the [`RunValueStore`] using
/// the node's declared `output_slots`.
pub type NodeExecutionOutputs = Vec<(reimagine_core::model::SlotId, Arc<RuntimeValue>)>;

/// Errors returned from a node executor.
///
/// The runner maps this into a runtime `NodeFailed` event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeExecutorError {
    /// Executor refused to run for a non-recoverable reason.
    Failed { message: String },
    /// Executor recognized the cancellation token mid-flight.
    Cancelled,
    /// The executor expected an input that was not supplied.
    MissingInput { slot_id: String },
    /// Generic infra failure (decode/load/etc).
    Infra { message: String },
}

impl std::fmt::Display for NodeExecutorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Failed { message } => write!(f, "node failed: {message}"),
            Self::Cancelled => write!(f, "node cancelled"),
            Self::MissingInput { slot_id } => write!(f, "missing input {slot_id}"),
            Self::Infra { message } => write!(f, "infra failure: {message}"),
        }
    }
}

impl std::error::Error for NodeExecutorError {}

/// Boundary for executing one plan node against resolved inputs and params.
///
/// V1 uses `async_trait` for a readable async trait-object surface. The
/// runtime stores `Box<dyn NodeExecutor>` keyed by `NodeTypeId`.
#[async_trait::async_trait]
pub trait NodeExecutor: Send + Sync + 'static {
    /// Run this executor. Should observe the cancellation token in the
    /// context and return [`NodeExecutorError::Cancelled`] if it observes a
    /// cancellation request.
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<NodeExecutionOutputs, NodeExecutorError>;
}

/// Convenience type alias for boxed node executors.
pub type BoxedNodeExecutor = Arc<dyn NodeExecutor>;

/// Errors from constructing or querying a registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeExecutorRegistryError {
    /// The registry already contains an executor for this node type.
    AlreadyRegistered { type_id: String },
    /// The requested type id has no registered executor.
    UnknownType { type_id: String },
}

impl std::fmt::Display for NodeExecutorRegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyRegistered { type_id } => {
                write!(f, "executor already registered for {type_id}")
            }
            Self::UnknownType { type_id } => write!(f, "no executor registered for {type_id}"),
        }
    }
}

impl std::error::Error for NodeExecutorRegistryError {}

/// Registry of node executors keyed by `NodeTypeId`.
///
/// Hosts assemble a registry at workspace startup and hand it to the
/// `RuntimeService`. The registry owns the executors; the runtime only
/// borrows them.
pub struct NodeExecutorRegistry {
    executors: HashMap<NodeTypeId, BoxedNodeExecutor>,
}

impl std::fmt::Debug for NodeExecutorRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeExecutorRegistry")
            .field("type_ids", &self.executors.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl Default for NodeExecutorRegistry {
    fn default() -> Self {
        Self {
            executors: HashMap::new(),
        }
    }
}

impl NodeExecutorRegistry {
    /// Register a new executor. Returns an error if a duplicate type id is
    /// provided.
    pub fn register(
        &mut self,
        type_id: impl Into<NodeTypeId>,
        executor: BoxedNodeExecutor,
    ) -> Result<(), NodeExecutorRegistryError> {
        let type_id = type_id.into();
        if self.executors.contains_key(&type_id) {
            return Err(NodeExecutorRegistryError::AlreadyRegistered {
                type_id: type_id.to_string(),
            });
        }
        self.executors.insert(type_id, executor);
        Ok(())
    }

    /// Look up the executor for a given node type id.
    pub fn get(&self, type_id: &NodeTypeId) -> Option<&BoxedNodeExecutor> {
        self.executors.get(type_id)
    }

    /// Number of registered executors.
    pub fn len(&self) -> usize {
        self.executors.len()
    }

    /// Returns `true` if no executors are registered.
    pub fn is_empty(&self) -> bool {
        self.executors.is_empty()
    }

    /// Build a shallow, shareable snapshot of the registry for a runner task.
    /// The cloned registry shares each `Arc<dyn NodeExecutor>` with the
    /// original so executors are not duplicated.
    pub fn clone_for_runner(&self) -> Arc<NodeExecutorRegistry> {
        Arc::new(NodeExecutorRegistry {
            executors: self.executors.clone(),
        })
    }
}
