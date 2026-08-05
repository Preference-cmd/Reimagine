//! Node executor trait and registry contract.
//!
//! The executor contract is owned by `reimagine-inference` (this crate)
//! so that built-in inference executors can implement it without
//! creating a `inference -> runtime` dependency edge. The runtime
//! composes an [`NodeExecutorRegistry`] (typically constructed by
//! app-host) and invokes `dyn NodeExecutor::execute` against an
//! inference-owned [`NodeExecutionContext`](crate::node_context::NodeExecutionContext).

use std::collections::HashMap;

use crate::ExecutionOutput;
use crate::capability::InferenceCapability;
use reimagine_core::model::NodeTypeId;

// Re-export the context type so executor modules can import it
// through `crate::executor::NodeExecutionContext` alongside the trait.
// `NodeInputs` / `NodeParams` remain available via
// `reimagine_inference::{NodeInputs, NodeParams}` (re-exported from
// `lib.rs`) — they don't need to live next to the trait.
pub use crate::node_context::NodeExecutionContext;

/// Result of executing one node.
///
/// V1 returns a `Vec<ExecutionOutput>` of declared outputs. Each output
/// bundles the produced value with the slot id it should be stored
/// under and the
/// [`ExecutionValueRetention`](crate::ExecutionValueRetention)
/// policy the executor intends. The runner task is responsible for
/// inserting these into the `RunValueStore` using the node's declared
/// `output_slots` and recording the retention alongside the value.
pub type NodeExecutionOutputs = Vec<ExecutionOutput>;

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

    /// Inference capabilities this executor requires from the backend
    /// to run.
    ///
    /// V1 executors that invoke a typed backend capability method
    /// declare the capability here so the [`NodeExecutorRegistry`]
    /// can build its capability index during registration. Executors
    /// that only transform values declare none.
    fn required_capabilities(&self) -> &'static [InferenceCapability] {
        &[]
    }
}

/// Convenience type alias for boxed node executors.
pub type BoxedNodeExecutor = std::sync::Arc<dyn NodeExecutor>;

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
///
/// A secondary index maps each [`InferenceCapability`] to the node
/// type ids whose executors require it, so callers can validate that
/// every required capability is available before starting execution.
#[derive(Default)]
pub struct NodeExecutorRegistry {
    executors: HashMap<NodeTypeId, BoxedNodeExecutor>,
    by_capability: HashMap<InferenceCapability, Vec<NodeTypeId>>,
}

impl std::fmt::Debug for NodeExecutorRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeExecutorRegistry")
            .field("type_ids", &self.executors.keys().collect::<Vec<_>>())
            .field("capabilities", &self.capability_union())
            .finish()
    }
}

impl NodeExecutorRegistry {
    /// Register a new executor. Returns an error if a duplicate type id is
    /// provided.
    ///
    /// The capability index is built from the executor's declared
    /// [`NodeExecutor::required_capabilities`].
    pub fn register(
        &mut self,
        type_id: impl Into<NodeTypeId>,
        executor: BoxedNodeExecutor,
    ) -> Result<(), NodeExecutorRegistryError> {
        let capabilities = executor.required_capabilities();
        self.register_with_capabilities(type_id, executor, capabilities)
    }

    /// Register a new executor with explicit capability requirements.
    ///
    /// This is for callers that know an executor's requirements
    /// statically and want the index populated without the executor
    /// itself overriding
    /// [`NodeExecutor::required_capabilities`]. The index records
    /// exactly the supplied capabilities.
    pub fn register_with_capabilities(
        &mut self,
        type_id: impl Into<NodeTypeId>,
        executor: BoxedNodeExecutor,
        capabilities: &[InferenceCapability],
    ) -> Result<(), NodeExecutorRegistryError> {
        let type_id = type_id.into();
        if self.executors.contains_key(&type_id) {
            return Err(NodeExecutorRegistryError::AlreadyRegistered {
                type_id: type_id.to_string(),
            });
        }
        for capability in capabilities {
            let bucket = self.by_capability.entry(*capability).or_default();
            if !bucket.contains(&type_id) {
                bucket.push(type_id.clone());
            }
        }
        self.executors.insert(type_id, executor);
        Ok(())
    }

    /// Look up the executor for a given node type id.
    pub fn get(&self, type_id: &NodeTypeId) -> Option<&BoxedNodeExecutor> {
        self.executors.get(type_id)
    }

    /// Borrow an iterator over every registered executor type id.
    ///
    /// This is for catalog/executor alignment reporting. The registry
    /// does not expose node metadata; it only enumerates the set of
    /// `NodeTypeId` values it knows how to execute.
    pub fn iter_type_ids(&self) -> impl Iterator<Item = &NodeTypeId> {
        self.executors.keys()
    }

    /// Number of registered executors.
    pub fn len(&self) -> usize {
        self.executors.len()
    }

    /// Returns `true` if no executors are registered.
    pub fn is_empty(&self) -> bool {
        self.executors.is_empty()
    }

    /// Node type ids whose registered executor requires the given
    /// capability, in registration order.
    pub fn query_by_capability(&self, capability: InferenceCapability) -> Vec<NodeTypeId> {
        self.by_capability
            .get(&capability)
            .cloned()
            .unwrap_or_default()
    }

    /// Returns `true` if at least one registered executor requires
    /// the given capability.
    pub fn has_capability(&self, capability: InferenceCapability) -> bool {
        self.by_capability.contains_key(&capability)
    }

    /// Union of capabilities required by the registered executors,
    /// in [`InferenceCapability::all_v1`] order.
    pub fn capability_union(&self) -> Vec<InferenceCapability> {
        InferenceCapability::all_v1()
            .iter()
            .copied()
            .filter(|capability| self.by_capability.contains_key(capability))
            .collect()
    }

    /// Build a shallow, shareable snapshot of the registry for a runner task.
    /// The cloned registry shares each `Arc<dyn NodeExecutor>` with the
    /// original so executors are not duplicated.
    pub fn clone_for_runner(&self) -> std::sync::Arc<NodeExecutorRegistry> {
        std::sync::Arc::new(NodeExecutorRegistry {
            executors: self.executors.clone(),
            by_capability: self.by_capability.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct CapableExecutor(&'static [InferenceCapability]);

    #[async_trait::async_trait]
    impl NodeExecutor for CapableExecutor {
        async fn execute(
            &self,
            _context: NodeExecutionContext,
        ) -> Result<NodeExecutionOutputs, NodeExecutorError> {
            Ok(Vec::new())
        }

        fn required_capabilities(&self) -> &'static [InferenceCapability] {
            self.0
        }
    }

    fn capable(capabilities: &'static [InferenceCapability]) -> BoxedNodeExecutor {
        Arc::new(CapableExecutor(capabilities))
    }

    #[test]
    fn register_builds_capability_index_from_executor_declarations() {
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "builtin.checkpoint_loader",
                capable(&[InferenceCapability::LoadBundle]),
            )
            .expect("register");
        registry
            .register(
                "builtin.ksampler",
                capable(&[InferenceCapability::DiffusionSample]),
            )
            .expect("register");

        assert_eq!(
            registry.query_by_capability(InferenceCapability::LoadBundle),
            vec![NodeTypeId::new("builtin.checkpoint_loader")]
        );
        assert_eq!(
            registry.query_by_capability(InferenceCapability::DiffusionSample),
            vec![NodeTypeId::new("builtin.ksampler")]
        );
        assert!(
            registry
                .query_by_capability(InferenceCapability::TextEncode)
                .is_empty()
        );
    }

    #[test]
    fn register_with_capabilities_indexes_explicit_capabilities() {
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register_with_capabilities(
                "builtin.clip_text_encode",
                capable(&[]),
                &[InferenceCapability::TextEncode],
            )
            .expect("register");

        assert_eq!(
            registry.query_by_capability(InferenceCapability::TextEncode),
            vec![NodeTypeId::new("builtin.clip_text_encode")]
        );
    }

    #[test]
    fn query_by_capability_returns_all_matching_type_ids_in_registration_order() {
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "builtin.ksampler",
                capable(&[InferenceCapability::DiffusionSample]),
            )
            .expect("register");
        registry
            .register(
                "builtin.ksampler_advanced",
                capable(&[InferenceCapability::DiffusionSample]),
            )
            .expect("register");

        assert_eq!(
            registry.query_by_capability(InferenceCapability::DiffusionSample),
            vec![
                NodeTypeId::new("builtin.ksampler"),
                NodeTypeId::new("builtin.ksampler_advanced"),
            ]
        );
    }

    #[test]
    fn has_capability_reflects_registered_executors() {
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "builtin.ksampler",
                capable(&[InferenceCapability::DiffusionSample]),
            )
            .expect("register");

        assert!(registry.has_capability(InferenceCapability::DiffusionSample));
        assert!(!registry.has_capability(InferenceCapability::TextEncode));
        assert!(!registry.has_capability(InferenceCapability::ImageSave));
    }

    #[test]
    fn capability_union_deduplicates_and_uses_v1_order() {
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "builtin.ksampler",
                capable(&[InferenceCapability::DiffusionSample]),
            )
            .expect("register");
        registry
            .register(
                "builtin.vae_decode",
                capable(&[InferenceCapability::LatentDecode]),
            )
            .expect("register");
        registry
            .register(
                "builtin.vae_encode",
                capable(&[
                    InferenceCapability::DiffusionSample,
                    InferenceCapability::LatentDecode,
                ]),
            )
            .expect("register");
        registry
            .register("builtin.string", capable(&[]))
            .expect("register");

        assert_eq!(
            registry.capability_union(),
            vec![
                InferenceCapability::DiffusionSample,
                InferenceCapability::LatentDecode,
            ]
        );
    }

    #[test]
    fn clone_for_runner_preserves_capability_index() {
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "builtin.ksampler",
                capable(&[InferenceCapability::DiffusionSample]),
            )
            .expect("register");

        let runner = registry.clone_for_runner();
        assert!(runner.has_capability(InferenceCapability::DiffusionSample));
        assert_eq!(
            runner.query_by_capability(InferenceCapability::DiffusionSample),
            vec![NodeTypeId::new("builtin.ksampler")]
        );
    }
}
