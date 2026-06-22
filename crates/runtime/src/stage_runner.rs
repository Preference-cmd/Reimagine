//! Helpers for preparing and executing same-stage node work.

use std::sync::Arc;

use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::model::{RunId, WorkflowId, WorkflowVersion};
use reimagine_core::readiness::ExecutionNode;
use reimagine_inference::{
    ArtifactPublisher, ExecutionOutput, NodeCancellation, NodeExecutionContext, NodeExecutorError,
    NodeExecutorRegistry, NodeInputs, NodeParams,
};
use tokio::sync::Mutex;

use crate::artifacts::{ArtifactStore, RuntimeNodeArtifactCapability};
use crate::cancellation::{CancellationToken, CombinedCancellation};
use crate::clock::Clock;
use crate::events::RunEventSink;

#[derive(Debug, Clone)]
pub struct PreparedNodeBindings {
    inputs: NodeInputs,
    params: NodeParams,
}

impl PreparedNodeBindings {
    pub fn new(inputs: NodeInputs, params: NodeParams) -> Self {
        Self { inputs, params }
    }

    pub fn into_parts(self) -> (NodeInputs, NodeParams) {
        (self.inputs, self.params)
    }
}

#[derive(Debug, Clone)]
pub enum StageNodePrepareError {
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct StageNodeWork {
    node: ExecutionNode,
    bindings: PreparedNodeBindings,
}

impl StageNodeWork {
    pub fn new(node: ExecutionNode, bindings: PreparedNodeBindings) -> Self {
        Self { node, bindings }
    }

    pub fn node(&self) -> &ExecutionNode {
        &self.node
    }
}

#[derive(Debug)]
pub enum StageNodeResult {
    Completed {
        node: ExecutionNode,
        outputs: Vec<ExecutionOutput>,
    },
    Failed {
        node: ExecutionNode,
        message: String,
    },
    Cancelled {
        node: ExecutionNode,
    },
}

#[derive(Clone)]
pub struct StageExecutionContext {
    pub run_id: RunId,
    pub workflow_id: WorkflowId,
    pub workflow_version: WorkflowVersion,
    pub correlation_id: Option<CorrelationId>,
    pub sink: Arc<dyn RunEventSink>,
    pub clock: Arc<dyn Clock>,
    pub registry: Arc<NodeExecutorRegistry>,
    pub cancellation: CancellationToken,
}

pub async fn execute_stage_node(
    context: StageExecutionContext,
    work: StageNodeWork,
    artifact_store: Arc<Mutex<ArtifactStore>>,
    failure_cancellation: CancellationToken,
) -> StageNodeResult {
    let node = work.node;
    let (inputs, params) = work.bindings.into_parts();
    let publisher: Arc<dyn ArtifactPublisher> = Arc::new(RuntimeNodeArtifactCapability::new(
        context.run_id.clone(),
        context.workflow_id.clone(),
        context.workflow_version,
        node.node_id().clone(),
        artifact_store,
        context.sink.clone(),
        context.clock.clone(),
        context.cancellation.clone(),
    ));
    let cancellation: Arc<dyn NodeCancellation> = Arc::new(CombinedCancellation::new(
        context.cancellation.clone(),
        failure_cancellation,
    ));
    let execution_context = NodeExecutionContext::new(
        context.run_id,
        context.workflow_id,
        context.workflow_version,
        context.correlation_id,
        node.node_id().clone(),
        node.type_id().clone(),
        inputs,
        params,
        publisher,
        cancellation,
        context.clock.now(),
    );

    let Some(executor) = context.registry.get(node.type_id()) else {
        let message = format!("no executor for {}", node.type_id().as_str());
        return StageNodeResult::Failed { node, message };
    };

    match executor.execute(execution_context).await {
        Ok(outputs) => StageNodeResult::Completed { node, outputs },
        Err(NodeExecutorError::Cancelled) => StageNodeResult::Cancelled { node },
        Err(NodeExecutorError::MissingInput { slot_id }) => StageNodeResult::Failed {
            node,
            message: format!("missing input {slot_id}"),
        },
        Err(NodeExecutorError::Failed { message }) | Err(NodeExecutorError::Infra { message }) => {
            StageNodeResult::Failed { node, message }
        }
    }
}

pub fn missing_upstream_value_message(from_node_id: &str, from_slot_id: &str) -> String {
    format!("missing upstream value for {from_node_id}:{from_slot_id}")
}

pub fn missing_workflow_input_message(workflow_input_id: &str, slot_id: &str) -> String {
    format!("missing workflow input {workflow_input_id} for slot {slot_id}")
}
