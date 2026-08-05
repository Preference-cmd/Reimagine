use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use reimagine_core::diagnostic::{CorrelationId, Diagnostic};
use reimagine_core::event::{RunEvent, RunEventId, RunEventKind};
use reimagine_core::model::{ArtifactId, NodeId, RunId, WorkflowId, WorkflowVersion};
use reimagine_core::readiness::ExecutionPlan;
use reimagine_inference::{
    BackendInstanceObservation, BackendInstanceRuntimeHooks, BackendInstanceSnapshot,
    NodeExecutorRegistry, ResourceHintSink, VramBudget,
};

use super::orchestrator::Runner;
use crate::cancellation::CancellationToken;
use crate::clock::{Clock, SystemClock};
use crate::error::RuntimeError;
use crate::events::RunEventSink;
use crate::handle::{RunHandle, RunState};
use crate::resources::NoopBackendInstanceRuntimeHooks;
use crate::run_inputs::RunInputs;
use crate::snapshot::{RunArtifactRef, RunSnapshot, RunSummary};
use crate::store::RunStore;

/// Options passed to [`RuntimeService::run`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RuntimeOptions {
    /// Optional correlation id propagated to events and node contexts.
    pub correlation_id: Option<CorrelationId>,
    /// Maximum number of same-stage node invocations admitted concurrently.
    ///
    /// `None` preserves V1's sequential compatibility behavior. `Some(1)` is
    /// also sequential. Values greater than one enable bounded same-stage
    /// concurrency. `Some(0)` is rejected by [`RuntimeService::run`].
    pub max_stage_concurrency: Option<usize>,
    /// Use the dynamic ready-set scheduler instead of sequential stages.
    ///
    /// When `true`, nodes are dispatched as soon as their inputs are
    /// satisfied rather than waiting for all nodes in a stage to complete.
    /// This enables better GPU utilization for multi-branch workflows.
    /// Default is `false` (sequential stages).
    pub use_ready_set: bool,
    /// VRAM budget sent to the backend via `ResourceHints` before each stage.
    ///
    /// `None` (default) means unlimited — the backend uses all available VRAM.
    /// When set, the backend should evict cached models to stay under budget.
    pub vram_budget: Option<VramBudget>,
    /// Default per-node execution deadline (BE-16/BE-27 merged).
    ///
    /// The deadline is armed when a node is admitted into a stage. A node
    /// that does not finish within the window is failed with a timeout
    /// diagnostic; per node-level semantics this is **not** a run failure —
    /// the run continues with the remaining nodes (downstream nodes that
    /// depend on the timed-out node's outputs fail on missing values, which
    /// does fail the run).
    ///
    /// Defaults to 5 minutes. `None` disables per-node deadlines entirely.
    pub default_node_timeout: Option<Duration>,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            correlation_id: None,
            max_stage_concurrency: None,
            use_ready_set: false,
            vram_budget: None,
            default_node_timeout: Some(Duration::from_secs(5 * 60)),
        }
    }
}

/// Public errors returned from `RuntimeService` operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeServiceError {
    UnknownRun { run_id: String },
    EmptyPlan { run_id: String },
    MissingExecutor { run_id: String, type_id: String },
    EventSink { message: String },
    InvalidStageConcurrency { value: usize },
}

impl std::fmt::Display for RuntimeServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownRun { run_id } => write!(f, "unknown run id: {run_id}"),
            Self::EmptyPlan { run_id } => write!(f, "empty execution plan for run {run_id}"),
            Self::MissingExecutor { run_id, type_id } => {
                write!(
                    f,
                    "no executor registered for node type {type_id} in run {run_id}"
                )
            }
            Self::EventSink { message } => write!(f, "run event sink failed: {message}"),
            Self::InvalidStageConcurrency { value } => {
                write!(
                    f,
                    "invalid max_stage_concurrency: {value}; expected at least 1"
                )
            }
        }
    }
}

impl std::error::Error for RuntimeServiceError {}

impl From<RuntimeError> for RuntimeServiceError {
    fn from(value: RuntimeError) -> Self {
        match value {
            RuntimeError::UnknownRun { run_id } => Self::UnknownRun { run_id },
            RuntimeError::EmptyPlan { run_id } => Self::EmptyPlan { run_id },
            RuntimeError::EmptyExecutionGraph { run_id } => Self::EmptyPlan { run_id },
            RuntimeError::MissingExecutor { run_id, type_id } => {
                Self::MissingExecutor { run_id, type_id }
            }
            RuntimeError::EventSink { message } => Self::EventSink { message },
        }
    }
}

pub struct RuntimeService {
    store: RunStore,
    registry: NodeExecutorRegistry,
    backend: Arc<dyn BackendInstanceRuntimeHooks>,
    hint_sink: Option<Arc<dyn ResourceHintSink>>,
    sink: Arc<dyn RunEventSink>,
    clock: Arc<dyn Clock>,
    next_run_seq: Arc<AtomicU64>,
    next_event_seq: Arc<AtomicU64>,
}

struct LifecycleEvent<'a> {
    run_id: &'a RunId,
    workflow_id: &'a WorkflowId,
    workflow_version: WorkflowVersion,
    kind: RunEventKind,
    node_id: Option<NodeId>,
    diagnostics: &'a [Diagnostic],
    correlation_id: Option<CorrelationId>,
}

impl std::fmt::Debug for RuntimeService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeService")
            .field("store", &self.store)
            .field("registry", &self.registry)
            .finish()
    }
}

impl RuntimeService {
    pub fn new(
        registry: NodeExecutorRegistry,
        backend: Arc<dyn BackendInstanceRuntimeHooks>,
        sink: Arc<dyn RunEventSink>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            store: RunStore::new(),
            registry,
            backend,
            hint_sink: None,
            sink,
            clock,
            next_run_seq: Arc::new(AtomicU64::new(0)),
            next_event_seq: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Wire a backend [`ResourceHintSink`] so runs transmit resource
    /// hints before each stage.
    ///
    /// Defaults to no sink, in which case the runner skips hint
    /// transmission entirely. Backend wiring code (e.g. worker-backed
    /// inference) sets this to the concrete backend instance.
    #[must_use]
    pub fn with_resource_hint_sink(mut self, sink: Option<Arc<dyn ResourceHintSink>>) -> Self {
        self.hint_sink = sink;
        self
    }

    pub fn with_defaults(registry: NodeExecutorRegistry, sink: Arc<dyn RunEventSink>) -> Self {
        Self::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            sink,
            Arc::new(SystemClock),
        )
    }

    pub fn store(&self) -> &RunStore {
        &self.store
    }

    pub fn registry(&self) -> &NodeExecutorRegistry {
        &self.registry
    }

    pub fn run(
        &self,
        plan: Arc<ExecutionPlan>,
        run_inputs: RunInputs,
        options: RuntimeOptions,
    ) -> Result<RunHandle, RuntimeServiceError> {
        if plan.nodes().is_empty() {
            return Err(RuntimeServiceError::EmptyPlan {
                run_id: String::new(),
            });
        }
        if options.max_stage_concurrency == Some(0) {
            return Err(RuntimeServiceError::InvalidStageConcurrency { value: 0 });
        }
        for node in plan.nodes() {
            if self.registry.get(node.type_id()).is_none() {
                return Err(RuntimeServiceError::MissingExecutor {
                    run_id: String::new(),
                    type_id: node.type_id().to_string(),
                });
            }
        }

        let run_seq = self.next_run_seq.fetch_add(1, Ordering::Relaxed);
        let run_id = RunId::new(format!("run-{run_seq}"));
        let cancellation = CancellationToken::new();
        let handle = RunHandle::new(
            run_id.clone(),
            plan.workflow_id().clone(),
            plan.workflow_version(),
            cancellation.clone(),
        );

        let started_at = self.clock.now();
        let initial_snapshot = RunSnapshot::new(
            run_id.clone(),
            plan.workflow_id().clone(),
            plan.workflow_version(),
            RunState::Queued,
            Default::default(),
            Vec::new(),
            Vec::new(),
            started_at.clone(),
            started_at,
        );
        self.store.put_snapshot(initial_snapshot);
        self.store.register_active(handle.clone());

        self.emit_lifecycle(LifecycleEvent {
            run_id: &run_id,
            workflow_id: plan.workflow_id(),
            workflow_version: plan.workflow_version(),
            kind: RunEventKind::RunQueued,
            node_id: None,
            diagnostics: &[],
            correlation_id: None,
        });
        self.emit_lifecycle(LifecycleEvent {
            run_id: &run_id,
            workflow_id: plan.workflow_id(),
            workflow_version: plan.workflow_version(),
            kind: RunEventKind::RunStarted,
            node_id: None,
            diagnostics: &[],
            correlation_id: options.correlation_id.clone(),
        });

        let runner = Runner {
            run_id: run_id.clone(),
            plan,
            run_inputs,
            options,
            cancellation,
            store: self.store.clone(),
            registry: self.registry.clone_for_runner(),
            backend: self.backend.clone(),
            hint_sink: self.hint_sink.clone(),
            sink: self.sink.clone(),
            clock: self.clock.clone(),
            next_event_seq: self.next_event_seq.clone(),
        };
        tokio::spawn(runner.run());

        Ok(handle)
    }

    pub fn cancel(&self, run_id: &RunId) -> Result<(), RuntimeServiceError> {
        match self.store.active_cancellation(run_id) {
            Some(token) => {
                token.cancel();
                Ok(())
            }
            None => Err(RuntimeServiceError::UnknownRun {
                run_id: run_id.to_string(),
            }),
        }
    }

    pub fn snapshot(&self, run_id: &RunId) -> Option<RunSnapshot> {
        self.store.snapshot(run_id)
    }

    pub fn summary(&self, run_id: &RunId) -> Option<RunSummary> {
        self.store.summary(run_id)
    }

    /// Search all active snapshots and terminal summaries for an artifact by id.
    pub fn find_artifact(&self, artifact_id: &ArtifactId) -> Option<RunArtifactRef> {
        self.store.find_artifact(artifact_id)
    }

    pub async fn backend_instance_snapshots(&self) -> Vec<BackendInstanceSnapshot> {
        BackendInstanceObservation::snapshots(self.backend.as_ref()).await
    }

    fn emit_lifecycle(&self, lifecycle: LifecycleEvent<'_>) {
        let event_id_index = self.next_event_seq.fetch_add(1, Ordering::Relaxed);
        let mut event = RunEvent::new(
            RunEventId::new(format!("{}-evt-{event_id_index}", lifecycle.run_id)),
            lifecycle.run_id.clone(),
            lifecycle.workflow_id.clone(),
            lifecycle.workflow_version,
            lifecycle.kind,
            self.clock.now(),
        );
        if let Some(nid) = lifecycle.node_id {
            event = event.with_node_id(nid);
        }
        for diag in lifecycle.diagnostics {
            event = event.with_diagnostic(diag.clone());
        }
        if let Some(cid) = lifecycle.correlation_id {
            event = event.with_correlation_id(cid);
        }
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.sink.emit(event))) {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                tracing::warn!(
                    target: "reimagine_runtime",
                    error = %error,
                    "run event sink failed; continuing without failing the run"
                );
            }
            Err(_) => {
                tracing::warn!(
                    target: "reimagine_runtime",
                    "run event sink panicked; continuing without failing the run"
                );
            }
        }
    }
}
