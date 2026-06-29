use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::event::RunEventKind;
use reimagine_core::model::{RunId, WorkflowVersion};
use reimagine_core::readiness::ExecutionPlan;
use reimagine_inference::{
    BackendInstanceRuntimeHooks, BackendRunLifecycleRequest, NodeExecutorRegistry,
};
use tokio::sync::Mutex;

use super::diagnostics::make_run_diagnostic;
use super::service::RuntimeOptions;
use crate::artifacts::ArtifactStore;
use crate::cancellation::CancellationToken;
use crate::clock::Clock;
use crate::consumer_index::PlanConsumerIndex;
use crate::events::RunEventSink;
use crate::handle::RunState;
use crate::run_inputs::RunInputs;
use crate::run_session::RunSession;
use crate::scheduler::StageExecutionPolicy;
use crate::store::RunStore;

pub(super) struct Runner {
    pub(super) run_id: RunId,
    pub(super) plan: Arc<ExecutionPlan>,
    pub(super) run_inputs: RunInputs,
    pub(super) options: RuntimeOptions,
    pub(super) cancellation: CancellationToken,
    pub(super) store: RunStore,
    pub(super) registry: Arc<NodeExecutorRegistry>,
    pub(super) backend: Arc<dyn BackendInstanceRuntimeHooks>,
    pub(super) sink: Arc<dyn RunEventSink>,
    pub(super) clock: Arc<dyn Clock>,
    pub(super) next_event_seq: Arc<AtomicU64>,
}

impl Runner {
    pub(super) async fn run(self) {
        let session = RunSession::new(
            self.run_id.clone(),
            self.plan.workflow_id().clone(),
            self.plan.workflow_version(),
            self.options.correlation_id.clone(),
            self.cancellation.clone(),
        );
        let artifact_store = Arc::new(Mutex::new(ArtifactStore::new()));
        let consumer_index = PlanConsumerIndex::from_plan(&self.plan);

        let request = BackendRunLifecycleRequest {
            run_id: self.run_id.clone(),
        };
        let mut lifecycle_diagnostics = Vec::new();
        match self.backend.begin_run(request.clone()).await {
            Ok(report) => lifecycle_diagnostics.extend(report.diagnostics),
            Err(err) => {
                tracing::warn!(%err, run_id = %self.run_id, "begin_run failed");
            }
        }
        let mut session = self
            .run_to_completion(session, artifact_store, &consumer_index)
            .await;
        session.values_mut().clear();
        match self.backend.cleanup_run(request).await {
            Ok(report) => lifecycle_diagnostics.extend(report.diagnostics),
            Err(err) => {
                tracing::warn!(%err, run_id = %self.run_id, "cleanup_run failed");
            }
        }
        if !lifecycle_diagnostics.is_empty() {
            self.store
                .append_diagnostics(&self.run_id, &lifecycle_diagnostics);
        }
    }

    async fn run_to_completion(
        &self,
        mut session: RunSession,
        artifact_store: Arc<Mutex<ArtifactStore>>,
        consumer_index: &PlanConsumerIndex,
    ) -> RunSession {
        let started_at = self.clock.now();
        let mut policy = StageExecutionPolicy::new();

        for stage in self.plan.stages() {
            if self.cancellation.is_cancelled() {
                self.handle_cancellation(&mut session, &started_at, &artifact_store)
                    .await;
                return session;
            }
            if self
                .run_stage(
                    stage.node_ids(),
                    &mut session,
                    &started_at,
                    &artifact_store,
                    consumer_index,
                    &mut policy,
                )
                .await
            {
                self.handle_cancellation(&mut session, &started_at, &artifact_store)
                    .await;
                return session;
            }
        }

        let finished_at = self.clock.now();
        let (state, lifecycle_kind, diagnostics) = if let Some(message) = policy.failed_message() {
            let diag = make_run_diagnostic(&self.run_id, message);
            (RunState::Failed, RunEventKind::RunFailed, vec![diag])
        } else {
            (RunState::Completed, RunEventKind::RunCompleted, Vec::new())
        };
        self.emit_lifecycle_event(lifecycle_kind, None, &diagnostics);
        self.publish_summary(
            &session,
            state,
            started_at.clone(),
            finished_at,
            &artifact_store,
            &diagnostics,
        )
        .await;
        self.publish_snapshot_with_state(&session, state, &started_at, &artifact_store)
            .await;
        self.store.finalize(&self.run_id);
        session
    }

    pub(super) fn workflow_version(&self) -> WorkflowVersion {
        self.plan.workflow_version()
    }

    pub(super) fn started_correlation_id(&self) -> Option<CorrelationId> {
        self.options.correlation_id.clone()
    }
}
