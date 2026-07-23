use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use reimagine_app_host::{
    RunCancellation, SwitchableWorker, WorkerSwitchError, WorkerSwitchService, WorkerSwitchTarget,
};
use reimagine_backend_worker_host::{WorkerProcessState, WorkerRunLeases};
use reimagine_backend_worker_protocol::WorkerIncarnationId;
use reimagine_core::model::RunId;
use reimagine_core::model::{NodeId, TensorDType, TensorShape, WorkflowId, WorkflowVersion};
use reimagine_inference::{
    Backend, BackendInstance, BackendPayloadKey, BackendTensorHandle, CannedCapabilityResponse,
    CreateEmptyLatentRequest, CreateEmptyLatentResponse, FakeBackend, InferenceBackend,
    InferenceRuntime, LatentContent, LatentSpaceMetadata, RuntimeLatent,
};

#[derive(Default)]
struct NoopRunCancellation;

impl RunCancellation for NoopRunCancellation {
    fn cancel(&self, _run_id: &RunId) -> Result<(), WorkerSwitchError> {
        Ok(())
    }
}

struct ReleasingRunCancellation {
    leases: Arc<WorkerRunLeases>,
    cancelled: Mutex<Vec<RunId>>,
}

struct TerminalWinsCancellation {
    leases: Arc<WorkerRunLeases>,
}

struct FailingRunCancellation;

impl RunCancellation for FailingRunCancellation {
    fn cancel(&self, run_id: &RunId) -> Result<(), WorkerSwitchError> {
        Err(WorkerSwitchError::Cancellation {
            run_id: run_id.clone(),
            message: "runtime cancellation failed".to_owned(),
        })
    }
}

impl RunCancellation for TerminalWinsCancellation {
    fn cancel(&self, run_id: &RunId) -> Result<(), WorkerSwitchError> {
        self.leases.release(run_id);
        Err(WorkerSwitchError::Cancellation {
            run_id: run_id.clone(),
            message: "run already reached a terminal state".to_owned(),
        })
    }
}

impl RunCancellation for ReleasingRunCancellation {
    fn cancel(&self, run_id: &RunId) -> Result<(), WorkerSwitchError> {
        self.cancelled
            .lock()
            .expect("cancelled")
            .push(run_id.clone());
        self.leases.release(run_id);
        Ok(())
    }
}

struct FakeWorker {
    instance: BackendInstance,
    incarnation: WorkerIncarnationId,
    leases: Arc<WorkerRunLeases>,
    shutdowns: AtomicUsize,
    fail_shutdown: AtomicBool,
    state: Mutex<WorkerProcessState>,
    backend: Option<Arc<dyn InferenceBackend>>,
}

impl FakeWorker {
    fn new(instance: &str, incarnation: &str) -> Self {
        Self {
            instance: BackendInstance::new(instance),
            incarnation: WorkerIncarnationId(incarnation.to_owned()),
            leases: Arc::new(WorkerRunLeases::new()),
            shutdowns: AtomicUsize::new(0),
            fail_shutdown: AtomicBool::new(false),
            state: Mutex::new(WorkerProcessState::Ready),
            backend: None,
        }
    }

    fn with_latent_backend(instance: &str, incarnation: &str, payload_key: &str) -> Self {
        let mut worker = Self::new(instance, incarnation);
        let latent = RuntimeLatent::new(
            BackendTensorHandle::with_instance(
                Backend::new("fake"),
                BackendInstance::new(instance),
                BackendPayloadKey::new(payload_key),
                TensorDType::F32,
                TensorShape::new(vec![1, 4, 8, 8]),
                "fake",
            ),
            64,
            64,
            1,
            4,
            LatentSpaceMetadata::sdxl_base(),
            LatentContent::EmptyGeometry,
        );
        worker.backend = Some(Arc::new(FakeBackend::new("fake").create_empty_latent(
            CannedCapabilityResponse::always(CreateEmptyLatentResponse::new(latent)),
        )));
        worker
    }
}

#[async_trait]
impl SwitchableWorker for FakeWorker {
    fn instance(&self) -> &BackendInstance {
        &self.instance
    }

    fn incarnation_id(&self) -> &WorkerIncarnationId {
        &self.incarnation
    }

    fn run_leases(&self) -> &Arc<WorkerRunLeases> {
        &self.leases
    }

    fn process_state(&self) -> WorkerProcessState {
        *self.state.lock().expect("state")
    }

    fn inference_backend(&self) -> Option<Arc<dyn InferenceBackend>> {
        self.backend.clone()
    }

    async fn shutdown(&self) -> Result<(), WorkerSwitchError> {
        self.shutdowns.fetch_add(1, Ordering::SeqCst);
        if self.fail_shutdown.load(Ordering::SeqCst) {
            return Err(WorkerSwitchError::Shutdown {
                instance: self.instance.clone(),
                message: "worker already crashed".to_owned(),
            });
        }
        Ok(())
    }
}

struct FakeTarget {
    worker: Arc<FakeWorker>,
    events: Arc<Mutex<Vec<&'static str>>>,
}

struct FailingTarget;

#[async_trait]
impl WorkerSwitchTarget for FailingTarget {
    async fn start(&self) -> Result<Arc<dyn SwitchableWorker>, WorkerSwitchError> {
        Err(WorkerSwitchError::Startup {
            message: "handshake failed".to_owned(),
        })
    }
}

#[async_trait]
impl WorkerSwitchTarget for FakeTarget {
    async fn start(&self) -> Result<Arc<dyn SwitchableWorker>, WorkerSwitchError> {
        self.events.lock().expect("events").push("target-ready");
        Ok(self.worker.clone())
    }
}

#[tokio::test]
async fn drain_switch_commits_after_target_ready_and_old_leases_finish() {
    let old = Arc::new(FakeWorker::new("burn:wgpu:default", "old-incarnation"));
    let target = Arc::new(FakeWorker::new("burn:wgpu:default", "new-incarnation"));
    let run_id = RunId::new("leased-run");
    old.leases.acquire(&run_id).expect("initial lease");
    let events = Arc::new(Mutex::new(Vec::new()));
    let service = Arc::new(WorkerSwitchService::new(
        old.clone(),
        Arc::new(NoopRunCancellation),
    ));
    let stale_handle = service.selected().await;

    let switch = {
        let service = Arc::clone(&service);
        let target = Arc::new(FakeTarget {
            worker: Arc::clone(&target),
            events: Arc::clone(&events),
        });
        tokio::spawn(async move {
            service
                .drain_and_switch(target, Duration::from_secs(1))
                .await
        })
    };

    tokio::task::yield_now().await;
    assert_eq!(events.lock().expect("events").as_slice(), &["target-ready"]);
    assert_eq!(
        old.leases.admission(),
        reimagine_backend_worker_host::WorkerAdmissionState::Draining
    );
    assert_eq!(service.selected().await, stale_handle);

    assert!(old.leases.release(&run_id));
    let selected = switch.await.expect("switch task").expect("switch succeeds");

    assert_eq!(selected.incarnation_id().0, "new-incarnation");
    assert_eq!(service.selected().await, selected);
    assert!(matches!(
        service.resolve(&stale_handle).await,
        Err(WorkerSwitchError::StaleHandle { .. })
    ));
    assert_eq!(old.shutdowns.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cancel_switch_cancels_every_leased_run_before_commit() {
    let old = Arc::new(FakeWorker::new("burn:wgpu:default", "old-incarnation"));
    let target = Arc::new(FakeWorker::new("burn:wgpu:default", "new-incarnation"));
    let run_a = RunId::new("run-a");
    let run_b = RunId::new("run-b");
    old.leases.acquire(&run_a).expect("lease a");
    old.leases.acquire(&run_b).expect("lease b");
    let cancellation = Arc::new(ReleasingRunCancellation {
        leases: Arc::clone(&old.leases),
        cancelled: Mutex::new(Vec::new()),
    });
    let service = WorkerSwitchService::new(old, cancellation.clone());
    let target = Arc::new(FakeTarget {
        worker: target,
        events: Arc::new(Mutex::new(Vec::new())),
    });

    let selected = service
        .cancel_and_switch(target, Duration::from_secs(1))
        .await
        .expect("cancel switch succeeds");

    let mut cancelled = cancellation.cancelled.lock().expect("cancelled").clone();
    cancelled.sort_by(|left, right| left.as_str().cmp(right.as_str()));
    assert_eq!(cancelled, vec![run_a, run_b]);
    assert_eq!(selected.incarnation_id().0, "new-incarnation");
}

#[tokio::test]
async fn cancel_switch_accepts_terminal_cleanup_winning_the_cancel_race() {
    let old = Arc::new(FakeWorker::new("burn:wgpu:default", "old-incarnation"));
    let target = Arc::new(FakeWorker::new("burn:wgpu:default", "new-incarnation"));
    let run_id = RunId::new("terminal-wins");
    old.leases.acquire(&run_id).expect("lease");
    let service = WorkerSwitchService::new(
        old.clone(),
        Arc::new(TerminalWinsCancellation {
            leases: Arc::clone(&old.leases),
        }),
    );
    let target = Arc::new(FakeTarget {
        worker: target,
        events: Arc::new(Mutex::new(Vec::new())),
    });

    let selected = service
        .cancel_and_switch(target, Duration::from_secs(1))
        .await
        .expect("terminal cleanup owns the race");

    assert_eq!(selected.incarnation_id().0, "new-incarnation");
}

#[tokio::test]
async fn failed_target_startup_preserves_old_ready_selection() {
    let old = Arc::new(FakeWorker::new("burn:wgpu:default", "old-incarnation"));
    let service = WorkerSwitchService::new(old.clone(), Arc::new(NoopRunCancellation));
    let selected_before = service.selected().await;

    let error = service
        .drain_and_switch(Arc::new(FailingTarget), Duration::from_secs(1))
        .await
        .expect_err("startup fails");

    assert!(matches!(error, WorkerSwitchError::Startup { .. }));
    assert_eq!(service.selected().await, selected_before);
    assert_eq!(
        old.leases.admission(),
        reimagine_backend_worker_host::WorkerAdmissionState::Ready
    );
    assert_eq!(old.shutdowns.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn drain_timeout_restores_old_admission_and_discards_target() {
    let old = Arc::new(FakeWorker::new("burn:wgpu:default", "old-incarnation"));
    let target = Arc::new(FakeWorker::new("burn:wgpu:default", "new-incarnation"));
    target.fail_shutdown.store(true, Ordering::SeqCst);
    old.leases
        .acquire(&RunId::new("still-running"))
        .expect("lease");
    let service = WorkerSwitchService::new(old.clone(), Arc::new(NoopRunCancellation));
    let selected_before = service.selected().await;
    let switch_target = Arc::new(FakeTarget {
        worker: target.clone(),
        events: Arc::new(Mutex::new(Vec::new())),
    });

    let error = service
        .drain_and_switch(switch_target, Duration::from_millis(10))
        .await
        .expect_err("drain times out");

    assert!(matches!(error, WorkerSwitchError::DrainTimeout { .. }));
    assert_eq!(service.selected().await, selected_before);
    assert_eq!(
        old.leases.admission(),
        reimagine_backend_worker_host::WorkerAdmissionState::Ready
    );
    assert_eq!(target.shutdowns.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cancel_timeout_is_not_replaced_by_target_shutdown_failure() {
    let old = Arc::new(FakeWorker::new("burn:wgpu:default", "old-incarnation"));
    let target = Arc::new(FakeWorker::new("burn:wgpu:default", "new-incarnation"));
    target.fail_shutdown.store(true, Ordering::SeqCst);
    old.leases
        .acquire(&RunId::new("cancel-still-running"))
        .expect("lease");
    let service = WorkerSwitchService::new(old.clone(), Arc::new(NoopRunCancellation));

    let error = service
        .cancel_and_switch(
            Arc::new(FakeTarget {
                worker: target,
                events: Arc::new(Mutex::new(Vec::new())),
            }),
            Duration::from_millis(10),
        )
        .await
        .expect_err("cancel drain times out");

    assert!(matches!(error, WorkerSwitchError::DrainTimeout { .. }));
    assert_eq!(
        old.leases.admission(),
        reimagine_backend_worker_host::WorkerAdmissionState::Ready
    );
}

#[tokio::test]
async fn crashed_old_worker_commits_only_after_terminal_cleanup_releases_lease() {
    let old = Arc::new(FakeWorker::new("burn:wgpu:default", "old-incarnation"));
    let target = Arc::new(FakeWorker::new("burn:wgpu:default", "new-incarnation"));
    let run_id = RunId::new("crashed-run");
    old.leases.acquire(&run_id).expect("lease");
    let service = Arc::new(WorkerSwitchService::new(
        old.clone(),
        Arc::new(NoopRunCancellation),
    ));
    let switch = {
        let service = service.clone();
        let target = Arc::new(FakeTarget {
            worker: target,
            events: Arc::new(Mutex::new(Vec::new())),
        });
        tokio::spawn(async move {
            service
                .drain_and_switch(target, Duration::from_secs(1))
                .await
        })
    };

    tokio::task::yield_now().await;
    *old.state.lock().expect("state") = WorkerProcessState::Failed;
    assert!(
        !switch.is_finished(),
        "crash alone must not bypass run cleanup"
    );
    assert!(old.leases.release(&run_id));

    let selected = switch.await.expect("switch task").expect("switch commits");
    assert_eq!(selected.incarnation_id().0, "new-incarnation");
}

#[tokio::test]
async fn committed_switch_is_successful_even_when_old_worker_retirement_fails() {
    let old = Arc::new(FakeWorker::new("burn:wgpu:default", "old-incarnation"));
    old.fail_shutdown.store(true, Ordering::SeqCst);
    let target = Arc::new(FakeWorker::new("burn:wgpu:default", "new-incarnation"));
    let service = WorkerSwitchService::new(old, Arc::new(NoopRunCancellation));
    let stale = service.selected().await;
    let switch_target = Arc::new(FakeTarget {
        worker: target,
        events: Arc::new(Mutex::new(Vec::new())),
    });

    let selected = service
        .drain_and_switch(switch_target, Duration::from_secs(1))
        .await
        .expect("commit remains successful");

    assert_eq!(selected.incarnation_id().0, "new-incarnation");
    assert!(matches!(
        service.resolve(&stale).await,
        Err(WorkerSwitchError::StaleHandle { .. })
    ));
}

#[tokio::test]
async fn inference_runtime_routes_atomically_to_the_committed_worker() {
    let old = Arc::new(FakeWorker::with_latent_backend(
        "burn:wgpu:default",
        "old-incarnation",
        "old-payload",
    ));
    let target = Arc::new(FakeWorker::with_latent_backend(
        "burn:wgpu:secondary",
        "new-incarnation",
        "new-payload",
    ));
    let workers = Arc::new(WorkerSwitchService::new(old, Arc::new(NoopRunCancellation)));
    let runtime = reimagine_app_host::SwitchingInferenceRuntime::new(workers.clone());
    let request = || {
        CreateEmptyLatentRequest::new(
            64,
            64,
            1,
            RunId::new("route-run"),
            WorkflowId::new("route-workflow"),
            WorkflowVersion::new(1),
            NodeId::new("route-node"),
        )
    };

    let before = runtime
        .create_empty_latent(request())
        .await
        .expect("old route");
    assert_eq!(
        before.latent().payload().payload_key().as_str(),
        "old-payload"
    );

    workers
        .drain_and_switch(
            Arc::new(FakeTarget {
                worker: target,
                events: Arc::new(Mutex::new(Vec::new())),
            }),
            Duration::from_secs(1),
        )
        .await
        .expect("switch");

    let after = runtime
        .create_empty_latent(request())
        .await
        .expect("new route");
    assert_eq!(
        after.latent().payload().payload_key().as_str(),
        "new-payload"
    );
    assert_eq!(
        after.latent().payload().backend_instance().as_str(),
        "burn:wgpu:secondary"
    );
}

#[tokio::test]
async fn cancellation_failure_rolls_back_old_admission_without_committing_target() {
    let old = Arc::new(FakeWorker::new("burn:wgpu:default", "old-incarnation"));
    let target = Arc::new(FakeWorker::new("burn:wgpu:default", "new-incarnation"));
    old.leases
        .acquire(&RunId::new("cancel-fails"))
        .expect("lease");
    let service = WorkerSwitchService::new(old.clone(), Arc::new(FailingRunCancellation));
    let selected_before = service.selected().await;

    let error = service
        .cancel_and_switch(
            Arc::new(FakeTarget {
                worker: target.clone(),
                events: Arc::new(Mutex::new(Vec::new())),
            }),
            Duration::from_secs(1),
        )
        .await
        .expect_err("cancellation fails");

    assert!(matches!(error, WorkerSwitchError::Cancellation { .. }));
    assert_eq!(service.selected().await, selected_before);
    assert_eq!(
        old.leases.admission(),
        reimagine_backend_worker_host::WorkerAdmissionState::Ready
    );
    assert_eq!(target.shutdowns.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn target_crash_before_ready_preserves_old_selection() {
    let old = Arc::new(FakeWorker::new("burn:wgpu:default", "old-incarnation"));
    let target = Arc::new(FakeWorker::new("burn:wgpu:default", "new-incarnation"));
    *target.state.lock().expect("state") = WorkerProcessState::Failed;
    let service = WorkerSwitchService::new(old.clone(), Arc::new(NoopRunCancellation));
    let selected_before = service.selected().await;

    let error = service
        .drain_and_switch(
            Arc::new(FakeTarget {
                worker: target.clone(),
                events: Arc::new(Mutex::new(Vec::new())),
            }),
            Duration::from_secs(1),
        )
        .await
        .expect_err("target crashed before ready");

    assert!(matches!(error, WorkerSwitchError::TargetNotReady { .. }));
    assert_eq!(service.selected().await, selected_before);
    assert_eq!(
        old.leases.admission(),
        reimagine_backend_worker_host::WorkerAdmissionState::Ready
    );
    assert_eq!(target.shutdowns.load(Ordering::SeqCst), 1);
}
