use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use reimagine_app_host::{
    RunCancellation, SwitchableWorker, WorkerSwitchError, WorkerSwitchService, WorkerSwitchTarget,
};
use reimagine_backend_worker_host::{WorkerAdmissionState, WorkerProcessState, WorkerRunLeases};
use reimagine_backend_worker_protocol::WorkerIncarnationId;
use reimagine_core::model::RunId;
use reimagine_inference::{BackendInstance, VramBudget};

/// The eviction budget the switch service applies to a retiring worker.
fn eviction_budget() -> VramBudget {
    VramBudget::unlimited().with_total_bytes(0)
}

#[derive(Default)]
struct NoopRunCancellation;

impl RunCancellation for NoopRunCancellation {
    fn cancel(&self, _run_id: &RunId) -> Result<(), WorkerSwitchError> {
        Ok(())
    }
}

struct ReleasingRunCancellation {
    leases: Arc<WorkerRunLeases>,
}

impl RunCancellation for ReleasingRunCancellation {
    fn cancel(&self, run_id: &RunId) -> Result<(), WorkerSwitchError> {
        self.leases.release(run_id);
        Ok(())
    }
}

struct FakeWorker {
    instance: BackendInstance,
    incarnation: WorkerIncarnationId,
    leases: Arc<WorkerRunLeases>,
    shutdowns: AtomicUsize,
    budgets: Mutex<Vec<VramBudget>>,
    fail_budget: AtomicBool,
    state: Mutex<WorkerProcessState>,
}

impl FakeWorker {
    fn new(instance: &str, incarnation: &str) -> Self {
        Self {
            instance: BackendInstance::new(instance),
            incarnation: WorkerIncarnationId(incarnation.to_owned()),
            leases: Arc::new(WorkerRunLeases::new()),
            shutdowns: AtomicUsize::new(0),
            budgets: Mutex::new(Vec::new()),
            fail_budget: AtomicBool::new(false),
            state: Mutex::new(WorkerProcessState::Ready),
        }
    }

    fn applied_budgets(&self) -> Vec<VramBudget> {
        self.budgets.lock().expect("budgets").clone()
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

    async fn set_model_cache_budget(&self, budget: VramBudget) -> Result<(), WorkerSwitchError> {
        self.budgets.lock().expect("budgets").push(budget);
        if self.fail_budget.load(Ordering::SeqCst) {
            return Err(WorkerSwitchError::Shutdown {
                instance: self.instance.clone(),
                message: "forced budget failure".to_owned(),
            });
        }
        Ok(())
    }

    async fn shutdown(&self) -> Result<(), WorkerSwitchError> {
        self.shutdowns.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct FakeTarget {
    worker: Arc<FakeWorker>,
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
        Ok(self.worker.clone())
    }
}

#[tokio::test]
async fn drain_switch_evicts_retiring_worker_cache_after_leases_drain() {
    let old = Arc::new(FakeWorker::new("burn:wgpu:default", "old-incarnation"));
    let target = Arc::new(FakeWorker::new("burn:wgpu:default", "new-incarnation"));
    let run_id = RunId::new("leased-run");
    old.leases.acquire(&run_id).expect("initial lease");
    let service = Arc::new(WorkerSwitchService::new(
        old.clone(),
        Arc::new(NoopRunCancellation),
    ));

    let switch = {
        let service = Arc::clone(&service);
        let target = Arc::new(FakeTarget { worker: target });
        tokio::spawn(async move {
            service
                .drain_and_switch(target, Duration::from_secs(1))
                .await
        })
    };

    tokio::task::yield_now().await;
    assert_eq!(old.leases.admission(), WorkerAdmissionState::Draining);
    assert!(
        old.applied_budgets().is_empty(),
        "cache eviction must wait for the drain to complete"
    );

    assert!(old.leases.release(&run_id));
    let selected = switch.await.expect("switch task").expect("switch succeeds");

    assert_eq!(selected.incarnation_id().0, "new-incarnation");
    assert_eq!(
        old.applied_budgets(),
        vec![eviction_budget()],
        "retiring worker must be told to evict its cross-run cache"
    );
    assert_eq!(old.shutdowns.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn drain_timeout_keeps_retiring_worker_cache_untouched() {
    let old = Arc::new(FakeWorker::new("burn:wgpu:default", "old-incarnation"));
    let target = Arc::new(FakeWorker::new("burn:wgpu:default", "new-incarnation"));
    old.leases
        .acquire(&RunId::new("still-running"))
        .expect("lease");
    let service = WorkerSwitchService::new(old.clone(), Arc::new(NoopRunCancellation));

    let error = service
        .drain_and_switch(
            Arc::new(FakeTarget { worker: target }),
            Duration::from_millis(10),
        )
        .await
        .expect_err("drain times out");

    assert!(matches!(error, WorkerSwitchError::DrainTimeout { .. }));
    assert_eq!(
        old.leases.admission(),
        WorkerAdmissionState::Ready,
        "worker that keeps serving must not have its cache evicted"
    );
    assert!(old.applied_budgets().is_empty());
}

#[tokio::test]
async fn cancel_switch_evicts_retiring_worker_cache_after_cancel_drain() {
    let old = Arc::new(FakeWorker::new("burn:wgpu:default", "old-incarnation"));
    let target = Arc::new(FakeWorker::new("burn:wgpu:default", "new-incarnation"));
    old.leases
        .acquire(&RunId::new("cancel-run"))
        .expect("lease");
    let service = WorkerSwitchService::new(
        old.clone(),
        Arc::new(ReleasingRunCancellation {
            leases: Arc::clone(&old.leases),
        }),
    );

    let selected = service
        .cancel_and_switch(
            Arc::new(FakeTarget { worker: target }),
            Duration::from_secs(1),
        )
        .await
        .expect("cancel switch succeeds");

    assert_eq!(selected.incarnation_id().0, "new-incarnation");
    assert_eq!(old.applied_budgets(), vec![eviction_budget()]);
}

#[tokio::test]
async fn shutdown_active_evicts_cache_before_shutdown() {
    let old = Arc::new(FakeWorker::new("burn:wgpu:default", "old-incarnation"));
    let service = WorkerSwitchService::new(old.clone(), Arc::new(NoopRunCancellation));

    service
        .shutdown_active(Duration::from_secs(1))
        .await
        .expect("graceful shutdown succeeds");

    assert_eq!(old.applied_budgets(), vec![eviction_budget()]);
    assert_eq!(old.shutdowns.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn failed_cache_eviction_does_not_fail_the_switch() {
    let old = Arc::new(FakeWorker::new("burn:wgpu:default", "old-incarnation"));
    old.fail_budget.store(true, Ordering::SeqCst);
    let target = Arc::new(FakeWorker::new("burn:wgpu:default", "new-incarnation"));
    let service = WorkerSwitchService::new(old.clone(), Arc::new(NoopRunCancellation));

    let selected = service
        .drain_and_switch(
            Arc::new(FakeTarget { worker: target }),
            Duration::from_secs(1),
        )
        .await
        .expect("budget failure is non-fatal");

    assert_eq!(selected.incarnation_id().0, "new-incarnation");
    assert_eq!(old.applied_budgets(), vec![eviction_budget()]);
}

#[tokio::test]
async fn failed_target_startup_does_not_evict_active_worker_cache() {
    let old = Arc::new(FakeWorker::new("burn:wgpu:default", "old-incarnation"));
    let service = WorkerSwitchService::new(old.clone(), Arc::new(NoopRunCancellation));

    let error = service
        .drain_and_switch(Arc::new(FailingTarget), Duration::from_secs(1))
        .await
        .expect_err("startup fails");

    assert!(matches!(error, WorkerSwitchError::Startup { .. }));
    assert!(old.applied_budgets().is_empty());
}
