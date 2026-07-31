use std::sync::Arc;
use std::sync::RwLock as SyncRwLock;
use std::time::Duration;

use async_trait::async_trait;
use reimagine_backend_worker_host::{
    ProcessInferenceBackend, StartedWorker, WorkerProcessState, WorkerRunLeases, WorkerSupervisor,
};
use reimagine_backend_worker_protocol::WorkerIncarnationId;
use reimagine_core::model::RunId;
use reimagine_inference::{
    Backend, BackendInstance, BackendInstanceObservation, BackendInstanceSnapshot,
    BackendRunLifecycle, BackendRunLifecycleReport, BackendRunLifecycleRequest, DeviceProfile,
    InferenceBackend, InferenceRouter, RouterRef,
};
use tokio::sync::{Mutex, RwLock};

use super::worker::WorkerBackendCandidate;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkerSelectionHandle {
    instance: BackendInstance,
    incarnation_id: WorkerIncarnationId,
    generation: u64,
}

impl WorkerSelectionHandle {
    pub fn instance(&self) -> &BackendInstance {
        &self.instance
    }

    pub fn incarnation_id(&self) -> &WorkerIncarnationId {
        &self.incarnation_id
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum WorkerSwitchError {
    NoActiveWorker,
    Startup {
        message: String,
    },
    TargetNotReady {
        instance: BackendInstance,
    },
    DrainTimeout {
        instance: BackendInstance,
    },
    Cancellation {
        run_id: RunId,
        message: String,
    },
    Shutdown {
        instance: BackendInstance,
        message: String,
    },
    StaleHandle {
        instance: BackendInstance,
    },
}

impl std::fmt::Display for WorkerSwitchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoActiveWorker => write!(formatter, "workspace has no active process worker"),
            Self::Startup { message } => write!(formatter, "worker startup failed: {message}"),
            Self::TargetNotReady { instance } => {
                write!(formatter, "target worker `{instance}` did not reach Ready")
            }
            Self::DrainTimeout { instance } => {
                write!(
                    formatter,
                    "worker `{instance}` did not drain before the deadline"
                )
            }
            Self::Cancellation { run_id, message } => {
                write!(formatter, "failed to cancel run `{run_id}`: {message}")
            }
            Self::Shutdown { instance, message } => {
                write!(formatter, "failed to stop worker `{instance}`: {message}")
            }
            Self::StaleHandle { instance } => {
                write!(formatter, "worker handle for `{instance}` is stale")
            }
        }
    }
}

impl std::error::Error for WorkerSwitchError {}

pub trait RunCancellation: Send + Sync + 'static {
    fn cancel(&self, run_id: &RunId) -> Result<(), WorkerSwitchError>;
}

impl RunCancellation for reimagine_runtime::RuntimeService {
    fn cancel(&self, run_id: &RunId) -> Result<(), WorkerSwitchError> {
        reimagine_runtime::RuntimeService::cancel(self, run_id).map_err(|error| {
            WorkerSwitchError::Cancellation {
                run_id: run_id.clone(),
                message: error.to_string(),
            }
        })
    }
}

#[async_trait]
pub trait SwitchableWorker: Send + Sync + 'static {
    fn instance(&self) -> &BackendInstance;
    fn incarnation_id(&self) -> &WorkerIncarnationId;
    fn run_leases(&self) -> &Arc<WorkerRunLeases>;
    fn process_state(&self) -> WorkerProcessState;
    fn inference_backend(&self) -> Option<Arc<dyn InferenceBackend>> {
        None
    }
    async fn cleanup_run(&self, run_id: &RunId) -> Result<(), WorkerSwitchError> {
        self.run_leases().release(run_id);
        Ok(())
    }
    async fn snapshot(&self) -> BackendInstanceSnapshot {
        BackendInstanceSnapshot {
            backend_instance: self.instance().clone(),
            backend: self
                .inference_backend()
                .map(|backend| backend.backend_kind().clone())
                .unwrap_or_else(|| Backend::new("worker")),
            plugin: None,
            extension: None,
            device: None,
            observations: std::collections::BTreeMap::from([(
                "process_state".to_owned(),
                format!("{:?}", self.process_state()),
            )]),
            diagnostics: Vec::new(),
        }
    }
    async fn shutdown(&self) -> Result<(), WorkerSwitchError>;
}

#[async_trait]
pub trait WorkerSwitchTarget: Send + Sync + 'static {
    async fn start(&self) -> Result<Arc<dyn SwitchableWorker>, WorkerSwitchError>;
}

pub struct ProcessSwitchableWorker {
    instance: BackendInstance,
    worker: Arc<StartedWorker>,
    backend: Arc<ProcessInferenceBackend>,
    run_leases: Arc<WorkerRunLeases>,
}

impl ProcessSwitchableWorker {
    pub(crate) fn new(
        instance: BackendInstance,
        worker: Arc<StartedWorker>,
        backend: Arc<ProcessInferenceBackend>,
    ) -> Self {
        let run_leases = backend.run_leases();
        Self {
            instance,
            worker,
            backend,
            run_leases,
        }
    }

    pub fn inference_backend(&self) -> Arc<ProcessInferenceBackend> {
        Arc::clone(&self.backend)
    }

    pub fn started_worker(&self) -> Arc<StartedWorker> {
        Arc::clone(&self.worker)
    }
}

#[async_trait]
impl SwitchableWorker for ProcessSwitchableWorker {
    fn instance(&self) -> &BackendInstance {
        &self.instance
    }

    fn incarnation_id(&self) -> &WorkerIncarnationId {
        self.worker.incarnation_id()
    }

    fn run_leases(&self) -> &Arc<WorkerRunLeases> {
        &self.run_leases
    }

    fn process_state(&self) -> WorkerProcessState {
        self.worker.state()
    }

    fn inference_backend(&self) -> Option<Arc<dyn InferenceBackend>> {
        let backend: Arc<dyn InferenceBackend> = self.backend.clone();
        Some(backend)
    }

    async fn cleanup_run(&self, run_id: &RunId) -> Result<(), WorkerSwitchError> {
        if !self.run_leases.owns(run_id) {
            return Ok(());
        }
        let cleanup = self
            .worker
            .cleanup(Some(run_id.to_string()), Vec::new())
            .await
            .map_err(|error| WorkerSwitchError::Shutdown {
                instance: self.instance.clone(),
                message: error.to_string(),
            });
        self.run_leases.release(run_id);
        cleanup.map(|_| ())
    }

    async fn snapshot(&self) -> BackendInstanceSnapshot {
        let live = self.worker.hello.profile.instances.iter().find(|profile| {
            profile.backend_instance_id == self.worker.hello.identity.backend_instance_id
        });
        BackendInstanceSnapshot {
            backend_instance: self.instance.clone(),
            backend: self.backend.backend_kind().clone(),
            plugin: Some(reimagine_plugin::Plugin::try_from("builtin.burn").expect("plugin id")),
            extension: Some(
                reimagine_plugin::Extension::try_from("backend.burn").expect("extension id"),
            ),
            device: live.map(|profile| DeviceProfile::new(profile.device_label.clone())),
            observations: std::collections::BTreeMap::from([
                ("process_backed".to_owned(), "true".to_owned()),
                (
                    "process_state".to_owned(),
                    format!("{:?}", self.process_state()),
                ),
            ]),
            diagnostics: Vec::new(),
        }
    }

    async fn shutdown(&self) -> Result<(), WorkerSwitchError> {
        self.worker
            .shutdown()
            .await
            .map_err(|error| WorkerSwitchError::Shutdown {
                instance: self.instance.clone(),
                message: error.to_string(),
            })
    }
}

#[async_trait]
impl WorkerSwitchTarget for WorkerBackendCandidate {
    async fn start(&self) -> Result<Arc<dyn SwitchableWorker>, WorkerSwitchError> {
        let worker = Arc::new(
            WorkerSupervisor::new(self.launch().clone())
                .start()
                .await
                .map_err(|error| WorkerSwitchError::Startup {
                    message: error.to_string(),
                })?,
        );
        let live_profile = worker.hello.profile.instances.iter().find(|profile| {
            profile.backend_instance_id == self.manifest_profile().backend_instance_id
        });
        if live_profile != Some(self.manifest_profile()) {
            let _ = worker.shutdown().await;
            return Err(WorkerSwitchError::Startup {
                message: format!(
                    "worker `{}` live hello does not match its inventory manifest profile",
                    self.backend_instance()
                ),
            });
        }

        let backend = Arc::new(ProcessInferenceBackend::new(Arc::clone(&worker)));
        Ok(Arc::new(ProcessSwitchableWorker::new(
            self.backend_instance(),
            worker,
            backend,
        )))
    }
}

struct WorkerSwitchState {
    active: Arc<dyn SwitchableWorker>,
    generation: u64,
}

pub struct WorkerSwitchService {
    state: RwLock<WorkerSwitchState>,
    transaction: Mutex<()>,
    run_cancellation: SyncRwLock<Arc<dyn RunCancellation>>,
    router_ref: SyncRwLock<Option<RouterRef>>,
}

impl std::fmt::Debug for WorkerSwitchService {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WorkerSwitchService")
            .finish_non_exhaustive()
    }
}

pub(crate) struct SwitchingWorkerRuntimeHooks {
    workers: Arc<WorkerSwitchService>,
    logical_instance: BackendInstance,
}

impl SwitchingWorkerRuntimeHooks {
    pub(crate) fn new(workers: Arc<WorkerSwitchService>) -> Self {
        Self {
            workers,
            logical_instance: BackendInstance::new("switching:worker"),
        }
    }
}

#[async_trait]
impl BackendRunLifecycle for SwitchingWorkerRuntimeHooks {
    fn backend_instance(&self) -> &BackendInstance {
        &self.logical_instance
    }

    async fn begin_run(
        &self,
        _request: BackendRunLifecycleRequest,
    ) -> Result<BackendRunLifecycleReport, reimagine_inference::InferenceError> {
        let instance = self.workers.state.read().await.active.instance().clone();
        Ok(BackendRunLifecycleReport {
            backend_instance: instance,
            diagnostics: Vec::new(),
        })
    }

    async fn cleanup_run(
        &self,
        request: BackendRunLifecycleRequest,
    ) -> Result<BackendRunLifecycleReport, reimagine_inference::InferenceError> {
        let active = Arc::clone(&self.workers.state.read().await.active);
        let instance = active.instance().clone();
        active.cleanup_run(&request.run_id).await.map_err(|error| {
            reimagine_inference::InferenceError::BackendExecutionFailed {
                message: error.to_string(),
            }
        })?;
        Ok(BackendRunLifecycleReport {
            backend_instance: instance,
            diagnostics: Vec::new(),
        })
    }
}

#[async_trait]
impl BackendInstanceObservation for SwitchingWorkerRuntimeHooks {
    fn backend_instance(&self) -> &BackendInstance {
        &self.logical_instance
    }

    async fn snapshot(&self) -> BackendInstanceSnapshot {
        let active = Arc::clone(&self.workers.state.read().await.active);
        active.snapshot().await
    }
}

impl WorkerSwitchService {
    pub fn new(
        active: Arc<dyn SwitchableWorker>,
        run_cancellation: Arc<dyn RunCancellation>,
    ) -> Self {
        Self {
            state: RwLock::new(WorkerSwitchState {
                active,
                generation: 0,
            }),
            transaction: Mutex::new(()),
            run_cancellation: SyncRwLock::new(run_cancellation),
            router_ref: SyncRwLock::new(None),
        }
    }

    /// Attach a `RouterRef` so that hot-swap can atomically replace the router.
    pub fn set_router_ref(&self, router_ref: RouterRef) {
        *self.router_ref.write().expect("router_ref poisoned") = Some(router_ref);
    }

    /// Atomically swap the current `InferenceRouter` for a new one.
    ///
    /// This is called after a worker switch to point executors at the
    /// new backend. In-flight requests on the old router continue to
    /// completion (they hold an `Arc` guard via `ArcSwap::load`).
    pub fn swap_router(&self, new_router: InferenceRouter) {
        if let Some(ref router_ref) = *self.router_ref.read().expect("router_ref poisoned") {
            router_ref.store(Arc::new(new_router));
        }
    }

    pub(crate) fn set_run_cancellation(&self, run_cancellation: Arc<dyn RunCancellation>) {
        *self
            .run_cancellation
            .write()
            .expect("worker run cancellation poisoned") = run_cancellation;
    }

    pub async fn selected(&self) -> WorkerSelectionHandle {
        let state = self.state.read().await;
        selection_handle(&state)
    }

    pub async fn resolve(
        &self,
        handle: &WorkerSelectionHandle,
    ) -> Result<Arc<dyn SwitchableWorker>, WorkerSwitchError> {
        let state = self.state.read().await;
        if handle != &selection_handle(&state) {
            return Err(WorkerSwitchError::StaleHandle {
                instance: handle.instance.clone(),
            });
        }
        Ok(Arc::clone(&state.active))
    }

    pub async fn drain_and_switch(
        &self,
        target: Arc<dyn WorkerSwitchTarget>,
        deadline: Duration,
    ) -> Result<WorkerSelectionHandle, WorkerSwitchError> {
        let _transaction = self.transaction.lock().await;
        let target = target.start().await?;
        if target.process_state() != WorkerProcessState::Ready {
            let instance = target.instance().clone();
            let _ = target.shutdown().await;
            return Err(WorkerSwitchError::TargetNotReady { instance });
        }

        let previous = Arc::clone(&self.state.read().await.active);
        previous.run_leases().begin_draining();
        if !previous.run_leases().wait_until_empty(deadline).await {
            previous.run_leases().restore_ready();
            let old_instance = previous.instance().clone();
            let _ = target.shutdown().await;
            return Err(WorkerSwitchError::DrainTimeout {
                instance: old_instance,
            });
        }

        let mut state = self.state.write().await;
        state.active = target;
        state.generation = state.generation.wrapping_add(1);
        let selected = selection_handle(&state);
        drop(state);
        let _ = previous.shutdown().await;
        Ok(selected)
    }

    /// Shut down the active worker gracefully.
    ///
    /// If graceful shutdown does not complete within `deadline`, the worker
    /// process is force-killed via the drop-based cleanup.
    pub async fn shutdown_active(&self, deadline: Duration) -> Result<(), WorkerSwitchError> {
        let active = Arc::clone(&self.state.read().await.active);
        match tokio::time::timeout(deadline, active.shutdown()).await {
            Ok(result) => result,
            Err(_elapsed) => {
                // Graceful shutdown timed out — drop the Arc so the
                // process supervisor can kill the child on drop.
                eprintln!(
                    "[app-host] worker `{}` graceful shutdown timed out after {deadline:?}; dropping reference",
                    active.instance(),
                );
                Ok(())
            }
        }
    }

    pub async fn cancel_and_switch(
        &self,
        target: Arc<dyn WorkerSwitchTarget>,
        deadline: Duration,
    ) -> Result<WorkerSelectionHandle, WorkerSwitchError> {
        let _transaction = self.transaction.lock().await;
        let target = target.start().await?;
        if target.process_state() != WorkerProcessState::Ready {
            let instance = target.instance().clone();
            let _ = target.shutdown().await;
            return Err(WorkerSwitchError::TargetNotReady { instance });
        }

        let previous = Arc::clone(&self.state.read().await.active);
        previous.run_leases().begin_draining();
        let run_cancellation = self
            .run_cancellation
            .read()
            .expect("worker run cancellation poisoned")
            .clone();
        for run_id in previous.run_leases().owned_run_ids() {
            if let Err(error) = run_cancellation.cancel(&run_id) {
                if !previous.run_leases().owns(&run_id) {
                    continue;
                }
                previous.run_leases().restore_ready();
                let _ = target.shutdown().await;
                return Err(error);
            }
        }
        if !previous.run_leases().wait_until_empty(deadline).await {
            previous.run_leases().restore_ready();
            let old_instance = previous.instance().clone();
            let _ = target.shutdown().await;
            return Err(WorkerSwitchError::DrainTimeout {
                instance: old_instance,
            });
        }

        let mut state = self.state.write().await;
        state.active = target;
        state.generation = state.generation.wrapping_add(1);
        let selected = selection_handle(&state);
        drop(state);
        let _ = previous.shutdown().await;
        Ok(selected)
    }
}

fn selection_handle(state: &WorkerSwitchState) -> WorkerSelectionHandle {
    WorkerSelectionHandle {
        instance: state.active.instance().clone(),
        incarnation_id: state.active.incarnation_id().clone(),
        generation: state.generation,
    }
}
