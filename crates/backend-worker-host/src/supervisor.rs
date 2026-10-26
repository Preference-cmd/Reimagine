use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::time::Duration;

use reimagine_backend_worker_protocol::client::WorkerClient;
use reimagine_backend_worker_protocol::conversion::{
    hello_from_proto, outcome_from_proto, progress_from_proto,
};
use reimagine_backend_worker_protocol::proto;
use reimagine_backend_worker_protocol::{
    ProgressFrame, TerminalOutcome, WorkerHello, WorkerIncarnationId,
};
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::timeout;

use crate::transport::{LocalProcessTransportFactory, WorkerConnection, WorkerTransportFactory};
use crate::{ExpectedWorkerIdentity, WorkerHostError, WorkerLaunchSpec};

const STATE_STOPPED: u8 = 0;
const STATE_READY: u8 = 1;
const STATE_FAILED: u8 = 2;
const STATE_STARTING: u8 = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkerProcessState {
    Stopped,
    Ready,
    Failed,
    Starting,
}

/// The terminal result of one executed operation: every progress frame
/// observed plus the single terminal outcome.
pub struct WorkerRequestResult {
    pub progress: Vec<ProgressFrame>,
    pub terminal: TerminalOutcome,
}

/// Liveness answer from the worker's `Health` RPC.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HealthReport {
    pub healthy: bool,
    pub message: Option<String>,
}

/// Shared per-request cancellation state between a handle and its
/// canceller.
struct RequestShared {
    cancelled: AtomicBool,
    terminal_seen: AtomicBool,
    abort: tokio::task::AbortHandle,
}

pub struct WorkerRequestHandle {
    inner: Arc<WorkerClientInner>,
    shared: Arc<RequestShared>,
    driver: JoinHandle<()>,
    receiver: mpsc::UnboundedReceiver<PendingEvent>,
}

impl Drop for WorkerRequestHandle {
    fn drop(&mut self) {
        // A dropped handle without `finish` cancels the worker-side
        // operation: aborting the driver drops the Execute stream, and
        // the gRPC cancellation reaches the worker.
        if !self.driver.is_finished() {
            self.driver.abort();
        }
    }
}

#[derive(Clone)]
pub struct WorkerRequestCanceller {
    inner: Arc<WorkerClientInner>,
    shared: Arc<RequestShared>,
}

impl WorkerRequestCanceller {
    pub async fn cancel(&self) -> Result<(), WorkerHostError> {
        cancel_request(&self.inner, &self.shared).await
    }
}

impl WorkerRequestHandle {
    pub async fn cancel(&self) -> Result<(), WorkerHostError> {
        cancel_request(&self.inner, &self.shared).await
    }

    pub fn canceller(&self) -> WorkerRequestCanceller {
        WorkerRequestCanceller {
            inner: Arc::clone(&self.inner),
            shared: Arc::clone(&self.shared),
        }
    }

    pub async fn finish(self) -> Result<WorkerRequestResult, WorkerHostError> {
        self.finish_with_progress(|_| {}).await
    }

    pub async fn finish_with_progress<F>(
        mut self,
        mut on_progress: F,
    ) -> Result<WorkerRequestResult, WorkerHostError>
    where
        F: FnMut(&ProgressFrame),
    {
        let request_timeout = self.inner.request_timeout;
        let receive = async {
            let mut progress = Vec::new();
            loop {
                match self.receiver.recv().await {
                    Some(PendingEvent::Progress(frame)) => {
                        on_progress(&frame);
                        progress.push(frame);
                    }
                    Some(PendingEvent::Terminal(outcome)) => {
                        self.shared.terminal_seen.store(true, Ordering::Release);
                        return Ok(WorkerRequestResult {
                            progress,
                            terminal: outcome,
                        });
                    }
                    Some(PendingEvent::TransportLost(message)) => {
                        return self.transport_lost_or_cancelled(progress, message);
                    }
                    None => {
                        return self.transport_lost_or_cancelled(
                            progress,
                            "worker response channel closed".to_owned(),
                        );
                    }
                }
            }
        };
        match timeout(request_timeout, receive).await {
            Ok(result) => result,
            Err(_) => {
                // Abort the driver: dropping the Execute stream cancels
                // the call and stops the operation on the worker.
                self.driver.abort();
                Err(WorkerHostError::RequestTimeout)
            }
        }
    }

    /// A cancel reported by the host yields the Cancelled terminal even
    /// when the worker never delivered one; anything else is a lost
    /// transport. Either way exactly one terminal is reported.
    fn transport_lost_or_cancelled(
        &self,
        progress: Vec<ProgressFrame>,
        message: String,
    ) -> Result<WorkerRequestResult, WorkerHostError> {
        if self.shared.cancelled.load(Ordering::Acquire) {
            Ok(WorkerRequestResult {
                progress,
                terminal: TerminalOutcome::Cancelled,
            })
        } else {
            Err(WorkerHostError::TransportLost { message })
        }
    }
}

enum PendingEvent {
    Progress(ProgressFrame),
    Terminal(TerminalOutcome),
    TransportLost(String),
}

struct ReaderLifecycle {
    state: Arc<AtomicU8>,
    current_generation: Arc<AtomicU64>,
    generation: u64,
    alive: Arc<AtomicBool>,
}

struct WorkerClientInner {
    client: WorkerClient,
    incarnation_id: WorkerIncarnationId,
    request_timeout: Duration,
    shutdown_timeout: Duration,
    state: Arc<AtomicU8>,
    current_generation: Arc<AtomicU64>,
    generation: u64,
    alive: Arc<AtomicBool>,
}

impl Drop for WorkerClientInner {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::Release);
        if self.current_generation.load(Ordering::Acquire) == self.generation {
            let _ = self.state.compare_exchange(
                STATE_READY,
                STATE_STOPPED,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
    }
}

pub struct StartedWorker {
    pub hello: WorkerHello,
    inner: Arc<WorkerClientInner>,
    transport: Arc<dyn crate::transport::WorkerLifecycle>,
    stderr_tail: Arc<Mutex<Vec<u8>>>,
    _stderr_task: Option<JoinHandle<()>>,
}

impl StartedWorker {
    pub fn state(&self) -> WorkerProcessState {
        match self.inner.state.load(Ordering::Acquire) {
            STATE_READY => WorkerProcessState::Ready,
            STATE_FAILED => WorkerProcessState::Failed,
            STATE_STARTING => WorkerProcessState::Starting,
            _ => WorkerProcessState::Stopped,
        }
    }

    pub fn incarnation_id(&self) -> &WorkerIncarnationId {
        &self.inner.incarnation_id
    }

    /// Returns the capabilities advertised by the worker during hello.
    #[must_use]
    pub fn capabilities(&self) -> Vec<String> {
        self.hello
            .profile
            .instances
            .iter()
            .find(|p| p.backend_instance_id == self.hello.identity.backend_instance_id)
            .map(|p| p.capabilities.clone())
            .unwrap_or_default()
    }

    /// Check whether the worker advertises a specific capability.
    #[must_use]
    pub fn has_capability(&self, cap: &str) -> bool {
        self.capabilities().iter().any(|c| c == cap)
    }

    pub async fn stderr_tail(&self) -> Vec<u8> {
        self.stderr_tail.lock().await.clone()
    }

    /// The transport endpoint the worker is reachable on (for local
    /// workers: the private socket or pipe URI).
    #[must_use]
    pub fn transport_endpoint(&self) -> String {
        self.transport.description().endpoint
    }

    pub async fn health(&self) -> Result<HealthReport, WorkerHostError> {
        ensure_alive(&self.inner)?;
        let response = call_with_timeout(
            self.inner.request_timeout,
            "health",
            self.inner.client.health(),
        )
        .await?;
        Ok(HealthReport {
            healthy: response.healthy,
            message: response.message,
        })
    }

    /// Release run-scoped payloads held by the worker; returns the
    /// number of released objects.
    pub async fn cleanup(
        &self,
        run_id: Option<String>,
        object_ids: Vec<String>,
    ) -> Result<u64, WorkerHostError> {
        ensure_alive(&self.inner)?;
        let response = call_with_timeout(
            self.inner.request_timeout,
            "cleanup",
            self.inner.client.cleanup(run_id, object_ids),
        )
        .await?;
        Ok(response.released_objects)
    }

    pub async fn shutdown(&self) -> Result<(), WorkerHostError> {
        ensure_alive(&self.inner)?;
        match timeout(self.inner.shutdown_timeout, self.inner.client.shutdown()).await {
            Ok(Ok(_)) => {}
            Ok(Err(status)) => {
                return Err(WorkerHostError::TransportLost {
                    message: format!("worker shutdown call failed: {status}"),
                });
            }
            Err(_) => {
                let _ = self.transport.shutdown().await;
                self.inner.alive.store(false, Ordering::Release);
                store_state_if_current(
                    &self.inner.state,
                    &self.inner.current_generation,
                    self.inner.generation,
                    STATE_STOPPED,
                );
                return Err(WorkerHostError::ShutdownTimeout);
            }
        }
        let _ = self.transport.shutdown().await;
        self.inner.alive.store(false, Ordering::Release);
        store_state_if_current(
            &self.inner.state,
            &self.inner.current_generation,
            self.inner.generation,
            STATE_STOPPED,
        );
        Ok(())
    }

    pub async fn request(
        &self,
        operation: impl Into<String>,
        payload: serde_json::Value,
    ) -> Result<WorkerRequestResult, WorkerHostError> {
        self.begin_request(operation, payload).await?.finish().await
    }

    pub async fn begin_request(
        &self,
        operation: impl Into<String>,
        payload: serde_json::Value,
    ) -> Result<WorkerRequestHandle, WorkerHostError> {
        ensure_alive(&self.inner)?;
        let payload_bytes =
            serde_json::to_vec(&payload).map_err(|error| WorkerHostError::TransportLost {
                message: format!("request payload is not serializable: {error}"),
            })?;
        let stream = self
            .inner
            .client
            .execute(operation, payload_bytes)
            .await
            .map_err(|status| {
                // A call that never got a stream means the transport is
                // gone (the worker died mid-call); mark the supervisor
                // failed the same way the stream driver would have.
                fail_client(&self.inner);
                WorkerHostError::TransportLost {
                    message: format!("worker rejected the execute call: {status}"),
                }
            })?;
        let (sender, receiver) = mpsc::unbounded_channel();
        let driver = tokio::spawn(drive_execute(
            stream,
            sender,
            ReaderLifecycle {
                state: Arc::clone(&self.inner.state),
                current_generation: Arc::clone(&self.inner.current_generation),
                generation: self.inner.generation,
                alive: Arc::clone(&self.inner.alive),
            },
        ));
        Ok(WorkerRequestHandle {
            inner: Arc::clone(&self.inner),
            shared: Arc::new(RequestShared {
                cancelled: AtomicBool::new(false),
                terminal_seen: AtomicBool::new(false),
                abort: driver.abort_handle(),
            }),
            driver,
            receiver,
        })
    }
}

async fn cancel_request(
    inner: &Arc<WorkerClientInner>,
    shared: &RequestShared,
) -> Result<(), WorkerHostError> {
    ensure_alive(inner)?;
    if shared.cancelled.swap(true, Ordering::AcqRel) {
        return Ok(());
    }
    if shared.terminal_seen.load(Ordering::Acquire) {
        return Ok(());
    }
    // Dropping the Execute stream is the cancellation: the worker sees
    // its response stream close and stops the operation.
    shared.abort.abort();
    Ok(())
}

/// Pump one `Execute` stream into the handle's event channel.
///
/// Dropping `stream` (through driver abort) is what cancels the call on
/// the worker, so this task must own the stream until it ends or the
/// caller gives up.
async fn drive_execute(
    mut stream: reimagine_backend_worker_protocol::Streaming<proto::ExecuteEvent>,
    sender: mpsc::UnboundedSender<PendingEvent>,
    lifecycle: ReaderLifecycle,
) {
    loop {
        match stream.message().await {
            Ok(Some(event)) => match event.event {
                Some(proto::execute_event::Event::Progress(progress)) => {
                    if sender
                        .send(PendingEvent::Progress(progress_from_proto(progress)))
                        .is_err()
                    {
                        // Receiver gone: the caller aborted us.
                        return;
                    }
                }
                Some(proto::execute_event::Event::Result(result)) => {
                    match outcome_from_proto(result) {
                        Some(outcome) => {
                            let _ = sender.send(PendingEvent::Terminal(outcome));
                        }
                        None => {
                            let _ = sender.send(PendingEvent::TransportLost(
                                "worker sent a result event without an outcome".to_owned(),
                            ));
                            fail_reader(
                                &lifecycle,
                                "worker result event had no outcome".to_owned(),
                            );
                        }
                    }
                    return;
                }
                None => {
                    let _ = sender.send(PendingEvent::TransportLost(
                        "worker closed the execute stream without a result".to_owned(),
                    ));
                    fail_reader(
                        &lifecycle,
                        "execute stream ended without a result".to_owned(),
                    );
                    return;
                }
            },
            Ok(None) => {
                let _ = sender.send(PendingEvent::TransportLost(
                    "worker closed the execute stream without a result".to_owned(),
                ));
                fail_reader(
                    &lifecycle,
                    "execute stream ended without a result".to_owned(),
                );
                return;
            }
            Err(status) => {
                let _ = sender.send(PendingEvent::TransportLost(format!(
                    "execute call failed: {status}"
                )));
                fail_reader(&lifecycle, format!("execute call failed: {status}"));
                return;
            }
        }
    }
}

async fn call_with_timeout<T, F>(
    deadline: Duration,
    operation: &'static str,
    call: F,
) -> Result<T, WorkerHostError>
where
    F: std::future::Future<Output = Result<T, reimagine_backend_worker_protocol::Status>>,
{
    match timeout(deadline, call).await {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(status)) => Err(WorkerHostError::TransportLost {
            message: format!("worker {operation} call failed: {status}"),
        }),
        Err(_) => Err(WorkerHostError::RequestTimeout),
    }
}

pub struct WorkerSupervisor {
    launch: WorkerLaunchSpec,
    transport_factory: Arc<dyn WorkerTransportFactory>,
    state: Arc<AtomicU8>,
    current_generation: Arc<AtomicU64>,
}

struct StartGuard {
    state: Arc<AtomicU8>,
    current_generation: Arc<AtomicU64>,
    generation: u64,
    committed: bool,
}

impl StartGuard {
    fn acquire(
        state: Arc<AtomicU8>,
        current_generation: Arc<AtomicU64>,
    ) -> Result<Self, WorkerHostError> {
        loop {
            let current = state.load(Ordering::Acquire);
            if matches!(current, STATE_READY | STATE_STARTING) {
                return Err(WorkerHostError::AlreadyStarted);
            }
            if state
                .compare_exchange(current, STATE_STARTING, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                let generation = current_generation.fetch_add(1, Ordering::AcqRel) + 1;
                return Ok(Self {
                    state,
                    current_generation,
                    generation,
                    committed: false,
                });
            }
        }
    }

    fn commit(mut self) {
        store_state_if_current(
            &self.state,
            &self.current_generation,
            self.generation,
            STATE_READY,
        );
        self.committed = true;
    }

    fn generation(&self) -> u64 {
        self.generation
    }
}

impl Drop for StartGuard {
    fn drop(&mut self) {
        if !self.committed {
            store_state_if_current(
                &self.state,
                &self.current_generation,
                self.generation,
                STATE_STOPPED,
            );
        }
    }
}

impl WorkerSupervisor {
    #[must_use]
    pub fn new(launch: WorkerLaunchSpec) -> Self {
        Self::new_with_factory(launch, Arc::new(LocalProcessTransportFactory))
    }

    #[must_use]
    pub fn new_with_factory(
        launch: WorkerLaunchSpec,
        transport_factory: Arc<dyn WorkerTransportFactory>,
    ) -> Self {
        Self {
            launch,
            transport_factory,
            state: Arc::new(AtomicU8::new(STATE_STOPPED)),
            current_generation: Arc::new(AtomicU64::new(0)),
        }
    }

    #[must_use]
    pub fn state(&self) -> WorkerProcessState {
        match self.state.load(Ordering::Acquire) {
            STATE_READY => WorkerProcessState::Ready,
            STATE_FAILED => WorkerProcessState::Failed,
            STATE_STARTING => WorkerProcessState::Starting,
            _ => WorkerProcessState::Stopped,
        }
    }

    pub async fn start(&self) -> Result<StartedWorker, WorkerHostError> {
        let start_guard = StartGuard::acquire(
            Arc::clone(&self.state),
            Arc::clone(&self.current_generation),
        )?;
        let generation = start_guard.generation();
        let startup = async {
            let WorkerConnection {
                transport,
                client,
                stderr_tail,
                stderr_task,
            } = self
                .transport_factory
                .connect(&self.launch)
                .await
                .map_err(WorkerHostError::Transport)?;
            let response = client.hello().await.map_err(|status| {
                WorkerHostError::Transport(
                    reimagine_backend_worker_protocol::TransportError::ConnectionFailed(format!(
                        "worker hello failed: {status}"
                    )),
                )
            })?;
            let hello = hello_from_proto(response).ok_or_else(|| {
                WorkerHostError::Transport(
                    reimagine_backend_worker_protocol::TransportError::ConnectionFailed(
                        "worker hello response is incomplete".to_owned(),
                    ),
                )
            })?;
            validate_hello(&hello, &self.launch.expected)?;
            if !self.launch.supported_protocols.contains(hello.protocol) {
                return Err(WorkerHostError::IdentityMismatch {
                    field: "protocol",
                    expected: self.launch.supported_protocols.to_string(),
                    actual: hello.protocol.0.to_string(),
                });
            }
            Ok((transport, client, hello, stderr_tail, stderr_task))
        };
        // The whole startup — spawn, READY line, connect, hello — is
        // bounded by one budget.
        let (transport, client, hello, stderr_tail, stderr_task) =
            timeout(self.launch.limits.startup_timeout, startup)
                .await
                .map_err(|_| WorkerHostError::StartupTimeout)??;
        let worker_caps: Vec<String> = hello
            .profile
            .instances
            .iter()
            .find(|p| p.backend_instance_id == hello.identity.backend_instance_id)
            .map(|p| p.capabilities.clone())
            .unwrap_or_default();
        tracing::info!(
            worker = %hello.identity.backend_instance_id.0,
            capabilities = ?worker_caps,
            "worker hello complete"
        );
        let alive = Arc::new(AtomicBool::new(true));
        let inner = Arc::new(WorkerClientInner {
            client,
            incarnation_id: hello.identity.incarnation_id.clone(),
            request_timeout: self.launch.limits.request_timeout,
            shutdown_timeout: self.launch.limits.shutdown_timeout,
            state: Arc::clone(&self.state),
            current_generation: Arc::clone(&self.current_generation),
            generation,
            alive: Arc::clone(&alive),
        });
        start_guard.commit();
        Ok(StartedWorker {
            hello,
            inner,
            transport,
            stderr_tail,
            _stderr_task: stderr_task,
        })
    }
}

/// Mark the worker behind `inner` failed: no new calls will succeed
/// and the supervisor reports a dead worker.
fn fail_client(inner: &WorkerClientInner) {
    inner.alive.store(false, Ordering::Release);
    store_state_if_current(
        &inner.state,
        &inner.current_generation,
        inner.generation,
        STATE_FAILED,
    );
}

fn fail_reader(lifecycle: &ReaderLifecycle, message: String) {
    lifecycle.alive.store(false, Ordering::Release);
    store_state_if_current(
        &lifecycle.state,
        &lifecycle.current_generation,
        lifecycle.generation,
        STATE_FAILED,
    );
    tracing::warn!("worker request stream failed: {message}");
}

fn ensure_alive(inner: &WorkerClientInner) -> Result<(), WorkerHostError> {
    if inner.alive.load(Ordering::Acquire) {
        return Ok(());
    }
    Err(WorkerHostError::TransportLost {
        message: format!(
            "worker incarnation `{}` is no longer alive",
            inner.incarnation_id.0
        ),
    })
}

fn store_state_if_current(
    state: &AtomicU8,
    current_generation: &AtomicU64,
    generation: u64,
    next_state: u8,
) {
    if current_generation.load(Ordering::Acquire) == generation {
        state.store(next_state, Ordering::Release);
    }
}

fn validate_hello(
    hello: &WorkerHello,
    expected: &ExpectedWorkerIdentity,
) -> Result<(), WorkerHostError> {
    validate_identity_field(
        "backend_instance_id",
        &expected.backend_instance_id.0,
        &hello.identity.backend_instance_id.0,
    )?;
    validate_identity_field(
        "installation_id",
        &expected.installation_id.0,
        &hello.identity.installation_id.0,
    )?;
    validate_identity_field(
        "backend_kind",
        &expected.backend_kind,
        &hello.identity.backend_kind,
    )?;
    validate_identity_field("target", &expected.target, &hello.identity.target)?;
    validate_identity_field(
        "manifest_digest",
        &expected.manifest_digest,
        &hello.identity.manifest_digest,
    )?;
    if hello
        .profile
        .instances
        .iter()
        .any(|instance| instance.backend_instance_id == expected.backend_instance_id)
    {
        return Ok(());
    }
    Err(WorkerHostError::IdentityMismatch {
        field: "profile.backend_instance_id",
        expected: expected.backend_instance_id.0.clone(),
        actual: hello
            .profile
            .instances
            .iter()
            .map(|instance| instance.backend_instance_id.0.as_str())
            .collect::<Vec<_>>()
            .join(","),
    })
}

fn validate_identity_field(
    field: &'static str,
    expected: &str,
    actual: &str,
) -> Result<(), WorkerHostError> {
    if expected == actual {
        return Ok(());
    }
    Err(WorkerHostError::IdentityMismatch {
        field,
        expected: expected.to_owned(),
        actual: actual.to_owned(),
    })
}
