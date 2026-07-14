use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::time::Duration;

use reimagine_backend_worker_protocol::{
    CancelAckFrame, CancelFrame, CleanupAckFrame, CleanupFrame, CodecError, ControlId,
    CorrelationId, FrameCodec, HealthAckFrame, HealthFrame, HostHello, MessageSender,
    ProgressFrame, ProtocolVersion, RequestFrame, RequestId, ShutdownFrame, TerminalFrame,
    WireMessage, WorkerHello, WorkerIncarnationId, validate_message_direction,
};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::timeout;

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

pub struct WorkerRequestResult {
    pub progress: Vec<ProgressFrame>,
    pub terminal: TerminalFrame,
    pub cancel_acknowledged: bool,
}

pub struct WorkerRequestHandle {
    inner: Arc<WorkerClientInner>,
    request_id: RequestId,
    correlation_id: CorrelationId,
    receiver: mpsc::UnboundedReceiver<PendingEvent>,
    cancel_sent: Arc<AtomicBool>,
}

#[derive(Clone)]
pub struct WorkerRequestCanceller {
    inner: Arc<WorkerClientInner>,
    request_id: RequestId,
    correlation_id: CorrelationId,
    cancel_sent: Arc<AtomicBool>,
}

impl WorkerRequestCanceller {
    pub async fn cancel(&self) -> Result<(), WorkerHostError> {
        send_cancel(
            &self.inner,
            &self.request_id,
            &self.correlation_id,
            &self.cancel_sent,
        )
        .await
    }
}

impl WorkerRequestHandle {
    pub async fn cancel(&self) -> Result<(), WorkerHostError> {
        self.canceller().cancel().await
    }

    pub fn canceller(&self) -> WorkerRequestCanceller {
        WorkerRequestCanceller {
            inner: Arc::clone(&self.inner),
            request_id: self.request_id.clone(),
            correlation_id: self.correlation_id.clone(),
            cancel_sent: Arc::clone(&self.cancel_sent),
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
        let receive = async {
            let mut progress = Vec::new();
            let mut cancel_acknowledged = false;
            loop {
                match self.receiver.recv().await {
                    Some(PendingEvent::Progress(frame)) => {
                        on_progress(&frame);
                        progress.push(frame);
                    }
                    Some(PendingEvent::CancelAck(_frame)) => cancel_acknowledged = true,
                    Some(PendingEvent::Terminal(terminal)) => {
                        return Ok(WorkerRequestResult {
                            progress,
                            terminal,
                            cancel_acknowledged,
                        });
                    }
                    Some(PendingEvent::TransportLost(message)) => {
                        return Err(WorkerHostError::TransportLost { message });
                    }
                    None => {
                        return Err(WorkerHostError::TransportLost {
                            message: "worker response channel closed".to_owned(),
                        });
                    }
                }
            }
        };
        match timeout(self.inner.request_timeout, receive).await {
            Ok(result) => result,
            Err(_) => {
                self.inner.pending.lock().await.remove(&self.request_id);
                Err(WorkerHostError::RequestTimeout {
                    request_id: self.request_id.0.clone(),
                })
            }
        }
    }
}

enum PendingEvent {
    Progress(ProgressFrame),
    CancelAck(CancelAckFrame),
    Terminal(TerminalFrame),
    TransportLost(String),
}

struct PendingRequest {
    correlation_id: CorrelationId,
    sender: mpsc::UnboundedSender<PendingEvent>,
}

type PendingRequests = Arc<Mutex<HashMap<RequestId, PendingRequest>>>;
type PendingControls = Arc<Mutex<HashMap<ControlId, oneshot::Sender<WireMessage>>>>;

struct ReaderLifecycle {
    state: Arc<AtomicU8>,
    current_generation: Arc<AtomicU64>,
    generation: u64,
    alive: Arc<AtomicBool>,
}

struct WorkerClientInner {
    child: Mutex<Child>,
    stdin: Mutex<ChildStdin>,
    pending: PendingRequests,
    controls: PendingControls,
    codec: FrameCodec,
    protocol_version: ProtocolVersion,
    incarnation_id: WorkerIncarnationId,
    request_timeout: Duration,
    shutdown_timeout: Duration,
    next_request: AtomicU64,
    next_control: AtomicU64,
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
    stderr_tail: Arc<Mutex<Vec<u8>>>,
    _reader_task: JoinHandle<()>,
    _stderr_task: JoinHandle<()>,
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

    pub async fn stderr_tail(&self) -> Vec<u8> {
        self.stderr_tail.lock().await.clone()
    }

    pub async fn health(&self) -> Result<HealthAckFrame, WorkerHostError> {
        let control_id = self.next_control_id("health");
        let response = self
            .send_control(
                control_id.clone(),
                WireMessage::Health(HealthFrame {
                    protocol_version: self.inner.protocol_version,
                    incarnation_id: self.inner.incarnation_id.clone(),
                    control_id,
                }),
                self.inner.request_timeout,
            )
            .await?;
        let WireMessage::HealthAck(frame) = response else {
            return Err(WorkerHostError::UnexpectedWorkerMessage {
                kind: response.kind(),
            });
        };
        Ok(frame)
    }

    pub async fn cleanup(
        &self,
        run_id: Option<String>,
        object_ids: Vec<String>,
    ) -> Result<CleanupAckFrame, WorkerHostError> {
        let control_id = self.next_control_id("cleanup");
        let response = self
            .send_control(
                control_id.clone(),
                WireMessage::Cleanup(CleanupFrame {
                    protocol_version: self.inner.protocol_version,
                    incarnation_id: self.inner.incarnation_id.clone(),
                    control_id,
                    run_id,
                    object_ids,
                }),
                self.inner.request_timeout,
            )
            .await?;
        let WireMessage::CleanupAck(frame) = response else {
            return Err(WorkerHostError::UnexpectedWorkerMessage {
                kind: response.kind(),
            });
        };
        Ok(frame)
    }

    fn next_control_id(&self, kind: &str) -> ControlId {
        let sequence = self.inner.next_control.fetch_add(1, Ordering::Relaxed);
        ControlId(format!("{kind}-{sequence}"))
    }

    async fn send_control(
        &self,
        control_id: ControlId,
        message: WireMessage,
        deadline: Duration,
    ) -> Result<WireMessage, WorkerHostError> {
        ensure_alive(&self.inner)?;
        let (sender, receiver) = oneshot::channel();
        self.inner
            .controls
            .lock()
            .await
            .insert(control_id.clone(), sender);
        {
            let mut stdin = self.inner.stdin.lock().await;
            if let Err(error) = write_frame(&mut *stdin, &self.inner.codec, &message).await {
                self.inner.controls.lock().await.remove(&control_id);
                return Err(error);
            }
        }
        match timeout(deadline, receiver).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => Err(WorkerHostError::TransportLost {
                message: format!(
                    "worker incarnation `{}` exited before control acknowledgement",
                    self.inner.incarnation_id.0
                ),
            }),
            Err(_) => {
                self.inner.controls.lock().await.remove(&control_id);
                Err(WorkerHostError::ControlTimeout {
                    control_id: control_id.0,
                })
            }
        }
    }

    pub async fn shutdown(&self) -> Result<(), WorkerHostError> {
        ensure_alive(&self.inner)?;
        let sequence = self.inner.next_control.fetch_add(1, Ordering::Relaxed);
        let control_id = ControlId(format!("shutdown-{sequence}"));
        let (sender, receiver) = oneshot::channel();
        self.inner
            .controls
            .lock()
            .await
            .insert(control_id.clone(), sender);
        let message = WireMessage::Shutdown(ShutdownFrame {
            protocol_version: self.inner.protocol_version,
            incarnation_id: self.inner.incarnation_id.clone(),
            control_id: control_id.clone(),
        });
        {
            let mut stdin = self.inner.stdin.lock().await;
            if let Err(error) = write_frame(&mut *stdin, &self.inner.codec, &message).await {
                self.inner.controls.lock().await.remove(&control_id);
                return Err(error);
            }
        }
        match timeout(self.inner.shutdown_timeout, receiver).await {
            Ok(Ok(WireMessage::ShutdownAck(_))) => {}
            Ok(Ok(message)) => {
                return Err(WorkerHostError::UnexpectedWorkerMessage {
                    kind: message.kind(),
                });
            }
            Ok(Err(_)) => {
                return Err(WorkerHostError::TransportLost {
                    message: "worker exited before shutdown acknowledgement".to_owned(),
                });
            }
            Err(_) => {
                let mut child = self.inner.child.lock().await;
                let _ = child.start_kill();
                let _ = child.wait().await;
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
        let mut child = self.inner.child.lock().await;
        if timeout(self.inner.shutdown_timeout, child.wait())
            .await
            .is_err()
        {
            let _ = child.start_kill();
            let _ = child.wait().await;
            self.inner.alive.store(false, Ordering::Release);
            store_state_if_current(
                &self.inner.state,
                &self.inner.current_generation,
                self.inner.generation,
                STATE_STOPPED,
            );
            return Err(WorkerHostError::ShutdownTimeout);
        }
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
        let sequence = self.inner.next_request.fetch_add(1, Ordering::Relaxed);
        let request_id = RequestId(format!("request-{sequence}"));
        let correlation_id = CorrelationId(format!("worker-{sequence}"));
        let (sender, receiver) = mpsc::unbounded_channel();
        self.inner.pending.lock().await.insert(
            request_id.clone(),
            PendingRequest {
                correlation_id: correlation_id.clone(),
                sender,
            },
        );
        let message = WireMessage::Request(RequestFrame {
            protocol_version: self.inner.protocol_version,
            incarnation_id: self.inner.incarnation_id.clone(),
            request_id: request_id.clone(),
            correlation_id,
            operation: operation.into(),
            payload,
        });
        let write_result = {
            let mut stdin = self.inner.stdin.lock().await;
            write_frame(&mut *stdin, &self.inner.codec, &message).await
        };
        if let Err(error) = write_result {
            self.inner.pending.lock().await.remove(&request_id);
            return Err(error);
        }
        Ok(WorkerRequestHandle {
            inner: Arc::clone(&self.inner),
            request_id,
            correlation_id: CorrelationId(format!("worker-{sequence}")),
            receiver,
            cancel_sent: Arc::new(AtomicBool::new(false)),
        })
    }
}

async fn send_cancel(
    inner: &Arc<WorkerClientInner>,
    request_id: &RequestId,
    correlation_id: &CorrelationId,
    cancel_sent: &AtomicBool,
) -> Result<(), WorkerHostError> {
    ensure_alive(inner)?;
    if cancel_sent.swap(true, Ordering::AcqRel) {
        return Ok(());
    }
    let message = WireMessage::Cancel(CancelFrame {
        protocol_version: inner.protocol_version,
        incarnation_id: inner.incarnation_id.clone(),
        request_id: request_id.clone(),
        correlation_id: correlation_id.clone(),
    });
    let mut stdin = inner.stdin.lock().await;
    let result = write_frame(&mut *stdin, &inner.codec, &message).await;
    if result.is_err() {
        cancel_sent.store(false, Ordering::Release);
    }
    result
}

pub struct WorkerSupervisor {
    launch: WorkerLaunchSpec,
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
        Self {
            launch,
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
        let mut command = Command::new(&self.launch.executable);
        command
            .env_clear()
            .envs(self.launch.environment.iter().cloned())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = command.spawn().map_err(|error| WorkerHostError::Spawn {
            path: self.launch.executable.clone(),
            message: error.to_string(),
        })?;
        let mut stdin = child.stdin.take().ok_or_else(|| WorkerHostError::Io {
            operation: "stdin setup",
            message: "child stdin was not piped".to_owned(),
        })?;
        let mut stdout = child.stdout.take().ok_or_else(|| WorkerHostError::Io {
            operation: "stdout setup",
            message: "child stdout was not piped".to_owned(),
        })?;
        let stderr = child.stderr.take().ok_or_else(|| WorkerHostError::Io {
            operation: "stderr setup",
            message: "child stderr was not piped".to_owned(),
        })?;
        let stderr_tail = Arc::new(Mutex::new(Vec::new()));
        let stderr_task = spawn_stderr_drain(
            stderr,
            Arc::clone(&stderr_tail),
            self.launch.limits.maximum_stderr_bytes,
        );
        let codec = FrameCodec::new(self.launch.limits.maximum_frame_bytes);
        let handshake = async {
            write_frame(
                &mut stdin,
                &codec,
                &WireMessage::HostHello(HostHello {
                    supported_protocols: self.launch.supported_protocols,
                }),
            )
            .await?;
            let message = read_frame(&mut stdout, &codec).await?;
            validate_message_direction(&message, MessageSender::Worker).map_err(|_| {
                WorkerHostError::UnexpectedStartupMessage {
                    kind: message.kind(),
                }
            })?;
            let WireMessage::WorkerHello(hello) = message else {
                return Err(WorkerHostError::UnexpectedStartupMessage {
                    kind: message.kind(),
                });
            };
            validate_hello(&hello, &self.launch.expected)?;
            if hello.selected_protocol < self.launch.supported_protocols.minimum
                || hello.selected_protocol > self.launch.supported_protocols.maximum
            {
                return Err(WorkerHostError::IdentityMismatch {
                    field: "selected_protocol",
                    expected: format!(
                        "{}..={}",
                        self.launch.supported_protocols.minimum.0,
                        self.launch.supported_protocols.maximum.0
                    ),
                    actual: hello.selected_protocol.0.to_string(),
                });
            }
            Ok(hello)
        };
        let hello = timeout(self.launch.limits.startup_timeout, handshake)
            .await
            .map_err(|_| WorkerHostError::StartupTimeout)??;
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let controls = Arc::new(Mutex::new(HashMap::new()));
        let alive = Arc::new(AtomicBool::new(true));
        let inner = Arc::new(WorkerClientInner {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
            pending: Arc::clone(&pending),
            controls: Arc::clone(&controls),
            codec: FrameCodec::new(self.launch.limits.maximum_frame_bytes),
            protocol_version: hello.selected_protocol,
            incarnation_id: hello.identity.incarnation_id.clone(),
            request_timeout: self.launch.limits.request_timeout,
            shutdown_timeout: self.launch.limits.shutdown_timeout,
            next_request: AtomicU64::new(1),
            next_control: AtomicU64::new(1),
            state: Arc::clone(&self.state),
            current_generation: Arc::clone(&self.current_generation),
            generation,
            alive: Arc::clone(&alive),
        });
        start_guard.commit();
        let reader_task = tokio::spawn(reader_loop(
            stdout,
            FrameCodec::new(self.launch.limits.maximum_frame_bytes),
            pending,
            controls,
            hello.selected_protocol,
            hello.identity.incarnation_id.clone(),
            ReaderLifecycle {
                state: Arc::clone(&self.state),
                current_generation: Arc::clone(&self.current_generation),
                generation,
                alive,
            },
        ));
        Ok(StartedWorker {
            hello,
            inner,
            stderr_tail,
            _reader_task: reader_task,
            _stderr_task: stderr_task,
        })
    }
}

async fn reader_loop(
    mut stdout: tokio::process::ChildStdout,
    codec: FrameCodec,
    pending: PendingRequests,
    controls: PendingControls,
    protocol_version: ProtocolVersion,
    incarnation_id: WorkerIncarnationId,
    lifecycle: ReaderLifecycle,
) {
    loop {
        let message = match read_frame(&mut stdout, &codec).await {
            Ok(message) => message,
            Err(error) => {
                fail_reader(&lifecycle, &pending, &controls, error.to_string()).await;
                return;
            }
        };
        if validate_message_direction(&message, MessageSender::Worker).is_err() {
            fail_reader(
                &lifecycle,
                &pending,
                &controls,
                format!("worker sent host-only message `{}`", message.kind()),
            )
            .await;
            return;
        }
        match message {
            WireMessage::Progress(frame) => {
                if !session_matches(
                    frame.protocol_version,
                    &frame.incarnation_id,
                    protocol_version,
                    &incarnation_id,
                ) {
                    fail_reader(
                        &lifecycle,
                        &pending,
                        &controls,
                        "worker progress used a stale session".to_owned(),
                    )
                    .await;
                    return;
                }
                let pending = pending.lock().await;
                if let Some(request) = pending.get(&frame.request_id)
                    && request.correlation_id == frame.correlation_id
                {
                    let _ = request.sender.send(PendingEvent::Progress(frame));
                }
            }
            WireMessage::Terminal(frame) => {
                if !session_matches(
                    frame.protocol_version,
                    &frame.incarnation_id,
                    protocol_version,
                    &incarnation_id,
                ) {
                    fail_reader(
                        &lifecycle,
                        &pending,
                        &controls,
                        "worker terminal used a stale session".to_owned(),
                    )
                    .await;
                    return;
                }
                let correlation_matches = pending
                    .lock()
                    .await
                    .get(&frame.request_id)
                    .is_none_or(|request| request.correlation_id == frame.correlation_id);
                if !correlation_matches {
                    fail_reader(
                        &lifecycle,
                        &pending,
                        &controls,
                        "worker terminal correlation mismatch".to_owned(),
                    )
                    .await;
                    return;
                }
                if let Some(request) = pending.lock().await.remove(&frame.request_id) {
                    let _ = request.sender.send(PendingEvent::Terminal(frame));
                }
            }
            WireMessage::Ping { .. } => {}
            WireMessage::HealthAck(frame) => {
                if !session_matches(
                    frame.protocol_version,
                    &frame.incarnation_id,
                    protocol_version,
                    &incarnation_id,
                ) {
                    fail_reader(
                        &lifecycle,
                        &pending,
                        &controls,
                        "worker health ack used a stale session".to_owned(),
                    )
                    .await;
                    return;
                }
                if let Some(sender) = controls.lock().await.remove(&frame.control_id) {
                    let _ = sender.send(WireMessage::HealthAck(frame));
                }
            }
            WireMessage::CleanupAck(frame) => {
                if !session_matches(
                    frame.protocol_version,
                    &frame.incarnation_id,
                    protocol_version,
                    &incarnation_id,
                ) {
                    fail_reader(
                        &lifecycle,
                        &pending,
                        &controls,
                        "worker cleanup ack used a stale session".to_owned(),
                    )
                    .await;
                    return;
                }
                if let Some(sender) = controls.lock().await.remove(&frame.control_id) {
                    let _ = sender.send(WireMessage::CleanupAck(frame));
                }
            }
            WireMessage::ShutdownAck(frame) => {
                if !session_matches(
                    frame.protocol_version,
                    &frame.incarnation_id,
                    protocol_version,
                    &incarnation_id,
                ) {
                    fail_reader(
                        &lifecycle,
                        &pending,
                        &controls,
                        "worker shutdown ack used a stale session".to_owned(),
                    )
                    .await;
                    return;
                }
                if let Some(sender) = controls.lock().await.remove(&frame.control_id) {
                    let _ = sender.send(WireMessage::ShutdownAck(frame));
                }
            }
            WireMessage::CancelAck(frame) => {
                if !session_matches(
                    frame.protocol_version,
                    &frame.incarnation_id,
                    protocol_version,
                    &incarnation_id,
                ) {
                    fail_reader(
                        &lifecycle,
                        &pending,
                        &controls,
                        "worker cancel ack used a stale session".to_owned(),
                    )
                    .await;
                    return;
                }
                let pending = pending.lock().await;
                if let Some(request) = pending.get(&frame.request_id)
                    && request.correlation_id == frame.correlation_id
                {
                    let _ = request.sender.send(PendingEvent::CancelAck(frame));
                }
            }
            message => {
                fail_reader(
                    &lifecycle,
                    &pending,
                    &controls,
                    format!("unexpected worker message `{}`", message.kind()),
                )
                .await;
                return;
            }
        }
    }
}

async fn fail_reader(
    lifecycle: &ReaderLifecycle,
    pending: &PendingRequests,
    controls: &PendingControls,
    message: String,
) {
    lifecycle.alive.store(false, Ordering::Release);
    fail_pending(pending, message).await;
    controls.lock().await.clear();
    store_state_if_current(
        &lifecycle.state,
        &lifecycle.current_generation,
        lifecycle.generation,
        STATE_FAILED,
    );
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

async fn fail_pending(pending: &PendingRequests, message: String) {
    let pending_requests = std::mem::take(&mut *pending.lock().await);
    for (_, request) in pending_requests {
        let _ = request
            .sender
            .send(PendingEvent::TransportLost(message.clone()));
    }
}

fn session_matches(
    actual_protocol: ProtocolVersion,
    actual_incarnation: &WorkerIncarnationId,
    expected_protocol: ProtocolVersion,
    expected_incarnation: &WorkerIncarnationId,
) -> bool {
    actual_protocol == expected_protocol && actual_incarnation == expected_incarnation
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

async fn write_frame(
    writer: &mut (impl AsyncWrite + Unpin),
    codec: &FrameCodec,
    message: &WireMessage,
) -> Result<(), WorkerHostError> {
    let payload = codec.encode_payload(message)?;
    let declared = u32::try_from(payload.len()).map_err(|_| {
        WorkerHostError::Protocol(CodecError::PayloadLengthOverflow {
            actual: payload.len(),
        })
    })?;
    writer
        .write_all(&declared.to_be_bytes())
        .await
        .map_err(|error| WorkerHostError::Io {
            operation: "write",
            message: error.to_string(),
        })?;
    writer
        .write_all(&payload)
        .await
        .map_err(|error| WorkerHostError::Io {
            operation: "write",
            message: error.to_string(),
        })?;
    writer.flush().await.map_err(|error| WorkerHostError::Io {
        operation: "flush",
        message: error.to_string(),
    })
}

async fn read_frame(
    reader: &mut (impl AsyncRead + Unpin),
    codec: &FrameCodec,
) -> Result<WireMessage, WorkerHostError> {
    let mut prefix = [0_u8; 4];
    read_exact_stage(reader, &mut prefix, "read prefix").await?;
    let declared = u32::from_be_bytes(prefix);
    if declared > codec.maximum_frame_bytes() {
        return Err(WorkerHostError::Protocol(CodecError::FrameTooLarge {
            declared,
            maximum: codec.maximum_frame_bytes(),
        }));
    }
    let mut payload = vec![0_u8; declared as usize];
    read_exact_stage(reader, &mut payload, "read payload").await?;
    codec
        .decode_payload(&payload)
        .map_err(WorkerHostError::from)
}

async fn read_exact_stage(
    reader: &mut (impl AsyncRead + Unpin),
    buffer: &mut [u8],
    operation: &'static str,
) -> Result<(), WorkerHostError> {
    let mut received = 0;
    while received < buffer.len() {
        match reader.read(&mut buffer[received..]).await {
            Ok(0) if received == 0 => return Err(WorkerHostError::CleanEof { operation }),
            Ok(0) => {
                return Err(WorkerHostError::IncompleteFrame {
                    operation,
                    received,
                    expected: buffer.len(),
                });
            }
            Ok(count) => received += count,
            Err(error) => {
                return Err(WorkerHostError::Io {
                    operation,
                    message: error.to_string(),
                });
            }
        }
    }
    Ok(())
}

fn spawn_stderr_drain(
    mut stderr: tokio::process::ChildStderr,
    tail: Arc<Mutex<Vec<u8>>>,
    maximum_bytes: usize,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut buffer = [0_u8; 4096];
        while let Ok(count) = stderr.read(&mut buffer).await {
            if count == 0 {
                break;
            }
            let mut tail = tail.lock().await;
            tail.extend_from_slice(&buffer[..count]);
            if tail.len() > maximum_bytes {
                let excess = tail.len() - maximum_bytes;
                tail.drain(..excess);
            }
        }
    })
}
