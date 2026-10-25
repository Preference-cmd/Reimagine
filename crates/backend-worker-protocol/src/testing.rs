//! Test-only fake worker service speaking the v2 contract.
//!
//! Used by worker fixture binaries and integration tests that need a
//! real gRPC worker without a real backend. Not part of the public
//! contract; guarded by `doc(hidden)`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use futures_core::Stream;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::proto::worker_server::Worker;
use crate::proto::{
    self, CleanupRequest, CleanupResponse, ExecuteEvent, ExecuteRequest, HealthRequest,
    HealthResponse, HelloRequest, HelloResponse, ShutdownRequest, ShutdownResponse,
};
use crate::{BackendExecutionError, TerminalOutcome, WorkerIdentity, WorkerProfile};

/// Behaviour knobs for [`FakeWorkerService`].
#[derive(Clone, Debug, Default)]
pub struct FakeWorkerConfig {
    pub identity: Option<WorkerIdentity>,
    pub profile: Option<WorkerProfile>,
    /// Bearer token required on every RPC; `None` disables the check.
    pub token: Option<String>,
    /// Protocol version reported by `Hello` (use 1 to simulate an old
    /// worker the host must reject).
    pub hello_protocol: u32,
    /// Shutdown handler sleeps past any host deadline instead of
    /// acknowledging.
    pub ignore_shutdown: bool,
    /// Capabilities reported by the `echo`-style operations.
    pub capabilities_echo: bool,
}

/// Response stream that cancels the running operation when the client
/// drops it (the gRPC cancellation path).
pub struct CancelOnDropStream {
    inner: ReceiverStream<Result<ExecuteEvent, tonic::Status>>,
    cancelled: Arc<AtomicBool>,
}

impl Stream for CancelOnDropStream {
    type Item = Result<ExecuteEvent, tonic::Status>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl Drop for CancelOnDropStream {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

/// A worker service whose operations are `echo`, `delay`, `progress`,
/// `latent.create_empty`, `resource.apply_hints`,
/// `model.set_cache_budget`, and `crash` (exits the process), mirroring
/// the scenarios the supervisor and adapter tests cover. Behaviour
/// knobs arrive through `FAKE_*` environment variables, like the
/// original frame-protocol fixture.
pub struct FakeWorkerService {
    config: FakeWorkerConfig,
    call_counter: AtomicU64,
}

impl FakeWorkerService {
    #[must_use]
    pub fn new(config: FakeWorkerConfig) -> Self {
        Self {
            config,
            call_counter: AtomicU64::new(1),
        }
    }
}

fn execute_event(event: proto::execute_event::Event) -> ExecuteEvent {
    ExecuteEvent { event: Some(event) }
}

fn progress_event(sequence: u64) -> ExecuteEvent {
    execute_event(proto::execute_event::Event::Progress(proto::Progress {
        sequence,
        completed: sequence,
        total: None,
        message: None,
    }))
}

fn unsupported(operation: &str) -> TerminalOutcome {
    TerminalOutcome::BackendError {
        error: BackendExecutionError {
            code: "unsupported_operation".to_owned(),
            message: format!("unsupported fake operation `{operation}`"),
            retryable: false,
        },
    }
}

#[tonic::async_trait]
impl Worker for FakeWorkerService {
    async fn hello(
        &self,
        request: tonic::Request<HelloRequest>,
    ) -> Result<tonic::Response<HelloResponse>, tonic::Status> {
        crate::server::require_bearer(&request, self.config.token.as_deref())?;
        let identity = self.config.identity.clone().unwrap_or(WorkerIdentity {
            backend_instance_id: "fake:cpu:default".into(),
            installation_id: "fake-installation".into(),
            incarnation_id: "fake-incarnation".into(),
            worker_version: "test".to_owned(),
            backend_kind: "fake".to_owned(),
            target: std::env::consts::ARCH.to_owned(),
            manifest_digest: "test-manifest".to_owned(),
        });
        let profile = self.config.profile.clone().unwrap_or(WorkerProfile {
            instances: vec![crate::WorkerInstanceProfile {
                backend_instance_id: identity.backend_instance_id.clone(),
                device_label: "cpu".to_owned(),
                capabilities: vec!["echo".to_owned(), "delay".to_owned()],
                operation_options: serde_json::json!({}),
            }],
        });
        Ok(tonic::Response::new(HelloResponse {
            protocol_version: self.config.hello_protocol.max(1),
            identity: Some(proto::WorkerIdentity {
                backend_instance_id: identity.backend_instance_id.0,
                installation_id: identity.installation_id.0,
                incarnation_id: identity.incarnation_id.0,
                worker_version: identity.worker_version,
                backend_kind: identity.backend_kind,
                target: identity.target,
                manifest_digest: identity.manifest_digest,
            }),
            profile: Some(crate::conversion::profile_to_proto(&profile)),
        }))
    }

    type ExecuteStream = CancelOnDropStream;

    async fn execute(
        &self,
        request: tonic::Request<ExecuteRequest>,
    ) -> Result<tonic::Response<Self::ExecuteStream>, tonic::Status> {
        crate::server::require_bearer(&request, self.config.token.as_deref())?;
        let request = request.into_inner();
        let operation = request.operation;
        let payload: serde_json::Value =
            serde_json::from_slice(&request.payload).unwrap_or(serde_json::json!({}));

        let (tx, rx) = mpsc::channel::<Result<ExecuteEvent, tonic::Status>>(16);
        let cancelled = Arc::new(AtomicBool::new(false));

        let thread_cancelled = Arc::clone(&cancelled);
        let progress_tx = tx.clone();
        let call = self.call_counter.fetch_add(1, Ordering::Relaxed);
        std::thread::spawn(move || {
            if operation == "crash" {
                std::process::exit(17);
            }
            let outcome = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                match operation.as_str() {
                    "echo" => TerminalOutcome::Success { output: payload },
                    "delay" => {
                        let milliseconds = payload
                            .get("milliseconds")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(25);
                        let steps = (milliseconds / 5).max(1);
                        for _ in 0..steps {
                            std::thread::sleep(Duration::from_millis(5));
                            if thread_cancelled.load(Ordering::Acquire) {
                                break;
                            }
                        }
                        if thread_cancelled.load(Ordering::Acquire) {
                            TerminalOutcome::Cancelled
                        } else {
                            TerminalOutcome::Success {
                                output: serde_json::json!({ "slept": milliseconds }),
                            }
                        }
                    }
                    "progress" => {
                        for sequence in 1..=3_u64 {
                            let _ = progress_tx.blocking_send(Ok(progress_event(sequence)));
                        }
                        TerminalOutcome::Success {
                            output: serde_json::json!({ "done": true }),
                        }
                    }
                    "latent.create_empty" => {
                        if std::env::var_os("FAKE_LATENT_PROGRESS").is_some() {
                            for sequence in 1..=3_u64 {
                                let _ = progress_tx.blocking_send(Ok(progress_event(sequence)));
                            }
                        }
                        let milliseconds = std::env::var("FAKE_LATENT_DELAY_MS")
                            .ok()
                            .and_then(|value| value.parse::<u64>().ok())
                            .unwrap_or(0);
                        let forced_terminal = std::env::var("FAKE_LATENT_TERMINAL").ok();
                        for _ in 0..(milliseconds / 5) {
                            std::thread::sleep(Duration::from_millis(5));
                            if forced_terminal.is_none() && thread_cancelled.load(Ordering::Acquire)
                            {
                                break;
                            }
                        }
                        if forced_terminal.as_deref() == Some("crash") {
                            std::process::exit(17);
                        } else if forced_terminal.as_deref() == Some("backend_error") {
                            TerminalOutcome::BackendError {
                                error: BackendExecutionError {
                                    code: "forced_backend_error".to_owned(),
                                    message: "forced backend terminal won the race".to_owned(),
                                    retryable: false,
                                },
                            }
                        } else if thread_cancelled.load(Ordering::Acquire) {
                            TerminalOutcome::Cancelled
                        } else {
                            TerminalOutcome::Success {
                                output: serde_json::json!({
                                    "worker_token": format!("latent:fake-{call}"),
                                    "width": payload["width"],
                                    "height": payload["height"],
                                    "batch_size": payload["batch_size"]
                                }),
                            }
                        }
                    }
                    "resource.apply_hints" => {
                        if std::env::var_os("FAKE_HINTS_FAIL").is_some() {
                            TerminalOutcome::BackendError {
                                error: BackendExecutionError {
                                    code: "forced_hints_error".to_owned(),
                                    message: "forced hints backend error".to_owned(),
                                    retryable: false,
                                },
                            }
                        } else {
                            TerminalOutcome::Success {
                                output: serde_json::json!({ "applied": true }),
                            }
                        }
                    }
                    "model.set_cache_budget" => {
                        if std::env::var_os("FAKE_BUDGET_FAIL").is_some() {
                            TerminalOutcome::BackendError {
                                error: BackendExecutionError {
                                    code: "forced_budget_error".to_owned(),
                                    message: "forced budget backend error".to_owned(),
                                    retryable: false,
                                },
                            }
                        } else {
                            TerminalOutcome::Success {
                                output: serde_json::json!({ "applied": true }),
                            }
                        }
                    }
                    other => unsupported(other),
                }
            })) {
                Ok(outcome) => outcome,
                Err(panic) => TerminalOutcome::BackendError {
                    error: BackendExecutionError {
                        code: "fake_worker_panic".to_owned(),
                        message: panic
                            .downcast_ref::<&str>()
                            .map(|message| (*message).to_owned())
                            .or_else(|| panic.downcast_ref::<String>().cloned())
                            .unwrap_or_else(|| "fake worker operation panicked".to_owned()),
                        retryable: false,
                    },
                },
            };
            let _ = progress_tx.blocking_send(Ok(execute_event(
                proto::execute_event::Event::Result(crate::conversion::outcome_to_proto(&outcome)),
            )));
        });

        Ok(tonic::Response::new(CancelOnDropStream {
            inner: ReceiverStream::new(rx),
            cancelled,
        }))
    }

    async fn health(
        &self,
        request: tonic::Request<HealthRequest>,
    ) -> Result<tonic::Response<HealthResponse>, tonic::Status> {
        crate::server::require_bearer(&request, self.config.token.as_deref())?;
        Ok(tonic::Response::new(HealthResponse {
            healthy: true,
            message: None,
        }))
    }

    async fn cleanup(
        &self,
        request: tonic::Request<CleanupRequest>,
    ) -> Result<tonic::Response<CleanupResponse>, tonic::Status> {
        crate::server::require_bearer(&request, self.config.token.as_deref())?;
        let released = request.into_inner().object_ids.len() as u64;
        Ok(tonic::Response::new(CleanupResponse {
            released_objects: released,
        }))
    }

    async fn shutdown(
        &self,
        request: tonic::Request<ShutdownRequest>,
    ) -> Result<tonic::Response<ShutdownResponse>, tonic::Status> {
        crate::server::require_bearer(&request, self.config.token.as_deref())?;
        if self.config.ignore_shutdown {
            // Outlast any host shutdown deadline so callers exercise
            // their kill path.
            std::thread::sleep(Duration::from_secs(60));
        }
        Ok(tonic::Response::new(ShutdownResponse {}))
    }
}
