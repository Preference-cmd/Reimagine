//! Host implementation running a mod in a child process over JSON-RPC 2.0 stdio.

use std::collections::HashMap;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use reimagine_core::diagnostic::Diagnostic;
use reimagine_core::task::CancelToken;
use serde_json::Value;
use tokio::io::AsyncBufReadExt;
use tokio::sync::{Mutex, mpsc, oneshot};

use crate::context::ModContext;
use crate::diagnostic::internal_error;
use crate::interface::Mod;
use crate::manifest::ModManifest;
use crate::protocol::{
    CancelParams, IncomingRpcMsg, InitializeParams, InitializeResult, InvokeParams,
    NotificationMsg, PROTOCOL_VERSION, PublishParams, RequestMsg,
};

type ResponseSender = oneshot::Sender<Result<Value, Diagnostic>>;
type PendingRequests = Arc<Mutex<HashMap<i64, ResponseSender>>>;

/// Configuration for launching a process-hosted mod.
#[derive(Debug, Clone)]
pub struct ProcessModSpec {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    /// Bounds the initialize handshake and the `mod/start` round-trip.
    pub startup_timeout: Duration,
    /// Bounds the `mod/stop` reply and the wait for the child to exit.
    pub shutdown_timeout: Duration,
}

impl ProcessModSpec {
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            startup_timeout: Duration::from_secs(5),
            shutdown_timeout: Duration::from_secs(3),
        }
    }
}

enum WriterCommand {
    Line(String),
    Close,
}

/// A mod running in an external child process communicating via JSON-RPC 2.0 stdio.
pub struct ProcessMod {
    manifest: ModManifest,
    writer_tx: mpsc::Sender<WriterCommand>,
    pending: PendingRequests,
    next_id: AtomicI64,
    ctx: Arc<RwLock<Option<ModContext>>>,
    child: Arc<Mutex<Option<tokio::process::Child>>>,
    stderr_tail: Arc<Mutex<Vec<u8>>>,
    exited: Arc<AtomicBool>,
    startup_timeout: Duration,
    shutdown_timeout: Duration,
}

impl std::fmt::Debug for ProcessMod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProcessMod")
            .field("manifest", &self.manifest)
            .finish()
    }
}

impl ProcessMod {
    /// Spawn a child process according to `spec` and perform `initialize`.
    pub async fn spawn(spec: ProcessModSpec) -> Result<ProcessMod, Diagnostic> {
        let initial_mod_id = spec
            .program
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("mod")
            .to_owned();

        let mut cmd = tokio::process::Command::new(&spec.program);
        cmd.args(&spec.args)
            .env_clear()
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return Err(internal_error(
                    &initial_mod_id,
                    "SPAWN_FAILED",
                    format!("failed to spawn `{}`: {e}", spec.program.display()),
                ));
            }
        };

        let stdin = child.stdin.take().ok_or_else(|| {
            internal_error(&initial_mod_id, "SPAWN_FAILED", "child stdin not piped")
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            internal_error(&initial_mod_id, "SPAWN_FAILED", "child stdout not piped")
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            internal_error(&initial_mod_id, "SPAWN_FAILED", "child stderr not piped")
        })?;

        let stderr_tail = Arc::new(Mutex::new(Vec::new()));
        let _stderr_drain = spawn_stderr_drain(stderr, Arc::clone(&stderr_tail), 64 * 1024);

        let (writer_tx, mut writer_rx) = mpsc::channel::<WriterCommand>(64);
        let mut stdin_writer = stdin;
        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            while let Some(cmd) = writer_rx.recv().await {
                match cmd {
                    WriterCommand::Line(line) => {
                        if stdin_writer.write_all(line.as_bytes()).await.is_err() {
                            break;
                        }
                        if stdin_writer.write_all(b"\n").await.is_err() {
                            break;
                        }
                        if stdin_writer.flush().await.is_err() {
                            break;
                        }
                    }
                    WriterCommand::Close => {
                        break;
                    }
                }
            }
        });

        let pending: PendingRequests = Arc::new(Mutex::new(HashMap::new()));
        let ctx: Arc<RwLock<Option<ModContext>>> = Arc::new(RwLock::new(None));
        let exited = Arc::new(AtomicBool::new(false));
        let mod_id_slot = Arc::new(RwLock::new(initial_mod_id.clone()));

        let reader_pending = Arc::clone(&pending);
        let reader_ctx = Arc::clone(&ctx);
        let reader_exited = Arc::clone(&exited);
        let reader_stderr = Arc::clone(&stderr_tail);
        let reader_mod_id = Arc::clone(&mod_id_slot);

        tokio::spawn(async move {
            let mut reader = tokio::io::BufReader::new(stdout).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let raw: Result<IncomingRpcMsg, _> = serde_json::from_str(trimmed);
                match raw {
                    Ok(msg) => {
                        if msg.id.is_none() || msg.id == Some(Value::Null) {
                            if msg.method.as_deref() == Some("kernel/publish")
                                && let Some(Ok(pub_params)) =
                                    msg.params.map(serde_json::from_value::<PublishParams>)
                            {
                                let cur_ctx: Option<ModContext> =
                                    reader_ctx.read().unwrap().clone();
                                if let Some(c) = cur_ctx
                                    && let Err(e) =
                                        c.publish(pub_params.event_type, pub_params.data)
                                {
                                    tracing::warn!("failed to publish mod event: {e:?}");
                                }
                            }
                            continue;
                        }

                        if let Some(req_id) = msg.id.and_then(|id_val| id_val.as_i64())
                            && let Some(sender) = reader_pending.lock().await.remove(&req_id)
                        {
                            let cur_id = reader_mod_id.read().unwrap().clone();
                            let result = if let Some(error) = msg.error {
                                let diag = if let Some(data) = error.data {
                                    serde_json::from_value::<Diagnostic>(data).unwrap_or_else(
                                        |_| internal_error(&cur_id, "PROTOCOL", error.message),
                                    )
                                } else {
                                    internal_error(&cur_id, "PROTOCOL", error.message)
                                };
                                Err(diag)
                            } else if let Some(res) = msg.result {
                                Ok(res)
                            } else {
                                Ok(Value::Null)
                            };
                            let _ = sender.send(result);
                        }
                    }
                    Err(e) => {
                        tracing::warn!("unparseable line from mod stdout: {e}");
                    }
                }
            }

            reader_exited.store(true, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(50)).await;
            let cur_id = reader_mod_id.read().unwrap().clone();
            let diag = exited_diagnostic(&cur_id, &reader_stderr).await;
            let mut map = reader_pending.lock().await;
            for (_, sender) in map.drain() {
                let _ = sender.send(Err(diag.clone()));
            }
        });

        // Perform initialize request
        let init_req_id = 1;
        let (init_tx, init_rx) = oneshot::channel();
        pending.lock().await.insert(init_req_id, init_tx);

        let init_msg = serde_json::to_string(&RequestMsg {
            jsonrpc: "2.0",
            id: init_req_id,
            method: "initialize",
            params: InitializeParams {
                protocol_version: PROTOCOL_VERSION.to_owned(),
            },
        })
        .expect("serialize initialize request");

        if writer_tx.send(WriterCommand::Line(init_msg)).await.is_err() {
            return Err(exited_diagnostic(&initial_mod_id, &stderr_tail).await);
        }

        let init_res = tokio::time::timeout(spec.startup_timeout, init_rx).await;
        let res_val = match init_res {
            Ok(Ok(Ok(val))) => val,
            Ok(Ok(Err(diag))) => {
                return Err(with_stderr(diag, &stderr_tail).await);
            }
            Ok(Err(_)) => {
                return Err(exited_diagnostic(&initial_mod_id, &stderr_tail).await);
            }
            Err(_) => {
                let diag = internal_error(
                    &initial_mod_id,
                    "TIMEOUT",
                    format!("startup timeout after {:?}", spec.startup_timeout),
                );
                return Err(with_stderr(diag, &stderr_tail).await);
            }
        };

        let init_result: InitializeResult = match serde_json::from_value(res_val) {
            Ok(r) => r,
            Err(e) => {
                let diag = internal_error(
                    &initial_mod_id,
                    "PROTOCOL",
                    format!(
                        "invalid initialize result from `{}`: {e}",
                        spec.program.display()
                    ),
                );
                return Err(with_stderr(diag, &stderr_tail).await);
            }
        };

        if init_result.protocol_version != PROTOCOL_VERSION {
            let diag = internal_error(
                &initial_mod_id,
                "PROTOCOL_MISMATCH",
                format!(
                    "protocol version mismatch: mod reported `{}`, expected `{PROTOCOL_VERSION}`",
                    init_result.protocol_version
                ),
            );
            return Err(with_stderr(diag, &stderr_tail).await);
        }

        *mod_id_slot.write().unwrap() = init_result.manifest.id.clone();

        Ok(ProcessMod {
            manifest: init_result.manifest,
            writer_tx,
            pending,
            next_id: AtomicI64::new(2),
            ctx,
            child: Arc::new(Mutex::new(Some(child))),
            stderr_tail,
            exited,
            startup_timeout: spec.startup_timeout,
            shutdown_timeout: spec.shutdown_timeout,
        })
    }

    /// Close the pipe, kill the child and reap it; used when a started process
    /// turns out to be unusable.
    async fn terminate(&self) {
        *self.ctx.write().unwrap() = None;
        let _ = self.writer_tx.try_send(WriterCommand::Close);
        let mut guard = self.child.lock().await;
        if let Some(mut child) = guard.take() {
            let _ = child.start_kill();
            let _ = child.wait().await;
        }
    }
}

#[async_trait]
impl Mod for ProcessMod {
    fn manifest(&self) -> &ModManifest {
        &self.manifest
    }

    async fn start(&self, ctx: ModContext) -> Result<(), Diagnostic> {
        *self.ctx.write().unwrap() = Some(ctx);
        if self.exited.load(Ordering::SeqCst) {
            return Err(exited_diagnostic(&self.manifest.id, &self.stderr_tail).await);
        }

        let req_id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(req_id, tx);

        let msg = serde_json::to_string(&RequestMsg {
            jsonrpc: "2.0",
            id: req_id,
            method: "mod/start",
            params: serde_json::json!({}),
        })
        .expect("serialize mod/start request");

        if self.writer_tx.send(WriterCommand::Line(msg)).await.is_err() {
            self.pending.lock().await.remove(&req_id);
            return Err(exited_diagnostic(&self.manifest.id, &self.stderr_tail).await);
        }

        let result = tokio::time::timeout(self.startup_timeout, rx).await;
        let error = match result {
            Ok(Ok(Ok(_))) => return Ok(()),
            Ok(Ok(Err(diagnostic))) => with_stderr(diagnostic, &self.stderr_tail).await,
            Ok(Err(_)) => exited_diagnostic(&self.manifest.id, &self.stderr_tail).await,
            Err(_) => {
                with_stderr(
                    internal_error(
                        &self.manifest.id,
                        "TIMEOUT",
                        format!(
                            "mod `{}` did not answer mod/start within {:?}",
                            self.manifest.id, self.startup_timeout
                        ),
                    ),
                    &self.stderr_tail,
                )
                .await
            }
        };

        // A mod that cannot start must not keep a live process: it would
        // otherwise publish events and hold resources while unavailable.
        self.pending.lock().await.remove(&req_id);
        self.terminate().await;
        Err(error)
    }

    async fn invoke(
        &self,
        operation: &str,
        payload: Value,
        cancel: CancelToken,
    ) -> Result<Value, Diagnostic> {
        if self.exited.load(Ordering::SeqCst) {
            return Err(exited_diagnostic(&self.manifest.id, &self.stderr_tail).await);
        }

        let req_id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(req_id, tx);

        let msg = serde_json::to_string(&RequestMsg {
            jsonrpc: "2.0",
            id: req_id,
            method: "mod/invoke",
            params: InvokeParams {
                operation: operation.to_owned(),
                payload,
            },
        })
        .expect("serialize mod/invoke request");

        if self.writer_tx.send(WriterCommand::Line(msg)).await.is_err() {
            self.pending.lock().await.remove(&req_id);
            return Err(exited_diagnostic(&self.manifest.id, &self.stderr_tail).await);
        }

        tokio::pin!(rx);

        tokio::select! {
            res = &mut rx => {
                match res {
                    Ok(val) => val,
                    Err(_) => Err(exited_diagnostic(&self.manifest.id, &self.stderr_tail).await),
                }
            }
            _ = cancel.cancelled() => {
                let cancel_msg = serde_json::to_string(&NotificationMsg {
                    jsonrpc: "2.0",
                    method: "$/cancelRequest",
                    params: CancelParams { id: req_id },
                })
                .expect("serialize cancel notification");
                let _ = self.writer_tx.send(WriterCommand::Line(cancel_msg)).await;
                match rx.await {
                    Ok(val) => val,
                    Err(_) => Err(exited_diagnostic(&self.manifest.id, &self.stderr_tail).await),
                }
            }
        }
    }

    async fn stop(&self) -> Result<(), Diagnostic> {
        let req_id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(req_id, tx);

        let stop_msg = serde_json::to_string(&RequestMsg {
            jsonrpc: "2.0",
            id: req_id,
            method: "mod/stop",
            params: serde_json::json!({}),
        })
        .expect("serialize mod/stop request");

        let _ = self.writer_tx.send(WriterCommand::Line(stop_msg)).await;
        let stop_res = tokio::time::timeout(self.shutdown_timeout, rx).await;

        // Close stdin
        let _ = self.writer_tx.send(WriterCommand::Close).await;

        // Wait for child to exit within shutdown_timeout, else kill it
        let mut child_guard = self.child.lock().await;
        if let Some(mut child) = child_guard.take() {
            let wait_res = tokio::time::timeout(self.shutdown_timeout, child.wait()).await;
            match wait_res {
                Ok(Ok(_)) => {}
                _ => {
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                }
            }
        }

        match stop_res {
            Ok(Ok(Ok(_))) => Ok(()),
            Ok(Ok(Err(diag))) => Err(diag),
            Ok(Err(_)) => Ok(()),
            Err(_) => Err(internal_error(
                &self.manifest.id,
                "TIMEOUT",
                "shutdown timeout waiting for mod/stop response",
            )),
        }
    }
}

fn spawn_stderr_drain(
    mut stderr: tokio::process::ChildStderr,
    tail: Arc<Mutex<Vec<u8>>>,
    maximum_bytes: usize,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        use tokio::io::AsyncReadExt;
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

async fn exited_diagnostic(mod_id: &str, stderr_tail: &Arc<Mutex<Vec<u8>>>) -> Diagnostic {
    let tail = stderr_tail.lock().await;
    let text = String::from_utf8_lossy(&tail).trim_end().to_owned();
    if text.is_empty() {
        internal_error(
            mod_id,
            "EXITED",
            format!("mod `{mod_id}` process exited unexpectedly"),
        )
    } else {
        internal_error(
            mod_id,
            "EXITED",
            format!("mod `{mod_id}` process exited unexpectedly; stderr:\n{text}"),
        )
    }
}

async fn with_stderr(diag: Diagnostic, stderr_tail: &Arc<Mutex<Vec<u8>>>) -> Diagnostic {
    let tail = stderr_tail.lock().await;
    let text = String::from_utf8_lossy(&tail).trim_end().to_owned();
    if text.is_empty() {
        diag
    } else {
        let msg = format!("{}; stderr:\n{text}", diag.message());
        let mut new_diag = Diagnostic::new(
            diag.id().clone(),
            diag.code().clone(),
            diag.severity(),
            diag.source().clone(),
            msg,
            diag.primary().clone(),
        );
        for related in diag.related() {
            new_diag = new_diag.with_related(related.clone());
        }
        for fix in diag.fixes() {
            new_diag = new_diag.with_fix(fix.clone());
        }
        new_diag
    }
}
