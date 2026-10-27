//! SDK for serving domain mods over JSON-RPC 2.0 stdio.

use std::any::Any;
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::task::Poll;

use reimagine_core::event::{Event, EventBus, EventReceiver};
use reimagine_core::task::CancelToken;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
use tokio::sync::broadcast::error::{RecvError, TryRecvError};
use tokio::sync::{Mutex, mpsc};
use tokio::task::JoinHandle;

use crate::context::ModContext;
use crate::diagnostic::internal_error;
use crate::interface::{Invocation, Mod};
use crate::protocol::{
    CancelParams, IncomingRpcMsg, InitializeParams, InitializeResult, InvokeParams,
    NotificationMsg, PROTOCOL_VERSION, PublishParams, ResponseErrorMsg, ResponseSuccessMsg,
    RpcError,
};

enum WriterMsg {
    AttachBus(EventReceiver),
    Response(String),
}

enum WriterStep {
    // Boxed because `Event` is much larger than `WriterMsg`.
    Event(Box<Result<Event, RecvError>>),
    Command(Option<WriterMsg>),
}

/// Serve a mod over standard I/O speaking JSON-RPC 2.0.
///
/// Reads lines from stdin until EOF. When stdin reaches EOF (the lifeline to
/// the host process), in-flight calls are cancelled and dropped, serving
/// completes, and the process can exit. A mod must write diagnostics to
/// stderr: anything on stdout is read as a protocol frame.
pub async fn serve(module: impl Mod + 'static) -> std::io::Result<()> {
    let module = Arc::new(module);
    let (writer_tx, mut writer_rx) = mpsc::channel::<WriterMsg>(64);

    let writer_handle = tokio::spawn(async move {
        let mut stdout = tokio::io::stdout();
        let mut bus_rx: Option<EventReceiver> = None;
        let mut bus_open = false;

        loop {
            let open = bus_open;
            let step = tokio::select! {
                event = async {
                    if open {
                        bus_rx
                            .as_mut()
                            .expect("bus receiver while the bus is open")
                            .recv()
                            .await
                    } else {
                        std::future::pending::<Result<Event, RecvError>>().await
                    }
                } => WriterStep::Event(Box::new(event)),
                command = writer_rx.recv() => WriterStep::Command(command),
            };

            match step {
                WriterStep::Event(event) => match *event {
                    Ok(Event::Mod { kind, payload }) => {
                        write_publish(&mut stdout, kind, payload).await?;
                    }
                    Ok(_) => {}
                    Err(RecvError::Closed) => {
                        // The mod dropped its context; stop polling a closed bus.
                        bus_open = false;
                    }
                    Err(RecvError::Lagged(skipped)) => {
                        tracing::warn!(skipped, "mod event bus lagged; events were dropped");
                    }
                },
                WriterStep::Command(Some(WriterMsg::AttachBus(receiver))) => {
                    bus_rx = Some(receiver);
                    bus_open = true;
                }
                WriterStep::Command(Some(WriterMsg::Response(line))) => {
                    drain_events(&mut stdout, &mut bus_rx, &mut bus_open).await?;
                    write_line(&mut stdout, &line).await?;
                }
                WriterStep::Command(None) => break,
            }
        }

        Ok::<(), std::io::Error>(())
    });

    let cancel_tokens = Arc::new(Mutex::new(HashMap::<i64, CancelToken>::new()));
    let stdin = tokio::io::stdin();
    let mut lines = tokio::io::BufReader::new(stdin).lines();
    let mut in_flight: Vec<(i64, JoinHandle<()>)> = Vec::new();

    while let Ok(Some(line)) = lines.next_line().await {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // A line that is not JSON at all is a parse error; valid JSON that is
        // not a JSON-RPC message is an invalid request.
        let value: Value = match serde_json::from_str(trimmed) {
            Ok(value) => value,
            Err(_) => {
                send_error(
                    &writer_tx,
                    None,
                    RpcError::parse_error("failed to parse JSON-RPC line"),
                )
                .await;
                continue;
            }
        };

        let message: IncomingRpcMsg = match serde_json::from_value(value) {
            Ok(message) => message,
            Err(_) => {
                send_error(
                    &writer_tx,
                    None,
                    RpcError::invalid_request("not a valid JSON-RPC 2.0 object"),
                )
                .await;
                continue;
            }
        };

        if message.jsonrpc != "2.0" {
            send_error(
                &writer_tx,
                None,
                RpcError::invalid_request("protocol envelope requires jsonrpc == '2.0'"),
            )
            .await;
            continue;
        }

        // Notifications have no id; we currently only support $/cancelRequest.
        let Some(req_id_val) = message.id else {
            if message.method.as_deref() == Some("$/cancelRequest")
                && let Some(params_val) = message.params
                && let Ok(cancel) = serde_json::from_value::<CancelParams>(params_val)
            {
                let tokens = cancel_tokens.lock().await;
                if let Some(token) = tokens.get(&cancel.id) {
                    token.cancel();
                }
            }
            continue;
        };

        let req_id = match req_id_val.as_i64() {
            Some(id) => id,
            None => {
                send_error(
                    &writer_tx,
                    None,
                    RpcError::invalid_request("request id must be an integer"),
                )
                .await;
                continue;
            }
        };

        let Some(method) = message.method.as_deref() else {
            send_error(
                &writer_tx,
                Some(req_id),
                RpcError::invalid_request("request is missing method"),
            )
            .await;
            continue;
        };

        match method {
            "initialize" => {
                let params: Result<InitializeParams, _> =
                    serde_json::from_value(message.params.unwrap_or(Value::Null));
                match params {
                    Ok(params) => {
                        if params.protocol_version != PROTOCOL_VERSION {
                            let diagnostic = internal_error(
                                &module.manifest().id,
                                "PROTOCOL_MISMATCH",
                                format!(
                                    "host requested protocol {} but mod requires {PROTOCOL_VERSION}",
                                    params.protocol_version
                                ),
                            );
                            send_error(&writer_tx, Some(req_id), RpcError::diagnostic(&diagnostic))
                                .await;
                            continue;
                        }

                        let result = InitializeResult {
                            protocol_version: PROTOCOL_VERSION.to_owned(),
                            manifest: module.manifest().clone(),
                        };
                        let response = ResponseSuccessMsg {
                            jsonrpc: "2.0",
                            id: req_id,
                            result,
                        };
                        let line =
                            serde_json::to_string(&response).expect("serialize initialize result");
                        let _ = writer_tx.send(WriterMsg::Response(line)).await;
                    }
                    Err(error) => {
                        send_error(
                            &writer_tx,
                            Some(req_id),
                            RpcError::invalid_params(format!("invalid initialize params: {error}")),
                        )
                        .await;
                    }
                }
            }
            "mod/start" => {
                // Run the lifecycle call in a task too, so a mod that hangs in
                // `start` cannot block the reader from noticing stdin EOF.
                let task_module = Arc::clone(&module);
                let task_writer_tx = writer_tx.clone();
                let handle = tokio::spawn(async move {
                    let task_mod_id = task_module.manifest().id.clone();
                    let bus = EventBus::new();
                    let receiver = bus.subscribe();
                    let _ = task_writer_tx.send(WriterMsg::AttachBus(receiver)).await;
                    let ctx = ModContext::new(Arc::new(task_module.manifest().clone()), bus);
                    let line = match catch_panic(task_module.start(ctx)).await {
                        Ok(Ok(())) => serde_json::to_string(&ResponseSuccessMsg {
                            jsonrpc: "2.0",
                            id: req_id,
                            result: serde_json::json!({}),
                        })
                        .expect("serialize start"),
                        Ok(Err(diagnostic)) => serde_json::to_string(&ResponseErrorMsg {
                            jsonrpc: "2.0",
                            id: Some(req_id),
                            error: RpcError::diagnostic(&diagnostic),
                        })
                        .expect("serialize error"),
                        Err(payload) => {
                            let diagnostic = internal_error(
                                &task_mod_id,
                                "PANIC",
                                format!(
                                    "mod `{task_mod_id}` panicked in start: {}",
                                    panic_message(payload)
                                ),
                            );
                            serde_json::to_string(&ResponseErrorMsg {
                                jsonrpc: "2.0",
                                id: Some(req_id),
                                error: RpcError::diagnostic(&diagnostic),
                            })
                            .expect("serialize error")
                        }
                    };
                    let _ = task_writer_tx.send(WriterMsg::Response(line)).await;
                });
                in_flight.push((req_id, handle));
                reap_finished(&mut in_flight, &writer_tx, &module.manifest().id).await;
            }
            "mod/invoke" => {
                let params: Result<InvokeParams, _> =
                    serde_json::from_value(message.params.unwrap_or(Value::Null));
                match params {
                    Ok(params) => {
                        let cancel_token = CancelToken::new();
                        cancel_tokens
                            .lock()
                            .await
                            .insert(req_id, cancel_token.clone());
                        let task_module = Arc::clone(&module);
                        let task_writer_tx = writer_tx.clone();
                        let cancel_tokens = Arc::clone(&cancel_tokens);
                        let task_mod_id = task_module.manifest().id.clone();
                        let invocation = Invocation {
                            operation: params.operation,
                            input: params.payload,
                            output_dir: params.output_dir,
                            cancel: cancel_token,
                        };
                        let handle = tokio::spawn(async move {
                            let result = match catch_panic(task_module.invoke(invocation)).await {
                                Ok(result) => result,
                                Err(payload) => {
                                    let message = panic_message(payload);
                                    Err(internal_error(
                                        &task_mod_id,
                                        "PANIC",
                                        format!(
                                            "mod `{task_mod_id}` panicked while handling the call: {message}"
                                        ),
                                    ))
                                }
                            };
                            cancel_tokens.lock().await.remove(&req_id);
                            let line = match result {
                                Ok(value) => serde_json::to_string(&ResponseSuccessMsg {
                                    jsonrpc: "2.0",
                                    id: req_id,
                                    result: value,
                                })
                                .expect("serialize invoke result"),
                                Err(diagnostic) => serde_json::to_string(&ResponseErrorMsg {
                                    jsonrpc: "2.0",
                                    id: Some(req_id),
                                    error: RpcError::diagnostic(&diagnostic),
                                })
                                .expect("serialize error"),
                            };
                            let _ = task_writer_tx.send(WriterMsg::Response(line)).await;
                        });
                        in_flight.push((req_id, handle));
                        reap_finished(&mut in_flight, &writer_tx, &module.manifest().id).await;
                    }
                    Err(error) => {
                        send_error(
                            &writer_tx,
                            Some(req_id),
                            RpcError::invalid_params(format!("invalid invoke params: {error}")),
                        )
                        .await;
                    }
                }
            }
            "mod/stop" => {
                // Spawned for the same reason as `mod/start`.
                let task_module = Arc::clone(&module);
                let task_writer_tx = writer_tx.clone();
                let handle = tokio::spawn(async move {
                    let task_mod_id = task_module.manifest().id.clone();
                    let line = match catch_panic(task_module.stop()).await {
                        Ok(Ok(())) => serde_json::to_string(&ResponseSuccessMsg {
                            jsonrpc: "2.0",
                            id: req_id,
                            result: serde_json::json!({}),
                        })
                        .expect("serialize stop"),
                        Ok(Err(diagnostic)) => serde_json::to_string(&ResponseErrorMsg {
                            jsonrpc: "2.0",
                            id: Some(req_id),
                            error: RpcError::diagnostic(&diagnostic),
                        })
                        .expect("serialize error"),
                        Err(payload) => {
                            let diagnostic = internal_error(
                                &task_mod_id,
                                "PANIC",
                                format!(
                                    "mod `{task_mod_id}` panicked in stop: {}",
                                    panic_message(payload)
                                ),
                            );
                            serde_json::to_string(&ResponseErrorMsg {
                                jsonrpc: "2.0",
                                id: Some(req_id),
                                error: RpcError::diagnostic(&diagnostic),
                            })
                            .expect("serialize error")
                        }
                    };
                    let _ = task_writer_tx.send(WriterMsg::Response(line)).await;
                });
                in_flight.push((req_id, handle));
                reap_finished(&mut in_flight, &writer_tx, &module.manifest().id).await;
            }
            other => {
                send_error(&writer_tx, Some(req_id), RpcError::method_not_found(other)).await;
            }
        }
    }

    // stdin EOF means the host is gone. Cancel in-flight calls so a mod waiting
    // on cancellation can return, then abort whatever is left (including a
    // lifecycle call stuck in start/stop): the writer task must be able to
    // finish, or the process would outlive its host.
    {
        let tokens = cancel_tokens.lock().await;
        for token in tokens.values() {
            token.cancel();
        }
    }
    for (_, handle) in in_flight.drain(..) {
        handle.abort();
        let _ = handle.await;
    }
    cancel_tokens.lock().await.clear();
    drop(writer_tx);
    let _ = writer_handle.await;
    Ok(())
}

async fn send_error(writer_tx: &mpsc::Sender<WriterMsg>, id: Option<i64>, error: RpcError) {
    let response = ResponseErrorMsg {
        jsonrpc: "2.0",
        id,
        error,
    };
    let line = serde_json::to_string(&response).expect("serialize error");
    let _ = writer_tx.send(WriterMsg::Response(line)).await;
}

/// Keep draining finished invoke tasks so their handles do not accumulate.
/// A task that failed outside the invocation itself still answers the caller.
async fn reap_finished(
    invokes: &mut Vec<(i64, JoinHandle<()>)>,
    writer_tx: &mpsc::Sender<WriterMsg>,
    mod_id: &str,
) {
    let mut index = 0;
    while index < invokes.len() {
        if invokes[index].1.is_finished() {
            let (req_id, handle) = invokes.swap_remove(index);
            if let Err(error) = handle.await {
                let diagnostic =
                    internal_error(mod_id, "TASK", format!("mod task failed: {error}"));
                send_error(writer_tx, Some(req_id), RpcError::diagnostic(&diagnostic)).await;
            }
        } else {
            index += 1;
        }
    }
}

async fn drain_events(
    stdout: &mut tokio::io::Stdout,
    bus_rx: &mut Option<EventReceiver>,
    bus_open: &mut bool,
) -> std::io::Result<()> {
    if !*bus_open {
        return Ok(());
    }
    let Some(receiver) = bus_rx.as_mut() else {
        return Ok(());
    };
    loop {
        match receiver.try_recv() {
            Ok(Event::Mod { kind, payload }) => write_publish(stdout, kind, payload).await?,
            Ok(_) => {}
            Err(TryRecvError::Empty) => return Ok(()),
            Err(TryRecvError::Lagged(skipped)) => {
                tracing::warn!(skipped, "mod event bus lagged; events were dropped");
            }
            Err(TryRecvError::Closed) => {
                *bus_open = false;
                return Ok(());
            }
        }
    }
}

async fn write_publish(
    stdout: &mut tokio::io::Stdout,
    event_type: String,
    data: Value,
) -> std::io::Result<()> {
    let notification = NotificationMsg {
        jsonrpc: "2.0",
        method: "kernel/publish",
        params: PublishParams { event_type, data },
    };
    let line = serde_json::to_string(&notification).expect("serialize notification");
    write_line(stdout, &line).await
}

async fn write_line(stdout: &mut tokio::io::Stdout, line: &str) -> std::io::Result<()> {
    stdout.write_all(line.as_bytes()).await?;
    stdout.write_all(b"\n").await?;
    stdout.flush().await?;
    Ok(())
}

/// Poll `future` to completion, turning a panic into an error value instead of
/// letting it kill the detached invoke task (which would never answer).
async fn catch_panic<F>(future: F) -> Result<F::Output, Box<dyn Any + Send>>
where
    F: std::future::Future,
{
    let mut future = std::pin::pin!(future);
    std::future::poll_fn(move |cx| {
        match catch_unwind(AssertUnwindSafe(|| future.as_mut().poll(cx))) {
            Ok(Poll::Ready(value)) => Poll::Ready(Ok(value)),
            Ok(Poll::Pending) => Poll::Pending,
            Err(payload) => Poll::Ready(Err(payload)),
        }
    })
    .await
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_owned()
    }
}
