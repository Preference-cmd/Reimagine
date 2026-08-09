//! Scripted mock of `reimagine-agent-daemon` for the agent bridge tests.
//!
//! Compiled as a `harness = false` test target (see `Cargo.toml`), so
//! `cargo test` builds it as a plain binary. Run without arguments during
//! a normal test run it exits immediately; the bridge test re-runs it with
//! `REIMAGINE_MOCK_DAEMON_SERVE` set, whereupon it serves a scripted
//! JSON-RPC session over stdio:
//!
//! - answers `initialize` with a fixed `server_info` and requires the
//!   `initialized` notification to arrive next (exits 3 otherwise),
//! - echoes `session.create` / `session.list` from internal state,
//! - answers `providers.list` with a fixed provider list,
//! - answers `turn.run` with `accepted` and then streams an
//!   `agent.content_delta` + `agent.turn_completed` pair,
//! - answers `turn.cancel` with `cancelled`.
//!
//! With `REIMAGINE_MOCK_DAEMON_EXIT_AFTER_INIT` set it exits 2 right after
//! receiving the `initialized` notification, so the bridge test can
//! exercise daemon-crash error propagation.

use std::io::{self, BufRead, Write};

use reimagine_agent_daemon::protocol::{
    ContentDeltaParams, JsonRpcError, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse,
    METHOD_AGENT_CONTENT_DELTA, METHOD_AGENT_TURN_COMPLETED, METHOD_INITIALIZE, METHOD_INITIALIZED,
    METHOD_PROVIDERS_LIST, METHOD_SESSION_CREATE, METHOD_SESSION_LIST, METHOD_TURN_CANCEL,
    METHOD_TURN_RUN, SessionInfo, TurnCompletedParams,
};
use serde_json::{Value, json};

const SERVE_ENV: &str = "REIMAGINE_MOCK_DAEMON_SERVE";
const EXIT_AFTER_INIT_ENV: &str = "REIMAGINE_MOCK_DAEMON_EXIT_AFTER_INIT";
const EXIT_AFTER_INIT: i32 = 2;
const EXIT_HANDSHAKE_VIOLATION: i32 = 3;

fn main() {
    if std::env::var_os(SERVE_ENV).is_none() {
        return;
    }
    std::process::exit(serve());
}

fn serve() -> i32 {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut sessions: Vec<SessionInfo> = Vec::new();
    let mut initialized = false;

    for line in stdin.lock().lines() {
        let Ok(line) = line else {
            break;
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(message) = serde_json::from_str::<Value>(line) else {
            continue;
        };

        if message.get("id").is_none() {
            match message.get("method").and_then(Value::as_str) {
                Some(METHOD_INITIALIZED) => {
                    initialized = true;
                    if std::env::var_os(EXIT_AFTER_INIT_ENV).is_some() {
                        return EXIT_AFTER_INIT;
                    }
                }
                _ => return EXIT_HANDSHAKE_VIOLATION,
            }
            continue;
        }

        let request: JsonRpcRequest<Value> = match serde_json::from_value(message) {
            Ok(request) => request,
            Err(_) => {
                write_line(
                    &mut stdout,
                    &JsonRpcResponse::<Value>::error(0, JsonRpcError::parse_error()),
                );
                continue;
            }
        };
        if request.method == METHOD_INITIALIZE {
            if initialized {
                return EXIT_HANDSHAKE_VIOLATION;
            }
        } else if !initialized {
            return EXIT_HANDSHAKE_VIOLATION;
        }

        let result: Result<Value, JsonRpcError> = match request.method.as_str() {
            METHOD_INITIALIZE => Ok(json!({
                "server_info": {
                    "name": "reimagine-agent-daemon-mock",
                    "version": "0.0.0",
                },
                "capabilities": {},
            })),
            METHOD_SESSION_CREATE => {
                let mode = request
                    .params
                    .get("mode")
                    .and_then(Value::as_str)
                    .unwrap_or("agent")
                    .to_owned();
                let provider = request
                    .params
                    .get("provider")
                    .and_then(Value::as_str)
                    .unwrap_or("openai")
                    .to_owned();
                let session_id = format!("sess-{}", sessions.len() + 1);
                sessions.push(SessionInfo {
                    session_id: session_id.clone(),
                    mode: mode.clone(),
                    provider: provider.clone(),
                    created_at: "2026-08-08T00:00:00Z".to_owned(),
                });
                Ok(json!({
                    "session_id": session_id,
                    "mode": mode,
                    "provider": provider,
                    "created_at": "2026-08-08T00:00:00Z",
                }))
            }
            METHOD_SESSION_LIST => Ok(json!({ "sessions": sessions })),
            METHOD_PROVIDERS_LIST => Ok(json!({
                "providers": [
                    { "id": "openai", "name": "OpenAI" },
                    { "id": "anthropic", "name": "Anthropic" },
                ]
            })),
            METHOD_TURN_CANCEL => Ok(json!({ "status": "cancelled" })),
            METHOD_TURN_RUN => {
                let session_id = request
                    .params
                    .get("session_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                let turn_id = request
                    .params
                    .get("turn_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                if session_id.is_empty() || turn_id.is_empty() {
                    Err(JsonRpcError::invalid_params())
                } else {
                    Ok(json!({
                        "status": "accepted",
                        "session_id": session_id,
                        "turn_id": turn_id,
                    }))
                }
            }
            _ => Err(JsonRpcError::method_not_found()),
        };

        match result {
            Ok(value) => write_line(&mut stdout, &JsonRpcResponse::success(request.id, value)),
            Err(error) => write_line(
                &mut stdout,
                &JsonRpcResponse::<Value>::error(request.id, error),
            ),
        }

        if request.method == METHOD_TURN_RUN {
            let session_id = request
                .params
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let turn_id = request
                .params
                .get("turn_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            write_line(
                &mut stdout,
                &JsonRpcNotification::new(
                    METHOD_AGENT_CONTENT_DELTA,
                    ContentDeltaParams {
                        session_id: session_id.to_owned(),
                        turn_id: turn_id.to_owned(),
                        text: "streamed".to_owned(),
                    },
                ),
            );
            write_line(
                &mut stdout,
                &JsonRpcNotification::new(
                    METHOD_AGENT_TURN_COMPLETED,
                    TurnCompletedParams {
                        session_id: session_id.to_owned(),
                        turn_id: turn_id.to_owned(),
                        result: json!({ "text": "done" }),
                    },
                ),
            );
        }
    }

    if !initialized {
        return EXIT_HANDSHAKE_VIOLATION;
    }
    0
}

fn write_line<T: serde::Serialize>(writer: &mut impl Write, message: &T) {
    let Ok(line) = serde_json::to_string(message) else {
        return;
    };
    if writeln!(writer, "{line}").is_err() {
        // stdout closed (the bridge is gone); the loop will exit on the
        // next read or write.
    }
    let _ = writer.flush();
}
