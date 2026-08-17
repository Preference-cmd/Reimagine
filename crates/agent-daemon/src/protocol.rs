//! JSON-RPC 2.0 wire contract for the agent daemon.
//!
//! Messages flow over stdio as newline-delimited JSON between the Tauri
//! host (client) and the daemon (server). Requests carry an `id` and
//! expect a matching response; notifications carry no `id`. V1 defines
//! the handshake, the session / turn / provider request methods, and the
//! daemon-to-client streaming notifications. `turn.steer` and
//! `agent.permissions.request_approval` are reserved interfaces that V2
//! will flesh out.
//!
//! This module is deliberately dependency-free beyond serde; future
//! transport and dispatch code in this crate adapts these types to the
//! `reimagine-agent` / `reimagine-app-host` domain model.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON-RPC protocol version field value, per JSON-RPC 2.0.
pub const JSONRPC_VERSION: &str = "2.0";

/// Standard JSON-RPC error code: request body failed to parse.
pub const ERROR_PARSE: i64 = -32700;
/// Standard JSON-RPC error code: request not a valid JSON-RPC message.
pub const ERROR_INVALID_REQUEST: i64 = -32600;
/// Standard JSON-RPC error code: method does not exist on the server.
pub const ERROR_METHOD_NOT_FOUND: i64 = -32601;
/// Standard JSON-RPC error code: params do not match the method.
pub const ERROR_INVALID_PARAMS: i64 = -32602;
/// Standard JSON-RPC error code: internal server failure.
pub const ERROR_INTERNAL: i64 = -32603;
/// Reimagine-specific error code: method exists but is not implemented
/// in this protocol version (e.g. `turn.steer` in V1).
pub const ERROR_NOT_SUPPORTED: i64 = -32604;

/// Handshake request sent by the client on connection startup.
pub const METHOD_INITIALIZE: &str = "initialize";
/// Client notification that handshake state is fully applied.
pub const METHOD_INITIALIZED: &str = "initialized";
/// Create a new agent session.
pub const METHOD_SESSION_CREATE: &str = "session.create";
/// Fetch a session's current info.
pub const METHOD_SESSION_GET: &str = "session.get";
/// List all live sessions.
pub const METHOD_SESSION_LIST: &str = "session.list";
/// Start a turn on a session.
pub const METHOD_TURN_RUN: &str = "turn.run";
/// Cancel a running turn.
pub const METHOD_TURN_CANCEL: &str = "turn.cancel";
/// Steer a running turn with new input. Reserved; V1 returns
/// `not_supported`.
pub const METHOD_TURN_STEER: &str = "turn.steer";
/// List configured providers.
pub const METHOD_PROVIDERS_LIST: &str = "providers.list";
/// Daemon streams incremental assistant text for a turn.
pub const METHOD_AGENT_CONTENT_DELTA: &str = "agent.content_delta";
pub const METHOD_AGENT_REASONING_DELTA: &str = "agent.reasoning_delta";
/// Daemon reports a summarization compaction replaced evicted history
/// with a sticky summary (CM-V2e).
pub const METHOD_AGENT_CONTEXT_COMPACTED: &str = "agent.context_compacted";
/// Daemon reports a tool call is being executed.
pub const METHOD_AGENT_TOOL_INVOKED: &str = "agent.tool_invoked";
/// Daemon reports a tool call completed successfully.
pub const METHOD_AGENT_TOOL_COMPLETED: &str = "agent.tool_completed";
/// Daemon reports a tool call failed.
pub const METHOD_AGENT_TOOL_FAILED: &str = "agent.tool_failed";
/// Daemon reports a turn finished.
pub const METHOD_AGENT_TURN_COMPLETED: &str = "agent.turn_completed";
/// Daemon reports a session started.
pub const METHOD_AGENT_SESSION_STARTED: &str = "agent.session_started";
/// Daemon reports a session stopped.
pub const METHOD_AGENT_SESSION_STOPPED: &str = "agent.session_stopped";
/// Daemon reports a build-mode proposal is ready for review.
pub const METHOD_AGENT_PROPOSAL_READY: &str = "agent.proposal_ready";
/// Daemon reports a session-scoped error.
pub const METHOD_AGENT_ERROR: &str = "agent.error";
/// Daemon asks the host to approve a tool call. Reserved; V2.
pub const METHOD_AGENT_REQUEST_APPROVAL: &str = "agent.permissions.request_approval";

/// Outgoing request envelope. `params` is the method-specific payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct JsonRpcRequest<T> {
    pub jsonrpc: String,
    pub id: u64,
    pub method: String,
    pub params: T,
}

impl<T> JsonRpcRequest<T> {
    pub fn new(method: impl Into<String>, id: u64, params: T) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_owned(),
            id,
            method: method.into(),
            params,
        }
    }
}

/// Response envelope. Exactly one of `result` / `error` is set; the
/// empty side is omitted on the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct JsonRpcResponse<T> {
    pub jsonrpc: String,
    pub id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

impl<T> JsonRpcResponse<T> {
    pub fn success(id: u64, result: T) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_owned(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: u64, error: JsonRpcError) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_owned(),
            id,
            result: None,
            error: Some(error),
        }
    }
}

/// Outgoing notification envelope (no `id`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct JsonRpcNotification<T> {
    pub jsonrpc: String,
    pub method: String,
    pub params: T,
}

impl<T> JsonRpcNotification<T> {
    pub fn new(method: impl Into<String>, params: T) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_owned(),
            method: method.into(),
            params,
        }
    }
}

/// Structured error carried by an error response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcError {
    pub fn new(code: i64, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }

    pub fn parse_error() -> Self {
        Self::new(ERROR_PARSE, "parse error")
    }

    pub fn invalid_request() -> Self {
        Self::new(ERROR_INVALID_REQUEST, "invalid request")
    }

    pub fn method_not_found() -> Self {
        Self::new(ERROR_METHOD_NOT_FOUND, "method not found")
    }

    pub fn invalid_params() -> Self {
        Self::new(ERROR_INVALID_PARAMS, "invalid params")
    }

    pub fn internal_error() -> Self {
        Self::new(ERROR_INTERNAL, "internal error")
    }

    /// `turn.steer` and other reserved V2 methods report this.
    pub fn not_supported() -> Self {
        Self::new(ERROR_NOT_SUPPORTED, "method not supported")
    }
}

/// Params for methods that take none (`session.list`, `providers.list`).
/// Serializes to an empty JSON object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct EmptyParams {}

/// `initialize` request params (client -> daemon handshake).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct InitializeRequest {
    pub client_info: ClientInfo,
    pub capabilities: ClientCapabilities,
}

/// Client identity reported during the handshake.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ClientInfo {
    pub name: String,
    pub version: String,
}

/// Client capability flags. V1 only declares the experimental API
/// toggle; future versions may add more.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ClientCapabilities {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experimental_api: Option<bool>,
}

/// `initialize` response result (daemon -> client handshake).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct InitializeResponse {
    pub server_info: ServerInfo,
    pub capabilities: ServerCapabilities,
}

/// Server identity reported during the handshake.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

/// Server capability flags. V1 keeps this open-ended so later tickets
/// can advertise session / turn / approval features.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ServerCapabilities {}

/// `session.create` request params.
///
/// Field audit (AR-35): V1 previously accepted `system_prompt` and
/// `workspace_dir` here but ignored both — the daemon session is always
/// rooted at the process-level `--workspace-dir` and has no per-session
/// prompt. The fields are removed instead of silently dropped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SessionCreateParams {
    /// Session mode: `"agent"` (auto-apply) or `"build"` (proposal-only).
    pub mode: String,
    /// Provider id the session is bound to.
    pub provider: String,
}

/// `session.create` response result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SessionCreateResult {
    pub session_id: String,
    pub mode: String,
    pub provider: String,
    /// Host-supplied creation timestamp (RFC 3339).
    pub created_at: String,
}

/// `session.get` request params.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SessionGetParams {
    pub session_id: String,
}

/// Session info returned by `session.get` and `session.list`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SessionInfo {
    pub session_id: String,
    pub mode: String,
    pub provider: String,
    /// Host-supplied creation timestamp (RFC 3339).
    pub created_at: String,
}

/// `session.list` response result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SessionListResult {
    pub sessions: Vec<SessionInfo>,
}

/// `turn.run` request params.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TurnRunParams {
    pub session_id: String,
    pub turn_id: String,
    /// Model id the turn should use.
    pub model: String,
    /// Free-form user input; kept as `serde_json::Value` so the host can
    /// attach extra fields without a protocol bump.
    pub input: Value,
}

/// Status reported by `turn.run`. V1 only accepts the turn; the turn
/// outcome streams back via `agent.*` notifications.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnRunStatus {
    Accepted,
}

/// `turn.run` response result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TurnRunResult {
    pub status: TurnRunStatus,
    pub session_id: String,
    pub turn_id: String,
}

/// `turn.cancel` request params.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TurnCancelParams {
    pub session_id: String,
    pub turn_id: String,
}

/// Status reported by `turn.cancel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnCancelStatus {
    Cancelled,
}

/// `turn.cancel` response result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TurnCancelResult {
    pub status: TurnCancelStatus,
}

/// `turn.steer` request params. Reserved interface: V1 answers every
/// steer with a `not_supported` error; V2 will implement live steering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TurnSteerParams {
    pub session_id: String,
    pub turn_id: String,
    /// Free-form steering input; same shape rules as `TurnRunParams.input`.
    pub input: Value,
}

/// Provider entry in a `providers.list` response. V1 exposes the id and
/// a display name; richer metadata can be added without a protocol bump.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ProviderInfo {
    pub id: String,
    pub name: String,
}

/// `providers.list` response result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ProvidersListResult {
    pub providers: Vec<ProviderInfo>,
}

/// `agent.content_delta` notification params.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ContentDeltaParams {
    pub session_id: String,
    pub turn_id: String,
    pub text: String,
}

/// `agent.reasoning_delta` notification params. Reasoning is display-only;
/// the daemon never persists it into session history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ReasoningDeltaParams {
    pub session_id: String,
    pub turn_id: String,
    pub text: String,
}

/// `agent.context_compacted` notification params (CM-V2e). `summary`
/// is the new sticky summary text; `tokens_before` / `tokens_after`
/// are estimated token counts of the replaced range and the summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ContextCompactedParams {
    pub session_id: String,
    pub turn_id: String,
    pub summary: String,
    pub tokens_before: u64,
    pub tokens_after: u64,
}

/// Params shared by `agent.tool_invoked` / `agent.tool_completed` /
/// `agent.tool_failed`. `error` is only set on `tool_failed`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ToolEventParams {
    pub session_id: String,
    pub turn_id: String,
    pub tool: String,
    pub tool_call_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// `agent.turn_completed` notification params.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TurnCompletedParams {
    pub session_id: String,
    pub turn_id: String,
    /// Free-form turn outcome (final text, tool results, usage); kept as
    /// `serde_json::Value` until turn payloads stabilize.
    pub result: Value,
}

/// `agent.session_started` notification params.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SessionStartedParams {
    pub session_id: String,
    pub mode: String,
    pub provider: String,
}

/// `agent.session_stopped` notification params.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SessionStoppedParams {
    pub session_id: String,
    /// Free-form stop reason (e.g. `"user_requested"`, `"error"`).
    pub reason: String,
}

/// `agent.proposal_ready` notification params.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ProposalReadyParams {
    pub session_id: String,
    pub proposal_id: String,
}

/// `agent.error` notification params.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AgentErrorParams {
    pub session_id: String,
    /// Machine-readable error code (e.g. `"provider_error"`).
    pub code: String,
    pub message: String,
}

/// `agent.permissions.request_approval` notification params. Reserved
/// interface: the V2 approval flow sends these; V1 never emits them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RequestApprovalParams {
    pub session_id: String,
    pub turn_id: String,
    pub tool: String,
    pub description: String,
    /// `"low"`, `"medium"`, or `"high"`.
    pub risk_level: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn assert_roundtrip<T>(value: &T)
    where
        T: Serialize + for<'de> Deserialize<'de> + PartialEq + std::fmt::Debug,
    {
        let json = serde_json::to_string(value).unwrap();
        let back: T = serde_json::from_str(&json).unwrap();
        assert_eq!(value, &back);
    }

    #[test]
    fn empty_params_serialize_to_empty_object() {
        let json = serde_json::to_value(EmptyParams {}).unwrap();
        assert_eq!(json, json!({}));
    }

    #[test]
    fn request_envelope_roundtrip() {
        let req = JsonRpcRequest::new(
            METHOD_SESSION_CREATE,
            1,
            SessionCreateParams {
                mode: "agent".into(),
                provider: "openai".into(),
            },
        );
        assert_eq!(req.jsonrpc, "2.0");
        assert_roundtrip(&req);
    }

    #[test]
    fn success_response_envelope_roundtrip() {
        let resp = JsonRpcResponse::success(
            2,
            SessionCreateResult {
                session_id: "s1".into(),
                mode: "agent".into(),
                provider: "openai".into(),
                created_at: "2026-08-08T00:00:00Z".into(),
            },
        );
        assert_roundtrip(&resp);
    }

    #[test]
    fn error_response_envelope_roundtrip() {
        let resp = JsonRpcResponse::<TurnSteerParams>::error(3, JsonRpcError::not_supported());
        assert_roundtrip(&resp);
        let wire = serde_json::to_value(&resp).unwrap();
        assert_eq!(wire["jsonrpc"], "2.0");
        assert_eq!(wire["error"]["code"], -32604);
        assert!(wire.get("result").is_none());
    }

    #[test]
    fn notification_envelope_roundtrip() {
        let notif = JsonRpcNotification::new(
            METHOD_AGENT_CONTENT_DELTA,
            ContentDeltaParams {
                session_id: "s1".into(),
                turn_id: "t1".into(),
                text: "hello".into(),
            },
        );
        assert_eq!(notif.jsonrpc, "2.0");
        assert!(!serde_json::to_string(&notif).unwrap().contains("\"id\""));
        assert_roundtrip(&notif);
    }

    #[test]
    fn initialize_handshake_roundtrip() {
        let req = JsonRpcRequest::new(
            METHOD_INITIALIZE,
            1,
            InitializeRequest {
                client_info: ClientInfo {
                    name: "reimagine".into(),
                    version: "0.1.0".into(),
                },
                capabilities: ClientCapabilities {
                    experimental_api: Some(true),
                },
            },
        );
        assert_roundtrip(&req);

        let resp = JsonRpcResponse::success(
            1,
            InitializeResponse {
                server_info: ServerInfo {
                    name: "reimagine-agent-daemon".into(),
                    version: "0.1.0".into(),
                },
                capabilities: ServerCapabilities {},
            },
        );
        assert_roundtrip(&resp);
    }

    #[test]
    fn initialize_wire_shape_uses_snake_case() {
        let req = JsonRpcRequest::new(
            METHOD_INITIALIZE,
            1,
            InitializeRequest {
                client_info: ClientInfo {
                    name: "reimagine".into(),
                    version: "0.1.0".into(),
                },
                capabilities: ClientCapabilities {
                    experimental_api: Some(true),
                },
            },
        );
        let wire = serde_json::to_value(&req).unwrap();
        assert_eq!(wire["params"]["client_info"]["name"], "reimagine");
        assert_eq!(wire["params"]["capabilities"]["experimental_api"], true);
    }

    #[test]
    fn session_methods_roundtrip() {
        let create_params = SessionCreateParams {
            mode: "build".into(),
            provider: "anthropic".into(),
        };
        assert_roundtrip(&create_params);

        // AR-35: fields the server used to accept and ignore must not
        // exist on the wire shape anymore.
        let wire = serde_json::to_value(&create_params).unwrap();
        assert!(wire.get("system_prompt").is_none());
        assert!(wire.get("workspace_dir").is_none());

        let create_result = SessionCreateResult {
            session_id: "s1".into(),
            mode: "build".into(),
            provider: "anthropic".into(),
            created_at: "2026-08-08T00:00:00Z".into(),
        };
        assert_roundtrip(&create_result);

        assert_roundtrip(&SessionGetParams {
            session_id: "s1".into(),
        });

        let info = SessionInfo {
            session_id: "s1".into(),
            mode: "build".into(),
            provider: "anthropic".into(),
            created_at: "2026-08-08T00:00:00Z".into(),
        };
        assert_roundtrip(&info);

        let list = SessionListResult {
            sessions: vec![info.clone()],
        };
        assert_roundtrip(&list);

        let wire = serde_json::to_value(&list).unwrap();
        assert_eq!(wire["sessions"][0]["session_id"], "s1");
        assert_eq!(wire["sessions"][0]["created_at"], "2026-08-08T00:00:00Z");
    }

    #[test]
    fn turn_methods_roundtrip() {
        assert_roundtrip(&JsonRpcRequest::new(
            METHOD_TURN_RUN,
            4,
            TurnRunParams {
                session_id: "s1".into(),
                turn_id: "t1".into(),
                model: "gpt-4o-mini".into(),
                input: json!({"text": "draw a cat"}),
            },
        ));

        let run_result = TurnRunResult {
            status: TurnRunStatus::Accepted,
            session_id: "s1".into(),
            turn_id: "t1".into(),
        };
        assert_roundtrip(&run_result);
        let wire = serde_json::to_value(&run_result).unwrap();
        assert_eq!(wire["status"], "accepted");

        assert_roundtrip(&TurnCancelParams {
            session_id: "s1".into(),
            turn_id: "t1".into(),
        });

        let cancel_result = TurnCancelResult {
            status: TurnCancelStatus::Cancelled,
        };
        assert_roundtrip(&cancel_result);
        let wire = serde_json::to_value(&cancel_result).unwrap();
        assert_eq!(wire["status"], "cancelled");

        assert_roundtrip(&TurnSteerParams {
            session_id: "s1".into(),
            turn_id: "t1".into(),
            input: json!("go faster"),
        });
    }

    #[test]
    fn turn_steer_answers_not_supported_in_v1() {
        let err = JsonRpcError::not_supported();
        assert_eq!(err.code, ERROR_NOT_SUPPORTED);
        assert_eq!(err.message, "method not supported");
        let wire = serde_json::to_value(&err).unwrap();
        assert_eq!(wire["code"], -32604);
        assert!(wire.get("data").is_none());
    }

    #[test]
    fn providers_list_roundtrip() {
        let result = ProvidersListResult {
            providers: vec![
                ProviderInfo {
                    id: "openai".into(),
                    name: "OpenAI".into(),
                },
                ProviderInfo {
                    id: "anthropic".into(),
                    name: "Anthropic".into(),
                },
            ],
        };
        assert_roundtrip(&result);
        let wire = serde_json::to_value(&result).unwrap();
        assert_eq!(wire["providers"][1]["id"], "anthropic");
    }

    #[test]
    fn stream_notifications_roundtrip() {
        assert_roundtrip(&ContentDeltaParams {
            session_id: "s1".into(),
            turn_id: "t1".into(),
            text: "partial".into(),
        });

        let invoked = ToolEventParams {
            session_id: "s1".into(),
            turn_id: "t1".into(),
            tool: "workflow.read".into(),
            tool_call_id: "call-1".into(),
            error: None,
        };
        assert_roundtrip(&invoked);

        let failed = ToolEventParams {
            error: Some("timeout".into()),
            ..invoked.clone()
        };
        assert_roundtrip(&failed);
        let wire = serde_json::to_value(&failed).unwrap();
        assert_eq!(wire["tool_call_id"], "call-1");
        assert_eq!(wire["error"], "timeout");

        assert_roundtrip(&TurnCompletedParams {
            session_id: "s1".into(),
            turn_id: "t1".into(),
            result: json!({"text": "done", "usage": {"input_tokens": 10}}),
        });
    }

    #[test]
    fn lifecycle_notifications_roundtrip() {
        assert_roundtrip(&SessionStartedParams {
            session_id: "s1".into(),
            mode: "agent".into(),
            provider: "openai".into(),
        });

        assert_roundtrip(&SessionStoppedParams {
            session_id: "s1".into(),
            reason: "user_requested".into(),
        });

        assert_roundtrip(&ProposalReadyParams {
            session_id: "s1".into(),
            proposal_id: "p1".into(),
        });

        assert_roundtrip(&AgentErrorParams {
            session_id: "s1".into(),
            code: "provider_error".into(),
            message: "upstream 429".into(),
        });
    }

    #[test]
    fn request_approval_notification_roundtrip() {
        let params = RequestApprovalParams {
            session_id: "s1".into(),
            turn_id: "t1".into(),
            tool: "shell.run".into(),
            description: "run cargo test".into(),
            risk_level: "high".into(),
        };
        assert_roundtrip(&params);
        let wire = serde_json::to_value(&params).unwrap();
        assert_eq!(wire["risk_level"], "high");
    }

    #[test]
    fn jsonrpc_error_helpers_use_standard_codes() {
        assert_eq!(JsonRpcError::parse_error().code, -32700);
        assert_eq!(JsonRpcError::invalid_request().code, -32600);
        assert_eq!(JsonRpcError::method_not_found().code, -32601);
        assert_eq!(JsonRpcError::invalid_params().code, -32602);
        assert_eq!(JsonRpcError::internal_error().code, -32603);
        assert_eq!(JsonRpcError::not_supported().code, -32604);
        let with_data = JsonRpcError::internal_error().with_data(json!({"detail": "x"}));
        assert_roundtrip(&with_data);
        assert_eq!(with_data.data.as_ref().unwrap()["detail"], "x");
    }
}
