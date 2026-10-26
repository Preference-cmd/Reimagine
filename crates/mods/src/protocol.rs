use reimagine_core::diagnostic::Diagnostic;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::manifest::ModManifest;

/// The wire protocol version supported by this crate.
pub const PROTOCOL_VERSION: &str = "0.1";

pub const PARSE_ERROR: i32 = -32700;
pub const INVALID_REQUEST: i32 = -32600;
pub const METHOD_NOT_FOUND: i32 = -32601;
pub const INVALID_PARAMS: i32 = -32602;
pub const DIAGNOSTIC_ERROR: i32 = -32000;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InitializeParams {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InitializeResult {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    pub manifest: ModManifest,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InvokeParams {
    pub operation: String,
    pub payload: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CancelParams {
    pub id: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PublishParams {
    #[serde(rename = "type")]
    pub event_type: String,
    pub data: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl RpcError {
    pub fn diagnostic(diag: &Diagnostic) -> Self {
        Self {
            code: DIAGNOSTIC_ERROR,
            message: diag.message().to_owned(),
            data: serde_json::to_value(diag).ok(),
        }
    }

    pub fn parse_error(msg: impl Into<String>) -> Self {
        Self {
            code: PARSE_ERROR,
            message: msg.into(),
            data: None,
        }
    }

    pub fn invalid_request(msg: impl Into<String>) -> Self {
        Self {
            code: INVALID_REQUEST,
            message: msg.into(),
            data: None,
        }
    }

    pub fn method_not_found(method: &str) -> Self {
        Self {
            code: METHOD_NOT_FOUND,
            message: format!("method `{method}` not found"),
            data: None,
        }
    }

    pub fn invalid_params(msg: impl Into<String>) -> Self {
        Self {
            code: INVALID_PARAMS,
            message: msg.into(),
            data: None,
        }
    }

    pub fn invalid_params_with_diag(msg: impl Into<String>, diag: &Diagnostic) -> Self {
        Self {
            code: INVALID_PARAMS,
            message: msg.into(),
            data: serde_json::to_value(diag).ok(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RequestMsg<'a, T> {
    pub jsonrpc: &'static str,
    pub id: i64,
    pub method: &'a str,
    pub params: T,
}

#[derive(Debug, Clone, Serialize)]
pub struct NotificationMsg<'a, T> {
    pub jsonrpc: &'static str,
    pub method: &'a str,
    pub params: T,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseSuccessMsg<T> {
    pub jsonrpc: &'static str,
    pub id: i64,
    pub result: T,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseErrorMsg {
    pub jsonrpc: &'static str,
    pub id: Option<i64>,
    pub error: RpcError,
}

#[derive(Debug, Deserialize)]
pub struct IncomingRpcMsg {
    pub jsonrpc: String,
    #[serde(default)]
    pub id: Option<Value>,
    pub method: Option<String>,
    pub params: Option<Value>,
    pub result: Option<Value>,
    pub error: Option<RpcError>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn serde_wire_shapes() {
        // Request shape
        let req = RequestMsg {
            jsonrpc: "2.0",
            id: 1,
            method: "initialize",
            params: InitializeParams {
                protocol_version: "0.1".to_owned(),
            },
        };
        let req_json = serde_json::to_value(&req).expect("serialize req");
        assert_eq!(
            req_json,
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "0.1"
                }
            })
        );

        // Cancel notification shape
        let cancel = NotificationMsg {
            jsonrpc: "2.0",
            method: "$/cancelRequest",
            params: CancelParams { id: 1 },
        };
        let cancel_json = serde_json::to_value(&cancel).expect("serialize cancel");
        assert_eq!(
            cancel_json,
            json!({
                "jsonrpc": "2.0",
                "method": "$/cancelRequest",
                "params": {
                    "id": 1
                }
            })
        );

        // Publish notification shape
        let publish = NotificationMsg {
            jsonrpc: "2.0",
            method: "kernel/publish",
            params: PublishParams {
                event_type: "sample.echoed".to_owned(),
                data: json!({ "text": "hello" }),
            },
        };
        let publish_json = serde_json::to_value(&publish).expect("serialize publish");
        assert_eq!(
            publish_json,
            json!({
                "jsonrpc": "2.0",
                "method": "kernel/publish",
                "params": {
                    "type": "sample.echoed",
                    "data": { "text": "hello" }
                }
            })
        );

        // Invoke request shape
        let invoke = RequestMsg {
            jsonrpc: "2.0",
            id: 2,
            method: "mod/invoke",
            params: InvokeParams {
                operation: "sample.echo".to_owned(),
                payload: json!({ "text": "hi" }),
            },
        };
        let invoke_json = serde_json::to_value(&invoke).expect("serialize invoke");
        assert_eq!(
            invoke_json,
            json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "mod/invoke",
                "params": {
                    "operation": "sample.echo",
                    "payload": { "text": "hi" }
                }
            })
        );

        // Error response shape
        let err_resp = ResponseErrorMsg {
            jsonrpc: "2.0",
            id: Some(2),
            error: RpcError::parse_error("syntax error"),
        };
        let err_json = serde_json::to_value(&err_resp).expect("serialize error");
        assert_eq!(
            err_json,
            json!({
                "jsonrpc": "2.0",
                "id": 2,
                "error": {
                    "code": -32700,
                    "message": "syntax error"
                }
            })
        );
    }
}
