//! Conversions between the `reimagine.worker.v2` protobuf messages and
//! the shared protocol types in `reimagine-backend-worker-protocol`.

use reimagine_backend_worker_protocol::{
    BackendExecutionError, BackendInstanceId, ProgressFrame, ProtocolVersion, TerminalOutcome,
    WorkerHello, WorkerIdentity, WorkerIncarnationId, WorkerInstallationId, WorkerInstanceProfile,
    WorkerProfile,
};
use serde_json::Value;

use crate::proto;

/// Map a `HelloResponse` onto the shared hello type.
#[must_use]
pub fn hello_from_proto(response: proto::HelloResponse) -> Option<WorkerHello> {
    let identity = response.identity?;
    let profile = profile_from_proto(response.profile?);
    Some(WorkerHello {
        protocol: ProtocolVersion(u16::try_from(response.protocol_version).ok()?),
        identity: WorkerIdentity {
            backend_instance_id: BackendInstanceId(identity.backend_instance_id),
            installation_id: WorkerInstallationId(identity.installation_id),
            incarnation_id: WorkerIncarnationId(identity.incarnation_id),
            worker_version: identity.worker_version,
            backend_kind: identity.backend_kind,
            target: identity.target,
            manifest_digest: identity.manifest_digest,
        },
        profile,
    })
}

/// Map the shared hello type onto a `HelloResponse`.
#[must_use]
pub fn hello_to_proto(hello: &WorkerHello) -> proto::HelloResponse {
    proto::HelloResponse {
        protocol_version: u32::from(hello.protocol.0),
        identity: Some(proto::WorkerIdentity {
            backend_instance_id: hello.identity.backend_instance_id.0.clone(),
            installation_id: hello.identity.installation_id.0.clone(),
            incarnation_id: hello.identity.incarnation_id.0.clone(),
            worker_version: hello.identity.worker_version.clone(),
            backend_kind: hello.identity.backend_kind.clone(),
            target: hello.identity.target.clone(),
            manifest_digest: hello.identity.manifest_digest.clone(),
        }),
        profile: Some(profile_to_proto(&hello.profile)),
    }
}

#[must_use]
pub fn profile_from_proto(profile: proto::WorkerProfile) -> WorkerProfile {
    WorkerProfile {
        instances: profile
            .instances
            .into_iter()
            .map(|instance| WorkerInstanceProfile {
                backend_instance_id: BackendInstanceId(instance.backend_instance_id),
                device_label: instance.device_label,
                capabilities: instance.capabilities,
                operation_options: decode_json(&instance.operation_options),
            })
            .collect(),
    }
}

#[must_use]
pub fn profile_to_proto(profile: &WorkerProfile) -> proto::WorkerProfile {
    proto::WorkerProfile {
        instances: profile
            .instances
            .iter()
            .map(|instance| proto::WorkerInstanceProfile {
                backend_instance_id: instance.backend_instance_id.0.clone(),
                device_label: instance.device_label.clone(),
                capabilities: instance.capabilities.clone(),
                operation_options: encode_json(&instance.operation_options),
            })
            .collect(),
    }
}

#[must_use]
pub fn progress_from_proto(progress: proto::Progress) -> ProgressFrame {
    ProgressFrame {
        sequence: progress.sequence,
        completed: progress.completed,
        total: progress.total,
        message: progress.message,
    }
}

/// Map the final `ExecuteEvent` outcome onto the shared terminal type.
///
/// `None` means the event was a progress frame or carried no outcome.
#[must_use]
pub fn outcome_from_proto(result: proto::ExecuteResult) -> Option<TerminalOutcome> {
    match result.outcome? {
        proto::execute_result::Outcome::Success(success) => Some(TerminalOutcome::Success {
            output: decode_json(&success.output),
        }),
        proto::execute_result::Outcome::Cancelled(_) => Some(TerminalOutcome::Cancelled),
        proto::execute_result::Outcome::BackendError(error) => {
            Some(TerminalOutcome::BackendError {
                error: BackendExecutionError {
                    code: error.code,
                    message: error.message,
                    retryable: error.retryable,
                },
            })
        }
    }
}

/// Map the shared terminal type onto the final `ExecuteEvent` outcome.
#[must_use]
pub fn outcome_to_proto(outcome: &TerminalOutcome) -> proto::ExecuteResult {
    let outcome = match outcome {
        TerminalOutcome::Success { output } => {
            proto::execute_result::Outcome::Success(proto::Success {
                output: encode_json(output),
            })
        }
        TerminalOutcome::Cancelled => {
            proto::execute_result::Outcome::Cancelled(proto::Cancelled {})
        }
        TerminalOutcome::BackendError { error } => {
            proto::execute_result::Outcome::BackendError(proto::BackendError {
                code: error.code.clone(),
                message: error.message.clone(),
                retryable: error.retryable,
            })
        }
    };
    proto::ExecuteResult {
        outcome: Some(outcome),
    }
}

fn encode_json(value: &Value) -> Vec<u8> {
    serde_json::to_vec(value).unwrap_or_default()
}

fn decode_json(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).unwrap_or(Value::Null)
}
