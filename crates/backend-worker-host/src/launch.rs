use std::path::PathBuf;
use std::time::Duration;

use reimagine_backend_worker_protocol::{
    BackendInstanceId, ProtocolRange, WORKER_PROTOCOL_VERSION, WorkerInstallationId,
};
use serde::{Deserialize, Serialize};

/// The protocol range this host accepts from workers. Everything the
/// host launches or registers must say hello inside this range.
pub const SUPPORTED_WORKER_PROTOCOLS: ProtocolRange = ProtocolRange {
    minimum: WORKER_PROTOCOL_VERSION,
    maximum: WORKER_PROTOCOL_VERSION,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExpectedWorkerIdentity {
    pub backend_instance_id: BackendInstanceId,
    pub installation_id: WorkerInstallationId,
    pub backend_kind: String,
    pub target: String,
    pub manifest_digest: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkerLimits {
    /// Bounds the whole startup: spawn, READY line, connect, and hello.
    pub startup_timeout: Duration,
    /// Bounds a single `Execute` call, progress included.
    pub request_timeout: Duration,
    /// Bounds the graceful `Shutdown` call before the process is killed.
    pub shutdown_timeout: Duration,
    pub maximum_stderr_bytes: usize,
}

impl Default for WorkerLimits {
    fn default() -> Self {
        Self {
            startup_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(30),
            shutdown_timeout: Duration::from_secs(3),
            maximum_stderr_bytes: 64 * 1024,
        }
    }
}

/// Everything the supervisor needs to launch a local worker process.
///
/// The supervisor itself chooses the private listen endpoint (Unix
/// socket inside a 0700 directory, or a named pipe on Windows) and the
/// per-launch bearer token; the worker receives both on its command
/// line and through the environment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkerLaunchSpec {
    pub executable: PathBuf,
    pub expected: ExpectedWorkerIdentity,
    /// Protocol versions this host accepts from the worker's `Hello`.
    pub supported_protocols: ProtocolRange,
    pub limits: WorkerLimits,
    pub environment: Vec<(String, String)>,
}
