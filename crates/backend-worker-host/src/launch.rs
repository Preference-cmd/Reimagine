use std::path::PathBuf;
use std::time::Duration;

use reimagine_backend_worker_protocol::{BackendInstanceId, ProtocolRange, WorkerInstallationId};
use serde::{Deserialize, Serialize};

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
    pub startup_timeout: Duration,
    pub request_timeout: Duration,
    pub shutdown_timeout: Duration,
    pub maximum_frame_bytes: u32,
    pub maximum_stderr_bytes: usize,
}

impl Default for WorkerLimits {
    fn default() -> Self {
        Self {
            startup_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(30),
            shutdown_timeout: Duration::from_secs(3),
            maximum_frame_bytes: 1024 * 1024,
            maximum_stderr_bytes: 64 * 1024,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkerLaunchSpec {
    pub executable: PathBuf,
    pub expected: ExpectedWorkerIdentity,
    pub supported_protocols: ProtocolRange,
    pub limits: WorkerLimits,
    pub environment: Vec<(String, String)>,
}
