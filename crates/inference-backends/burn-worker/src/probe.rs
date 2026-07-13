//! Authoritative device profile and identity construction for the Burn worker.
//!
//! MB03 mandates that exactly one compute backend is compiled in (wgpu or flex).
//! This module reports the active backend kind, device label, and capabilities
//! in the worker handshake. A device probe failure must keep the process
//! unready rather than silently falling back.

use reimagine_backend_worker_protocol::{
    BackendInstanceId, WorkerIdentity, WorkerIncarnationId, WorkerInstallationId,
    WorkerInstanceProfile, WorkerProfile,
};
use reimagine_inference_burn::BurnBackend;

/// Build the worker's identity and capability profile from the running backend.
///
/// The identity reports the currently active feature/variant so the host
/// adapter can verify the launch expectation against the compiled binary.
pub fn build(backend: &BurnBackend) -> (WorkerIdentity, WorkerProfile) {
    let backend_instance_id =
        BackendInstanceId(backend.backend_instance().as_str().to_string());

    let identity = WorkerIdentity {
        backend_instance_id: backend_instance_id.clone(),
        installation_id: WorkerInstallationId(
            std::env::var("REIMAGINE_INSTALLATION_ID")
                .unwrap_or_else(|_| "dev".to_string()),
        ),
        incarnation_id: WorkerIncarnationId(format!("inc-{}", std::process::id())),
        worker_version: env!("CARGO_PKG_VERSION").to_string(),
        backend_kind: "burn".to_string(),
        target: format!(
            "{}-{}",
            std::env::consts::ARCH,
            std::env::consts::OS
        ),
        manifest_digest: std::env::var("REIMAGINE_MANIFEST_DIGEST")
            .unwrap_or_else(|_| "dev".to_string()),
    };

    let instance_profile = WorkerInstanceProfile {
        backend_instance_id,
        device_label: backend.device_label().to_string(),
        capabilities: vec![
            "model.load_bundle".to_string(),
            "latent.create_empty".to_string(),
            "text.encode".to_string(),
            "diffusion.sample".to_string(),
            "latent.decode".to_string(),
            "image.save".to_string(),
            "image.preview".to_string(),
        ],
        operation_options: serde_json::json!({}),
    };

    let profile = WorkerProfile {
        instances: vec![instance_profile],
    };

    (identity, profile)
}