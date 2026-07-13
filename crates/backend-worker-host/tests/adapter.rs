use std::path::PathBuf;
use std::sync::Arc;

use reimagine_backend_worker_host::{
    ExpectedWorkerIdentity, ProcessInferenceBackend, WorkerLaunchSpec, WorkerLimits,
    WorkerSupervisor,
};
use reimagine_backend_worker_protocol::{BackendInstanceId, ProtocolRange, WorkerInstallationId};
use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};
use reimagine_inference::{CreateEmptyLatentRequest, InferenceBackend, InferenceCapability};

fn launch_spec() -> WorkerLaunchSpec {
    WorkerLaunchSpec {
        executable: PathBuf::from(env!("CARGO_BIN_EXE_fake-backend-worker-host-fixture")),
        expected: ExpectedWorkerIdentity {
            backend_instance_id: BackendInstanceId::from("fake:cpu:default"),
            installation_id: WorkerInstallationId::from("fake-installation"),
            backend_kind: "fake".to_owned(),
            target: std::env::consts::ARCH.to_owned(),
            manifest_digest: "test-manifest".to_owned(),
        },
        supported_protocols: ProtocolRange::new(1, 1),
        limits: WorkerLimits::default(),
        environment: Vec::new(),
    }
}

#[tokio::test]
async fn adapter_advertises_and_maps_only_create_empty_latent() {
    let worker = Arc::new(WorkerSupervisor::new(launch_spec()).start().await.unwrap());
    let backend = ProcessInferenceBackend::new(worker);
    assert!(
        backend
            .capabilities()
            .supports_capability(InferenceCapability::CreateEmptyLatent)
    );
    assert!(
        !backend
            .capabilities()
            .supports_capability(InferenceCapability::TextEncode)
    );

    let response = backend
        .create_empty_latent(CreateEmptyLatentRequest::new(
            64,
            64,
            1,
            RunId::new("run-1"),
            WorkflowId::new("workflow-1"),
            WorkflowVersion::new(1),
            NodeId::new("node-1"),
        ))
        .await
        .unwrap();
    let latent = response.into_latent();
    assert_eq!(latent.width(), 64);
    assert_eq!(latent.height(), 64);
    assert_eq!(
        latent.payload().backend_instance().as_str(),
        "fake:cpu:default"
    );
}

#[tokio::test]
async fn adapter_does_not_advertise_or_invoke_undeclared_operation() {
    let mut spec = launch_spec();
    spec.environment.push((
        "FAKE_WORKER_MODE".to_owned(),
        "profile_without_latent".to_owned(),
    ));
    let worker = Arc::new(WorkerSupervisor::new(spec).start().await.unwrap());
    let backend = ProcessInferenceBackend::new(worker);

    assert!(
        !backend
            .capabilities()
            .supports_capability(InferenceCapability::CreateEmptyLatent)
    );
    assert!(matches!(
        backend
            .create_empty_latent(CreateEmptyLatentRequest::new(
                64,
                64,
                1,
                RunId::new("run-2"),
                WorkflowId::new("workflow-2"),
                WorkflowVersion::new(1),
                NodeId::new("node-2"),
            ))
            .await,
        Err(reimagine_inference::InferenceError::BackendNotImplemented {
            capability: InferenceCapability::CreateEmptyLatent,
            ..
        })
    ));
}
