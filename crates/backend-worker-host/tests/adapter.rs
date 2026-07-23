use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use reimagine_backend_worker_host::{
    ExpectedWorkerIdentity, ProcessInferenceBackend, WorkerLaunchSpec, WorkerLimits,
    WorkerSupervisor,
};
use reimagine_backend_worker_protocol::{BackendInstanceId, ProtocolRange, WorkerInstallationId};
use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};
use reimagine_inference::{CreateEmptyLatentRequest, InferenceBackend, InferenceCapability};
use reimagine_inference::{
    InferenceError, InferenceInvocation, NoopInferenceProgressSink, NoopNodeCancellation,
};
use reimagine_inference::{InferenceProgress, InferenceProgressSink};

#[derive(Default)]
struct RecordingProgressSink(Mutex<Vec<InferenceProgress>>);

impl InferenceProgressSink for RecordingProgressSink {
    fn report(&self, progress: InferenceProgress) {
        self.0
            .lock()
            .expect("progress sink poisoned")
            .push(progress);
    }
}

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

#[tokio::test]
async fn invocation_cancellation_reaches_worker_and_wins_once() {
    let mut spec = launch_spec();
    spec.environment
        .push(("FAKE_LATENT_DELAY_MS".to_owned(), "250".to_owned()));
    let worker = Arc::new(WorkerSupervisor::new(spec).start().await.unwrap());
    let backend = Arc::new(ProcessInferenceBackend::new(worker));
    let cancellation = Arc::new(NoopNodeCancellation::new());
    let invocation = InferenceInvocation::new(
        RunId::new("run-cancel"),
        NodeId::new("node-cancel"),
        None,
        cancellation.clone(),
        Arc::new(NoopInferenceProgressSink),
    );

    let request_task = {
        let backend = Arc::clone(&backend);
        tokio::spawn(async move {
            backend
                .create_empty_latent_with_invocation(
                    &invocation,
                    CreateEmptyLatentRequest::new(
                        64,
                        64,
                        1,
                        RunId::new("run-cancel"),
                        WorkflowId::new("workflow-cancel"),
                        WorkflowVersion::new(1),
                        NodeId::new("node-cancel"),
                    ),
                )
                .await
        })
    };

    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    cancellation.cancel();

    assert!(matches!(
        request_task.await.unwrap(),
        Err(InferenceError::Cancelled)
    ));
}

#[tokio::test]
async fn backend_error_terminal_wins_over_late_cancel_ack_once() {
    let mut spec = launch_spec();
    spec.environment
        .push(("FAKE_LATENT_DELAY_MS".to_owned(), "50".to_owned()));
    spec.environment.push((
        "FAKE_LATENT_TERMINAL".to_owned(),
        "backend_error".to_owned(),
    ));
    let worker = Arc::new(WorkerSupervisor::new(spec).start().await.unwrap());
    let backend = Arc::new(ProcessInferenceBackend::new(worker));
    let cancellation = Arc::new(NoopNodeCancellation::new());
    let invocation = InferenceInvocation::new(
        RunId::new("run-backend-error-race"),
        NodeId::new("node-backend-error-race"),
        None,
        cancellation.clone(),
        Arc::new(NoopInferenceProgressSink),
    );
    let request_task = {
        let backend = backend.clone();
        tokio::spawn(async move {
            backend
                .create_empty_latent_with_invocation(
                    &invocation,
                    CreateEmptyLatentRequest::new(
                        64,
                        64,
                        1,
                        RunId::new("run-backend-error-race"),
                        WorkflowId::new("workflow-race"),
                        WorkflowVersion::new(1),
                        NodeId::new("node-backend-error-race"),
                    ),
                )
                .await
        })
    };

    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    cancellation.cancel();

    let error = request_task.await.unwrap().expect_err("backend error wins");
    assert!(matches!(
        error,
        InferenceError::BackendExecutionFailed { .. }
    ));
    assert!(error.to_string().contains("forced_backend_error"));
}

#[tokio::test]
async fn success_terminal_wins_over_late_cancel_ack_once() {
    let mut spec = launch_spec();
    spec.environment
        .push(("FAKE_LATENT_DELAY_MS".to_owned(), "50".to_owned()));
    spec.environment
        .push(("FAKE_LATENT_TERMINAL".to_owned(), "success".to_owned()));
    let worker = Arc::new(WorkerSupervisor::new(spec).start().await.unwrap());
    let backend = Arc::new(ProcessInferenceBackend::new(worker));
    let cancellation = Arc::new(NoopNodeCancellation::new());
    let invocation = InferenceInvocation::new(
        RunId::new("run-success-race"),
        NodeId::new("node-success-race"),
        None,
        cancellation.clone(),
        Arc::new(NoopInferenceProgressSink),
    );
    let request_task = {
        let backend = backend.clone();
        tokio::spawn(async move {
            backend
                .create_empty_latent_with_invocation(
                    &invocation,
                    CreateEmptyLatentRequest::new(
                        64,
                        64,
                        1,
                        RunId::new("run-success-race"),
                        WorkflowId::new("workflow-race"),
                        WorkflowVersion::new(1),
                        NodeId::new("node-success-race"),
                    ),
                )
                .await
        })
    };

    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    cancellation.cancel();

    let response = request_task.await.unwrap().expect("success wins");
    assert_eq!(response.latent().width(), 64);
}

#[tokio::test]
async fn worker_crash_after_cancel_request_is_transport_lost_not_cancelled() {
    let mut spec = launch_spec();
    spec.environment
        .push(("FAKE_LATENT_DELAY_MS".to_owned(), "50".to_owned()));
    spec.environment
        .push(("FAKE_LATENT_TERMINAL".to_owned(), "crash".to_owned()));
    let worker = Arc::new(WorkerSupervisor::new(spec).start().await.unwrap());
    let backend = Arc::new(ProcessInferenceBackend::new(worker));
    let cancellation = Arc::new(NoopNodeCancellation::new());
    let invocation = InferenceInvocation::new(
        RunId::new("run-crash-race"),
        NodeId::new("node-crash-race"),
        None,
        cancellation.clone(),
        Arc::new(NoopInferenceProgressSink),
    );
    let request_task = {
        let backend = backend.clone();
        tokio::spawn(async move {
            backend
                .create_empty_latent_with_invocation(
                    &invocation,
                    CreateEmptyLatentRequest::new(
                        64,
                        64,
                        1,
                        RunId::new("run-crash-race"),
                        WorkflowId::new("workflow-race"),
                        WorkflowVersion::new(1),
                        NodeId::new("node-crash-race"),
                    ),
                )
                .await
        })
    };

    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    cancellation.cancel();

    let error = request_task.await.unwrap().expect_err("crash wins");
    assert!(matches!(
        error,
        InferenceError::BackendExecutionFailed { .. }
    ));
    assert!(error.to_string().contains("transport lost"));
}

#[tokio::test]
async fn invocation_progress_is_forwarded_before_worker_terminal() {
    let mut spec = launch_spec();
    spec.environment
        .push(("FAKE_LATENT_PROGRESS".to_owned(), "1".to_owned()));
    spec.environment
        .push(("FAKE_LATENT_DELAY_MS".to_owned(), "250".to_owned()));
    let worker = Arc::new(WorkerSupervisor::new(spec).start().await.unwrap());
    let backend = Arc::new(ProcessInferenceBackend::new(worker));
    let progress = Arc::new(RecordingProgressSink::default());
    let invocation = InferenceInvocation::new(
        RunId::new("run-progress"),
        NodeId::new("node-progress"),
        None,
        Arc::new(NoopNodeCancellation::new()),
        progress.clone(),
    );
    let request_task = {
        let backend = Arc::clone(&backend);
        tokio::spawn(async move {
            backend
                .create_empty_latent_with_invocation(
                    &invocation,
                    CreateEmptyLatentRequest::new(
                        64,
                        64,
                        1,
                        RunId::new("run-progress"),
                        WorkflowId::new("workflow-progress"),
                        WorkflowVersion::new(1),
                        NodeId::new("node-progress"),
                    ),
                )
                .await
        })
    };

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert!(
        !request_task.is_finished(),
        "worker terminal should still be pending"
    );
    assert_eq!(
        progress
            .0
            .lock()
            .expect("progress sink poisoned")
            .iter()
            .map(|progress| progress.sequence)
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    request_task.await.unwrap().expect("worker success");
}
