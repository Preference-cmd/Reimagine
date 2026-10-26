use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use reimagine_backend_worker_host::{
    ExpectedWorkerIdentity, WorkerLaunchSpec, WorkerLimits, WorkerProcessState, WorkerSupervisor,
};
use reimagine_backend_worker_protocol::{
    BackendInstanceId, ProtocolRange, TerminalOutcome, WorkerInstallationId,
};

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
        supported_protocols: ProtocolRange::new(2, 2),
        limits: WorkerLimits {
            startup_timeout: Duration::from_secs(2),
            ..WorkerLimits::default()
        },
        environment: Vec::new(),
    }
}

fn launch_spec_with_mode(mode: &str) -> WorkerLaunchSpec {
    let mut spec = launch_spec();
    spec.environment
        .push(("FAKE_WORKER_MODE".to_owned(), mode.to_owned()));
    spec
}

#[tokio::test]
async fn construction_does_not_start_and_start_returns_validated_hello() {
    let supervisor = WorkerSupervisor::new(launch_spec());
    assert_eq!(supervisor.state(), WorkerProcessState::Stopped);

    let started = supervisor.start().await.unwrap();
    assert_eq!(started.hello.identity.backend_kind, "fake");
    assert_eq!(
        started.hello.identity.backend_instance_id,
        BackendInstanceId::from("fake:cpu:default")
    );
    assert!(!started.hello.profile.instances.is_empty());
    started.shutdown().await.unwrap();
}

#[tokio::test]
async fn one_client_multiplexes_concurrent_requests_and_progress() {
    let supervisor = WorkerSupervisor::new(launch_spec());
    let worker = Arc::new(supervisor.start().await.unwrap());
    let slow_worker = Arc::clone(&worker);
    let slow = tokio::spawn(async move {
        slow_worker
            .request("delay", serde_json::json!({ "milliseconds": 150 }))
            .await
    });
    let fast_worker = Arc::clone(&worker);
    let fast = tokio::spawn(async move {
        fast_worker
            .request("echo", serde_json::json!({ "value": 7 }))
            .await
    });

    let fast = tokio::time::timeout(Duration::from_millis(100), fast)
        .await
        .expect("fast request was blocked behind slow request")
        .unwrap()
        .unwrap();
    assert_eq!(
        fast.terminal,
        TerminalOutcome::Success {
            output: serde_json::json!({ "value": 7 })
        }
    );
    assert!(
        !slow.is_finished(),
        "slow request should still be pending when fast completes"
    );
    slow.await.unwrap().unwrap();

    let progress = worker
        .request("progress", serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(
        progress
            .progress
            .iter()
            .map(|frame| frame.sequence)
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );

    worker.shutdown().await.unwrap();
}

#[tokio::test]
async fn cancellation_is_idempotent_and_reports_one_cancelled_terminal() {
    let supervisor = WorkerSupervisor::new(launch_spec());
    let worker = supervisor.start().await.unwrap();
    let request = worker
        .begin_request("delay", serde_json::json!({ "milliseconds": 500 }))
        .await
        .unwrap();

    request.cancel().await.unwrap();
    request.cancel().await.unwrap();
    let result = request.finish().await.unwrap();
    assert_eq!(result.terminal, TerminalOutcome::Cancelled);

    worker.shutdown().await.unwrap();
}

#[tokio::test]
async fn startup_timeout_identity_and_protocol_mismatches_are_distinct() {
    let mut hanging = launch_spec_with_mode("hang");
    hanging.limits.startup_timeout = Duration::from_millis(500);
    let error = match WorkerSupervisor::new(hanging).start().await {
        Ok(_) => panic!("hang mode must not start"),
        Err(error) => error,
    };
    assert!(
        matches!(
            error,
            reimagine_backend_worker_host::WorkerHostError::StartupTimeout
        ),
        "expected StartupTimeout, got: {error}"
    );

    assert!(matches!(
        WorkerSupervisor::new(launch_spec_with_mode("wrong_identity"))
            .start()
            .await,
        Err(
            reimagine_backend_worker_host::WorkerHostError::IdentityMismatch {
                field: "manifest_digest",
                ..
            }
        )
    ));

    assert!(matches!(
        WorkerSupervisor::new(launch_spec_with_mode("wrong_protocol"))
            .start()
            .await,
        Err(
            reimagine_backend_worker_host::WorkerHostError::IdentityMismatch {
                field: "protocol",
                ..
            }
        )
    ));
}

#[tokio::test]
async fn unexpected_stdout_is_rejected() {
    let mut spec = launch_spec_with_mode("stdout_noise");
    spec.limits.startup_timeout = Duration::from_secs(2);
    let error = match WorkerSupervisor::new(spec).start().await {
        Ok(_) => panic!("unexpected stdout must fail startup"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        reimagine_backend_worker_host::WorkerHostError::Transport(_)
    ));
    assert!(
        error.to_string().contains("stdout"),
        "unexpected stdout must be reported: {error}"
    );
}

#[tokio::test]
async fn stderr_is_drained_and_bounded_without_blocking_hello() {
    let mut spec = launch_spec_with_mode("stderr_flood");
    spec.limits.maximum_stderr_bytes = 1024;
    let worker = WorkerSupervisor::new(spec).start().await.unwrap();
    // The drain task runs concurrently with startup; poll until the
    // bounded tail is full.
    let tail = {
        let mut tail = worker.stderr_tail().await;
        for _ in 0..100 {
            if tail.len() == 1024 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
            tail = worker.stderr_tail().await;
        }
        tail
    };
    assert_eq!(tail.len(), 1024);
    assert!(tail.iter().all(|byte| *byte == b'x'));

    worker.shutdown().await.unwrap();
}

#[tokio::test]
async fn crash_fails_every_pending_request_and_marks_supervisor_failed() {
    let supervisor = Arc::new(WorkerSupervisor::new(launch_spec()));
    let worker = supervisor.start().await.unwrap();
    let delayed = worker
        .begin_request("delay", serde_json::json!({ "milliseconds": 500 }))
        .await
        .unwrap();
    // The worker may die before or after the Execute call is
    // established; either way the request reports a lost transport.
    let crash = match worker.begin_request("crash", serde_json::json!({})).await {
        Ok(handle) => handle.finish().await.map(|_| ()),
        Err(error) => Err(error),
    };
    assert!(matches!(
        crash,
        Err(reimagine_backend_worker_host::WorkerHostError::TransportLost { .. })
    ));
    assert!(matches!(
        delayed.finish().await,
        Err(reimagine_backend_worker_host::WorkerHostError::TransportLost { .. })
    ));
    assert_eq!(supervisor.state(), WorkerProcessState::Failed);
}

#[tokio::test]
async fn graceful_shutdown_acknowledges_reaps_and_stops() {
    let supervisor = WorkerSupervisor::new(launch_spec());
    let worker = supervisor.start().await.unwrap();
    worker.shutdown().await.unwrap();
    assert_eq!(supervisor.state(), WorkerProcessState::Stopped);
}

#[tokio::test]
async fn ready_supervisor_rejects_a_second_process_start() {
    let supervisor = WorkerSupervisor::new(launch_spec());
    let worker = supervisor.start().await.unwrap();
    assert!(matches!(
        supervisor.start().await,
        Err(reimagine_backend_worker_host::WorkerHostError::AlreadyStarted)
    ));
    worker.shutdown().await.unwrap();
}

#[tokio::test]
async fn restart_uses_new_incarnation_and_old_client_stays_stale() {
    let mut spec = launch_spec();
    spec.limits.request_timeout = Duration::from_millis(100);
    let supervisor = WorkerSupervisor::new(spec);
    let old_worker = supervisor.start().await.unwrap();
    let old_incarnation = old_worker.hello.identity.incarnation_id.clone();
    assert!(
        old_worker
            .request("crash", serde_json::json!({}))
            .await
            .is_err()
    );
    assert_eq!(supervisor.state(), WorkerProcessState::Failed);

    let new_worker = supervisor.start().await.unwrap();
    assert_ne!(new_worker.hello.identity.incarnation_id, old_incarnation);
    assert!(matches!(
        old_worker.request("echo", serde_json::json!({})).await,
        Err(reimagine_backend_worker_host::WorkerHostError::TransportLost { .. })
    ));
    assert!(
        new_worker
            .request("echo", serde_json::json!({}))
            .await
            .is_ok()
    );
    new_worker.shutdown().await.unwrap();
}

#[tokio::test]
async fn health_and_cleanup_controls_succeed() {
    let supervisor = WorkerSupervisor::new(launch_spec());
    let worker = supervisor.start().await.unwrap();

    let health = worker.health().await.unwrap();
    assert!(health.healthy);
    assert_eq!(health.message, None);

    let released = worker
        .cleanup(
            Some("run-1".to_owned()),
            vec!["latent-1".to_owned(), "latent-2".to_owned()],
        )
        .await
        .unwrap();
    assert_eq!(released, 2);

    worker.shutdown().await.unwrap();
}

#[tokio::test]
async fn shutdown_timeout_kills_worker_and_invalidates_client() {
    let mut spec = launch_spec_with_mode("ignore_shutdown");
    spec.limits.shutdown_timeout = Duration::from_millis(50);
    let supervisor = WorkerSupervisor::new(spec);
    let worker = supervisor.start().await.unwrap();

    assert!(matches!(
        worker.shutdown().await,
        Err(reimagine_backend_worker_host::WorkerHostError::ShutdownTimeout)
    ));
    assert_eq!(supervisor.state(), WorkerProcessState::Stopped);
    assert!(matches!(
        worker.request("echo", serde_json::json!({})).await,
        Err(reimagine_backend_worker_host::WorkerHostError::TransportLost { .. })
    ));
}

#[tokio::test]
async fn dropping_last_worker_handle_stops_owned_process_and_removes_socket_dir() {
    let supervisor = WorkerSupervisor::new(launch_spec());
    let worker = supervisor.start().await.unwrap();
    let endpoint = worker.transport_endpoint();
    let socket_path = PathBuf::from(
        endpoint
            .strip_prefix("unix://")
            .expect("local workers expose a unix:// endpoint"),
    );
    assert!(socket_path.exists(), "socket file must exist while running");
    let socket_dir = socket_path
        .parent()
        .expect("socket lives in its own directory")
        .to_path_buf();

    drop(worker);

    assert_eq!(supervisor.state(), WorkerProcessState::Stopped);
    for _ in 0..100 {
        if !socket_path.exists() && !socket_dir.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        !socket_path.exists(),
        "dropping the worker must remove its socket file"
    );
    assert!(
        !socket_dir.exists(),
        "dropping the worker must remove its private socket directory"
    );

    let restarted = supervisor.start().await.unwrap();
    restarted.shutdown().await.unwrap();
}

#[tokio::test]
async fn startup_rejects_profile_without_expected_instance() {
    assert!(matches!(
        WorkerSupervisor::new(launch_spec_with_mode("wrong_profile"))
            .start()
            .await,
        Err(
            reimagine_backend_worker_host::WorkerHostError::IdentityMismatch {
                field: "profile.backend_instance_id",
                ..
            }
        )
    ));
}

#[tokio::test]
async fn dropping_stale_client_does_not_overwrite_new_worker_state() {
    let supervisor = WorkerSupervisor::new(launch_spec());
    let stale_worker = supervisor.start().await.unwrap();
    assert!(
        stale_worker
            .request("crash", serde_json::json!({}))
            .await
            .is_err()
    );
    let current_worker = supervisor.start().await.unwrap();

    drop(stale_worker);
    assert_eq!(supervisor.state(), WorkerProcessState::Ready);
    assert!(
        current_worker
            .request("echo", serde_json::json!({}))
            .await
            .is_ok()
    );
    current_worker.shutdown().await.unwrap();
}

#[tokio::test]
async fn connect_failure_reports_transport_error_and_stays_stopped() {
    let mut spec = launch_spec();
    spec.executable = PathBuf::from("/nonexistent/reimagine-fake-worker");
    let supervisor = WorkerSupervisor::new(spec);
    assert!(matches!(
        supervisor.start().await,
        Err(reimagine_backend_worker_host::WorkerHostError::Transport(_))
    ));
    assert_eq!(supervisor.state(), WorkerProcessState::Stopped);
}
