#![cfg(feature = "aigc")]

//! Integration tests for runtime AIGC failure isolation.
//!
//! Verifies that:
//! 1. A failed inference bootstrap does not abort `WorkspaceHostBuilder::build`.
//! 2. Core workspace features (projects, boards, workflows, documents, model management) continue working.
//! 3. `WorkspaceHost::aigc_status()` reports unavailable with the real bootstrap failure reason.
//! 4. Gated operations depending on inference runtime (worker switch, workflow execution with nodes)
//!    return `AppHostError::AigcUnavailable` with error code `"aigc_unavailable"`.
//! 5. `AppHostErrorCode::AigcUnavailable` wire string is `"aigc_unavailable"`.

use std::sync::Arc;
use std::time::Duration;

use reimagine_app_host::WorkspaceScope;
use reimagine_app_host::{
    AigcStatus, AppHostError, AppHostErrorCode, EmptyWorkerInventoryProvider, RunWorkflowRequest,
    WorkerBackendCandidate, WorkspaceHostBuilder,
};
use reimagine_backend_worker_host::{ExpectedWorkerIdentity, WorkerLaunchSpec, WorkerLimits};
use reimagine_backend_worker_protocol::{
    BackendInstanceId, ProtocolRange, WorkerInstallationId, WorkerInstanceProfile,
};
use reimagine_config::{InferenceBackendConfig, WorkerEndpointConfig};
use reimagine_core::model::{DocumentVersion, NodeId, ProjectId, SlotId, WorkflowId};
use reimagine_core::readiness::{RunTarget, RunTargetSelection};
use reimagine_core::workflow::{Workflow, WorkflowNode};
use reimagine_nodes::BUILTIN_CHECKPOINT_LOADER;
use reimagine_runtime::VecRunEventSink;

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-aigc-isolation-{prefix}-{nonce}"))
}

/// An inference backend configuration with a deliberately invalid worker transport.
///
/// This triggers a real, deterministic bootstrap failure inside `build_topology` ->
/// `ConfigDiscovery::from_config` without requiring test-only hooks in production code.
fn failing_inference_backend_config() -> InferenceBackendConfig {
    InferenceBackendConfig {
        workers: vec![WorkerEndpointConfig {
            id: "invalid-worker".to_owned(),
            transport: "carrier-pigeon".to_owned(),
            address: "quic://127.0.0.1:9999".to_owned(),
            capabilities: Vec::new(),
            device_label: String::new(),
            fingerprint: None,
        }],
        ..InferenceBackendConfig::default()
    }
}

fn dummy_switch_candidate() -> WorkerBackendCandidate {
    let launch = WorkerLaunchSpec {
        executable: std::path::PathBuf::from("/bin/true"),
        expected: ExpectedWorkerIdentity {
            backend_instance_id: BackendInstanceId("burn:wgpu:default".to_owned()),
            installation_id: WorkerInstallationId("dummy-inst".to_owned()),
            backend_kind: "burn".to_owned(),
            target: "test".to_owned(),
            manifest_digest: "seed".to_owned(),
        },
        supported_protocols: ProtocolRange::new(1, 1),
        limits: WorkerLimits::default(),
        environment: Vec::new(),
        transport: Default::default(),
    };
    let profile = WorkerInstanceProfile {
        backend_instance_id: BackendInstanceId("burn:wgpu:default".to_owned()),
        device_label: "wgpu:default".to_owned(),
        capabilities: vec!["model.load_bundle".to_owned()],
        operation_options: serde_json::json!({}),
    };
    WorkerBackendCandidate::try_new(launch, profile).expect("valid candidate")
}

#[test]
fn error_code_string_is_aigc_unavailable() {
    assert_eq!(
        AppHostErrorCode::AigcUnavailable.as_str(),
        "aigc_unavailable"
    );

    let error = AppHostError::AigcUnavailable {
        reason: "transport carrier-pigeon unsupported".to_owned(),
    };
    assert_eq!(error.code(), AppHostErrorCode::AigcUnavailable);
    assert_eq!(error.code().as_str(), "aigc_unavailable");
    let details = error.details().expect("details must be present");
    assert_eq!(
        details.get("reason").and_then(|v| v.as_str()),
        Some("transport carrier-pigeon unsupported")
    );
}

#[tokio::test]
async fn workspace_builds_and_functions_normally_when_aigc_bootstrap_fails() {
    let base = temp_dir("degraded-build");
    let config = failing_inference_backend_config();

    let workspace = WorkspaceHostBuilder::new(WorkspaceScope::new("isolation-test"), &base)
        .backend_config(config)
        .event_sink(Arc::new(VecRunEventSink::new()))
        .worker_inventory(Arc::new(EmptyWorkerInventoryProvider))
        .build()
        .await
        .expect("workspace must build successfully even when inference bootstrap fails");

    // 1. Verify AigcStatus
    assert!(!workspace.aigc_status().is_available());
    assert!(matches!(
        workspace.aigc_status(),
        AigcStatus::Unavailable { .. }
    ));
    let reason = workspace
        .aigc_status()
        .unavailable_reason()
        .expect("reason must be present");
    assert!(
        reason.contains("carrier-pigeon") || reason.contains("unknown transport"),
        "failure reason should describe the bootstrap failure, got: {reason}"
    );

    // 2. Core project/board/workflow documents work
    let projects = workspace
        .project_service()
        .list_projects()
        .await
        .expect("list projects must succeed");
    assert!(
        projects.iter().any(|p| p.id().as_str() == "default"),
        "default project must be discoverable"
    );

    let board = workspace
        .board_service()
        .load_or_create(&ProjectId::new("default"))
        .await
        .expect("load board must succeed");
    assert_eq!(board.project_id().as_str(), "default");

    let wf_id = WorkflowId::new("test-wf");
    let initial_wf = Workflow::new(wf_id.as_str(), DocumentVersion::new(1));
    workspace
        .workflow_service()
        .save_workflow_snapshot(&initial_wf)
        .await
        .expect("save workflow must succeed");

    let loaded_wf = workspace
        .workflow_service()
        .load_workflow(&wf_id)
        .await
        .expect("load workflow must succeed");
    assert_eq!(loaded_wf.as_str(), "test-wf");

    // 3. Model management does not depend on bootstrap and continues to function
    let models = workspace
        .model_service()
        .list_models()
        .await
        .expect("list models must succeed");
    assert!(models.is_empty() || !models.is_empty()); // Just ensure it returns Ok

    // 4. Gated operations return AppHostError::AigcUnavailable
    let selected_err = workspace
        .selected_worker()
        .await
        .expect_err("selected_worker must fail when AIGC unavailable");
    assert_eq!(selected_err.code().as_str(), "aigc_unavailable");

    let switch_err = workspace
        .drain_and_switch_worker(dummy_switch_candidate(), Duration::from_secs(1))
        .await
        .expect_err("drain_and_switch_worker must fail when AIGC unavailable");
    assert_eq!(switch_err.code().as_str(), "aigc_unavailable");

    let cancel_err = workspace
        .cancel_and_switch_worker(dummy_switch_candidate(), Duration::from_secs(1))
        .await
        .expect_err("cancel_and_switch_worker must fail when AIGC unavailable");
    assert_eq!(cancel_err.code().as_str(), "aigc_unavailable");

    // Topology and discovery accessors remain Ungated (they return Option)
    assert!(workspace.topology().is_none());
    assert!(workspace.discovery().is_none());

    // 5. Workflow execution that needs an inference executor is gated
    let wf_with_node_id = WorkflowId::new("wf-with-node");
    let wf_with_node = Workflow::new(wf_with_node_id.as_str(), DocumentVersion::new(1))
        .with_node(WorkflowNode::new("loader", BUILTIN_CHECKPOINT_LOADER));
    workspace
        .workflow_service()
        .save_workflow_snapshot(&wf_with_node)
        .await
        .expect("save workflow with node must succeed");
    workspace
        .workflow_service()
        .load_workflow(&wf_with_node_id)
        .await
        .expect("load workflow with node must succeed");

    let target_selection = RunTargetSelection::ExplicitTargets(vec![RunTarget::NodeOutput {
        node_id: NodeId::new("loader"),
        slot_id: SlotId::new("model"),
    }]);
    let run_request = RunWorkflowRequest::new(wf_with_node_id, target_selection);
    let run_err = workspace
        .run_workflow(run_request)
        .await
        .expect_err("run_workflow must return aigc_unavailable when workflow has nodes");
    assert_eq!(run_err.code().as_str(), "aigc_unavailable");

    let _ = tokio::fs::remove_dir_all(&base).await;
}
