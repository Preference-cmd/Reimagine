//! AR-38: real app-host workspace-tool JSON schemas.
//!
//! Verifies every registered workspace tool advertises a
//! field-accurate (non-placeholder) input schema, and that invalid
//! parameters are rejected before the handler runs.

use std::sync::Arc;

use reimagine_agent_harness::{
    AgentMode, AgentSessionId, PermissionSet, ToolContext, ToolErrorCode, ToolName, ToolPermission,
    WorkspaceScope,
};
use reimagine_app_host::WorkspaceHost;
use serde_json::json;

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-ar38-{prefix}-{nonce}"))
}

#[test]
fn workspace_tools_advertise_input_schemas() {
    let base = temp_dir("schemas");
    let host = Arc::new(WorkspaceHost::with_defaults(
        WorkspaceScope::new("ws-ar38"),
        &base,
    ));
    let registry = Arc::clone(host.agent_service().registry());
    let specs = registry.list();
    let ws_names = [
        "workflow.get",
        "workflow.preview_commands",
        "workflow.propose_commands",
        "workflow.apply_commands",
        "model.list",
        "model.resolve_ref",
        "diagnostics.for_workflow",
        "model.download",
    ];
    for name in ws_names {
        let spec = specs
            .iter()
            .find(|s| s.name().as_str() == name)
            .unwrap_or_else(|| panic!("missing tool {name}"));
        let input = spec.input_schema().expect("input schema present");
        // Non-placeholder: must declare properties (empty object is
        // still marked so a bare {type:object} placeholder is gone).
        assert!(
            input.get("properties").is_some(),
            "{name} input schema must declare properties"
        );
        assert!(
            spec.output_schema().is_some(),
            "{name} output schema present"
        );
    }
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn invalid_workflow_get_input_rejected_before_handler() {
    let base = temp_dir("invalid");
    let host = Arc::new(WorkspaceHost::with_defaults(
        WorkspaceScope::new("ws-ar38"),
        &base,
    ));
    let registry = Arc::clone(host.agent_service().registry());
    let ctx = ToolContext::new(
        WorkspaceScope::new("ws-ar38"),
        AgentSessionId::new("t-1"),
        AgentMode::Agent,
    )
    .with_permissions(PermissionSet::from_iter([ToolPermission::new(
        "workflow.read",
    )]));

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(registry.invoke(
        &ToolName::new("workflow.get"),
        &ctx,
        json!({ "wrong_field": "x" }),
    ));
    assert!(result.is_err(), "missing required field must be rejected");
    if let Err(err) = result {
        let code = match err {
            reimagine_agent_harness::ToolRegistryError::ToolReturned(e) => e.code(),
            other => panic!("expected ToolReturned, got {other:?}"),
        };
        assert_eq!(
            code,
            ToolErrorCode::InvalidInput,
            "missing required field must be an InvalidInput validation error"
        );
    }
    let _ = std::fs::remove_dir_all(&base);
}
