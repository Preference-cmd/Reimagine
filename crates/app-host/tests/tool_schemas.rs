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
        // AR-38/AR-40: field-accurate — type object, declared
        // properties, and every property typed + described. This is what
        // keeps the catalog from drifting back into placeholders.
        assert_eq!(
            input.get("type").and_then(|t| t.as_str()),
            Some("object"),
            "{name} input schema must be an object"
        );
        let properties = input
            .get("properties")
            .and_then(|p| p.as_object())
            .unwrap_or_else(|| panic!("{name} input schema must declare properties"));
        for (field, prop_schema) in properties {
            assert!(
                prop_schema.get("type").is_some(),
                "{name}.{field} must declare a type"
            );
            assert!(
                prop_schema.get("description").is_some(),
                "{name}.{field} must declare a description"
            );
        }
        // AR-40: declared fields must be pinned as `required` (V1
        // subset). `model.list` is the one tool with no inputs; every
        // other workspace tool pins its fields so drift back toward a
        // transparent `{}` is caught here.
        if !properties.is_empty() {
            let required = input
                .get("required")
                .and_then(|r| r.as_array())
                .unwrap_or_else(|| panic!("{name} input schema must declare required"));
            for entry in required {
                let field = entry
                    .as_str()
                    .unwrap_or_else(|| panic!("{name} required entry must be a string"));
                assert!(
                    properties.contains_key(field),
                    "{name} required field '{field}' must be declared in properties"
                );
            }
        }
        let output = spec.output_schema().expect("output schema present");
        assert_eq!(
            output.get("type").and_then(|t| t.as_str()),
            Some("object"),
            "{name} output schema must be an object"
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
