//! Tests for shared ID newtypes — ergonomics through the public facade.

use reimagine_core::model::*;

macro_rules! id_ergonomics_test {
    ($test_name:ident, $ty:ty, $sample:literal) => {
        #[test]
        fn $test_name() {
            let from_new = <$ty>::new($sample);
            assert_eq!(from_new.as_str(), $sample);
            assert_eq!(from_new.to_string(), $sample);

            let from_string = <$ty>::from(String::from($sample));
            assert_eq!(from_string.as_str(), $sample);

            let from_str = <$ty>::from($sample);
            assert_eq!(from_str.as_str(), $sample);

            let clone = from_str.clone();
            assert_eq!(from_str, clone);

            let mut set = std::collections::HashSet::new();
            set.insert(from_str);
            set.insert(clone);
            assert_eq!(set.len(), 1);
        }
    };
}

id_ergonomics_test!(workflow_id_ergonomics, WorkflowId, "workflow-1");
id_ergonomics_test!(node_id_ergonomics, NodeId, "node-1");
id_ergonomics_test!(edge_id_ergonomics, EdgeId, "edge-1");
id_ergonomics_test!(node_type_id_ergonomics, NodeTypeId, "builtin.ksampler");
id_ergonomics_test!(slot_id_ergonomics, SlotId, "latent");
id_ergonomics_test!(
    workflow_input_id_ergonomics,
    WorkflowInputId,
    "positive_prompt"
);
id_ergonomics_test!(workflow_output_id_ergonomics, WorkflowOutputId, "image");
id_ergonomics_test!(run_id_ergonomics, RunId, "run-1");
id_ergonomics_test!(artifact_id_ergonomics, ArtifactId, "artifact-1");
id_ergonomics_test!(diagnostic_id_ergonomics, DiagnosticId, "diagnostic-1");
id_ergonomics_test!(history_entry_id_ergonomics, HistoryEntryId, "history-1");
id_ergonomics_test!(command_batch_id_ergonomics, CommandBatchId, "batch-1");
id_ergonomics_test!(proposal_id_ergonomics, ProposalId, "proposal-1");
id_ergonomics_test!(model_id_ergonomics, ModelId, "model-1");

// ProjectId tests
id_ergonomics_test!(project_id_ergonomics, ProjectId, "my-project");

// -----------------------------------------------------------
// Serde round-trip: IDs serialise as plain strings.
// -----------------------------------------------------------
#[test]
fn id_serde_roundtrip() {
    let wid = WorkflowId::new("wf-1");
    let json = serde_json::to_string(&wid).expect("serialize");
    assert_eq!(json, r#""wf-1""#);
    let back: WorkflowId = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, wid);

    // One more type for good measure
    let mid = ModelId::new("sd-xl-base");
    let json = serde_json::to_string(&mid).expect("serialize");
    assert_eq!(json, r#""sd-xl-base""#);
    let back: ModelId = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, mid);
}

#[test]
fn workflow_version_is_numeric() {
    let version = WorkflowVersion::new(7);
    let json = serde_json::to_string(&version).expect("serialize");
    assert_eq!(json, "7");
    let back: WorkflowVersion = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, version);
    assert_eq!(back.get(), 7);
}

// -----------------------------------------------------------
// Display
// -----------------------------------------------------------
#[test]
fn id_display() {
    assert_eq!(RunId::new("r42").to_string(), "r42");
    assert_eq!(ProposalId::new("p-1").to_string(), "p-1");
}

// -----------------------------------------------------------
// Validation: new() rejects invalid IDs
// -----------------------------------------------------------
#[test]
#[should_panic(expected = "ID must not be empty")]
fn id_new_rejects_empty_string() {
    NodeId::new("");
}

#[test]
#[should_panic(expected = "ID must be ASCII")]
fn id_new_rejects_non_ascii() {
    NodeId::new("nodeétest");
}

#[test]
#[should_panic(expected = "ID contains invalid characters")]
fn id_new_rejects_forward_slash() {
    NodeId::new("../../../etc/passwd");
}

#[test]
#[should_panic(expected = "ID contains invalid characters")]
fn id_new_rejects_backslash() {
    NodeId::new("node\\test");
}

#[test]
#[should_panic(expected = "ID contains invalid characters")]
fn id_new_rejects_null_byte() {
    NodeId::new("node\0test");
}

#[test]
fn id_from_str_bypasses_validation() {
    // From<&str> bypasses validation for backwards compatibility
    let id = NodeId::from("../../../etc/passwd");
    assert_eq!(id.as_str(), "../../../etc/passwd");
}

// -----------------------------------------------------------
// ProjectId validation tests
// -----------------------------------------------------------
#[test]
#[should_panic(expected = "ID must not be empty")]
fn project_id_new_rejects_empty_string() {
    ProjectId::new("");
}

#[test]
#[should_panic(expected = "ID must be ASCII")]
fn project_id_new_rejects_non_ascii() {
    ProjectId::new("projetétest");
}

#[test]
#[should_panic(expected = "ID contains invalid characters")]
fn project_id_new_rejects_forward_slash() {
    ProjectId::new("../../../etc/passwd");
}

#[test]
#[should_panic(expected = "ID contains invalid characters")]
fn project_id_new_rejects_backslash() {
    ProjectId::new("project\\test");
}

#[test]
#[should_panic(expected = "ID contains invalid characters")]
fn project_id_new_rejects_null_byte() {
    ProjectId::new("project\0test");
}
