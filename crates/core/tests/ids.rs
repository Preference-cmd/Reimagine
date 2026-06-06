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
id_ergonomics_test!(run_id_ergonomics, RunId, "run-1");
id_ergonomics_test!(artifact_id_ergonomics, ArtifactId, "artifact-1");
id_ergonomics_test!(diagnostic_id_ergonomics, DiagnosticId, "diagnostic-1");
id_ergonomics_test!(history_entry_id_ergonomics, HistoryEntryId, "history-1");
id_ergonomics_test!(command_batch_id_ergonomics, CommandBatchId, "batch-1");
id_ergonomics_test!(proposal_id_ergonomics, ProposalId, "proposal-1");
id_ergonomics_test!(model_id_ergonomics, ModelId, "model-1");

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

// -----------------------------------------------------------
// Display
// -----------------------------------------------------------
#[test]
fn id_display() {
    assert_eq!(RunId::new("r42").to_string(), "r42");
    assert_eq!(ProposalId::new("p-1").to_string(), "p-1");
}
