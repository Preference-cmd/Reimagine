//! Integration tests for the snapshot watch channel (BE-34) and the
//! incremental delta stream (BE-17).
//!
//! These drive the store update methods directly (the orchestrator/publisher
//! call sites are owned by a concurrent task).

use std::collections::HashMap;
use std::time::Duration;

use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::event::Timestamp;
use reimagine_core::model::{
    ArtifactId, ArtifactRef, DiagnosticId, NodeId, RunId, WorkflowId, WorkflowVersion,
};
use reimagine_runtime::{
    NodeState, RunArtifactRef, RunSnapshot, RunSnapshotUpdate, RunState, RunStore,
};

fn snapshot(
    run_id: &RunId,
    state: RunState,
    nodes: &[(&str, NodeState)],
    artifacts: &[(&str, &str, &str)],
    ts: &str,
) -> RunSnapshot {
    RunSnapshot::new(
        run_id.clone(),
        WorkflowId::new("wf"),
        WorkflowVersion::new(1),
        state,
        nodes
            .iter()
            .map(|(id, state)| (NodeId::new(*id), *state))
            .collect(),
        Vec::new(),
        artifacts
            .iter()
            .map(|(id, node, reference)| {
                RunArtifactRef::new(
                    ArtifactId::new(*id),
                    NodeId::new(*node),
                    ArtifactRef::new(*reference),
                )
            })
            .collect(),
        Timestamp::new(ts),
        Timestamp::new(ts),
    )
}

fn run_id(id: &str) -> RunId {
    RunId::new(id)
}

fn diagnostic(message: &str) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new(format!("diag-{message}")),
        DiagnosticCode::new("RUNTIME/CLEANUP"),
        DiagnosticSeverity::Warning,
        DiagnosticSourceName::new("runtime"),
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new("runtime")),
    )
}

async fn next_update(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<RunSnapshotUpdate>,
) -> RunSnapshotUpdate {
    tokio::time::timeout(Duration::from_millis(250), rx.recv())
        .await
        .expect("timed out waiting for delta stream update")
        .expect("delta stream closed unexpectedly")
}

async fn expect_no_update(rx: &mut tokio::sync::mpsc::UnboundedReceiver<RunSnapshotUpdate>) {
    assert!(
        tokio::time::timeout(Duration::from_millis(100), rx.recv())
            .await
            .is_err(),
        "expected no update on the delta stream"
    );
}

#[tokio::test]
async fn watch_subscriber_observes_every_put_without_polling() {
    let store = RunStore::new();
    let run = run_id("watch-1");

    store.put_snapshot(snapshot(
        &run,
        RunState::Queued,
        &[("n1", NodeState::Queued)],
        &[],
        "t0",
    ));

    let mut rx = store
        .subscribe(&run)
        .expect("subscribe after first snapshot");
    assert_eq!(rx.borrow().state, RunState::Queued);

    store.put_snapshot(snapshot(
        &run,
        RunState::Running,
        &[("n1", NodeState::Running)],
        &[],
        "t1",
    ));
    assert!(rx.changed().await.is_ok());
    assert_eq!(rx.borrow().state, RunState::Running);

    store.put_snapshot(snapshot(
        &run,
        RunState::Completed,
        &[("n1", NodeState::Completed)],
        &[],
        "t2",
    ));
    assert!(rx.changed().await.is_ok());
    assert_eq!(rx.borrow().state, RunState::Completed);
}

#[tokio::test]
async fn snapshot_returns_latest_value_and_keeps_backward_compat_semantics() {
    let store = RunStore::new();
    let run = run_id("poll-1");

    assert_eq!(store.snapshot(&run), None);
    assert_eq!(store.snapshot(&run_id("never-written")), None);

    store.put_snapshot(snapshot(
        &run,
        RunState::Running,
        &[("n1", NodeState::Running)],
        &[],
        "t1",
    ));
    let snap = store.snapshot(&run).expect("snapshot present after put");
    assert_eq!(snap.state, RunState::Running);
    assert_eq!(
        snap.node_states.get(&NodeId::new("n1")),
        Some(&NodeState::Running)
    );

    store.put_snapshot(snapshot(
        &run,
        RunState::Completed,
        &[("n1", NodeState::Completed)],
        &[],
        "t2",
    ));
    assert_eq!(store.snapshot(&run).unwrap().state, RunState::Completed);
}

#[tokio::test]
async fn subscribe_before_first_snapshot_returns_none() {
    let store = RunStore::new();
    assert!(store.subscribe(&run_id("ghost")).is_none());
    assert!(store.delta_stream(&run_id("ghost")).is_none());
}

#[tokio::test]
async fn delta_stream_emits_full_baseline_then_incremental_deltas() {
    let store = RunStore::new();
    let run = run_id("delta-1");

    store.put_snapshot(snapshot(
        &run,
        RunState::Running,
        &[("n1", NodeState::Running), ("n2", NodeState::Queued)],
        &[("a1", "n1", "samples/a1.png")],
        "t0",
    ));
    store.put_snapshot(snapshot(
        &run,
        RunState::Running,
        &[("n1", NodeState::Completed), ("n2", NodeState::Queued)],
        &[("a1", "n1", "samples/a1.png")],
        "t1",
    ));

    let mut rx = store
        .delta_stream(&run)
        .expect("delta stream for existing run");

    // Late joiner baseline: the current full snapshot, not just the last delta.
    match next_update(&mut rx).await {
        RunSnapshotUpdate::Full(full) => {
            assert_eq!(full.run_id, run);
            assert_eq!(
                full.node_states.get(&NodeId::new("n1")),
                Some(&NodeState::Completed)
            );
        }
        other => panic!("expected Full baseline, got {other:?}"),
    }

    // Next put yields a delta carrying only the changed node.
    store.put_snapshot(snapshot(
        &run,
        RunState::Running,
        &[("n1", NodeState::Completed), ("n2", NodeState::Running)],
        &[("a1", "n1", "samples/a1.png")],
        "t2",
    ));
    match next_update(&mut rx).await {
        RunSnapshotUpdate::Delta {
            run_id,
            changed_nodes,
            new_artifacts,
            timestamp,
        } => {
            assert_eq!(run_id, run);
            assert_eq!(changed_nodes.len(), 1);
            assert_eq!(
                changed_nodes.get(&NodeId::new("n2")),
                Some(&NodeState::Running)
            );
            assert!(new_artifacts.is_empty());
            assert_eq!(timestamp.as_str(), "t2");
        }
        other => panic!("expected Delta, got {other:?}"),
    }

    expect_no_update(&mut rx).await;
}

#[tokio::test]
async fn delta_artifacts_are_incremental_across_puts() {
    let store = RunStore::new();
    let run = run_id("delta-art");

    store.put_snapshot(snapshot(
        &run,
        RunState::Running,
        &[("n1", NodeState::Running)],
        &[("a1", "n1", "samples/a1.png")],
        "t0",
    ));
    let mut rx = store.delta_stream(&run).unwrap();
    assert!(matches!(
        next_update(&mut rx).await,
        RunSnapshotUpdate::Full(_)
    ));

    store.put_snapshot(snapshot(
        &run,
        RunState::Running,
        &[("n1", NodeState::Completed)],
        &[
            ("a1", "n1", "samples/a1.png"),
            ("a2", "n1", "samples/a2.png"),
        ],
        "t1",
    ));
    match next_update(&mut rx).await {
        RunSnapshotUpdate::Delta { new_artifacts, .. } => {
            assert_eq!(new_artifacts.len(), 1, "only the new artifact appears");
            assert_eq!(new_artifacts[0].id, ArtifactId::new("a2"));
        }
        other => panic!("expected Delta, got {other:?}"),
    }
}

#[tokio::test]
async fn unchanged_snapshot_publishes_to_watch_but_emits_no_delta() {
    let store = RunStore::new();
    let run = run_id("noop");

    store.put_snapshot(snapshot(
        &run,
        RunState::Running,
        &[("n1", NodeState::Running)],
        &[],
        "t0",
    ));
    let mut rx = store.delta_stream(&run).unwrap();
    assert!(matches!(
        next_update(&mut rx).await,
        RunSnapshotUpdate::Full(_)
    ));

    // Identical node states/artifacts, only the updated_at timestamp changes.
    store.put_snapshot(snapshot(
        &run,
        RunState::Running,
        &[("n1", NodeState::Running)],
        &[],
        "t1",
    ));
    expect_no_update(&mut rx).await;
}

#[tokio::test]
async fn runs_are_isolated_per_run_id() {
    let store = RunStore::new();
    let run_a = run_id("run-a");
    let run_b = run_id("run-b");

    assert!(store.subscribe(&run_b).is_none());

    store.put_snapshot(snapshot(
        &run_a,
        RunState::Running,
        &[("n1", NodeState::Running)],
        &[],
        "t0",
    ));
    assert!(
        store.delta_stream(&run_b).is_none(),
        "no stream for a run without snapshots"
    );

    store.put_snapshot(snapshot(
        &run_b,
        RunState::Queued,
        &[("n1", NodeState::Queued)],
        &[],
        "t1",
    ));
    let mut rx_a = store.delta_stream(&run_a).unwrap();
    let mut rx_b = store.delta_stream(&run_b).unwrap();
    assert!(matches!(
        next_update(&mut rx_a).await,
        RunSnapshotUpdate::Full(_)
    ));
    assert!(matches!(
        next_update(&mut rx_b).await,
        RunSnapshotUpdate::Full(_)
    ));

    // Update run_a only; run_b's stream stays silent.
    store.put_snapshot(snapshot(
        &run_a,
        RunState::Running,
        &[("n1", NodeState::Completed)],
        &[],
        "t2",
    ));
    assert!(matches!(
        next_update(&mut rx_a).await,
        RunSnapshotUpdate::Delta { .. }
    ));
    expect_no_update(&mut rx_b).await;
}

#[tokio::test]
async fn watch_reflects_appended_diagnostics_without_emitting_deltas() {
    let store = RunStore::new();
    let run = run_id("diag-1");

    store.put_snapshot(snapshot(
        &run,
        RunState::Completed,
        &[("n1", NodeState::Completed)],
        &[],
        "t0",
    ));
    let mut rx = store.subscribe(&run).unwrap();
    assert!(rx.borrow().diagnostics.is_empty());

    // Diagnostics are append-only and land on the watch so subscribers see
    // the refreshed snapshot, but no delta is emitted for them.
    store.append_diagnostics(&run, std::slice::from_ref(&diagnostic("cleanup note")));

    assert!(rx.changed().await.is_ok());
    assert_eq!(rx.borrow().diagnostics.len(), 1);

    let mut deltas = store.delta_stream(&run).unwrap();
    assert!(matches!(
        next_update(&mut deltas).await,
        RunSnapshotUpdate::Full(_)
    ));
    expect_no_update(&mut deltas).await;
}

#[tokio::test]
async fn multiple_delta_subscribers_each_receive_the_same_updates() {
    let store = RunStore::new();
    let run = run_id("fanout");

    store.put_snapshot(snapshot(
        &run,
        RunState::Running,
        &[("n1", NodeState::Running)],
        &[],
        "t0",
    ));
    let mut rx1 = store.delta_stream(&run).unwrap();
    let mut rx2 = store.delta_stream(&run).unwrap();
    assert!(matches!(
        next_update(&mut rx1).await,
        RunSnapshotUpdate::Full(_)
    ));
    assert!(matches!(
        next_update(&mut rx2).await,
        RunSnapshotUpdate::Full(_)
    ));

    store.put_snapshot(snapshot(
        &run,
        RunState::Running,
        &[("n1", NodeState::Completed)],
        &[],
        "t1",
    ));
    assert!(matches!(
        next_update(&mut rx1).await,
        RunSnapshotUpdate::Delta { .. }
    ));
    assert!(matches!(
        next_update(&mut rx2).await,
        RunSnapshotUpdate::Delta { .. }
    ));
}

#[tokio::test]
async fn delta_baseline_is_never_overtaken_by_subsequent_deltas() {
    let store = RunStore::new();
    let run = run_id("ordering");

    store.put_snapshot(snapshot(
        &run,
        RunState::Running,
        &[("n1", NodeState::Running)],
        &[],
        "t0",
    ));
    let mut rx = store.delta_stream(&run).unwrap();

    // Fire several puts back-to-back; the stream must still deliver the Full
    // baseline before any Delta, in FIFO order.
    for (node, ts) in [("n1", "t1"), ("n2", "t2"), ("n3", "t3")] {
        store.put_snapshot(snapshot(
            &run,
            RunState::Running,
            &[("n1", NodeState::Running), (node, NodeState::Completed)],
            &[],
            ts,
        ));
    }

    assert!(
        matches!(next_update(&mut rx).await, RunSnapshotUpdate::Full(_)),
        "baseline must arrive before deltas"
    );
    for _ in 0..3 {
        assert!(matches!(
            next_update(&mut rx).await,
            RunSnapshotUpdate::Delta { .. }
        ));
    }
}

#[tokio::test]
async fn delta_carries_only_the_nodes_that_changed_across_many() {
    let store = RunStore::new();
    let run = run_id("many-nodes");

    let ids: Vec<String> = (0..100).map(|i| format!("n{i}")).collect();
    let all_queued: Vec<(&str, NodeState)> = ids
        .iter()
        .map(|id| (id.as_str(), NodeState::Queued))
        .collect();
    store.put_snapshot(snapshot(&run, RunState::Running, &all_queued, &[], "t0"));
    let mut rx = store.delta_stream(&run).unwrap();
    assert!(matches!(
        next_update(&mut rx).await,
        RunSnapshotUpdate::Full(_)
    ));

    // Flip exactly one of the 100 nodes.
    let mut one_changed = all_queued.clone();
    one_changed[0].1 = NodeState::Running;
    store.put_snapshot(snapshot(&run, RunState::Running, &one_changed, &[], "t1"));
    match next_update(&mut rx).await {
        RunSnapshotUpdate::Delta { changed_nodes, .. } => {
            assert_eq!(changed_nodes.len(), 1, "delta carries 1 of 100 nodes");
            assert_eq!(
                changed_nodes.get(&NodeId::new("n0")),
                Some(&NodeState::Running)
            );
        }
        other => panic!("expected Delta, got {other:?}"),
    }
}

#[tokio::test]
async fn watch_borrow_returns_the_shared_arc_without_copying() {
    let store = RunStore::new();
    let run = run_id("arc");

    store.put_snapshot(snapshot(
        &run,
        RunState::Running,
        &[("n1", NodeState::Running)],
        &[],
        "t0",
    ));
    let rx = store.subscribe(&run).unwrap();

    // Borrows share the underlying snapshot; cloning the arc is cheap.
    let first: HashMap<NodeId, NodeState> = rx.borrow().node_states.clone();
    let second: HashMap<NodeId, NodeState> = rx.borrow().node_states.clone();
    assert_eq!(first, second);
    assert_eq!(first.len(), 1);
    let _ = (first, second);
}
