use serde_json::json;

use reimagine_backend_worker_protocol::{
    BackendExecutionError, CancelDisposition, CorrelationId, LifecycleError, ProgressFrame,
    RequestId, RequestTracker, TerminalFrame, TerminalOutcome, WorkerIncarnationId,
};

fn terminal(request: &str, correlation: &str, outcome: TerminalOutcome) -> TerminalFrame {
    TerminalFrame {
        request_id: RequestId::from(request),
        correlation_id: CorrelationId::from(correlation),
        outcome,
    }
}

#[test]
fn exactly_one_terminal_is_accepted_per_request() {
    let mut tracker = RequestTracker::new();
    tracker
        .register(RequestId::from("r1"), CorrelationId::from("c1"))
        .unwrap();
    let success = terminal("r1", "c1", TerminalOutcome::Success { output: json!(1) });
    tracker.record_terminal(&success).unwrap();

    assert_eq!(
        tracker.record_terminal(&success),
        Err(LifecycleError::DuplicateTerminal(RequestId::from("r1")))
    );
}

#[test]
fn cancellation_is_idempotent_and_does_not_create_a_terminal() {
    let mut tracker = RequestTracker::new();
    let request_id = RequestId::from("r1");
    let correlation_id = CorrelationId::from("c1");
    tracker
        .register(request_id.clone(), correlation_id.clone())
        .unwrap();

    assert_eq!(
        tracker
            .request_cancel(&request_id, &correlation_id)
            .unwrap(),
        CancelDisposition::Requested
    );
    assert_eq!(
        tracker
            .request_cancel(&request_id, &correlation_id)
            .unwrap(),
        CancelDisposition::AlreadyRequested
    );
    assert_eq!(
        tracker
            .transport_lost(WorkerIncarnationId::from("inc-1"))
            .pending_request_ids,
        vec![request_id]
    );
}

#[test]
fn progress_and_terminal_must_preserve_correlation() {
    let mut tracker = RequestTracker::new();
    tracker
        .register(RequestId::from("r1"), CorrelationId::from("c1"))
        .unwrap();
    let progress = ProgressFrame {
        request_id: RequestId::from("r1"),
        correlation_id: CorrelationId::from("wrong"),
        sequence: 1,
        completed: 1,
        total: Some(2),
        message: None,
    };
    assert!(matches!(
        tracker.record_progress(&progress),
        Err(LifecycleError::CorrelationMismatch { .. })
    ));

    let backend_error = terminal(
        "r1",
        "c1",
        TerminalOutcome::BackendError {
            error: BackendExecutionError {
                code: "execution_failed".to_owned(),
                message: "model rejected input".to_owned(),
                retryable: false,
            },
        },
    );
    tracker.record_terminal(&backend_error).unwrap();
    assert!(
        tracker
            .transport_lost(WorkerIncarnationId::from("inc-1"))
            .pending_request_ids
            .is_empty()
    );
}

#[test]
fn completed_request_tombstones_are_bounded() {
    let mut tracker = RequestTracker::new();
    for index in 0..=1024 {
        let request_id = RequestId(format!("r{index}"));
        let correlation_id = CorrelationId(format!("c{index}"));
        tracker
            .register(request_id.clone(), correlation_id.clone())
            .unwrap();
        tracker
            .record_terminal(&TerminalFrame {
                request_id,
                correlation_id,
                outcome: TerminalOutcome::Cancelled,
            })
            .unwrap();
    }

    assert_eq!(
        tracker.register(RequestId::from("r0"), CorrelationId::from("reused")),
        Ok(())
    );
}

#[test]
fn mismatched_terminal_does_not_consume_the_active_request() {
    let mut tracker = RequestTracker::new();
    tracker
        .register(RequestId::from("r1"), CorrelationId::from("c1"))
        .unwrap();

    assert!(matches!(
        tracker.record_terminal(&terminal("r1", "wrong", TerminalOutcome::Cancelled)),
        Err(LifecycleError::CorrelationMismatch { .. })
    ));
    assert_eq!(
        tracker.record_terminal(&terminal("r1", "c1", TerminalOutcome::Cancelled)),
        Ok(())
    );
}
