use std::io::{BufReader, ErrorKind};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use reimagine_backend_worker_protocol::{
    CancelFrame, CodecError, CorrelationId, FrameCodec, HostHello, ProtocolRange, ProtocolVersion,
    RequestFrame, RequestId, RequestTracker, TerminalOutcome, WireMessage, WorkerHello,
    WorkerIncarnationId,
};
use serde_json::json;

const MAXIMUM_FRAME_BYTES: u32 = 1024 * 1024;

struct WorkerProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    hello: WorkerHello,
}

impl WorkerProcess {
    fn spawn() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_fake-backend-worker"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        codec()
            .write(
                &mut stdin,
                &WireMessage::HostHello(HostHello {
                    supported_protocols: ProtocolRange::new(1, 1),
                }),
            )
            .unwrap();
        let WireMessage::WorkerHello(hello) = codec().read(&mut stdout).unwrap() else {
            panic!("worker did not answer with worker_hello");
        };
        Self {
            child,
            stdin,
            stdout,
            hello,
        }
    }

    fn request(
        &mut self,
        request: &str,
        correlation: &str,
        operation: &str,
        payload: serde_json::Value,
    ) {
        codec()
            .write(
                &mut self.stdin,
                &WireMessage::Request(RequestFrame {
                    protocol_version: ProtocolVersion(1),
                    incarnation_id: self.hello.identity.incarnation_id.clone(),
                    request_id: RequestId::from(request),
                    correlation_id: CorrelationId::from(correlation),
                    operation: operation.to_owned(),
                    payload,
                }),
            )
            .unwrap();
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn codec() -> FrameCodec {
    FrameCodec::new(MAXIMUM_FRAME_BYTES)
}

#[test]
fn fake_worker_multiplexes_progress_and_idempotent_cancellation() {
    let mut worker = WorkerProcess::spawn();
    assert_eq!(worker.hello.selected_protocol, ProtocolVersion(1));
    assert_eq!(worker.hello.identity.backend_kind, "fake");
    assert!(!worker.hello.identity.manifest_digest.is_empty());

    worker.request("slow", "flow-slow", "delay", json!({ "milliseconds": 200 }));
    worker.request("fast", "flow-fast", "echo", json!({ "value": 7 }));
    let WireMessage::Terminal(fast) = codec().read(&mut worker.stdout).unwrap() else {
        panic!("fast request should complete before slow request");
    };
    assert_eq!(fast.request_id, RequestId::from("fast"));
    assert_eq!(fast.correlation_id, CorrelationId::from("flow-fast"));

    worker.request("progress", "flow-progress", "progress", json!({}));
    for expected_sequence in 1..=3 {
        let WireMessage::Progress(progress) = codec().read(&mut worker.stdout).unwrap() else {
            panic!("expected progress frame");
        };
        assert_eq!(progress.request_id, RequestId::from("progress"));
        assert_eq!(progress.sequence, expected_sequence);
    }
    assert!(matches!(
        codec().read(&mut worker.stdout).unwrap(),
        WireMessage::Terminal(terminal)
            if terminal.request_id == RequestId::from("progress")
                && matches!(terminal.outcome, TerminalOutcome::Success { .. })
    ));

    codec()
        .write(
            &mut worker.stdin,
            &WireMessage::Cancel(CancelFrame {
                protocol_version: worker.hello.selected_protocol,
                incarnation_id: worker.hello.identity.incarnation_id.clone(),
                request_id: RequestId::from("slow"),
                correlation_id: CorrelationId::from("flow-slow"),
            }),
        )
        .unwrap();
    assert!(matches!(
        codec().read(&mut worker.stdout).unwrap(),
        WireMessage::CancelAck(ack) if ack.accepted && !ack.already_terminal
    ));
    assert!(matches!(
        codec().read(&mut worker.stdout).unwrap(),
        WireMessage::Terminal(terminal)
            if terminal.request_id == RequestId::from("slow")
                && terminal.outcome == TerminalOutcome::Cancelled
    ));

    codec()
        .write(
            &mut worker.stdin,
            &WireMessage::Cancel(CancelFrame {
                protocol_version: worker.hello.selected_protocol,
                incarnation_id: worker.hello.identity.incarnation_id.clone(),
                request_id: RequestId::from("slow"),
                correlation_id: CorrelationId::from("flow-slow"),
            }),
        )
        .unwrap();
    assert!(matches!(
        codec().read(&mut worker.stdout).unwrap(),
        WireMessage::CancelAck(ack) if !ack.accepted && ack.already_terminal
    ));
}

#[test]
fn abrupt_exit_is_transport_lost_for_the_specific_incarnation() {
    let mut worker = WorkerProcess::spawn();
    let incarnation = worker.hello.identity.incarnation_id.clone();
    let mut tracker = RequestTracker::new();
    tracker
        .register(RequestId::from("crash"), CorrelationId::from("flow-crash"))
        .unwrap();
    worker.request("crash", "flow-crash", "crash", json!({}));

    assert!(matches!(
        codec().read(&mut worker.stdout),
        Err(CodecError::Io(error)) if error.kind() == ErrorKind::UnexpectedEof
    ));
    let lost = tracker.transport_lost(incarnation.clone());
    assert_eq!(lost.incarnation_id, incarnation);
    assert_eq!(lost.pending_request_ids, vec![RequestId::from("crash")]);
}

#[test]
fn every_process_start_has_a_new_incarnation() {
    let first = WorkerProcess::spawn();
    let second = WorkerProcess::spawn();
    assert_ne!(
        first.hello.identity.incarnation_id,
        second.hello.identity.incarnation_id
    );
    assert_ne!(
        first.hello.identity.incarnation_id,
        WorkerIncarnationId::from("")
    );
}
