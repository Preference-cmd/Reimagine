use std::net::{Ipv4Addr, SocketAddr};

use reimagine_backend_worker_protocol::{
    RequestFrame, RequestId, CorrelationId, ProtocolVersion,
    WireMessage, TerminalOutcome, WorkerTransport,
};
use reimagine_backend_worker_transport_quic::{QuicTransport, tls::SelfSignedCert};
use reimagine_backend_worker_transport_quic::listener::QuicWorkerListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn read_wire_message(recv: &mut quinn::RecvStream) -> Result<WireMessage, String> {
    let mut prefix = [0u8; 4];
    recv.read_exact(&mut prefix)
        .await
        .map_err(|e| format!("read prefix: {e}"))?;
    let len = u32::from_be_bytes(prefix) as usize;
    let mut payload = vec![0u8; len];
    recv.read_exact(&mut payload)
        .await
        .map_err(|e| format!("read payload: {e}"))?;
    serde_json::from_slice(&payload).map_err(|e| format!("deserialize: {e}"))
}

async fn write_wire_message(send: &mut quinn::SendStream, message: &WireMessage) -> Result<(), String> {
    let json = serde_json::to_vec(message).map_err(|e| format!("serialize: {e}"))?;
    send.write_all(&(json.len() as u32).to_be_bytes())
        .await
        .map_err(|e| format!("write length: {e}"))?;
    send.write_all(&json)
        .await
        .map_err(|e| format!("write payload: {e}"))?;
    Ok(())
}

#[tokio::test]
async fn quic_worker_listener_handshake_and_request() {
    let cert = SelfSignedCert::generate("localhost").unwrap();

    // Start a QUIC worker listener
    let listener = QuicWorkerListener::start(
        SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
        &cert,
    )
    .unwrap();
    let listen_addr = listener.local_addr().unwrap();

    // Server task: accept connection, handle requests
    let server = tokio::spawn(async move {
        let (_connection, mut send, mut recv, hello) = listener.accept().await.unwrap();
        assert_eq!(hello.identity.backend_kind, "fake");

        // Read a request from the host
        let request = read_wire_message(&mut recv).await.unwrap();

        if let WireMessage::Request(req) = request {
            assert_eq!(req.operation, "echo");
            // Send a terminal response
            let response = WireMessage::Terminal(reimagine_backend_worker_protocol::TerminalFrame {
                protocol_version: req.protocol_version,
                incarnation_id: req.incarnation_id,
                request_id: req.request_id,
                correlation_id: req.correlation_id,
                outcome: TerminalOutcome::Success {
                    output: serde_json::json!({ "echoed": req.payload }),
                },
            });
            write_wire_message(&mut send, &response).await.unwrap();
        }

        // Keep connection alive until client closes
        let _ = recv.read_to_end(1024 * 1024).await;
    });

    // Client: connect, perform handshake, send request
    let client_addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0);
    let transport = QuicTransport::connect(client_addr, listen_addr, "localhost", &cert)
        .await
        .unwrap();

    // Open a bidirectional stream
    let (mut send, mut recv) = transport.open_bi().await.unwrap();

    // Perform handshake: send HostHello, read WorkerHello
    let host_hello = WireMessage::HostHello(reimagine_backend_worker_protocol::HostHello {
        supported_protocols: reimagine_backend_worker_protocol::ProtocolRange::new(1, 1),
    });
    write_wire_message(&mut send, &host_hello).await.unwrap();

    // Read WorkerHello
    let worker_hello = read_wire_message(&mut recv).await.unwrap();
    assert!(matches!(worker_hello, WireMessage::WorkerHello(_)));

    // Send a request
    let request = WireMessage::Request(RequestFrame {
        protocol_version: ProtocolVersion(1),
        incarnation_id: reimagine_backend_worker_protocol::WorkerIncarnationId("test".into()),
        request_id: RequestId::from("req-1"),
        correlation_id: CorrelationId::from("flow-1"),
        operation: "echo".into(),
        payload: serde_json::json!({ "value": 42 }),
    });
    write_wire_message(&mut send, &request).await.unwrap();

    // Read response
    let response = read_wire_message(&mut recv).await.unwrap();

    if let WireMessage::Terminal(terminal) = response {
        match terminal.outcome {
            TerminalOutcome::Success { output } => {
                assert_eq!(output, serde_json::json!({ "echoed": { "value": 42 } }));
            }
            other => panic!("expected success, got {:?}", other),
        }
    } else {
        panic!("expected terminal, got {:?}", response);
    }

    transport.shutdown().await.unwrap();
    server.await.unwrap();
}
