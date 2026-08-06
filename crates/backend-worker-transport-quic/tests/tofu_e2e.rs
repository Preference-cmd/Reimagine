//! End-to-end tests for the QUIC TOFU/pinning trust model (T19).
//!
//! A fake worker listens with a self-signed cert; the host connects via
//! `QuicTransport::connect_with_trust` and the `TrustedKeyStore`.

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Mutex;

use reimagine_backend_worker_protocol::WorkerTransport;
use reimagine_backend_worker_transport_quic::tls::SelfSignedCert;
use reimagine_backend_worker_transport_quic::{ConnectTrust, QuicTransport, TrustedKeyStore};

/// Start a QUIC server with `cert` on an ephemeral port; returns the
/// listener address and a handle that accepts one connection and keeps it
/// alive until closed.
async fn start_worker(cert: &SelfSignedCert) -> SocketAddr {
    let endpoint = reimagine_backend_worker_transport_quic::tls::server_endpoint(
        SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
        cert,
    )
    .unwrap();
    let addr = endpoint.local_addr().unwrap();
    tokio::spawn(async move {
        if let Some(incoming) = endpoint.accept().await {
            if let Ok(conn) = incoming.await {
                let _ = conn.closed().await;
            }
        }
    });
    addr
}

/// Establish a QUIC connection with the given trust config and return the
/// description endpoint (proves the handshake succeeded).
async fn connect_ok(addr: SocketAddr, trust: ConnectTrust<'_>) -> String {
    let transport = QuicTransport::connect_with_trust(
        SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
        addr,
        "localhost",
        trust,
    )
    .await
    .expect("connect should succeed");
    let description = transport.description().endpoint;
    transport.shutdown().await.unwrap();
    description
}

#[tokio::test]
async fn tofu_records_first_connect_and_pins_later_connects() {
    let cert = SelfSignedCert::generate("localhost").unwrap();
    let addr = start_worker(&cert).await;

    let dir = tempfile::tempdir().unwrap();
    let store = Mutex::new(TrustedKeyStore::load(dir.path().join("trusted-keys.json")).unwrap());

    // First connect: TOFU records the fingerprint.
    let endpoint = connect_ok(addr, ConnectTrust::new(&store, "worker-1")).await;
    assert!(endpoint.contains("quic://"));

    let pin = store
        .lock()
        .unwrap()
        .pinned("worker-1")
        .expect("pin should be recorded")
        .clone();
    assert_eq!(pin, cert.fingerprint());

    // Reload from disk: the pin persisted.
    let reloaded = TrustedKeyStore::load(dir.path().join("trusted-keys.json")).unwrap();
    assert_eq!(reloaded.pinned("worker-1"), Some(&cert.fingerprint()));

    // Second connect (new listener, same cert): pin matches, connects.
    let addr2 = start_worker(&cert).await;
    connect_ok(addr2, ConnectTrust::new(&store, "worker-1")).await;
}

#[tokio::test]
async fn changed_cert_is_rejected_after_pin() {
    let dir = tempfile::tempdir().unwrap();
    let store = Mutex::new(TrustedKeyStore::load(dir.path().join("trusted-keys.json")).unwrap());

    // First worker: TOFU records cert A.
    let cert_a = SelfSignedCert::generate("localhost").unwrap();
    let addr_a = start_worker(&cert_a).await;
    connect_ok(addr_a, ConnectTrust::new(&store, "worker-1")).await;

    // Attacker/rotated worker presents cert B from a different endpoint.
    let cert_b = SelfSignedCert::generate("localhost").unwrap();
    let addr_b = start_worker(&cert_b).await;

    let result = QuicTransport::connect_with_trust(
        SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
        addr_b,
        "localhost",
        ConnectTrust::new(&store, "worker-1"),
    )
    .await;

    assert!(
        result.is_err(),
        "pinned connect to a different cert must fail"
    );
    let message = result.unwrap_err().to_string();
    assert!(
        message.contains("handshake failed"),
        "expected handshake failure, got: {message}"
    );
}

#[tokio::test]
async fn require_pin_refuses_unknown_worker() {
    let cert = SelfSignedCert::generate("localhost").unwrap();
    let addr = start_worker(&cert).await;

    let dir = tempfile::tempdir().unwrap();
    let store = Mutex::new(TrustedKeyStore::load(dir.path().join("trusted-keys.json")).unwrap());

    let result = QuicTransport::connect_with_trust(
        SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
        addr,
        "localhost",
        ConnectTrust::new(&store, "unknown").require_pin(),
    )
    .await;

    assert!(result.is_err());
    let message = result.unwrap_err().to_string();
    assert!(
        message.contains("no pinned certificate"),
        "expected pin-required error, got: {message}"
    );
}

#[tokio::test]
async fn insecure_skip_bypasses_pinning() {
    let cert = SelfSignedCert::generate("localhost").unwrap();
    let addr = start_worker(&cert).await;

    let dir = tempfile::tempdir().unwrap();
    let store = Mutex::new(TrustedKeyStore::load(dir.path().join("trusted-keys.json")).unwrap());

    // Skip pinning: connect succeeds even though no pin exists, and
    // nothing is recorded.
    connect_ok(
        addr,
        ConnectTrust::new(&store, "worker-skip").insecure_skip(),
    )
    .await;
    assert!(store.lock().unwrap().is_empty());
}
