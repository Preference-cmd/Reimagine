//! Remote-`https://` trust and authentication tests: TOFU pins the
//! first-seen certificate, a rotated certificate is rejected, and a
//! remote worker rejects missing or wrong bearer tokens like a local
//! one.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use reimagine_backend_worker_protocol::auth::GrpcAuth;
use reimagine_backend_worker_protocol::remote::{ConnectTrust, connect_with_trust};
use reimagine_backend_worker_protocol::server::{
    BoundListen, ListenEndpoint, serve_bound, worker_server,
};
use reimagine_backend_worker_protocol::testing::{FakeWorkerConfig, FakeWorkerService};
use reimagine_backend_worker_protocol::tls::{SelfSignedIdentity, generate_self_signed_identity};
use reimagine_backend_worker_protocol::trust::TrustedKeyStore;
use reimagine_backend_worker_protocol::{
    BackendInstanceId, WorkerIdentity, WorkerIncarnationId, WorkerInstallationId,
    WorkerInstanceProfile, WorkerProfile,
};

const TOKEN: &str = "remote-secret";
const WORKER_ID: &str = "worker-1";

fn fake_identity() -> WorkerIdentity {
    WorkerIdentity {
        backend_instance_id: BackendInstanceId::from("burn:remote:default"),
        installation_id: WorkerInstallationId::from("remote-install"),
        incarnation_id: WorkerIncarnationId::from("remote-incarnation"),
        worker_version: "test".to_owned(),
        backend_kind: "burn".to_owned(),
        target: std::env::consts::ARCH.to_owned(),
        manifest_digest: "test-manifest".to_owned(),
    }
}

fn fake_profile() -> WorkerProfile {
    WorkerProfile {
        instances: vec![WorkerInstanceProfile {
            backend_instance_id: BackendInstanceId::from("burn:remote:default"),
            device_label: "cuda:0".to_owned(),
            capabilities: vec!["echo".to_owned()],
            operation_options: serde_json::json!({}),
        }],
    }
}

/// Serve the fake worker over TLS on an ephemeral loopback port with the
/// given identity; returns the `https://` endpoint.
async fn serve_tls_worker(
    token: Option<String>,
    identity: SelfSignedIdentity,
) -> (String, tokio::task::JoinHandle<()>) {
    let service = FakeWorkerService::new(FakeWorkerConfig {
        identity: Some(fake_identity()),
        profile: Some(fake_profile()),
        token,
        hello_protocol: 2,
        ..FakeWorkerConfig::default()
    });
    let bound =
        BoundListen::bind_https_with_identity(SocketAddr::from(([127, 0, 0, 1], 0)), identity)
            .await
            .unwrap();
    let endpoint = bound.uri().to_owned();
    let handle = tokio::spawn(async move {
        let _ = serve_bound(bound, worker_server(service)).await;
    });
    // Give the accept loop a moment; the listener is already bound.
    tokio::time::sleep(Duration::from_millis(50)).await;
    (endpoint, handle)
}

fn temp_store() -> (tempfile::TempDir, Arc<Mutex<TrustedKeyStore>>) {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(Mutex::new(
        TrustedKeyStore::load(dir.path().join("trusted-keys.json")).unwrap(),
    ));
    (dir, store)
}

#[tokio::test]
async fn tofu_records_first_connect_and_pins_later_connects() {
    let identity = generate_self_signed_identity("localhost").unwrap();
    let (endpoint, _server) = serve_tls_worker(Some(TOKEN.to_owned()), identity.clone()).await;
    let (_dir, store) = temp_store();
    let auth = GrpcAuth {
        token: Some(TOKEN.to_owned()),
        tls: None,
    };

    // First connect: TOFU records the presented fingerprint under
    // worker_id@endpoint-host.
    let client = connect_with_trust(&endpoint, ConnectTrust::new(&store, WORKER_ID), &auth)
        .await
        .expect("first connect");
    let hello = client.hello().await.expect("hello");
    assert_eq!(hello.protocol_version, 2);

    // The pin key is `worker_id@host` (no port): a worker restart that
    // changes the port must keep its pin.
    let host = endpoint
        .trim_start_matches("https://")
        .rsplit_once(':')
        .expect("endpoint carries a port")
        .0;
    let pin_key = format!("{WORKER_ID}@{host}");
    let pin = store
        .lock()
        .unwrap()
        .pinned(&pin_key)
        .expect("pin recorded")
        .clone();
    assert_eq!(pin, identity.fingerprint());

    // The pin persisted to disk.
    let path = store.lock().unwrap().path().to_owned();
    let reloaded = TrustedKeyStore::load(path).unwrap();
    assert_eq!(reloaded.pinned(&pin_key), Some(&identity.fingerprint()));

    // Second connect: the recorded pin matches and the handshake
    // succeeds.
    connect_with_trust(&endpoint, ConnectTrust::new(&store, WORKER_ID), &auth)
        .await
        .expect("pinned connect")
        .hello()
        .await
        .expect("authenticated hello");
}

#[tokio::test]
async fn changed_cert_is_rejected_after_pin() {
    let identity = generate_self_signed_identity("localhost").unwrap();
    let (endpoint, _server) = serve_tls_worker(Some(TOKEN.to_owned()), identity).await;
    let (_dir, store) = temp_store();
    let auth = GrpcAuth {
        token: Some(TOKEN.to_owned()),
        tls: None,
    };
    connect_with_trust(&endpoint, ConnectTrust::new(&store, WORKER_ID), &auth)
        .await
        .expect("first connect pins the certificate");

    // A rotated worker presents a different certificate from the same
    // host (same composite pin key): the connect must fail.
    let rotated = generate_self_signed_identity("localhost").unwrap();
    let (rotated_endpoint, _rotated_server) =
        serve_tls_worker(Some(TOKEN.to_owned()), rotated).await;
    let error = connect_with_trust(
        &rotated_endpoint,
        ConnectTrust::new(&store, WORKER_ID),
        &auth,
    )
    .await
    .expect_err("pinned connect to a different cert must fail");
    assert_eq!(error.code(), tonic::Code::Unavailable);
    assert!(
        error.to_string().contains("connect"),
        "expected connect/handshake failure, got: {error}"
    );
}

#[tokio::test]
async fn require_pin_refuses_unknown_worker() {
    let identity = generate_self_signed_identity("localhost").unwrap();
    let (endpoint, _server) = serve_tls_worker(Some(TOKEN.to_owned()), identity).await;
    let (_dir, store) = temp_store();
    let auth = GrpcAuth {
        token: Some(TOKEN.to_owned()),
        tls: None,
    };
    let error = connect_with_trust(
        &endpoint,
        ConnectTrust::new(&store, WORKER_ID).require_pin(),
        &auth,
    )
    .await
    .expect_err("require_pin must refuse an unknown worker");
    assert!(error.to_string().contains("no pinned certificate"));
    assert!(store.lock().unwrap().is_empty(), "nothing recorded");
}

#[tokio::test]
async fn pinning_requires_an_https_endpoint() {
    let (_dir, store) = temp_store();
    let error = connect_with_trust(
        "http://127.0.0.1:59998",
        ConnectTrust::new(&store, WORKER_ID),
        &GrpcAuth::plain(),
    )
    .await
    .expect_err("pinning over cleartext must be refused");
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
}

#[tokio::test]
async fn missing_token_is_rejected_on_the_remote_endpoint() {
    let identity = generate_self_signed_identity("localhost").unwrap();
    let (endpoint, _server) = serve_tls_worker(Some(TOKEN.to_owned()), identity).await;
    let (_dir, store) = temp_store();
    let client = connect_with_trust(
        &endpoint,
        ConnectTrust::new(&store, WORKER_ID),
        &GrpcAuth::plain(),
    )
    .await
    .expect("channel connects without a token");
    let error = client.hello().await.expect_err("hello without token");
    assert_eq!(error.code(), tonic::Code::Unauthenticated);
}

#[tokio::test]
async fn wrong_token_is_rejected_on_the_remote_endpoint() {
    let identity = generate_self_signed_identity("localhost").unwrap();
    let (endpoint, _server) = serve_tls_worker(Some(TOKEN.to_owned()), identity).await;
    let (_dir, store) = temp_store();
    let auth = GrpcAuth {
        token: Some("not-the-token".to_owned()),
        tls: None,
    };
    let client = connect_with_trust(&endpoint, ConnectTrust::new(&store, WORKER_ID), &auth)
        .await
        .expect("channel connects with a wrong token");
    let error = client.hello().await.expect_err("hello with wrong token");
    assert_eq!(error.code(), tonic::Code::Unauthenticated);
}

#[tokio::test]
async fn insecure_skip_connects_without_recording_a_pin() {
    let identity = generate_self_signed_identity("localhost").unwrap();
    let (endpoint, _server) = serve_tls_worker(None, identity).await;
    let (_dir, store) = temp_store();
    let client = connect_with_trust(
        &endpoint,
        ConnectTrust::new(&store, WORKER_ID).insecure_skip(),
        &GrpcAuth::plain(),
    )
    .await
    .expect("skip-pinning connect");
    client.hello().await.expect("hello");
    assert!(store.lock().unwrap().is_empty(), "nothing recorded");
}

#[tokio::test]
async fn listen_endpoint_parse_accepts_only_the_three_schemes() {
    let unix = ListenEndpoint::parse("unix:///tmp/worker.sock").unwrap();
    assert_eq!(unix.to_uri(), "unix:///tmp/worker.sock");
    assert!(matches!(
        ListenEndpoint::parse("https://127.0.0.1:9100").unwrap(),
        ListenEndpoint::Https(_)
    ));
    assert_eq!(
        ListenEndpoint::parse("https://127.0.0.1:9100")
            .unwrap()
            .to_uri(),
        "https://127.0.0.1:9100"
    );

    for rejected in ["http://127.0.0.1:9100", "quic://0.0.0.0:9100", "tcp://x"] {
        let error = ListenEndpoint::parse(rejected).unwrap_err();
        assert!(
            error.to_string().contains("unix://, pipe:// or https://"),
            "{rejected} must be rejected, got: {error}"
        );
    }
}
