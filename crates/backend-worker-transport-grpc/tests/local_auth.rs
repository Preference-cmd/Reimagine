//! Local-socket authentication tests: a worker serving a private Unix
//! socket must reject missing or wrong bearer tokens.

use std::time::Duration;

use reimagine_backend_worker_transport_grpc::auth::GrpcAuth;
use reimagine_backend_worker_transport_grpc::client::WorkerClient;
use reimagine_backend_worker_transport_grpc::local::{LocalEndpoint, bind, connect_local};
use reimagine_backend_worker_transport_grpc::server::{
    parse_listen_endpoint, serve_local, worker_server,
};
use reimagine_backend_worker_transport_grpc::testing::{FakeWorkerConfig, FakeWorkerService};

const TOKEN: &str = "local-secret";

async fn serve_fake_worker(endpoint: &LocalEndpoint) {
    let service = FakeWorkerService::new(FakeWorkerConfig {
        token: Some(TOKEN.to_owned()),
        hello_protocol: 2,
        ..FakeWorkerConfig::default()
    });
    let incoming = bind(endpoint).await.unwrap();
    tokio::spawn(async move {
        let _ = serve_local(incoming, worker_server(service)).await;
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
}

fn temp_endpoint(name: &str) -> LocalEndpoint {
    let path = std::env::temp_dir().join(format!("reimagine-auth-{name}-{}", std::process::id()));
    let _ = std::fs::remove_file(&path);
    parse_listen_endpoint(&format!("unix://{}", path.display())).unwrap()
}

#[tokio::test]
async fn missing_token_is_rejected_on_the_local_socket() {
    let endpoint = temp_endpoint("missing");
    serve_fake_worker(&endpoint).await;

    let client = connect_local(&endpoint, &GrpcAuth::plain())
        .await
        .expect("the channel connects without a token");
    let error = client.hello().await.expect_err("hello without token");
    assert_eq!(error.code(), tonic::Code::Unauthenticated);
}

#[tokio::test]
async fn wrong_token_is_rejected_on_the_local_socket() {
    let endpoint = temp_endpoint("wrong");
    serve_fake_worker(&endpoint).await;

    let auth = GrpcAuth {
        token: Some("not-the-token".to_owned()),
        tls: None,
    };
    let client = connect_local(&endpoint, &auth)
        .await
        .expect("the channel connects with a wrong token");
    let error = client.hello().await.expect_err("hello with wrong token");
    assert_eq!(error.code(), tonic::Code::Unauthenticated);
}

#[tokio::test]
async fn correct_token_is_accepted_on_the_local_socket() {
    let endpoint = temp_endpoint("correct");
    serve_fake_worker(&endpoint).await;

    let auth = GrpcAuth {
        token: Some(TOKEN.to_owned()),
        tls: None,
    };
    let client = connect_local(&endpoint, &auth).await.expect("connect");
    let hello = client.hello().await.expect("authenticated hello");
    assert_eq!(hello.protocol_version, 2);
}

#[tokio::test]
async fn execute_also_requires_the_token() {
    let endpoint = temp_endpoint("execute");
    serve_fake_worker(&endpoint).await;

    let client = connect_local(&endpoint, &GrpcAuth::plain())
        .await
        .expect("the channel connects without a token");
    let error = client
        .execute("echo", serde_json::to_vec(&serde_json::json!({})).unwrap())
        .await
        .expect_err("execute without token");
    assert_eq!(error.code(), tonic::Code::Unauthenticated);
}

#[tokio::test]
async fn cleartext_tcp_with_a_token_is_refused() {
    let error = WorkerClient::connect(
        "http://127.0.0.1:59999",
        &GrpcAuth {
            token: Some("s3cret".to_owned()),
            tls: None,
        },
    )
    .await
    .expect_err("cleartext token refused before any I/O");
    assert!(error.to_string().contains("cleartext"), "got: {error}");
}
