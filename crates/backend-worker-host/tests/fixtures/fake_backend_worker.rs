//! Fake worker used by the supervisor and adapter tests: a real gRPC
//! worker process speaking `reimagine.worker.v2` over a private local
//! socket, built on the shared fake service from the transport crate.
//!
//! Behaviour is selected with `FAKE_WORKER_MODE`; operations (`echo`,
//! `delay`, `progress`, `crash`) mirror the old frame-protocol fixture
//! so the same scenarios stay covered.

use std::io::Write as _;
use std::process;
use std::time::Duration;

use reimagine_backend_worker_protocol::{
    BackendInstanceId, WorkerIdentity, WorkerIncarnationId, WorkerInstallationId,
    WorkerInstanceProfile, WorkerProfile,
};
use reimagine_backend_worker_transport_grpc::local::{LocalEndpoint, bind};
use reimagine_backend_worker_transport_grpc::server::{
    parse_listen_endpoint, serve_local, worker_server,
};
use reimagine_backend_worker_transport_grpc::testing::{FakeWorkerConfig, FakeWorkerService};

/// Worker stderr must never panic on a dead pipe: all writes ignore
/// errors.
fn log(message: &str) {
    let _ = std::io::stderr().write_all(message.as_bytes());
}

fn fixture_identity(mode: &str, incarnation: &str) -> WorkerIdentity {
    WorkerIdentity {
        backend_instance_id: BackendInstanceId::from("fake:cpu:default"),
        installation_id: WorkerInstallationId::from("fake-installation"),
        incarnation_id: WorkerIncarnationId(incarnation.to_owned()),
        worker_version: env!("CARGO_PKG_VERSION").to_owned(),
        backend_kind: "fake".to_owned(),
        target: std::env::consts::ARCH.to_owned(),
        manifest_digest: if mode == "wrong_identity" {
            "wrong-manifest".to_owned()
        } else {
            "test-manifest".to_owned()
        },
    }
}

fn fixture_profile(mode: &str) -> WorkerProfile {
    let mut capabilities = vec!["echo".to_owned(), "delay".to_owned(), "progress".to_owned()];
    if mode != "profile_without_latent" {
        capabilities.push("latent.create_empty".to_owned());
    }
    WorkerProfile {
        instances: vec![WorkerInstanceProfile {
            backend_instance_id: BackendInstanceId::from(if mode == "wrong_profile" {
                "fake:cpu:other"
            } else {
                "fake:cpu:default"
            }),
            device_label: "cpu".to_owned(),
            capabilities,
            operation_options: serde_json::json!({}),
        }],
    }
}

fn main() {
    log("host fake worker fixture started\n");
    let mode = std::env::var("FAKE_WORKER_MODE").unwrap_or_default();
    if mode == "hang" {
        std::thread::sleep(Duration::from_secs(30));
    }
    if mode == "stdout_noise" {
        println!("hello there");
    }

    let listen = std::env::args()
        .skip_while(|arg| arg != "--listen")
        .nth(1)
        .unwrap_or_default();
    let endpoint: LocalEndpoint = match parse_listen_endpoint(&listen) {
        Ok(endpoint) => endpoint,
        Err(error) => {
            log(&format!("FATAL: invalid --listen `{listen}`: {error}\n"));
            process::exit(1);
        }
    };
    let token = std::env::var("REIMAGINE_WORKER_TOKEN")
        .ok()
        .filter(|value| !value.is_empty());

    let incarnation = format!(
        "fixture-{}-{}",
        process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or(0)
    );
    let service = FakeWorkerService::new(FakeWorkerConfig {
        identity: Some(fixture_identity(&mode, &incarnation)),
        profile: Some(fixture_profile(&mode)),
        token,
        hello_protocol: if mode == "wrong_protocol" { 1 } else { 2 },
        ignore_shutdown: mode == "ignore_shutdown",
        ..FakeWorkerConfig::default()
    });

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("fixture runtime");

    // Lifeline: stdin EOF (host death) exits the process.
    std::thread::spawn(|| {
        use std::io::Read;
        let mut stdin = std::io::stdin().lock();
        let mut buffer = [0_u8; 256];
        loop {
            match stdin.read(&mut buffer) {
                Ok(0) | Err(_) => process::exit(0),
                Ok(_) => {}
            }
        }
    });

    runtime.block_on(async move {
        let incoming = match bind(&endpoint).await {
            Ok(incoming) => incoming,
            Err(error) => {
                log(&format!("FATAL: cannot bind {endpoint:?}: {error}\n"));
                process::exit(1);
            }
        };
        // READY line: the only stdout output of a worker, printed only
        // after the endpoint is bound.
        let mut stdout = std::io::stdout().lock();
        if writeln!(stdout, "READY {}", endpoint.to_uri()).is_err() || stdout.flush().is_err() {
            process::exit(1);
        }
        drop(stdout);

        // The stderr flood runs while serving: the host drains it
        // concurrently, mirroring a chatty worker. Logs must never
        // panic on a broken pipe.
        if mode == "stderr_flood" {
            std::thread::spawn(|| {
                let mut stderr = std::io::stderr().lock();
                let _ = stderr.write_all(&vec![b'x'; 256 * 1024]);
                let _ = stderr.flush();
            });
        }

        if let Err(error) = serve_local(incoming, worker_server(service)).await {
            log(&format!("fixture server error: {error}\n"));
        }
    });
}
