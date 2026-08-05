use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use clap::Parser;
use reimagine_backend_worker_protocol::{
    FrameCodec, HostHello, ProtocolRange, ProtocolVersion, WireMessage,
};
use reimagine_backend_worker_transport_quic::{
    discovery::MdnsWorkerRegister, listener::QuicWorkerListener, tls::SelfSignedCert,
};
use reimagine_inference_burn::{BurnBackend, BurnBackendConfig};

mod mapping;
mod probe;
mod quic_adapter;
mod server;
mod shutdown;

const WORKER_PROTOCOL_VERSION: ProtocolVersion = ProtocolVersion(1);
const MAX_FRAME_BYTES: u32 = 64 * 1024 * 1024; // 64 MB

#[derive(Parser)]
#[command(
    name = "reimagine-inference-burn-worker",
    about = "Reimagine Burn inference worker"
)]
struct Cli {
    /// Start a QUIC listener on this address (e.g. quic://0.0.0.0:9100).
    /// When set, the worker accepts remote QUIC connections instead of stdio.
    #[arg(long = "listen")]
    listen: Option<String>,
}

fn main() {
    let cli = Cli::parse();

    // ----- config and authoritative filesystem roots -----
    let models_dir = startup_or_exit(prepare_root_env("REIMAGINE_MODELS_DIR", false));
    let output_dir = startup_or_exit(prepare_root_env("REIMAGINE_OUTPUT_DIR", true));
    let allowed_model_roots = startup_or_exit(parse_allowlist_value(
        "REIMAGINE_ALLOWED_MODEL_ROOTS",
        std::env::var_os("REIMAGINE_ALLOWED_MODEL_ROOTS"),
    ));
    let allowed_output_roots = startup_or_exit(parse_allowlist_value(
        "REIMAGINE_ALLOWED_OUTPUT_ROOTS",
        std::env::var_os("REIMAGINE_ALLOWED_OUTPUT_ROOTS"),
    ));

    if !is_within_allowed_roots(&models_dir, &allowed_model_roots) {
        eprintln!(
            "FATAL: models dir '{}' is not within allowed MODEL roots",
            models_dir.display(),
        );
        std::process::exit(1);
    }
    if !is_within_allowed_roots(&output_dir, &allowed_output_roots) {
        eprintln!(
            "FATAL: output dir '{}' is not within allowed OUTPUT roots",
            output_dir.display(),
        );
        std::process::exit(1);
    }

    // ----- initialize Burn backend -----
    let config = BurnBackendConfig::new(&models_dir, &output_dir);
    let backend = match BurnBackend::new(config) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("FATAL: failed to initialize Burn backend: {e}");
            std::process::exit(1);
        }
    };

    // ----- tokio runtime for async backend calls -----
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("FATAL: failed to create tokio runtime: {e}");
            std::process::exit(1);
        }
    };

    match cli.listen {
        Some(listen_url) => {
            run_quic_mode(rt, &backend, &listen_url);
        }
        None => {
            run_stdio_mode(rt, &backend);
        }
    }
}

// ---------------------------------------------------------------------------
// Stdio mode (default)
// ---------------------------------------------------------------------------

fn run_stdio_mode(rt: tokio::runtime::Runtime, backend: &BurnBackend) {
    // ----- protocol handshake over stdio -----
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = BufWriter::new(stdout.lock());
    let codec = FrameCodec::new(MAX_FRAME_BYTES);
    // Read HostHello
    let host_hello: HostHello = match codec.read(&mut reader) {
        Ok(WireMessage::HostHello(hello)) => hello,
        Ok(other) => {
            eprintln!("FATAL: expected HostHello, got {}", other.kind());
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("FATAL: failed to read HostHello: {e}");
            std::process::exit(1);
        }
    };

    // Negotiate protocol version
    let worker_range = ProtocolRange::new(WORKER_PROTOCOL_VERSION.0, WORKER_PROTOCOL_VERSION.0);
    let selected = match reimagine_backend_worker_protocol::negotiate_protocol(
        host_hello.supported_protocols,
        worker_range,
    ) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("FATAL: protocol negotiation failed: {e}");
            std::process::exit(1);
        }
    };

    // Build identity and profile from the active backend
    let (identity, profile) = probe::build(backend);
    let incarnation_id = identity.incarnation_id.clone();

    // Send WorkerHello
    let worker_hello = reimagine_backend_worker_protocol::WorkerHello {
        selected_protocol: selected,
        identity,
        profile,
    };
    if let Err(e) = codec.write(&mut writer, &WireMessage::WorkerHello(worker_hello)) {
        eprintln!("FATAL: failed to send WorkerHello: {e}");
        std::process::exit(1);
    }
    if let Err(e) = writer.flush() {
        eprintln!("FATAL: failed to flush stdout after WorkerHello: {e}");
        std::process::exit(1);
    }

    // ----- serve loop -----
    // The handshake `BufReader` is dropped here: the reader thread in
    // the serve loop re-locks stdin itself, and the host sends no
    // frames between WorkerHello and the first request, so no buffered
    // bytes are lost.
    drop(reader);
    server::serve_loop(
        rt,
        backend,
        codec,
        std::io::stdin(),
        &mut writer,
        selected,
        &incarnation_id,
    );

    eprintln!("worker: serve loop exited, terminating");
}

// ---------------------------------------------------------------------------
// QUIC mode
// ---------------------------------------------------------------------------

fn run_quic_mode(rt: tokio::runtime::Runtime, backend: &BurnBackend, listen_url: &str) {
    let listen_addr = parse_quic_listen_url(listen_url);

    // Generate self-signed certificate for LAN development
    let hostname = listen_addr.ip().to_string();
    let cert = match SelfSignedCert::generate(&hostname) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FATAL: failed to generate TLS certificate: {e}");
            std::process::exit(1);
        }
    };

    // Start QUIC endpoint
    let listener = match QuicWorkerListener::start(listen_addr, &cert) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("FATAL: failed to start QUIC listener: {e}");
            std::process::exit(1);
        }
    };

    let actual_addr = match listener.local_addr() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("FATAL: failed to get listener address: {e}");
            std::process::exit(1);
        }
    };

    // Build identity for mDNS registration
    let (identity, profile) = probe::build(backend);

    // Register mDNS service
    let capabilities: Vec<&str> = profile
        .instances
        .iter()
        .flat_map(|inst| inst.capabilities.iter().map(|s| s.as_str()))
        .collect();
    let device_labels: Vec<&str> = profile
        .instances
        .iter()
        .map(|inst| inst.device_label.as_str())
        .collect();

    let mut mdns_props = HashMap::new();
    mdns_props.insert("endpoint".to_string(), format!("quic://{actual_addr}"));
    mdns_props.insert("backend".to_string(), identity.backend_kind.clone());
    mdns_props.insert("devices".to_string(), device_labels.join(","));
    mdns_props.insert("capabilities".to_string(), capabilities.join(","));

    let mdns = match MdnsWorkerRegister::register(
        &identity.backend_instance_id.0,
        actual_addr,
        mdns_props,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("FATAL: failed to register mDNS service: {e}");
            std::process::exit(1);
        }
    };

    eprintln!(
        "worker: QUIC listener on {actual_addr}, mDNS registered as '{}'",
        identity.backend_instance_id.0,
    );

    // Spawn QUIC accept loop on a dedicated thread with its own tokio runtime.
    // This avoids interfering with the stdio serve loop's runtime (if dual
    // mode is added later) and keeps the async QUIC I/O isolated.
    let quic_rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("quic-accept")
        .worker_threads(2)
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("FATAL: failed to create QUIC runtime: {e}");
            std::process::exit(1);
        }
    };

    let quic_backend = backend.clone();
    let quic_handle = std::thread::Builder::new()
        .name("quic-accept".to_owned())
        .spawn(move || {
            run_quic_accept_loop(quic_rt, rt, &quic_backend, listener);
        });

    match quic_handle {
        Ok(handle) => {
            // Block the main thread until the QUIC thread exits.
            // The QUIC thread runs until the endpoint is closed or
            // an unrecoverable error occurs.
            if let Err(e) = handle.join() {
                eprintln!("FATAL: QUIC accept thread panicked: {e:?}");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("FATAL: failed to spawn QUIC accept thread: {e}");
            std::process::exit(1);
        }
    }

    // Cleanup: drop mDNS registration
    drop(mdns);
    eprintln!("worker: QUIC mode terminated, cleaning up");
}

/// Run the QUIC accept loop on the provided tokio runtime.
///
/// Each accepted connection performs the worker handshake and then
/// enters a `serve_loop` via `spawn_blocking`, reusing the same
/// synchronous dispatch logic as stdio mode.
fn run_quic_accept_loop(
    quic_rt: tokio::runtime::Runtime,
    serve_rt: tokio::runtime::Runtime,
    backend: &BurnBackend,
    listener: QuicWorkerListener,
) {
    let serve_rt = Arc::new(serve_rt);
    let backend = Arc::new(backend.clone());

    quic_rt.block_on(async {
        loop {
            match listener.accept().await {
                Ok((connection, send, recv, worker_hello)) => {
                    let remote = connection.remote_address();
                    eprintln!("worker: QUIC connection from {remote}");

                    let protocol_version = worker_hello.selected_protocol;
                    let serve_rt = Arc::clone(&serve_rt);
                    let backend = Arc::clone(&backend);

                    // Handle each connection on a blocking thread, reusing
                    // the synchronous serve_loop with QUIC stream adapters.
                    tokio::task::spawn_blocking(move || {
                        let incarnation = worker_hello.identity.incarnation_id.clone();

                        let read_adapter =
                            quic_adapter::QuicReadAdapter::new(recv, Arc::clone(&serve_rt));
                        let mut write_adapter = quic_adapter::QuicWriteAdapter::new(send, serve_rt);

                        let codec = FrameCodec::new(MAX_FRAME_BYTES);

                        // Create a dedicated current-thread runtime for this
                        // connection's dispatch threads. The serve_loop wraps
                        // it in an Arc and dispatch threads call rt.block_on()
                        // to run async backend operations.
                        let conn_rt = tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                            .expect("failed to create connection runtime");

                        server::serve_loop(
                            conn_rt,
                            &backend,
                            codec,
                            read_adapter,
                            &mut write_adapter,
                            protocol_version,
                            &incarnation,
                        );

                        eprintln!("worker: QUIC connection from {remote} closed");
                    });
                }
                Err(e) => {
                    eprintln!("worker: QUIC accept error: {e}");
                    break;
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// CLI helpers
// ---------------------------------------------------------------------------

/// Parse a `quic://host:port` URL into a `SocketAddr`.
///
/// Accepts `quic://0.0.0.0:9100` or `quic://127.0.0.1:9100`.
fn parse_quic_listen_url(url: &str) -> SocketAddr {
    let stripped = url.strip_prefix("quic://").unwrap_or(url);

    stripped.parse::<SocketAddr>().unwrap_or_else(|e| {
        eprintln!("FATAL: invalid QUIC listen address '{url}': {e}");
        std::process::exit(1);
    })
}

/// Parse the `:`-separated allowlist from an environment variable
/// into a `Vec` of canonical `PathBuf` entries.
fn parse_allowlist_value(
    var: &str,
    raw: Option<std::ffi::OsString>,
) -> Result<Vec<PathBuf>, String> {
    let raw = raw.ok_or_else(|| format!("{var} is required"))?;
    let mut roots: Vec<PathBuf> = Vec::new();
    for path in std::env::split_paths(&raw) {
        if path.as_os_str().is_empty() {
            continue;
        }
        if path.is_absolute() {
            match path.canonicalize() {
                Ok(canon) => roots.push(canon),
                Err(e) => {
                    eprintln!(
                        "worker: {var} entry '{}' cannot be canonicalized: {e}",
                        path.display()
                    );
                }
            }
        } else {
            eprintln!(
                "worker: {var} entry '{}' is not absolute — skipping",
                path.display()
            );
        }
    }
    if roots.is_empty() {
        return Err(format!("{var} contains no valid canonical roots"));
    }
    Ok(roots)
}

/// Check whether `path` (which must already be canonical) is a child
/// of any allowlisted root.
fn is_within_allowed_roots(path: &Path, roots: &[PathBuf]) -> bool {
    roots.iter().any(|root| path.starts_with(root))
}

/// Read a path from the environment, defaulting to `fallback`,
/// canonicalize it, and return the canonical form.
///
/// If the path is relative it is resolved against the current working
/// directory *before* canonicalization. The canonical form must match
/// a registered allowlisted root (checked by the caller) unless the
/// allowlist is empty.
fn prepare_root_env(var: &str, create: bool) -> Result<PathBuf, String> {
    let raw = std::env::var_os(var).ok_or_else(|| format!("{var} is required"))?;
    prepare_root_path(var, Path::new(&raw), create)
}

fn prepare_root_path(var: &str, path: &Path, create: bool) -> Result<PathBuf, String> {
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| format!("cannot resolve {var}: {error}"))?
            .join(path)
    };
    if create {
        std::fs::create_dir_all(&resolved)
            .map_err(|error| format!("cannot create {var} '{}': {error}", resolved.display()))?;
    }
    resolved.canonicalize().map_err(|error| {
        format!(
            "cannot canonicalize {var} '{}': {error}",
            resolved.display()
        )
    })
}

fn startup_or_exit<T>(result: Result<T, String>) -> T {
    match result {
        Ok(value) => value,
        Err(message) => {
            eprintln!("FATAL: {message}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_allowlist_with_no_valid_roots_is_rejected() {
        let missing = std::env::temp_dir().join("reimagine-missing-allowlist-root");
        let raw = std::env::join_paths([missing]).unwrap();
        assert!(parse_allowlist_value("TEST_ROOTS", Some(raw)).is_err());
    }

    #[test]
    fn absent_required_allowlist_is_rejected() {
        assert!(parse_allowlist_value("TEST_ROOTS", None).is_err());
    }

    #[test]
    fn output_root_is_created_before_canonicalization() {
        let parent = tempfile::tempdir().unwrap();
        let output = parent.path().join("new-output");
        let canonical = prepare_root_path("TEST_OUTPUT", &output, true).unwrap();
        assert!(canonical.is_dir());
        assert_eq!(canonical, output.canonicalize().unwrap());
    }

    #[test]
    fn parse_quic_listen_url_basic() {
        let addr = parse_quic_listen_url("quic://0.0.0.0:9100");
        assert_eq!(addr, "0.0.0.0:9100".parse::<SocketAddr>().unwrap());
    }

    #[test]
    fn parse_quic_listen_url_localhost() {
        let addr = parse_quic_listen_url("quic://127.0.0.1:8080");
        assert_eq!(addr, "127.0.0.1:8080".parse::<SocketAddr>().unwrap());
    }
}
