//! Burn inference worker serving the `reimagine.worker.v2` gRPC
//! contract.
//!
//! The host launches the worker with `--listen <endpoint>` and a
//! per-launch bearer token (`REIMAGINE_WORKER_TOKEN`). The worker binds
//! the endpoint, prints one `READY <endpoint>` line on stdout (its only
//! stdout output), and serves until the host closes its stdin. Endpoints:
//! a private local socket (`unix://<path>`, Windows `pipe://<name>`) for
//! host-launched workers, or `https://<ip>:<port>` for remote workers,
//! which additionally serves TLS with a self-signed certificate and
//! registers itself with mDNS for discovery.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use clap::Parser;
use reimagine_backend_worker_protocol::auth::TOKEN_ENV;
use reimagine_backend_worker_protocol::discovery::MdnsWorkerRegister;
use reimagine_backend_worker_protocol::server::{
    ListenEndpoint, bind_listen, serve_bound, worker_server,
};
use reimagine_inference_burn::{BurnBackend, BurnBackendConfig};

mod mapping;
mod probe;
mod service;
mod shutdown;

/// Grace period between a shutdown acknowledgement and process exit,
/// giving the ack time to reach the host.
const SHUTDOWN_EXIT_GRACE: Duration = Duration::from_millis(200);

#[derive(Parser)]
#[command(
    name = "reimagine-inference-burn-worker",
    about = "Reimagine Burn inference worker"
)]
struct Cli {
    /// Endpoint to serve the v2 contract on: `unix://<path>` or, on
    /// Windows, `pipe://<name>` for a host-launched worker; `https://<ip>:<port>`
    /// for a remote worker.
    #[arg(long = "listen")]
    listen: String,
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
        log(&format!(
            "FATAL: models dir '{}' is not within allowed MODEL roots",
            models_dir.display()
        ));
        std::process::exit(1);
    }
    if !is_within_allowed_roots(&output_dir, &allowed_output_roots) {
        log(&format!(
            "FATAL: output dir '{}' is not within allowed OUTPUT roots",
            output_dir.display()
        ));
        std::process::exit(1);
    }

    // ----- initialize Burn backend -----
    let config = BurnBackendConfig::new(&models_dir, &output_dir);
    let backend = match BurnBackend::new(config) {
        Ok(backend) => backend,
        Err(error) => {
            log(&format!(
                "FATAL: failed to initialize Burn backend: {error}"
            ));
            std::process::exit(1);
        }
    };

    let endpoint: ListenEndpoint = match ListenEndpoint::parse(&cli.listen) {
        Ok(endpoint) => endpoint,
        Err(error) => {
            log(&format!(
                "FATAL: invalid --listen `{}`: {error}",
                cli.listen
            ));
            std::process::exit(1);
        }
    };
    let token = std::env::var(TOKEN_ENV)
        .ok()
        .filter(|value| !value.is_empty());

    // Identity for the mDNS registration of a remote (`https://`)
    // worker; computed before the backend moves into the service.
    let mdns_identity = if matches!(endpoint, ListenEndpoint::Https(_)) {
        Some(probe::build(&backend))
    } else {
        None
    };

    let shutdown_requested = Arc::new(AtomicBool::new(false));
    let service = service::BurnWorkerService::new(backend, token, Arc::clone(&shutdown_requested));

    // Lifeline: the host holds our stdin open; EOF means the host is
    // gone and this process must exit instead of becoming an orphan.
    std::thread::spawn(|| {
        use std::io::Read;
        let mut stdin = std::io::stdin().lock();
        let mut buffer = [0_u8; 256];
        loop {
            match stdin.read(&mut buffer) {
                Ok(0) | Err(_) => std::process::exit(0),
                Ok(_) => {}
            }
        }
    });

    // Graceful shutdown: leave a grace period after the shutdown ack so
    // the response still reaches the host.
    {
        let shutdown_requested = Arc::clone(&shutdown_requested);
        std::thread::spawn(move || {
            while !shutdown_requested.load(Ordering::Acquire) {
                std::thread::sleep(Duration::from_millis(20));
            }
            std::thread::sleep(SHUTDOWN_EXIT_GRACE);
            std::process::exit(0);
        });
    }

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            log(&format!("FATAL: failed to create tokio runtime: {error}"));
            std::process::exit(1);
        }
    };

    runtime.block_on(async move {
        let bound = match bind_listen(&endpoint).await {
            Ok(bound) => bound,
            Err(error) => {
                log(&format!(
                    "FATAL: cannot bind {}: {error}",
                    endpoint.to_uri()
                ));
                std::process::exit(1);
            }
        };
        // A remote worker advertises itself on the LAN so hosts can
        // discover it; the advertised fingerprint lets a host pre-validate
        // its TOFU pin. Registration failure is not fatal: the worker
        // stays reachable at its endpoint with token authentication. The
        // registration handle is held for the worker's whole lifetime;
        // dropping it would unregister the service.
        let mut mdns_registration = None;
        if let (Some((identity, profile)), Some(tls_identity)) = (&mdns_identity, bound.identity())
        {
            let capabilities: Vec<&str> = profile
                .instances
                .iter()
                .flat_map(|instance| instance.capabilities.iter().map(String::as_str))
                .collect();
            let devices: Vec<&str> = profile
                .instances
                .iter()
                .map(|instance| instance.device_label.as_str())
                .collect();
            let mut properties = HashMap::new();
            properties.insert("endpoint".to_owned(), bound.uri().to_owned());
            properties.insert("backend".to_owned(), identity.backend_kind.clone());
            properties.insert("devices".to_owned(), devices.join(","));
            properties.insert("capabilities".to_owned(), capabilities.join(","));
            properties.insert(
                "fingerprint".to_owned(),
                tls_identity.fingerprint().as_str().to_owned(),
            );
            match MdnsWorkerRegister::register(
                &identity.backend_instance_id.0,
                bound.https_addr().expect("https listener has an address"),
                properties,
            ) {
                Ok(registration) => {
                    log(&format!(
                        "worker: mDNS registered as '{}'",
                        identity.backend_instance_id.0
                    ));
                    mdns_registration = Some(registration);
                }
                Err(error) => {
                    log(&format!(
                        "worker: mDNS registration failed (worker stays reachable at {}): {error}",
                        bound.uri()
                    ));
                }
            }
        }
        // The READY line is the only stdout output of a worker and is
        // printed only once the endpoint is bound.
        let mut stdout = std::io::stdout().lock();
        if writeln!(stdout, "READY {}", bound.uri()).is_err() || stdout.flush().is_err() {
            std::process::exit(1);
        }
        drop(stdout);

        if let Err(error) = serve_bound(bound, worker_server(service)).await {
            log(&format!("worker server error: {error}"));
        }
        drop(mdns_registration);
    });

    log("worker: server stopped, terminating");
}

/// Write to stderr, ignoring a broken pipe: the host may already be
/// gone, and panicking on a dead log sink would kill the worker before
/// it can honor its lifeline.
fn log(message: &str) {
    let _ = std::io::stderr().write_all(message.as_bytes());
    let _ = std::io::stderr().write_all(b"\n");
}

// ---------------------------------------------------------------------------
// CLI helpers
// ---------------------------------------------------------------------------

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
                    log(&format!(
                        "worker: {var} entry '{}' cannot be canonicalized: {e}",
                        path.display()
                    ));
                }
            }
        } else {
            log(&format!(
                "worker: {var} entry '{}' is not absolute — skipping",
                path.display()
            ));
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
            log(&format!("FATAL: {message}"));
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
}
