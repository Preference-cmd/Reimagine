use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use reimagine_backend_worker_protocol::{
    FrameCodec, HostHello, ProtocolRange, ProtocolVersion, WireMessage,
};
use reimagine_inference_burn::{BurnBackend, BurnBackendConfig};

mod mapping;
mod probe;
mod server;
mod shutdown;

const WORKER_PROTOCOL_VERSION: ProtocolVersion = ProtocolVersion(1);
const MAX_FRAME_BYTES: u32 = 64 * 1024 * 1024; // 64 MB

fn main() {
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
    let (identity, profile) = probe::build(&backend);
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
    server::serve_loop(
        &rt,
        &backend,
        &codec,
        &mut reader,
        &mut writer,
        selected,
        &incarnation_id,
    );

    eprintln!("worker: serve loop exited, terminating");
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
}
