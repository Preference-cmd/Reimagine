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
    // ----- validate allowlisted roots (authoritative directory anchors) -----
    let allowed_model_roots = parse_allowlist("REIMAGINE_ALLOWED_MODEL_ROOTS");
    let allowed_output_roots = parse_allowlist("REIMAGINE_ALLOWED_OUTPUT_ROOTS");

    // ----- config from environment (must be inside allowed roots) -----
    let models_dir = canonicalize_env("REIMAGINE_MODELS_DIR", ".");
    let output_dir = canonicalize_env("REIMAGINE_OUTPUT_DIR", ".");

    if !allowed_model_roots.is_empty() {
        if !is_within_allowed_roots(&models_dir, &allowed_model_roots) {
            eprintln!(
                "FATAL: models dir '{}' is not within allowed MODEL roots",
                models_dir.display(),
            );
            std::process::exit(1);
        }
    }
    if !allowed_output_roots.is_empty() {
        if !is_within_allowed_roots(&output_dir, &allowed_output_roots) {
            eprintln!(
                "FATAL: output dir '{}' is not within allowed OUTPUT roots",
                output_dir.display(),
            );
            std::process::exit(1);
        }
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
    let worker_range = ProtocolRange::new(
        WORKER_PROTOCOL_VERSION.0,
        WORKER_PROTOCOL_VERSION.0,
    );
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
    server::serve_loop(&rt, &backend, &codec, &mut reader, &mut writer);

    eprintln!("worker: serve loop exited, terminating");
}

/// Parse the `:`-separated allowlist from an environment variable
/// into a `Vec` of canonical `PathBuf` entries.
fn parse_allowlist(var: &str) -> Vec<PathBuf> {
    let raw = match std::env::var(var) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let mut roots: Vec<PathBuf> = Vec::new();
    for entry in raw.split(':') {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            continue;
        }
        let path = PathBuf::from(trimmed);
        if path.is_absolute() {
            // Canonicalize to resolve symlinks so the comparison is
            // robust against symlink-based path variations.
            match path.canonicalize() {
                Ok(canon) => roots.push(canon),
                Err(e) => {
                    eprintln!(
                        "worker: {var} entry '{trimmed}' cannot be canonicalized: {e}"
                    );
                }
            }
        } else {
            eprintln!(
                "worker: {var} entry '{trimmed}' is not absolute — skipping"
            );
        }
    }
    roots
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
fn canonicalize_env(var: &str, fallback: &str) -> PathBuf {
    let raw = std::env::var(var).unwrap_or_else(|_| fallback.to_string());
    let path = PathBuf::from(&raw);
    let resolved = if path.is_absolute() {
        path.clone()
    } else {
        match std::env::current_dir() {
            Ok(cwd) => {
                let abs = cwd.join(&path);
                eprintln!(
                    "worker: {var}='{raw}' is relative, resolved to '{}'",
                    abs.display()
                );
                abs
            }
            Err(e) => {
                eprintln!("FATAL: cannot resolve {var}='{raw}': {e}");
                std::process::exit(1);
            }
        }
    };
    match resolved.canonicalize() {
        Ok(canon) => canon,
        Err(e) => {
            eprintln!(
                "FATAL: cannot canonicalize {var}='{}': {e}",
                resolved.display()
            );
            std::process::exit(1);
        }
    }
}