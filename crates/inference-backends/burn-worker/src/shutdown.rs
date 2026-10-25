//! Deterministic cleanup and graceful exit for the Burn worker.
//!
//! On shutdown the worker releases stored payloads and cached model
//! bundles before the process exits. After `execute` returns the
//! process must exit deterministically (no background threads, no
//! lingering GPU resources).

use reimagine_inference_burn::BurnBackend;

/// Release all stored payloads and cached model bundles. After this
/// call the process is ready to terminate.
pub fn cleanup(backend: &BurnBackend) {
    let payloads = backend.store().cleanup_all();
    let bundles = backend.model_cache().clear();
    writeln_stderr(format!(
        "worker shutdown: {payloads} payloads released, {bundles} cached bundles"
    ));
}

/// Write to stderr, ignoring a broken pipe: the host may already be
/// gone and a panic here would take the worker down mid-exit.
pub fn writeln_stderr(message: String) {
    use std::io::Write;
    let _ = std::io::stderr().write_all(message.as_bytes());
    let _ = std::io::stderr().write_all(b"\n");
}
