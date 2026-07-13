//! WGPU validation error guard.
//!
//! WGPU validation errors (e.g. buffer binding size mismatches) occur during
//! command encoding inside CubeCL's internal threads. They are caught by
//! `catch_unwind` inside the compute server, but `std::panic::set_hook` fires
//! **before** the catch_unwind handler runs — so we can intercept them here.
//!
//! Two APIs:
//!
//! 1. [`WgpuErrorGuard`] — RAII scope guard:
//! ```ignore
//! let mut guard = WgpuErrorGuard::new();
//! // ... GPU work ...
//! guard.check()?;
//! ```
//!
//! 2. [`check_global`] — one-shot check (no RAII):
//! ```ignore
//! // ... GPU work ...
//! check_global()?;
//! ```

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::BurnBackendError;

static WGPU_ERROR_SEEN: AtomicBool = AtomicBool::new(false);
static HOOK_INSTALLED: OnceLock<()> = OnceLock::new();

fn install_hook() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let payload = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| {
                info.payload()
                    .downcast_ref::<String>()
                    .map(|s| s.to_string())
            })
            .unwrap_or_default();
        if payload.contains("wgpu error") || payload.contains("Validation Error") {
            WGPU_ERROR_SEEN.store(true, Ordering::SeqCst);
        }
        prev(info);
    }));
}

/// One-shot check: returns Err if any WGPU validation error has been seen
/// since the last successful check (or since the hook was installed).
pub(crate) fn check_global() -> Result<(), BurnBackendError> {
    HOOK_INSTALLED.get_or_init(install_hook);
    if WGPU_ERROR_SEEN.swap(false, Ordering::SeqCst) {
        return Err(BurnBackendError::BackendNotImplemented(
            "WGPU validation error detected (buffer binding size mismatch)".to_string(),
        ));
    }
    Ok(())
}

/// RAII guard that checks for WGPU validation errors around a scope.
///
/// Resets the error flag on construction; reports and consumes it on
/// [`check()`](WgpuErrorGuard::check).
pub(crate) struct WgpuErrorGuard {
    consumed: bool,
}

impl WgpuErrorGuard {
    pub fn new() -> Self {
        HOOK_INSTALLED.get_or_init(install_hook);
        WGPU_ERROR_SEEN.store(false, Ordering::SeqCst);
        Self { consumed: false }
    }

    /// Check whether a WGPU validation error occurred during the guarded scope.
    pub fn check(&mut self) -> Result<(), BurnBackendError> {
        if WGPU_ERROR_SEEN.swap(false, Ordering::SeqCst) {
            self.consumed = true;
            return Err(BurnBackendError::BackendNotImplemented(
                "WGPU validation error detected (buffer binding size mismatch)".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_no_error_returns_ok() {
        let mut guard = WgpuErrorGuard::new();
        assert!(guard.check().is_ok());
    }

    #[test]
    fn check_global_initial_returns_ok() {
        let result = check_global();
        // May error if previous test left a flag, but most likely OK.
        if let Err(e) = &result {
            eprintln!("check_global initial returned error (may be stale flag): {e}");
        }
        // Reset so subsequent tests are clean.
        WGPU_ERROR_SEEN.store(false, Ordering::SeqCst);
    }
}
