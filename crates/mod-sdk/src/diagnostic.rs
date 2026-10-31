//! Mod diagnostics. A failed call is either the caller's fault (bad input,
//! unknown mod or operation) or the mod's own.

use reimagine_core::diagnostic::{
    DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::model::DiagnosticId;

/// The error returned by every mod call, re-exported so a mod can name it
/// (`mod_sdk::diagnostic::Diagnostic`) without depending on `reimagine-core`.
pub use reimagine_core::diagnostic::Diagnostic;

/// The caller sent something the mod or the kernel cannot accept.
/// The code becomes `MOD/CALLER_<code>`.
pub fn caller_error(mod_id: &str, code: &str, message: impl Into<String>) -> Diagnostic {
    error(mod_id, "CALLER", code, message)
}

/// The mod failed while handling a valid request.
/// The code becomes `MOD/INTERNAL_<code>`.
pub fn internal_error(mod_id: &str, code: &str, message: impl Into<String>) -> Diagnostic {
    error(mod_id, "INTERNAL", code, message)
}

/// The operation was cancelled by the caller.
/// The code becomes `MOD/CALLER_CANCELLED`.
pub fn cancelled(mod_id: &str) -> Diagnostic {
    caller_error(
        mod_id,
        "CANCELLED",
        format!("operation on mod `{mod_id}` was cancelled"),
    )
}

fn error(mod_id: &str, class: &str, code: &str, message: impl Into<String>) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new(format!("mod:{mod_id}:{code}")),
        DiagnosticCode::new(format!("MOD/{class}_{code}")),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("mods"),
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new(format!("mod.{mod_id}"))),
    )
}
