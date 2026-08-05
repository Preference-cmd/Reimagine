//! Cooperative cancellation for Burn operations.
//!
//! The worker installs a per-request `CancellationToken` into the
//! current thread via [`with_request_cancellation`] before dispatching
//! an operation; operations poll [`is_cancelled`] at cheap check
//! points (text.encode entry, between denoising steps). When no
//! request-scoped token is installed the backend-wide token held by
//! [`BurnRuntime`](crate::runtime::BurnRuntime) is consulted instead.

use std::cell::RefCell;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::backend::BurnBackend;
use crate::error::BurnBackendError;

thread_local! {
    static REQUEST_SCOPE: RefCell<Option<Arc<CancellationToken>>> =
        const { RefCell::new(None) };
}

/// Run `operation` with `token` installed as the request-scoped
/// cancellation token for the current thread.
pub fn with_request_cancellation<T>(
    token: Arc<CancellationToken>,
    operation: impl FnOnce() -> T,
) -> T {
    REQUEST_SCOPE.with(|slot| {
        let previous = slot.replace(Some(token));
        let result = operation();
        slot.replace(previous);
        result
    })
}

/// Check whether the current operation should abort. A request-scoped
/// token installed via [`with_request_cancellation`] takes precedence;
/// otherwise `fallback` (the backend-wide token) is consulted.
pub fn is_cancelled(fallback: &CancellationToken) -> bool {
    REQUEST_SCOPE.with(|slot| {
        slot.borrow()
            .as_ref()
            .map_or_else(|| fallback.is_cancelled(), |token| token.is_cancelled())
    })
}

/// Fail with [`BurnBackendError::Cancelled`] if the operation has
/// been cancelled.
pub fn ensure_not_cancelled(backend: &BurnBackend) -> Result<(), BurnBackendError> {
    if is_cancelled(&backend.cancellation()) {
        Err(BurnBackendError::Cancelled)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BurnBackendConfig;

    #[test]
    fn request_scope_overrides_fallback_token() {
        let backend = BurnBackend::new(BurnBackendConfig::new("/models", "/output")).unwrap();
        let scope = Arc::new(CancellationToken::new());
        assert!(!is_cancelled(&backend.cancellation()));

        with_request_cancellation(scope.clone(), || {
            assert!(!is_cancelled(&backend.cancellation()));
            scope.cancel();
            assert!(is_cancelled(&backend.cancellation()));
            assert!(matches!(
                ensure_not_cancelled(&backend),
                Err(BurnBackendError::Cancelled)
            ));
        });

        assert!(
            !is_cancelled(&backend.cancellation()),
            "scope must be removed after with_request_cancellation returns"
        );
    }

    #[test]
    fn fallback_token_is_consulted_without_request_scope() {
        let backend = BurnBackend::new(BurnBackendConfig::new("/models", "/output")).unwrap();
        assert!(!is_cancelled(&backend.cancellation()));

        backend.cancellation().cancel();
        assert!(is_cancelled(&backend.cancellation()));
        assert!(matches!(
            ensure_not_cancelled(&backend),
            Err(BurnBackendError::Cancelled)
        ));
    }
}
