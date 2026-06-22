//! Scheduler-aware cancellation token used by the runtime.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use reimagine_inference::NodeCancellation;
use tokio::sync::Notify;

/// Shared, cloneable cancellation token used by the runtime.
///
/// Internally uses an [`AtomicBool`] for fast poll checks and a [`Notify`]
/// so callers can also `await` cancellation. Implements the inference-side
/// [`NodeCancellation`] trait so the runner can wrap it in an
/// `Arc<dyn NodeCancellation>` and pass it through [`reimagine_inference::NodeExecutionContext`].
#[derive(Debug, Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
    notify: Arc<Notify>,
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(Notify::new()),
        }
    }

    /// Returns `true` if cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Request cancellation. All current and future observers will see it.
    pub fn cancel(&self) {
        if !self.cancelled.swap(true, Ordering::SeqCst) {
            self.notify.notify_waiters();
        }
    }

    /// Await cancellation. Returns immediately if already cancelled.
    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        // Box::pin the Notified future so we can poll it once with a
        // no-op waker to register the waiter before any re-check.
        let mut notified = Box::pin(self.notify.notified());
        {
            let noop_waker = noop_waker();
            let mut cx = std::task::Context::from_waker(&noop_waker);
            if std::future::Future::poll(notified.as_mut(), &mut cx).is_ready() {
                return;
            }
        }
        if self.is_cancelled() {
            return;
        }
        notified.await;
    }
}

#[async_trait]
impl NodeCancellation for CancellationToken {
    fn is_cancelled(&self) -> bool {
        CancellationToken::is_cancelled(self)
    }

    async fn cancelled(&self) {
        CancellationToken::cancelled(self).await;
    }
}

/// Cancellation view that trips when either underlying token is cancelled.
#[derive(Debug, Clone)]
pub struct CombinedCancellation {
    primary: CancellationToken,
    secondary: CancellationToken,
}

impl CombinedCancellation {
    pub fn new(primary: CancellationToken, secondary: CancellationToken) -> Self {
        Self { primary, secondary }
    }
}

#[async_trait]
impl NodeCancellation for CombinedCancellation {
    fn is_cancelled(&self) -> bool {
        self.primary.is_cancelled() || self.secondary.is_cancelled()
    }

    async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }

        tokio::select! {
            _ = self.primary.cancelled() => {}
            _ = self.secondary.cancelled() => {}
        }
    }
}

fn noop_waker() -> std::task::Waker {
    use std::sync::Arc;
    use std::task::{Wake, Waker};
    struct Noop;
    impl Wake for Noop {
        fn wake(self: Arc<Self>) {}
        fn wake_by_ref(self: &Arc<Self>) {}
    }
    Waker::from(Arc::new(Noop))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_token_is_not_cancelled() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn cancel_marks_token() {
        let token = CancellationToken::new();
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn cancel_is_idempotent() {
        let token = CancellationToken::new();
        token.cancel();
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[tokio::test]
    async fn cancelled_awaiter_resolves() {
        let token = CancellationToken::new();
        let waiter = {
            let token = token.clone();
            tokio::spawn(async move { token.cancelled().await })
        };
        // Give the waiter a chance to subscribe.
        tokio::task::yield_now().await;
        token.cancel();
        waiter.await.unwrap();
    }
}
