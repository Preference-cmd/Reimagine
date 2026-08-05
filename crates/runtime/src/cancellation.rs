//! Scheduler-aware hierarchical cancellation token used by the runtime.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use reimagine_core::model::NodeId;
use reimagine_inference::NodeCancellation;
use tokio::sync::Notify;

/// Shared, cloneable hierarchical cancellation token used by the runtime.
///
/// Two levels:
/// - a run-level token ([`AtomicBool`] + [`Notify`]) — cancelling it
///   cancels every node in the run (and is what hosts drive today);
/// - per-node tokens (`NodeId` → [`CancellationToken`]) — cancelling a
///   node affects only that node's executor;
/// - per-node deadlines (`NodeId` → [`Instant`]) — a node past its
///   deadline is failed with a timeout diagnostic by the stage reducer.
///
/// Implements the inference-side [`NodeCancellation`] trait at run level
/// so the runner can wrap it in an `Arc<dyn NodeCancellation>` and pass
/// it through [`reimagine_inference::NodeExecutionContext`].
#[derive(Debug, Clone)]
pub struct CancellationToken {
    run_token: Arc<AtomicBool>,
    notify: Arc<Notify>,
    node_tokens: Arc<Mutex<HashMap<NodeId, CancellationToken>>>,
    deadlines: Arc<Mutex<HashMap<NodeId, Instant>>>,
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            run_token: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(Notify::new()),
            node_tokens: Arc::new(Mutex::new(HashMap::new())),
            deadlines: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Returns `true` if the run has been asked to stop.
    pub fn is_cancelled(&self) -> bool {
        self.run_token.load(Ordering::SeqCst)
    }

    /// Request cancellation of the whole run. All current and future
    /// observers — including every node token — will see it.
    pub fn cancel(&self) {
        self.cancel_run();
    }

    /// Request cancellation of the whole run. Explicitly run-scoped
    /// sibling of [`Self::cancel`].
    pub fn cancel_run(&self) {
        if !self.run_token.swap(true, Ordering::SeqCst) {
            self.notify.notify_waiters();
        }
    }

    /// Await run-level cancellation. Returns immediately if already
    /// cancelled.
    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        // Box::pin the Notified future so we can poll it once with a
        // no-op waker to register the waiter before any re-check.
        let mut notified = Box::pin(self.notify.notified());
        {
            let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
            if std::future::Future::poll(notified.as_mut(), &mut cx).is_ready() {
                return;
            }
        }
        if self.is_cancelled() {
            return;
        }
        notified.await;
    }

    /// Request cancellation of a single node.
    ///
    /// The node's executor observes the cancellation through its
    /// `NodeCancellation` and returns `NodeExecutorError::Cancelled`; the
    /// stage reducer classifies that result as node-level (the run
    /// continues) as long as the run token itself is not cancelled.
    pub fn cancel_node(&self, node_id: &NodeId) {
        self.node_cancellation(node_id).cancel();
    }

    /// Returns `true` if the node is cancelled — either because the run
    /// was cancelled or because this specific node was cancelled.
    pub fn is_node_cancelled(&self, node_id: &NodeId) -> bool {
        self.is_cancelled()
            || self
                .node_tokens
                .lock()
                .expect("node token map poisoned")
                .get(node_id)
                .is_some_and(CancellationToken::is_cancelled)
    }

    /// Get-or-create the node-scoped token for `node_id`.
    ///
    /// The returned token is independent: cancelling it only affects this
    /// node, and it stays shared across all clones of this token (the
    /// runner's admission loop and the executor wiring see the same flag).
    pub fn node_cancellation(&self, node_id: &NodeId) -> CancellationToken {
        let mut tokens = self.node_tokens.lock().expect("node token map poisoned");
        tokens.entry(node_id.clone()).or_default().clone()
    }

    /// Arm a deadline for `node_id` starting now: the node must complete
    /// within `timeout`, or the stage reducer fails it with a timeout
    /// diagnostic.
    pub fn set_deadline(&self, node_id: &NodeId, timeout: Duration) {
        self.set_deadline_at(node_id, Instant::now() + timeout);
    }

    /// Arm an absolute deadline for `node_id`.
    pub fn set_deadline_at(&self, node_id: &NodeId, deadline: Instant) {
        self.deadlines
            .lock()
            .expect("deadline map poisoned")
            .insert(node_id.clone(), deadline);
    }

    /// The deadline currently armed for `node_id`, if any.
    pub fn node_deadline(&self, node_id: &NodeId) -> Option<Instant> {
        self.deadlines
            .lock()
            .expect("deadline map poisoned")
            .get(node_id)
            .copied()
    }

    /// Returns `true` when `node_id` has a deadline in the past.
    pub fn is_node_expired(&self, node_id: &NodeId) -> bool {
        self.node_deadline(node_id)
            .is_some_and(|deadline| Instant::now() >= deadline)
    }

    /// Disarm any deadline for `node_id`.
    pub fn clear_deadline(&self, node_id: &NodeId) {
        self.deadlines
            .lock()
            .expect("deadline map poisoned")
            .remove(node_id);
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

/// Cancellation view that trips when any underlying token is cancelled.
#[derive(Debug, Clone)]
pub struct CombinedCancellation {
    primary: CancellationToken,
    secondary: CancellationToken,
    tertiary: CancellationToken,
}

impl CombinedCancellation {
    /// Combine three independent tokens (run-level, node-level, and
    /// stage-level fail-fast) into a single cancellation view.
    pub fn triple(
        primary: CancellationToken,
        secondary: CancellationToken,
        tertiary: CancellationToken,
    ) -> Self {
        Self {
            primary,
            secondary,
            tertiary,
        }
    }
}

#[async_trait]
impl NodeCancellation for CombinedCancellation {
    fn is_cancelled(&self) -> bool {
        self.primary.is_cancelled() || self.secondary.is_cancelled() || self.tertiary.is_cancelled()
    }

    async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }

        tokio::select! {
            _ = self.primary.cancelled() => {}
            _ = self.secondary.cancelled() => {}
            _ = self.tertiary.cancelled() => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_token_is_not_cancelled() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
        assert!(!token.is_node_cancelled(&NodeId::new("a")));
        assert!(!token.is_node_expired(&NodeId::new("a")));
    }

    #[test]
    fn cancel_marks_run_token() {
        let token = CancellationToken::new();
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn cancel_run_is_an_alias_for_cancel() {
        let token = CancellationToken::new();
        token.cancel_run();
        assert!(token.is_cancelled());
    }

    #[test]
    fn cancel_is_idempotent() {
        let token = CancellationToken::new();
        token.cancel();
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn cancel_node_affects_only_that_node() {
        let token = CancellationToken::new();
        token.cancel_node(&NodeId::new("a"));

        assert!(token.is_node_cancelled(&NodeId::new("a")));
        assert!(!token.is_node_cancelled(&NodeId::new("b")));
        // The run token is untouched.
        assert!(!token.is_cancelled());
    }

    #[test]
    fn node_tokens_are_shared_across_clones() {
        let token = CancellationToken::new();
        let clone = token.clone();
        token.cancel_node(&NodeId::new("a"));

        assert!(clone.is_node_cancelled(&NodeId::new("a")));
        assert!(clone.node_cancellation(&NodeId::new("a")).is_cancelled());
        assert!(!clone.node_cancellation(&NodeId::new("b")).is_cancelled());
    }

    #[test]
    fn run_cancel_propagates_to_node_tokens() {
        let token = CancellationToken::new();
        token.cancel_run();

        assert!(token.is_node_cancelled(&NodeId::new("a")));
        assert!(token.is_node_cancelled(&NodeId::new("b")));
    }

    #[test]
    fn deadline_expiry_is_time_based() {
        let token = CancellationToken::new();
        let node = NodeId::new("a");

        token.set_deadline(&node, Duration::from_secs(3600));
        assert!(!token.is_node_expired(&node));

        token.set_deadline_at(&node, Instant::now() - Duration::from_secs(1));
        assert!(token.is_node_expired(&node));

        token.clear_deadline(&node);
        assert!(!token.is_node_expired(&node));
    }

    #[test]
    fn deadlines_are_per_node() {
        let token = CancellationToken::new();
        let a = NodeId::new("a");
        let b = NodeId::new("b");

        token.set_deadline_at(&a, Instant::now() - Duration::from_secs(1));
        assert!(token.is_node_expired(&a));
        assert!(!token.is_node_expired(&b));
        assert_eq!(token.node_deadline(&b), None);
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

    #[tokio::test]
    async fn combined_cancellation_awaiter_resolves_on_any_token() {
        let run = CancellationToken::new();
        let node = CancellationToken::new();
        let combined =
            CombinedCancellation::triple(run.clone(), node.clone(), CancellationToken::new());
        let waiter = {
            let combined = combined.clone();
            tokio::spawn(async move { combined.cancelled().await })
        };
        tokio::task::yield_now().await;
        node.cancel();
        waiter.await.unwrap();
        assert!(combined.is_cancelled());
    }

    #[tokio::test]
    async fn node_cancelled_awaiter_resolves() {
        let token = CancellationToken::new();
        let node = token.node_cancellation(&NodeId::new("a"));
        let waiter = {
            let node = node.clone();
            tokio::spawn(async move { node.cancelled().await })
        };
        tokio::task::yield_now().await;
        token.cancel_node(&NodeId::new("a"));
        waiter.await.unwrap();
    }
}
