//! Cooperative task cancellation token with parent-child propagation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use tokio::sync::Notify;

/// Cheaply cloneable, cooperative cancellation token.
///
/// Cancellation propagates downward: cancelling a token cancels every token
/// derived from it with [`CancelToken::child`] or [`CancelToken::any`].
/// Cancelling a derived token never cancels its parents.
#[derive(Debug, Clone)]
pub struct CancelToken(Arc<CancelTokenInner>);

struct CancelTokenInner {
    cancelled: AtomicBool,
    notify: Notify,
    children: Mutex<Vec<Weak<CancelTokenInner>>>,
}

impl std::fmt::Debug for CancelTokenInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancelTokenInner")
            .field("cancelled", &self.cancelled.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancelToken {
    /// Creates a fresh, uncancelled cancellation token.
    pub fn new() -> Self {
        Self(Arc::new(CancelTokenInner {
            cancelled: AtomicBool::new(false),
            notify: Notify::new(),
            children: Mutex::new(Vec::new()),
        }))
    }

    /// Returns `true` once this token or any token it derives from is cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.0.cancelled.load(Ordering::SeqCst)
    }

    /// Cancels this token and every token derived from it.
    pub fn cancel(&self) {
        if self.0.cancelled.swap(true, Ordering::SeqCst) {
            return;
        }
        self.0.notify.notify_waiters();
        let children: Vec<Arc<CancelTokenInner>> = self
            .0
            .children
            .lock()
            .expect("cancel token children lock poisoned")
            .drain(..)
            .filter_map(|child| child.upgrade())
            .collect();
        for child in children {
            CancelToken(child).cancel();
        }
    }

    /// Resolves once the token is cancelled; immediately if it already is.
    pub async fn cancelled(&self) {
        let mut notified = std::pin::pin!(self.0.notify.notified());
        // Register as a waiter before re-checking the flag, so a cancel that
        // lands between the check and the await still wakes this future.
        notified.as_mut().enable();
        if self.is_cancelled() {
            return;
        }
        notified.await;
    }

    /// Creates a token that is cancelled when `self` is cancelled.
    pub fn child(&self) -> Self {
        Self::any(std::slice::from_ref(self))
    }

    /// Creates a token that is cancelled when any of `parents` is cancelled.
    pub fn any(parents: &[CancelToken]) -> Self {
        let child = Self::new();
        for parent in parents {
            let mut children = parent
                .0
                .children
                .lock()
                .expect("cancel token children lock poisoned");
            children.retain(|existing| existing.strong_count() > 0);
            children.push(Arc::downgrade(&child.0));
        }
        // A parent cancelled before registration drained its list without
        // this child, so check once more after registering.
        if parents.iter().any(CancelToken::is_cancelled) {
            child.cancel();
        }
        child
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_token_is_not_cancelled() {
        let token = CancelToken::new();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn cancel_marks_token_and_is_idempotent() {
        let token = CancelToken::new();
        token.cancel();
        assert!(token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn parent_cancel_propagates_to_child() {
        let parent = CancelToken::new();
        let child = parent.child();

        assert!(!child.is_cancelled());
        parent.cancel();
        assert!(child.is_cancelled());
    }

    #[test]
    fn child_cancel_does_not_propagate_to_parent() {
        let parent = CancelToken::new();
        let child = parent.child();

        child.cancel();
        assert!(child.is_cancelled());
        assert!(!parent.is_cancelled());
    }

    #[test]
    fn child_of_already_cancelled_parent_is_immediately_cancelled() {
        let parent = CancelToken::new();
        parent.cancel();
        let child = parent.child();
        assert!(child.is_cancelled());
    }

    #[test]
    fn linked_token_cancels_when_any_parent_cancels() {
        let p1 = CancelToken::new();
        let p2 = CancelToken::new();
        let p3 = CancelToken::new();
        let linked = CancelToken::any(&[p1.clone(), p2.clone(), p3.clone()]);

        assert!(!linked.is_cancelled());
        p2.cancel();
        assert!(linked.is_cancelled());
        assert!(!p1.is_cancelled());
        assert!(!p3.is_cancelled());
    }

    #[test]
    fn linked_token_with_precancelled_parent_is_immediately_cancelled() {
        let p1 = CancelToken::new();
        let p2 = CancelToken::new();
        p2.cancel();
        let linked = CancelToken::any(&[p1.clone(), p2]);
        assert!(linked.is_cancelled());
        assert!(!p1.is_cancelled());
    }

    #[tokio::test]
    async fn cancelled_awaiter_resolves_on_cancel() {
        let token = CancelToken::new();
        let waiter = {
            let token = token.clone();
            tokio::spawn(async move { token.cancelled().await })
        };
        tokio::task::yield_now().await;
        token.cancel();
        waiter.await.unwrap();
    }

    #[tokio::test]
    async fn child_cancelled_awaiter_resolves_when_parent_cancels() {
        let parent = CancelToken::new();
        let child = parent.child();
        let waiter = {
            let child = child.clone();
            tokio::spawn(async move { child.cancelled().await })
        };
        tokio::task::yield_now().await;
        parent.cancel();
        waiter.await.unwrap();
        assert!(child.is_cancelled());
    }

    #[tokio::test]
    async fn linked_token_awaiter_resolves_when_any_parent_cancels() {
        let p1 = CancelToken::new();
        let p2 = CancelToken::new();
        let linked = CancelToken::any(&[p1.clone(), p2.clone()]);
        let waiter = {
            let linked = linked.clone();
            tokio::spawn(async move { linked.cancelled().await })
        };
        tokio::task::yield_now().await;
        p2.cancel();
        waiter.await.unwrap();
        assert!(linked.is_cancelled());
    }
}
