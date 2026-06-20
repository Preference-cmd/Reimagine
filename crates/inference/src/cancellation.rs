//! Cancellation abstraction exposed to node executors.
//!
//! The executor contract owns the *shape* (this trait) so that node
//! executors can observe cancellation without depending on the
//! runtime's concrete
//! [`CancellationToken`](reimagine_runtime::CancellationToken). The
//! runtime provides the impl at context construction time and wraps it
//! in an `Arc<dyn NodeCancellation>` before handing it to the executor.

/// Abstract cancellation token handed to node executors.
///
/// Runtime provides a concrete impl (its own
/// [`CancellationToken`](reimagine_runtime::CancellationToken));
/// executors only see this trait.
///
/// Uses `async_trait` for object-safety so the executor context can
/// hold `Arc<dyn NodeCancellation>`.
#[async_trait::async_trait]
pub trait NodeCancellation: Send + Sync + 'static {
    /// Returns `true` if cancellation has been requested.
    fn is_cancelled(&self) -> bool;

    /// Await cancellation. Returns immediately if already cancelled.
    async fn cancelled(&self);
}
