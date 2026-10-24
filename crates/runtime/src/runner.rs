//! Runtime service facade and background runner internals.

pub mod deadlines;
mod diagnostics;
mod orchestrator;
mod publisher;
mod reduction;
mod service;
mod stage_reducer;

pub use deadlines::NodeDeadlines;
pub use service::{RuntimeOptions, RuntimeService, RuntimeServiceError};
