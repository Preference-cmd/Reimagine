//! Runtime service facade and background runner internals.

mod diagnostics;
mod orchestrator;
mod publisher;
mod reduction;
mod service;
mod stage_reducer;

pub use service::{RuntimeOptions, RuntimeService, RuntimeServiceError};
