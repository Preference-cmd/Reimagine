//! App-host owned inference composition.
//!
//! This module is the composition root for concrete inference backends and the
//! executor-facing runtime/router wiring that sits between app-host bootstrap
//! and the generic runtime service.

pub(crate) mod candidate;
pub(crate) mod compose;
pub mod discovery;
pub(crate) mod grpc_worker;
pub mod health;
pub(crate) mod image_source_resolver;
pub mod pool;
pub(crate) mod quic_worker;
pub(crate) mod resolver;
pub(crate) mod selection;
pub mod switch;
pub mod topology;
pub(crate) mod worker;
