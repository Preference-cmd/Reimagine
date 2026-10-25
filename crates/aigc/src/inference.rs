//! AIGC inference composition module.
//!
//! This module is the composition root for concrete inference backends and the
//! executor-facing runtime/router wiring that sits between host bootstrap
//! and the generic runtime service.

pub mod candidate;
pub mod compose;
pub mod discovery;
pub mod health;
pub mod image_source_resolver;
pub mod pool;
pub mod remote_worker;
pub mod resolver;
pub mod selection;
pub mod switch;
pub mod topology;
pub mod worker;
