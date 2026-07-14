//! App-host owned inference composition.
//!
//! This module is the composition root for concrete inference backends and the
//! executor-facing runtime/router wiring that sits between app-host bootstrap
//! and the generic runtime service.

pub(crate) mod candidate;
pub(crate) mod compose;
pub(crate) mod image_source_resolver;
pub(crate) mod resolver;
pub(crate) mod selection;
pub(crate) mod worker;
