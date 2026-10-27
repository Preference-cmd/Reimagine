//! Conformance kit for mod authors, behind the `testing` feature.
//!
//! - [`check_manifest`]: static checks on a manifest, beyond what the
//!   registry enforces at registration.
//! - [`validate`]: check an output or event payload against its schema.
//! - [`SampleMod`]: a reference mod that exercises every interface behavior.

mod check;
mod sample;

pub use check::{check_manifest, validate};
pub use sample::SampleMod;
