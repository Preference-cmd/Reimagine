//! Library target for the Burn inference worker.
//!
//! Re-exports the `probe` module so integration tests can
//! verify device profile and identity construction without
//! spawning a subprocess. The binary entry point lives in
//! `main.rs` and is linked separately.

pub mod probe;
