//! Fingerprint verification facade.

mod refresh;
mod sha256;

pub use refresh::{FingerprintRefresh, ModelFingerprintVerifier};
