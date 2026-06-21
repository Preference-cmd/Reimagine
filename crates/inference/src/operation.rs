//! V1 capability identifiers.
//!
//! [`InferenceCapability`] is the closed V1 capability identity used
//! for diagnostics, capability reports, tracing, and bridge policy
//! context. The primary execution dispatch is the typed method call,
//! not the capability identifier — see
//! `crate::request` and `crate::response`.

pub use crate::InferenceCapability;
