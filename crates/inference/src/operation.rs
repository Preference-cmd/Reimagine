//! V1 capability identifiers.
//!
//! [`InferenceCapability`] is the closed V1 capability identity used
//! for diagnostics, capability reports, tracing, and bridge policy
//! context. The primary execution dispatch is the typed method call,
//! not the capability identifier — see
//! `reimagine_inference_core::request` and `reimagine_inference_core::response`.

pub use reimagine_inference_core::InferenceCapability;
