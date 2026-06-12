//! Translation between Reimagine DTOs and provider-native DTOs.
//!
//! The translation layer is intentionally provider-SDK-free. It operates on
//! Reimagine DTOs and `serde_json::Value` payloads so concrete adapters can use
//! Rig or direct HTTP without leaking provider-native types into
//! `crates/agent`.

pub mod listing;
pub mod request;
pub mod response;
pub mod streaming;
pub mod tools;
