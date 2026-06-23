//! HTTP DTOs (request/response payloads) for the Axum host.
//!
//! V1 re-exports the cross-host DTOs from `reimagine_app_host::dto`.
//! If the Axum host ever needs HTTP-specific DTO extensions (e.g.
//! pagination wrappers), they go in this module alongside the
//! re-exports.

pub use reimagine_app_host::dto::*;
