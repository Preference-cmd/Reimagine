//! HTTP API handlers, grouped by resource.
//!
//! Each submodule owns a single resource family and is the only place
//! that knows how to translate between HTTP DTOs and app-host facade
//! calls. See `dto.rs` for the request/response shapes.

pub mod artifacts;
pub mod compute_profile;
pub mod health;
pub mod models;
pub mod nodes;
pub mod runs;
pub mod workflows;
