//! `image.save`, `image.preview`, and `image.import` operations.
//!
//! This module keeps the backend capability entrypoints small and
//! delegates import decoding, artifact persistence, and tensor/image
//! encoding to focused internal modules. Host-side workspace path
//! authorization still lives outside Candle; this layer only consumes
//! already-resolved sources and backend-owned image payloads.

mod encoding;
mod import;
mod persistence;

pub use import::execute_image_import;
pub use persistence::{execute_image_preview, execute_image_save};
