//! `reimagine-candle-integration`
//!
//! Candle-specific implementation of the inference backend defined in
//! `reimagine-core`. M0 stub: types compile and a model can be loaded into
//! a session, but `infer` returns `NotImplemented`. Real Candle wiring
//! lands at M1.

pub mod loader;
pub mod models;
pub mod runtime;

pub use loader::load;
pub use runtime::Session;