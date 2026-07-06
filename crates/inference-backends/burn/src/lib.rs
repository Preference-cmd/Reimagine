//! Burn inference backend adapter skeleton.

pub use backend::BurnBackend;
pub use config::BurnBackendConfig;
pub use device::BurnDevice;
pub use error::BurnBackendError;
pub use profile::BurnProfileProvider;
pub use resource::BurnBackendInstanceRuntimeHooks;

mod active_backend;
mod backend;
mod config;
mod device;
mod error;
pub mod models;
mod operation;
mod profile;
mod resource;
mod runtime;
mod store;
pub mod text_encoder;
