//! Burn inference backend adapter skeleton.

pub use backend::BurnBackend;
pub use config::BurnBackendConfig;
pub use device::BurnDevice;
pub use error::BurnBackendError;
pub use profile::BurnProfileProvider;
pub use resource::BurnBackendInstanceRuntimeHooks;

mod backend;
mod config;
mod device;
mod error;
mod profile;
mod resource;
