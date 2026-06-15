#![deny(unsafe_code)]

mod backend;
mod config;
mod device;
mod error;
mod models;
mod operation;
mod resource;
mod store;

pub use backend::CandleBackend;
pub use config::CandleBackendConfig;
pub use device::CandleDevice;
pub use error::{BackendNotImplementedError, CandleBackendError};
pub use resource::CandleRunResourceBackend;
pub use store::{CandleLatent, CandleModelCache, CandlePayload, CandleStore, StoreError};

pub use candle_core::{DType, Tensor};
pub use models::{LoadedModelBundle, LoadedSdxlBundle};
