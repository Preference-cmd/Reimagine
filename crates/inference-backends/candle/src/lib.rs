#![deny(unsafe_code)]

mod backend;
mod config;
mod device;
mod error;
mod graph;
mod models;
mod operation;
mod profile;
mod resource;
mod store;

pub use backend::CandleBackend;
pub use config::CandleBackendConfig;
pub use device::CandleDevice;
pub use error::{BackendNotImplementedError, CandleBackendError};
pub use models::stable_diffusion::sdxl::checkpoint_import::{
    CANDLE_EXAMPLE_SPLIT_LAYOUT, SDXL_CHECKPOINT_IMPORT_CONVERTER_VERSION,
    SdxlCheckpointConversionManifest, SdxlCheckpointImportError, SdxlCheckpointImportRequest,
    SdxlCheckpointImportResult, SdxlConvertedComponent,
    import_sdxl_checkpoint_to_candle_example_split,
};
pub use profile::CandleProfileProvider;
pub use resource::CandleBackendInstanceRuntimeHooks;
pub use store::{
    CandleConditioning, CandleImage, CandleLatent, CandleModelCache, CandlePayload, CandleStore,
    StoreError,
};

pub use candle_core::{DType, Tensor};
pub use models::{LoadedModelBundle, LoadedSdxlBundle};
