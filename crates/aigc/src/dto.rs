//! Cross-host shared data transfer objects for AIGC capabilities.

pub mod compute_profile;
pub mod model_acquisition;
pub mod models;

pub use compute_profile::{
    BackendInstanceProfileDto, BackendProfileDto, ComputeProfileDto, DTypeProfileDto,
    DeviceProfileDto, DiagnosticDto, MemoryProfileDto, TopologyWorkerDto,
    topology_workers_from_pool,
};
pub use model_acquisition::{
    FileEntryDto, ModelAcquireConversionReport, ModelAcquireDownloadReport, ModelAcquireInput,
    ModelAcquireOutput, ModelCardDto, ModelCatalogEntryDto, ModelDownloadInput,
    ModelDownloadOutput, ModelFilters,
};
pub use models::{
    ModelComponentDto, ModelConvertConversionReport, ModelConvertInput, ModelConvertOutput,
    ModelDetailDto, ModelInfoDto, ModelListEntry, ModelListOutput, ModelRemoveOutput,
    format_status,
};
