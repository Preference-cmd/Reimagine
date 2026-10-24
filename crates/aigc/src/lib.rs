//! The AIGC domain module (inference composition, worker management, model services).
//!
//! This crate owns the domain logic for AI generative computing:
//! - Inference composition, topology, and backend switching
//! - Worker inventory and out-of-process worker management
//! - Model management, conversion, and acquisition services
//!
//! Services in this crate are assembled by `reimagine-app-host` to expose
//! workspace capabilities to host adapters. Invariants:
//! - All inference operations validate backend readiness and compatibility.
//! - Out-of-process worker switches maintain drain and cancellation semantics.
//! - Model downloads and checkpoint conversions preserve manifest integrity.

#![deny(unsafe_code)]

pub mod dto;
mod error;
pub mod inference;
mod inference_backend;
mod model_acquisition_service;
mod model_conversion;
mod model_service;
mod readiness;
mod worker_management;

pub use error::{AigcError, AigcResult};
pub use inference::grpc_worker::{
    GrpcSwitchableWorker, GrpcWorkerCandidate, GrpcWorkerCandidateConfig,
};
pub use inference::quic_worker::{
    QuicSwitchableWorker, QuicWorkerCandidate, QuicWorkerCandidateConfig,
};
pub use inference::switch::{
    ProcessSwitchableWorker, RunCancellation, SwitchableWorker, WorkerSelectionHandle,
    WorkerSwitchError, WorkerSwitchService, WorkerSwitchTarget,
};
pub use inference::worker::{
    EmptyWorkerInventoryProvider, InstalledWorkerInventoryProvider, StaticWorkerInventoryProvider,
    WorkerActivationError, WorkerBackendCandidate, WorkerInventoryProvider,
    WorkerInventorySnapshot,
};
pub use inference_backend::BackendSelection;
pub use model_acquisition_service::ModelAcquisitionService;
pub use model_conversion::{
    BurnCheckpointConverter, BurnConversionComponent, BurnConversionComponentRole,
    BurnConversionReport,
};
pub use model_service::{AcquireAndConvertReport, AcquireAndConvertRequest, ModelService};
pub use readiness::SnapshotExternalReadinessProvider;
pub use reimagine_backend_worker_transport_grpc::{GrpcAuth, GrpcTls};
#[cfg(feature = "candle")]
pub use reimagine_inference_candle;
pub use worker_management::{
    WorkerCatalogItemDto, WorkerInstallationDto, WorkerManagementError, WorkerManagementService,
};
