mod adapter;
mod authority;
pub mod catalog;
mod error;
pub mod install;
pub mod inventory;
pub mod launch;
mod leases;
pub mod package;
pub mod store_paths;
mod supervisor;

pub use adapter::ProcessInferenceBackend;
pub use catalog::{
    CatalogClient, CatalogError, CatalogResult, CatalogTarget, CompatibilityFilter, HostInfo,
    TargetCustomMetadata, VerifiedCatalog,
};
pub use error::WorkerHostError;
pub use install::{
    InstallConfig, InstallEngine, InstallError, InstallJournal, InstallResult, JournalManager,
    PromotionEngine, SelfCheckConfig, SelfCheckRunner, StagingManager,
};
pub use inventory::{
    InstallationRecord, InventoryError, InventoryResult, InventorySnapshot, InventoryStore,
    VersionPolicy,
};
pub use launch::{ExpectedWorkerIdentity, WorkerLaunchSpec, WorkerLimits};
pub use leases::{WorkerAdmissionState, WorkerRunLeaseError, WorkerRunLeases};
pub use package::{
    ExtractionLimits, PackageError, PackageExtractor, PackageFileEntry, PackageManifest,
    PackageResult,
};
pub use store_paths::WorkerStorePaths;
pub use supervisor::{
    StartedWorker, WorkerProcessState, WorkerRequestCanceller, WorkerRequestHandle,
    WorkerRequestResult, WorkerSupervisor,
};

// Re-export worker protocol types used in our public API
pub use reimagine_backend_worker_protocol::{
    BackendInstanceId, ProtocolRange, ProtocolVersion, WorkerIdentity, WorkerIncarnationId,
    WorkerInstallationId, WorkerInstanceProfile, WorkerProfile,
};
