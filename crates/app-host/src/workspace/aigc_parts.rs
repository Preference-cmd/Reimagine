//! AIGC subsystem components and operating status.

#[cfg(feature = "aigc")]
use std::sync::Arc;
#[cfg(feature = "aigc")]
use tokio::sync::Mutex;

#[cfg(feature = "aigc")]
use aigc::inference::discovery::DiscoveryOrchestrator;
#[cfg(feature = "aigc")]
use aigc::inference::topology::ConnectionTopologyManager;
use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::health::Health;
use reimagine_core::model::DiagnosticId;
#[cfg(feature = "aigc")]
use reimagine_inference::{BackendInstance, WorkspaceComputeProfile};

#[cfg(feature = "aigc")]
use crate::{ModelService, WorkerInventoryProvider, WorkerSwitchService};

/// Diagnostic for when AIGC inference runtime bootstrap fails.
#[cfg(feature = "aigc")]
pub(crate) fn aigc_bootstrap_failed_diagnostic(message: impl Into<String>) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new("app-host:aigc:bootstrap_failed"),
        DiagnosticCode::new("AIGC/BOOTSTRAP_FAILED"),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("app-host"),
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new("app-host.aigc")),
    )
}

/// Diagnostic for when AIGC is not installed in the current build.
#[cfg(not(feature = "aigc"))]
pub(crate) fn aigc_not_installed_diagnostic() -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new("app-host:aigc:not_installed"),
        DiagnosticCode::new("AIGC/NOT_INSTALLED"),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("app-host"),
        "AIGC is not installed",
        DiagnosticTarget::new(DiagnosticTargetDomain::new("app-host.aigc")),
    )
}

#[cfg(not(feature = "aigc"))]
pub(crate) static AIGC_NOT_INSTALLED_HEALTH: std::sync::LazyLock<Health> =
    std::sync::LazyLock::new(|| Health::Unavailable(aigc_not_installed_diagnostic()));

/// Subsystems and runtime state belonging to the AIGC subsystem.
#[cfg(feature = "aigc")]
pub(crate) struct AigcParts {
    pub(crate) status: Health,
    pub(crate) model_service: Arc<ModelService>,
    pub(crate) compute_profile: Arc<WorkspaceComputeProfile>,
    pub(crate) resolved_backend_instance: BackendInstance,
    pub(crate) worker_switch: Option<Arc<WorkerSwitchService>>,
    pub(crate) worker_inventory: Arc<dyn WorkerInventoryProvider>,
    pub(crate) topology: Option<Arc<Mutex<ConnectionTopologyManager>>>,
    pub(crate) discovery: Option<Arc<DiscoveryOrchestrator>>,
}

#[cfg(feature = "aigc")]
impl std::fmt::Debug for AigcParts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AigcParts")
            .field("status", &self.status)
            .field("compute_profile", &self.compute_profile)
            .field("resolved_backend_instance", &self.resolved_backend_instance)
            .field("worker_switch", &self.worker_switch)
            .finish_non_exhaustive()
    }
}
