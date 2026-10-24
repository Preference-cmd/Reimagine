//! AIGC subsystem components and operating status.

use std::sync::Arc;
use tokio::sync::Mutex;

use aigc::inference::discovery::DiscoveryOrchestrator;
use aigc::inference::topology::ConnectionTopologyManager;
use reimagine_inference::{BackendInstance, WorkspaceComputeProfile};

use crate::{ModelService, WorkerInventoryProvider, WorkerSwitchService};

/// Operating status of the workspace AIGC subsystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AigcStatus {
    /// AIGC inference runtime bootstrapped successfully.
    Available,
    /// AIGC inference runtime bootstrap failed.
    Unavailable {
        /// Reason for bootstrap failure.
        reason: String,
    },
}

impl AigcStatus {
    /// Whether AIGC is currently available.
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available)
    }

    /// The failure reason if unavailable.
    pub fn unavailable_reason(&self) -> Option<&str> {
        match self {
            Self::Available => None,
            Self::Unavailable { reason } => Some(reason),
        }
    }
}

/// Subsystems and runtime state belonging to the AIGC subsystem.
pub(crate) struct AigcParts {
    pub(crate) status: AigcStatus,
    pub(crate) model_service: Arc<ModelService>,
    pub(crate) compute_profile: Arc<WorkspaceComputeProfile>,
    pub(crate) resolved_backend_instance: BackendInstance,
    pub(crate) worker_switch: Option<Arc<WorkerSwitchService>>,
    pub(crate) worker_inventory: Arc<dyn WorkerInventoryProvider>,
    pub(crate) topology: Option<Arc<Mutex<ConnectionTopologyManager>>>,
    pub(crate) discovery: Option<Arc<DiscoveryOrchestrator>>,
}

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
