//! `model.load_bundle` response DTO.

use reimagine_core::RuntimeClipHandle;
use reimagine_core::RuntimeModelHandle;
use reimagine_core::RuntimeVaeHandle;

/// `model.load_bundle` response.
///
/// Returns three lightweight handles for the workflow's `model`,
/// `clip`, and `vae` outputs. The executor is responsible for
/// mapping these into the right `SlotId` outputs.
#[derive(Debug, Clone)]
pub struct LoadBundleResponse {
    model: RuntimeModelHandle,
    clip: RuntimeClipHandle,
    vae: RuntimeVaeHandle,
}

impl LoadBundleResponse {
    pub fn new(model: RuntimeModelHandle, clip: RuntimeClipHandle, vae: RuntimeVaeHandle) -> Self {
        Self { model, clip, vae }
    }

    pub fn model(&self) -> &RuntimeModelHandle {
        &self.model
    }

    pub fn clip(&self) -> &RuntimeClipHandle {
        &self.clip
    }

    pub fn vae(&self) -> &RuntimeVaeHandle {
        &self.vae
    }
}
