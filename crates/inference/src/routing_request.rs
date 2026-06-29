//! Crate-private routing metadata for typed inference requests.
//!
//! Public request DTOs stay focused on their operation payloads. This
//! module teaches the default router how to derive backend-selection
//! metadata from each typed request without reintroducing stringly
//! operation dispatch.

use reimagine_core::model::NodeId;

use crate::backend_selection::{
    BackendInstance, BackendInstanceDescriptor, BackendSelectionOverlay, BackendSelectionRequest,
};
use crate::capability::InferenceCapability;
use crate::request::diffusion::DiffusionSampleRequest;
use crate::request::image::{ImagePreviewRequest, ImageSaveRequest};
use crate::request::image_import::ImageImportRequest;
use crate::request::latent::{CreateEmptyLatentRequest, LatentDecodeRequest};
use crate::request::latent_encode::LatentEncodeRequest;
use crate::request::model::LoadBundleRequest;
use crate::request::text::TextEncodeRequest;

pub(crate) trait RoutableInferenceRequest {
    const CAPABILITY: InferenceCapability;

    fn node_id(&self) -> &NodeId;

    fn backend_affinities(&self) -> Vec<BackendInstance>;

    fn backend_selection_overlay(&self) -> &BackendSelectionOverlay;

    fn selection_request(
        &self,
        registered: Vec<BackendInstanceDescriptor>,
    ) -> BackendSelectionRequest {
        BackendSelectionRequest {
            capability: Self::CAPABILITY,
            node_id: Some(self.node_id().clone()),
            affinities: self.backend_affinities(),
            registered,
            explicit_override: self.backend_selection_overlay().explicit_override.clone(),
        }
    }
}

macro_rules! impl_routable_request {
    ($request:ty, $capability:expr) => {
        impl RoutableInferenceRequest for $request {
            const CAPABILITY: InferenceCapability = $capability;

            fn node_id(&self) -> &NodeId {
                self.node_id()
            }

            fn backend_affinities(&self) -> Vec<BackendInstance> {
                self.backend_affinities()
            }

            fn backend_selection_overlay(&self) -> &BackendSelectionOverlay {
                self.backend_selection_overlay()
            }
        }
    };
}

impl_routable_request!(LoadBundleRequest, InferenceCapability::LoadBundle);
impl_routable_request!(TextEncodeRequest, InferenceCapability::TextEncode);
impl_routable_request!(
    CreateEmptyLatentRequest,
    InferenceCapability::CreateEmptyLatent
);
impl_routable_request!(DiffusionSampleRequest, InferenceCapability::DiffusionSample);
impl_routable_request!(LatentDecodeRequest, InferenceCapability::LatentDecode);
impl_routable_request!(LatentEncodeRequest, InferenceCapability::LatentEncode);
impl_routable_request!(ImageImportRequest, InferenceCapability::ImageImport);
impl_routable_request!(ImageSaveRequest, InferenceCapability::ImageSave);
impl_routable_request!(ImagePreviewRequest, InferenceCapability::ImagePreview);
