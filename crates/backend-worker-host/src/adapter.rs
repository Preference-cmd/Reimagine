use std::sync::Arc;

use reimagine_backend_worker_protocol::TerminalOutcome;
use reimagine_core::model::TensorShape;
use reimagine_inference::{
    Backend, BackendInstance, BackendTensorHandle, CreateEmptyLatentRequest,
    CreateEmptyLatentResponse, DiffusionSampleRequest, DiffusionSampleResponse, ImageImportRequest,
    ImageImportResponse, ImagePreviewRequest, ImagePreviewResponse, ImageSaveRequest,
    ImageSaveResponse, InferenceBackend, InferenceBackendCapabilities, InferenceCapability,
    InferenceCapabilitySupport, InferenceError, LatentContent, LatentDecodeRequest,
    LatentDecodeResponse, LatentEncodeRequest, LatentEncodeResponse, LoadBundleRequest,
    LoadBundleResponse, RuntimeLatent, TextEncodeRequest, TextEncodeResponse,
};

use crate::StartedWorker;
use crate::authority::WorkerAuthorityTable;

pub struct ProcessInferenceBackend {
    backend: Backend,
    instance: BackendInstance,
    worker: Arc<StartedWorker>,
    authority: WorkerAuthorityTable,
    supports_create_empty_latent: bool,
}

impl ProcessInferenceBackend {
    #[must_use]
    pub fn new(worker: Arc<StartedWorker>) -> Self {
        let supports_create_empty_latent = worker
            .hello
            .profile
            .instances
            .iter()
            .find(|profile| {
                profile.backend_instance_id == worker.hello.identity.backend_instance_id
            })
            .is_some_and(|profile| {
                profile
                    .capabilities
                    .iter()
                    .any(|capability| capability == InferenceCapability::CreateEmptyLatent.as_str())
            });
        Self {
            backend: Backend::new(worker.hello.identity.backend_kind.clone()),
            instance: BackendInstance::new(worker.hello.identity.backend_instance_id.0.clone()),
            authority: WorkerAuthorityTable::new(worker.hello.identity.incarnation_id.clone()),
            worker,
            supports_create_empty_latent,
        }
    }

    fn not_implemented(&self, capability: InferenceCapability) -> InferenceError {
        InferenceError::BackendNotImplemented {
            capability,
            backend_kind: self.backend.to_string(),
            message: Some("worker backend does not advertise this capability".to_owned()),
        }
    }
}

#[async_trait::async_trait]
impl InferenceBackend for ProcessInferenceBackend {
    fn backend_kind(&self) -> &Backend {
        &self.backend
    }

    fn capabilities(&self) -> InferenceBackendCapabilities {
        let capabilities = InferenceBackendCapabilities::new(self.backend.clone());
        if self.supports_create_empty_latent {
            capabilities.with_support(InferenceCapabilitySupport::new(
                InferenceCapability::CreateEmptyLatent,
            ))
        } else {
            capabilities
        }
    }

    async fn load_bundle(
        &self,
        _request: LoadBundleRequest,
    ) -> Result<LoadBundleResponse, InferenceError> {
        Err(self.not_implemented(InferenceCapability::LoadBundle))
    }

    async fn text_encode(
        &self,
        _request: TextEncodeRequest,
    ) -> Result<TextEncodeResponse, InferenceError> {
        Err(self.not_implemented(InferenceCapability::TextEncode))
    }

    async fn create_empty_latent(
        &self,
        request: CreateEmptyLatentRequest,
    ) -> Result<CreateEmptyLatentResponse, InferenceError> {
        if !self.supports_create_empty_latent {
            return Err(self.not_implemented(InferenceCapability::CreateEmptyLatent));
        }
        let width = request.width();
        let height = request.height();
        let batch_size = request.batch_size();
        let latent_space = request.latent_space().clone();
        let result = self
            .worker
            .request(
                InferenceCapability::CreateEmptyLatent.as_str(),
                serde_json::json!({
                    "width": width,
                    "height": height,
                    "batch_size": batch_size,
                }),
            )
            .await
            .map_err(|error| InferenceError::BackendExecutionFailed {
                message: error.to_string(),
            })?;
        let output = match result.terminal.outcome {
            TerminalOutcome::Success { output } => output,
            TerminalOutcome::Cancelled => {
                return Err(InferenceError::BackendExecutionFailed {
                    message: "worker request was cancelled".to_owned(),
                });
            }
            TerminalOutcome::BackendError { error } => {
                return Err(InferenceError::BackendExecutionFailed {
                    message: format!("{}: {}", error.code, error.message),
                });
            }
        };
        let worker_token = output
            .get("worker_token")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| InferenceError::InvalidResponse {
                reason: "latent.create_empty response omitted worker_token".to_owned(),
            })?;
        for (field, expected) in [
            ("width", width as u64),
            ("height", height as u64),
            ("batch_size", batch_size as u64),
        ] {
            if output.get(field).and_then(serde_json::Value::as_u64) != Some(expected) {
                return Err(InferenceError::InvalidResponse {
                    reason: format!("latent.create_empty response changed `{field}`"),
                });
            }
        }
        let scale = latent_space.spatial_scale_factor();
        let channels = latent_space.channels();
        let payload_key = self.authority.register(worker_token.to_owned());
        let resolved_token = self
            .authority
            .resolve(self.authority.incarnation_id(), &payload_key)
            .map_err(|error| InferenceError::InvalidResponse {
                reason: error.to_string(),
            })?;
        if resolved_token != worker_token {
            return Err(InferenceError::InvalidResponse {
                reason: "worker authority resolved a different payload token".to_owned(),
            });
        }
        let payload = BackendTensorHandle::with_instance(
            self.backend.clone(),
            self.instance.clone(),
            payload_key,
            latent_space.dtype(),
            TensorShape::new(vec![
                batch_size as usize,
                channels as usize,
                (height / scale) as usize,
                (width / scale) as usize,
            ]),
            "worker",
        );
        Ok(CreateEmptyLatentResponse::new(RuntimeLatent::new(
            payload,
            width,
            height,
            batch_size,
            channels,
            latent_space,
            LatentContent::EmptyGeometry,
        )))
    }

    async fn diffusion_sample(
        &self,
        _request: DiffusionSampleRequest,
    ) -> Result<DiffusionSampleResponse, InferenceError> {
        Err(self.not_implemented(InferenceCapability::DiffusionSample))
    }

    async fn latent_decode(
        &self,
        _request: LatentDecodeRequest,
    ) -> Result<LatentDecodeResponse, InferenceError> {
        Err(self.not_implemented(InferenceCapability::LatentDecode))
    }

    async fn latent_encode(
        &self,
        _request: LatentEncodeRequest,
    ) -> Result<LatentEncodeResponse, InferenceError> {
        Err(self.not_implemented(InferenceCapability::LatentEncode))
    }

    async fn image_import(
        &self,
        _request: ImageImportRequest,
    ) -> Result<ImageImportResponse, InferenceError> {
        Err(self.not_implemented(InferenceCapability::ImageImport))
    }

    async fn image_save(
        &self,
        _request: ImageSaveRequest,
    ) -> Result<ImageSaveResponse, InferenceError> {
        Err(self.not_implemented(InferenceCapability::ImageSave))
    }

    async fn image_preview(
        &self,
        _request: ImagePreviewRequest,
    ) -> Result<ImagePreviewResponse, InferenceError> {
        Err(self.not_implemented(InferenceCapability::ImagePreview))
    }
}
