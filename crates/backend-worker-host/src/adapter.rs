use std::sync::Arc;

use reimagine_backend_worker_protocol::TerminalOutcome;
use reimagine_core::model::{ArtifactRef, TensorShape};
use reimagine_inference::{
    Backend, BackendInstance, BackendTensorHandle, CreateEmptyLatentRequest,
    CreateEmptyLatentResponse, DiffusionSampleRequest, DiffusionSampleResponse,
    ExecutionConditioning, ImagePreviewRequest, ImagePreviewResponse, ImageSaveRequest,
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
    capabilities: Vec<String>,
}

impl ProcessInferenceBackend {
    #[must_use]
    pub fn new(worker: Arc<StartedWorker>) -> Self {
        // Collect capabilities advertised by the worker's profile for
        // this backend instance.
        let capabilities: Vec<String> = worker
            .hello
            .profile
            .instances
            .iter()
            .find(|profile| {
                profile.backend_instance_id == worker.hello.identity.backend_instance_id
            })
            .map(|profile| profile.capabilities.clone())
            .unwrap_or_default();

        Self {
            backend: Backend::new(worker.hello.identity.backend_kind.clone()),
            instance: BackendInstance::new(worker.hello.identity.backend_instance_id.0.clone()),
            authority: WorkerAuthorityTable::new(worker.hello.identity.incarnation_id.clone()),
            worker,
            capabilities,
        }
    }

    /// Check whether a capability is advertised by the worker.
    fn supports(&self, cap: &InferenceCapability) -> bool {
        self.capabilities
            .iter()
            .any(|c| c == cap.as_str())
    }

    fn not_implemented(&self, capability: InferenceCapability) -> InferenceError {
        InferenceError::BackendNotImplemented {
            capability,
            backend_kind: self.backend.to_string(),
            message: Some("worker backend does not advertise this capability".to_owned()),
        }
    }

    /// Register a worker token and return the host-side backend payload key.
    fn register_token(&self, worker_token: &str) -> reimagine_inference::BackendPayloadKey {
        self.authority.register(worker_token.to_owned())
    }

    /// Resolve a host-side payload key back to a worker token.
    fn resolve_token(
        &self,
        host_key: &reimagine_inference::BackendPayloadKey,
    ) -> Result<String, InferenceError> {
        self.authority
            .resolve(self.authority.incarnation_id(), host_key)
            .map_err(|e| InferenceError::InvalidResponse {
                reason: e.to_string(),
            })
    }

    /// Send a request to the worker and parse the terminal response.
    async fn send_request(
        &self,
        operation: &str,
        payload: serde_json::Value,
    ) -> Result<serde_json::Value, InferenceError> {
        let result = self
            .worker
            .request(operation, payload)
            .await
            .map_err(|error| InferenceError::BackendExecutionFailed {
                message: error.to_string(),
            })?;
        match result.terminal.outcome {
            TerminalOutcome::Success { output } => Ok(output),
            TerminalOutcome::Cancelled => Err(InferenceError::BackendExecutionFailed {
                message: "worker request was cancelled".to_owned(),
            }),
            TerminalOutcome::BackendError { error } => {
                Err(InferenceError::BackendExecutionFailed {
                    message: format!("{}: {}", error.code, error.message),
                })
            }
        }
    }

    /// Build a BackendTensorHandle from a worker response token.
    fn make_handle(
        &self,
        token: &str,
        dtype: reimagine_core::model::TensorDType,
        shape: TensorShape,
    ) -> BackendTensorHandle {
        BackendTensorHandle::with_instance(
            self.backend.clone(),
            self.instance.clone(),
            self.register_token(token),
            dtype,
            shape,
            self.instance.as_str(),
        )
    }
}

#[async_trait::async_trait]
impl InferenceBackend for ProcessInferenceBackend {
    fn backend_kind(&self) -> &Backend {
        &self.backend
    }

    fn capabilities(&self) -> InferenceBackendCapabilities {
        let mut caps = InferenceBackendCapabilities::new(self.backend.clone());
        for capability_str in &self.capabilities {
            let capability = match InferenceCapability::from_label(capability_str) {
                Some(c) => c,
                None => continue,
            };
            caps = caps.with_support(InferenceCapabilitySupport::new(capability));
        }
        caps
    }

    async fn load_bundle(
        &self,
        request: LoadBundleRequest,
    ) -> Result<LoadBundleResponse, InferenceError> {
        if !self.supports(&InferenceCapability::LoadBundle) {
            return Err(self.not_implemented(InferenceCapability::LoadBundle));
        }

        let resolved = request.resolved_model();
        let output = self
            .send_request(
                "model.load_bundle",
                serde_json::json!({
                    "model_id": resolved.model_id().as_str(),
                    "series": resolved.series().as_str(),
                    "variant": resolved.variant().as_str(),
                    "role": format!("{:?}", resolved.role()),
                    "source_path": resolved.source_path().to_string_lossy().to_string(),
                }),
            )
            .await?;

        let model_token = output
            .get("model_token")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| InferenceError::InvalidResponse {
                reason: "load_bundle response omitted model_token".to_owned(),
            })?;

        let clip_token = output
            .get("clip_token")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| InferenceError::InvalidResponse {
                reason: "load_bundle response omitted clip_token".to_owned(),
            })?;

        let vae_token = output
            .get("vae_token")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| InferenceError::InvalidResponse {
                reason: "load_bundle response omitted vae_token".to_owned(),
            })?;

        let model_key = self.register_token(model_token);
        let clip_key = self.register_token(clip_token);
        let vae_key = self.register_token(vae_token);

        let model_handle = reimagine_inference::RuntimeModelHandle::with_instance(
            resolved.model_id().clone(),
            resolved.role(),
            self.backend.clone(),
            self.instance.clone(),
            model_key,
        );

        let clip_handle = reimagine_inference::RuntimeClipHandle::with_instance(
            resolved.model_id().clone(),
            self.backend.clone(),
            self.instance.clone(),
            clip_key,
        );

        let vae_handle = reimagine_inference::RuntimeVaeHandle::with_instance(
            resolved.model_id().clone(),
            self.backend.clone(),
            self.instance.clone(),
            vae_key,
        );

        Ok(LoadBundleResponse::new(model_handle, clip_handle, vae_handle))
    }

    async fn text_encode(
        &self,
        request: TextEncodeRequest,
    ) -> Result<TextEncodeResponse, InferenceError> {
        if !self.supports(&InferenceCapability::TextEncode) {
            return Err(self.not_implemented(InferenceCapability::TextEncode));
        }

        let prompt = request
            .prompt_string()
            .ok_or_else(|| InferenceError::BackendExecutionFailed {
                message: "text_encode requires a string prompt".to_owned(),
            })?;

        let clip_token = request.clip().payload_key().as_str().to_owned();

        let output = self
            .send_request(
                "text.encode",
                serde_json::json!({
                    "clip_token": clip_token,
                    "prompt_text": prompt,
                }),
            )
            .await?;

        let cond_token = output
            .get("conditioning_token")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| InferenceError::InvalidResponse {
                reason: "text_encode response omitted conditioning_token".to_owned(),
            })?;

        let text_embedding = self.make_handle(
            cond_token,
            reimagine_core::model::TensorDType::F32,
            TensorShape::new(vec![1, 77, 2048]),
        );

        let mut conditioning = ExecutionConditioning::new(
            text_embedding,
            reimagine_inference::ConditioningMetadata::new(1024, 1024),
        );

        if let Some(pooled_token) = output.get("pooled_token").and_then(serde_json::Value::as_str) {
            let pooled_embedding = self.make_handle(
                pooled_token,
                reimagine_core::model::TensorDType::F32,
                TensorShape::new(vec![1, 1280]),
            );
            conditioning = conditioning.with_pooled_embedding(pooled_embedding);
        }

        Ok(TextEncodeResponse::new(conditioning))
    }

    async fn create_empty_latent(
        &self,
        request: CreateEmptyLatentRequest,
    ) -> Result<CreateEmptyLatentResponse, InferenceError> {
        if !self.supports(&InferenceCapability::CreateEmptyLatent) {
            return Err(self.not_implemented(InferenceCapability::CreateEmptyLatent));
        }

        let width = request.width();
        let height = request.height();
        let batch_size = request.batch_size();
        let latent_space = request.latent_space().clone();

        let output = self
            .send_request(
                "latent.create_empty",
                serde_json::json!({
                    "width": width,
                    "height": height,
                    "batch_size": batch_size,
                }),
            )
            .await?;

        let worker_token = output
            .get("worker_token")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| InferenceError::InvalidResponse {
                reason: "create_empty_latent response omitted worker_token".to_owned(),
            })?;

        // Validate response shape reflection
        for (field, expected) in [
            ("width", width as u64),
            ("height", height as u64),
            ("batch_size", batch_size as u64),
        ] {
            if output.get(field).and_then(serde_json::Value::as_u64) != Some(expected) {
                return Err(InferenceError::InvalidResponse {
                    reason: format!("create_empty_latent response changed `{field}`"),
                });
            }
        }

        let scale = latent_space.spatial_scale_factor();
        let channels = latent_space.channels();
        let payload_key = self.register_token(worker_token);

        // Verify authority registration roundtrips
        let resolved_token = self.resolve_token(&payload_key)?;
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
        request: DiffusionSampleRequest,
    ) -> Result<DiffusionSampleResponse, InferenceError> {
        if !self.supports(&InferenceCapability::DiffusionSample) {
            return Err(self.not_implemented(InferenceCapability::DiffusionSample));
        }

        let model_token = request.model().payload_key().as_str().to_owned();
        let pos_cond_token = request.positive().text_embedding().payload_key().as_str().to_owned();
        let neg_cond_token = request.negative().text_embedding().payload_key().as_str().to_owned();
        let latent_token = request.latent().payload().payload_key().as_str().to_owned();

        let output = self
            .send_request(
                "diffusion.sample",
                serde_json::json!({
                    "model_token": model_token,
                    "pos_cond_token": pos_cond_token,
                    "neg_cond_token": neg_cond_token,
                    "latent_token": latent_token,
                    "seed": request.seed(),
                    "steps": request.steps(),
                    "cfg": request.cfg(),
                    "denoise": request.denoise(),
                    "sampler": request.sampler().as_str(),
                    "scheduler": request.scheduler().as_str(),
                }),
            )
            .await?;

        let result_latent_token = output
            .get("latent_token")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| InferenceError::InvalidResponse {
                reason: "diffusion_sample response omitted latent_token".to_owned(),
            })?;

        let latent_payload = self.make_handle(
            result_latent_token,
            reimagine_core::model::TensorDType::F32,
            TensorShape::new(vec![1, 4, 128, 128]),
        );

        let latent = RuntimeLatent::new(
            latent_payload,
            1024,
            1024,
            request.latent().batch(),
            4,
            request.latent().latent_space().clone(),
            LatentContent::Sampled,
        );

        Ok(DiffusionSampleResponse::new(latent))
    }

    async fn latent_decode(
        &self,
        request: LatentDecodeRequest,
    ) -> Result<LatentDecodeResponse, InferenceError> {
        if !self.supports(&InferenceCapability::LatentDecode) {
            return Err(self.not_implemented(InferenceCapability::LatentDecode));
        }

        let vae_token = request.vae().payload_key().as_str().to_owned();
        let latent_token = request.latent().payload().payload_key().as_str().to_owned();

        let output = self
            .send_request(
                InferenceCapability::LatentDecode.as_str(),
                serde_json::json!({
                    "vae_token": vae_token,
                    "latent_token": latent_token,
                }),
            )
            .await?;

        let image_token = output
            .get("image_token")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| InferenceError::InvalidResponse {
                reason: "latent_decode response omitted image_token".to_owned(),
            })?;

        let width = output
            .get("width")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1024) as u32;
        let height = output
            .get("height")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1024) as u32;

        let image_payload = self.make_handle(
            image_token,
            reimagine_core::model::TensorDType::F32,
            TensorShape::new(vec![1, 3, height as usize, width as usize]),
        );

        let image = reimagine_inference::RuntimeImage::new(
            image_payload,
            width,
            height,
            1,
            "rgb",
        );

        Ok(LatentDecodeResponse::new(image))
    }

    async fn latent_encode(
        &self,
        _request: LatentEncodeRequest,
    ) -> Result<LatentEncodeResponse, InferenceError> {
        Err(self.not_implemented(InferenceCapability::LatentEncode))
    }

    async fn image_save(
        &self,
        request: ImageSaveRequest,
    ) -> Result<ImageSaveResponse, InferenceError> {
        if !self.supports(&InferenceCapability::ImageSave) {
            return Err(self.not_implemented(InferenceCapability::ImageSave));
        }

        let image_token = request.image().payload().payload_key().as_str().to_owned();

        let output = self
            .send_request(
                InferenceCapability::ImageSave.as_str(),
                serde_json::json!({
                    "image_token": image_token,
                    "filename_prefix": request.filename_prefix().as_str(),
                }),
            )
            .await?;

        let artifact_path = output
            .get("artifact_path")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| InferenceError::InvalidResponse {
                reason: "image_save response omitted artifact_path".to_owned(),
            })?;

        Ok(ImageSaveResponse::new(ArtifactRef::new(artifact_path)))
    }

    async fn image_preview(
        &self,
        request: ImagePreviewRequest,
    ) -> Result<ImagePreviewResponse, InferenceError> {
        if !self.supports(&InferenceCapability::ImagePreview) {
            return Err(self.not_implemented(InferenceCapability::ImagePreview));
        }

        let image_token = request.image().payload().payload_key().as_str().to_owned();

        let output = self
            .send_request(
                InferenceCapability::ImagePreview.as_str(),
                serde_json::json!({
                    "image_token": image_token,
                }),
            )
            .await?;

        let artifact_path = output
            .get("artifact_path")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| InferenceError::InvalidResponse {
                reason: "image_preview response omitted artifact_path".to_owned(),
            })?;

        Ok(ImagePreviewResponse::new(ArtifactRef::new(artifact_path)))
    }

    async fn image_import(
        &self,
        _request: reimagine_inference::ImageImportRequest,
    ) -> Result<reimagine_inference::ImageImportResponse, InferenceError> {
        Err(self.not_implemented(InferenceCapability::ImageImport))
    }
}
