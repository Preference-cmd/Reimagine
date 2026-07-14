use std::future::Future;
use std::sync::Arc;

use reimagine_backend_worker_protocol::TerminalOutcome;
use reimagine_core::model::{ArtifactRef, TensorShape};
use reimagine_inference::{
    Backend, BackendInstance, BackendTensorHandle, CreateEmptyLatentRequest,
    CreateEmptyLatentResponse, DiffusionSampleRequest, DiffusionSampleResponse,
    ExecutionConditioning, ImageImportRequest, ImageImportResponse, ImagePreviewRequest,
    ImagePreviewResponse, ImageSaveRequest, ImageSaveResponse, InferenceBackend,
    InferenceBackendCapabilities, InferenceCapability, InferenceCapabilitySupport, InferenceError,
    InferenceInvocation, InferenceProgress, LatentContent, LatentDecodeRequest,
    LatentDecodeResponse, LatentEncodeRequest, LatentEncodeResponse, LoadBundleRequest,
    LoadBundleResponse, RuntimeLatent, TextEncodeRequest, TextEncodeResponse,
};

use crate::StartedWorker;
use crate::authority::WorkerAuthorityTable;

tokio::task_local! {
    static PROCESS_INVOCATION: InferenceInvocation;
}

pub struct ProcessInferenceBackend {
    backend: Backend,
    instance: BackendInstance,
    worker: Arc<StartedWorker>,
    authority: WorkerAuthorityTable,
    capabilities: Vec<String>,
    run_leases: Arc<crate::WorkerRunLeases>,
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
            run_leases: Arc::new(crate::WorkerRunLeases::new()),
        }
    }

    pub fn run_leases(&self) -> Arc<crate::WorkerRunLeases> {
        Arc::clone(&self.run_leases)
    }

    /// Check whether a capability is advertised by the worker.
    fn supports(&self, cap: &InferenceCapability) -> bool {
        self.capabilities.iter().any(|c| c == cap.as_str())
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
        let invocation = PROCESS_INVOCATION.try_with(Clone::clone).ok();
        if let Some(invocation) = invocation {
            return self
                .send_request_with_invocation(&invocation, operation, payload)
                .await;
        }
        let result = self
            .worker
            .request(operation, payload)
            .await
            .map_err(|error| InferenceError::BackendExecutionFailed {
                message: error.to_string(),
            })?;
        Self::map_worker_result(result)
    }

    async fn send_request_with_invocation(
        &self,
        invocation: &InferenceInvocation,
        operation: &str,
        payload: serde_json::Value,
    ) -> Result<serde_json::Value, InferenceError> {
        let handle = self
            .worker
            .begin_request(operation, payload)
            .await
            .map_err(|error| InferenceError::BackendExecutionFailed {
                message: error.to_string(),
            })?;
        let canceller = handle.canceller();
        let finish = handle.finish_with_progress(|frame| {
            invocation.progress().report(InferenceProgress {
                sequence: frame.sequence,
                completed: frame.completed,
                total: frame.total,
                message: frame.message.clone(),
            });
        });
        tokio::pin!(finish);
        let result = tokio::select! {
            biased;
            result = &mut finish => result,
            () = invocation.cancellation().cancelled() => {
                canceller.cancel().await.map_err(|error| {
                    InferenceError::BackendExecutionFailed { message: error.to_string() }
                })?;
                finish.await
            }
        }
        .map_err(|error| InferenceError::BackendExecutionFailed {
            message: error.to_string(),
        })?;
        Self::map_worker_result(result)
    }

    fn map_worker_result(
        result: crate::WorkerRequestResult,
    ) -> Result<serde_json::Value, InferenceError> {
        match result.terminal.outcome {
            TerminalOutcome::Success { output } => Ok(output),
            TerminalOutcome::Cancelled => Err(InferenceError::Cancelled),
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

    async fn invoke<T, F>(
        &self,
        invocation: &InferenceInvocation,
        operation: F,
    ) -> Result<T, InferenceError>
    where
        F: Future<Output = Result<T, InferenceError>>,
    {
        self.run_leases
            .acquire(invocation.run_id())
            .map_err(|error| InferenceError::BackendExecutionFailed {
                message: error.to_string(),
            })?;
        PROCESS_INVOCATION
            .scope(invocation.clone(), operation)
            .await
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

    async fn load_bundle_with_invocation(
        &self,
        invocation: &InferenceInvocation,
        request: LoadBundleRequest,
    ) -> Result<LoadBundleResponse, InferenceError> {
        self.invoke(invocation, self.load_bundle(request)).await
    }

    async fn text_encode_with_invocation(
        &self,
        invocation: &InferenceInvocation,
        request: TextEncodeRequest,
    ) -> Result<TextEncodeResponse, InferenceError> {
        self.invoke(invocation, self.text_encode(request)).await
    }

    async fn create_empty_latent_with_invocation(
        &self,
        invocation: &InferenceInvocation,
        request: CreateEmptyLatentRequest,
    ) -> Result<CreateEmptyLatentResponse, InferenceError> {
        self.invoke(invocation, self.create_empty_latent(request))
            .await
    }

    async fn diffusion_sample_with_invocation(
        &self,
        invocation: &InferenceInvocation,
        request: DiffusionSampleRequest,
    ) -> Result<DiffusionSampleResponse, InferenceError> {
        self.invoke(invocation, self.diffusion_sample(request))
            .await
    }

    async fn latent_decode_with_invocation(
        &self,
        invocation: &InferenceInvocation,
        request: LatentDecodeRequest,
    ) -> Result<LatentDecodeResponse, InferenceError> {
        self.invoke(invocation, self.latent_decode(request)).await
    }

    async fn latent_encode_with_invocation(
        &self,
        invocation: &InferenceInvocation,
        request: LatentEncodeRequest,
    ) -> Result<LatentEncodeResponse, InferenceError> {
        self.invoke(invocation, self.latent_encode(request)).await
    }

    async fn image_import_with_invocation(
        &self,
        invocation: &InferenceInvocation,
        request: ImageImportRequest,
    ) -> Result<ImageImportResponse, InferenceError> {
        self.invoke(invocation, self.image_import(request)).await
    }

    async fn image_save_with_invocation(
        &self,
        invocation: &InferenceInvocation,
        request: ImageSaveRequest,
    ) -> Result<ImageSaveResponse, InferenceError> {
        self.invoke(invocation, self.image_save(request)).await
    }

    async fn image_preview_with_invocation(
        &self,
        invocation: &InferenceInvocation,
        request: ImagePreviewRequest,
    ) -> Result<ImagePreviewResponse, InferenceError> {
        self.invoke(invocation, self.image_preview(request)).await
    }

    async fn load_bundle(
        &self,
        request: LoadBundleRequest,
    ) -> Result<LoadBundleResponse, InferenceError> {
        if !self.supports(&InferenceCapability::LoadBundle) {
            return Err(self.not_implemented(InferenceCapability::LoadBundle));
        }

        let resolved = request.resolved_model();
        let components = resolved.source_set().map(|source_set| {
            source_set
                .sources()
                .iter()
                .map(|source| {
                    serde_json::json!({
                        "kind": source.kind(),
                        "role": source.role(),
                        "path": source.path().to_string_lossy(),
                        "format": source.format(),
                        "metadata": source.metadata(),
                    })
                })
                .collect::<Vec<_>>()
        });
        let output = self
            .send_request(
                "model.load_bundle",
                serde_json::json!({
                    "model_id": resolved.model_id().as_str(),
                    "series": resolved.series().as_str(),
                    "variant": resolved.variant().as_str(),
                    "role": format!("{:?}", resolved.role()),
                    "source_path": resolved.source_path().to_string_lossy().to_string(),
                    "components": components,
                    "run_id": request.run_id().as_str(),
                    "workflow_id": request.workflow_id().as_str(),
                    "workflow_version": request.workflow_version().get(),
                    "node_id": request.node_id().as_str(),
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

        Ok(LoadBundleResponse::new(
            model_handle,
            clip_handle,
            vae_handle,
        ))
    }

    async fn text_encode(
        &self,
        request: TextEncodeRequest,
    ) -> Result<TextEncodeResponse, InferenceError> {
        if !self.supports(&InferenceCapability::TextEncode) {
            return Err(self.not_implemented(InferenceCapability::TextEncode));
        }

        let prompt =
            request
                .prompt_string()
                .ok_or_else(|| InferenceError::BackendExecutionFailed {
                    message: "text_encode requires a string prompt".to_owned(),
                })?;

        let clip_token = self.resolve_token(request.clip().payload_key())?;

        let output = self
            .send_request(
                "text.encode",
                serde_json::json!({
                    "clip_token": clip_token,
                    "model_id": request.clip().model_id().as_str(),
                    "prompt_text": prompt,
                    "run_id": request.run_id().as_str(),
                    "workflow_id": request.workflow_id().as_str(),
                    "workflow_version": request.workflow_version().get(),
                    "node_id": request.node_id().as_str(),
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

        if let Some(pooled_token) = output
            .get("pooled_token")
            .and_then(serde_json::Value::as_str)
        {
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
                    "run_id": request.run_id().as_str(),
                    "workflow_id": request.workflow_id().as_str(),
                    "workflow_version": request.workflow_version().get(),
                    "node_id": request.node_id().as_str(),
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

        let model_token = self.resolve_token(request.model().payload_key())?;
        let pos_cond_token =
            self.resolve_token(request.positive().text_embedding().payload_key())?;
        let neg_cond_token =
            self.resolve_token(request.negative().text_embedding().payload_key())?;
        let pos_pooled_token = request
            .positive()
            .pooled_embedding()
            .map(|handle| self.resolve_token(handle.payload_key()))
            .transpose()?;
        let neg_pooled_token = request
            .negative()
            .pooled_embedding()
            .map(|handle| self.resolve_token(handle.payload_key()))
            .transpose()?;
        let latent_token = self.resolve_token(request.latent().payload().payload_key())?;
        let input_width = request.latent().width();
        let input_height = request.latent().height();
        let input_batch = request.latent().batch();
        let input_channels = request.latent().channels();

        let output = self
            .send_request(
                "diffusion.sample",
                serde_json::json!({
                    "model_token": model_token,
                    "model_id": request.model().model_id().as_str(),
                    "pos_cond_token": pos_cond_token,
                    "neg_cond_token": neg_cond_token,
                    "pos_pooled_token": pos_pooled_token,
                    "neg_pooled_token": neg_pooled_token,
                    "latent_token": latent_token,
                    "width": input_width,
                    "height": input_height,
                    "batch_size": input_batch,
                    "channels": input_channels,
                    "seed": request.seed(),
                    "steps": request.steps(),
                    "cfg": request.cfg(),
                    "denoise": request.denoise(),
                    "sampler": request.sampler().as_str(),
                    "scheduler": request.scheduler().as_str(),
                    "run_id": request.run_id().as_str(),
                    "workflow_id": request.workflow_id().as_str(),
                    "workflow_version": request.workflow_version().get(),
                    "node_id": request.node_id().as_str(),
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
            TensorShape::new(vec![
                input_batch as usize,
                input_channels as usize,
                (input_height / 8) as usize,
                (input_width / 8) as usize,
            ]),
        );

        let latent = RuntimeLatent::new(
            latent_payload,
            input_width,
            input_height,
            input_batch,
            input_channels,
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

        let vae_token = self.resolve_token(request.vae().payload_key())?;
        let latent_token = self.resolve_token(request.latent().payload().payload_key())?;

        let output = self
            .send_request(
                InferenceCapability::LatentDecode.as_str(),
                serde_json::json!({
                    "vae_token": vae_token,
                    "model_id": request.vae().model_id().as_str(),
                    "latent_token": latent_token,
                    "width": request.latent().width(),
                    "height": request.latent().height(),
                    "batch_size": request.latent().batch(),
                    "channels": request.latent().channels(),
                    "run_id": request.run_id().as_str(),
                    "workflow_id": request.workflow_id().as_str(),
                    "workflow_version": request.workflow_version().get(),
                    "node_id": request.node_id().as_str(),
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

        let image = reimagine_inference::RuntimeImage::new(image_payload, width, height, 1, "rgb");

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

        let image_token = self.resolve_token(request.image().payload().payload_key())?;

        let output = self
            .send_request(
                InferenceCapability::ImageSave.as_str(),
                serde_json::json!({
                    "image_token": image_token,
                    "filename_prefix": request.filename_prefix().as_str(),
                    "width": request.image().width(),
                    "height": request.image().height(),
                    "batch_size": request.image().batch(),
                    "color_space": request.image().color_space(),
                    "run_id": request.run_id().as_str(),
                    "workflow_id": request.workflow_id().as_str(),
                    "workflow_version": request.workflow_version().get(),
                    "node_id": request.node_id().as_str(),
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

        let image_token = self.resolve_token(request.image().payload().payload_key())?;

        let output = self
            .send_request(
                InferenceCapability::ImagePreview.as_str(),
                serde_json::json!({
                    "image_token": image_token,
                    "width": request.image().width(),
                    "height": request.image().height(),
                    "batch_size": request.image().batch(),
                    "color_space": request.image().color_space(),
                    "run_id": request.run_id().as_str(),
                    "workflow_id": request.workflow_id().as_str(),
                    "workflow_version": request.workflow_version().get(),
                    "node_id": request.node_id().as_str(),
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
