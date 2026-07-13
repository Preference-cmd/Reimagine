//! Wire operation <-> Burn typed DTO mapping.
//!
//! Maps incoming wire `RequestFrame` operation strings to the
//! corresponding Burn backend operation. Each dispatch function
//! deserializes the JSON payload, constructs the typed request,
//! executes it via the `InferenceBackend` trait (through tokio
//! `block_on`), and serializes the result back to a JSON terminal
//! payload.
//!
//! Worker tokens are stored as `BackendPayloadKey` values in the
//! shared `BurnStore`. The host adapter maintains its own
//! `WorkerAuthorityTable` that maps between host-side and
//! worker-side keys.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use reimagine_backend_worker_protocol::BackendExecutionError;
use reimagine_core::model::{
    ModelId, ModelRole, ModelSeries, ModelVariant, NodeId, ParamValue, RunId,
    WorkflowId, WorkflowVersion,
};
use reimagine_inference::{
    CreateEmptyLatentRequest, DiffusionSampleRequest, ExecutionConditioning, ImagePreviewRequest,
    ImageSaveRequest, InferenceBackend, LatentDecodeRequest, LoadBundleRequest,
    ModelFormat, ResolvedInferenceModel, ResolvedInferenceModelSource, ModelSourceKind,
    ResolvedInferenceModelSourceSet, TextEncodeRequest,
};
use reimagine_inference_burn::BurnBackend;

/// Thread-safe worker-local token generator.
///
/// Each generated token is unique within the process lifetime and
/// maps 1:1 to a `BackendPayloadKey` in the `BurnStore`.
pub struct TokenGenerator {
    counter: AtomicU64,
}

impl TokenGenerator {
    pub fn new() -> Self {
        Self {
            counter: AtomicU64::new(1),
        }
    }

    pub fn next(&self) -> u64 {
        self.counter.fetch_add(1, Ordering::Relaxed)
    }
}

/// Result of a mapped operation.
#[allow(dead_code)]
pub enum MappingResult {
    Success(serde_json::Value),
    BackendError(BackendExecutionError),
    NotImplemented,
}

/// Dispatch an incoming wire operation to the Burn backend.
///
/// Called from the synchronous serve loop. Each async backend call
/// is run to completion via `rt.block_on`.
pub fn dispatch(
    rt: &tokio::runtime::Runtime,
    backend: &BurnBackend,
    tokens: &TokenGenerator,
    operation: &str,
    payload: &serde_json::Value,
) -> MappingResult {
    match operation {
        "latent.create_empty" => {
            dispatch_create_empty_latent(rt, backend, tokens, payload)
        }
        "model.load_bundle" => {
            dispatch_load_bundle(rt, backend, tokens, payload)
        }
        "text.encode" => {
            dispatch_text_encode(rt, backend, tokens, payload)
        }
        "diffusion.sample" => {
            dispatch_diffusion_sample(rt, backend, tokens, payload)
        }
        "latent.decode" => {
            dispatch_latent_decode(rt, backend, tokens, payload)
        }
        "image.save" => {
            dispatch_image_save(rt, backend, tokens, payload)
        }
        "image.preview" => {
            dispatch_image_preview(rt, backend, tokens, payload)
        }
        other => {
            MappingResult::BackendError(BackendExecutionError {
                code: "unknown_operation".to_string(),
                message: format!("unknown operation: {other}"),
                retryable: false,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: extract a u32 from a JSON field
// ---------------------------------------------------------------------------

fn extract_u32(value: &serde_json::Value, field: &str) -> Result<u32, String> {
    value
        .get(field)
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .ok_or_else(|| format!("missing or invalid field `{field}`"))
}

fn extract_string(value: &serde_json::Value, field: &str) -> Result<String, String> {
    value
        .get(field)
        .and_then(|v| v.as_str())
        .map(|v| v.to_owned())
        .ok_or_else(|| format!("missing or invalid field `{field}`"))
}

fn extract_f32(value: &serde_json::Value, field: &str) -> Result<f32, String> {
    value
        .get(field)
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .ok_or_else(|| format!("missing or invalid field `{field}`"))
}

// ---------------------------------------------------------------------------
// create_empty_latent
// ---------------------------------------------------------------------------

/// Handle `create_empty_latent` — allocate a zero-filled latent tensor.
fn dispatch_create_empty_latent(
    rt: &tokio::runtime::Runtime,
    backend: &BurnBackend,
    tokens: &TokenGenerator,
    payload: &serde_json::Value,
) -> MappingResult {
    let width = match extract_u32(payload, "width") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };
    let height = match extract_u32(payload, "height") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };
    let batch_size = match extract_u32(payload, "batch_size") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };

    let token = tokens.next();
    let request = CreateEmptyLatentRequest::new(
        width,
        height,
        batch_size,
        RunId::new(format!("wrk:{token}")),
        WorkflowId::new("burn-worker"),
        WorkflowVersion::new(1),
        NodeId::new("op"),
    );

    let response = match rt.block_on(backend.create_empty_latent(request)) {
        Ok(r) => r,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "backend_error".to_string(),
                message: e.to_string(),
                retryable: false,
            });
        }
    };

    let latent = response.into_latent();
    let worker_token = latent.payload().payload_key().as_str().to_string();

    MappingResult::Success(serde_json::json!({
        "worker_token": worker_token,
        "width": width,
        "height": height,
        "batch_size": batch_size,
    }))
}

// ---------------------------------------------------------------------------
// load_bundle
// ---------------------------------------------------------------------------

/// Handle `load_bundle` — load a model bundle (SDXL checkpoint) into the
/// model cache and return handle tokens for model, clip, and vae.
fn dispatch_load_bundle(
    rt: &tokio::runtime::Runtime,
    backend: &BurnBackend,
    tokens: &TokenGenerator,
    payload: &serde_json::Value,
) -> MappingResult {
    let model_id = match extract_string(payload, "model_id") {
        Ok(v) => ModelId::new(v),
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };
    let series_str = match extract_string(payload, "series") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };
    let variant_str = match extract_string(payload, "variant") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };
    let role_str = match extract_string(payload, "role") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };
    let source_path = match extract_string(payload, "source_path") {
        Ok(v) => PathBuf::from(v),
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };

    // Parse role from string
    let role = match role_str.as_str() {
        "CheckpointBundle" => ModelRole::CheckpointBundle,
        "DiffusionModel" => ModelRole::DiffusionModel,
        "TextEncoder" => ModelRole::TextEncoder,
        "Vae" => ModelRole::Vae,
        "Scheduler" => ModelRole::Scheduler,
        "Lora" => ModelRole::Lora,
        "ControlNet" => ModelRole::ControlNet,
        "Upscaler" => ModelRole::Upscaler,
        _ => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: format!("unknown model role: {role_str}"),
                retryable: false,
            });
        }
    };

    // Build ResolvedInferenceModel. The load_bundle operation requires a
    // converted SplitComponent source set in production (burn/04). For the
    // worker path we accept a flat source_path and build a checkpoint-bundle
    // source set.
    let resolved = ResolvedInferenceModel::new(
        model_id.clone(),
        ModelSeries::new(series_str),
        ModelVariant::new(variant_str),
        role,
        source_path.clone(),
        ModelFormat::SafeTensors,
    );

    // Build a source set from individual component paths if provided,
    // otherwise use the checkpoint-bundle source set from the resolved model.
    let source_set = if let Some(components) = payload.get("components").and_then(|v| v.as_array()) {
        let mut sources = Vec::new();
        for comp in components {
            let comp_role_str = match comp.get("role").and_then(|v| v.as_str()) {
                Some(r) => r,
                None => continue,
            };
            let comp_path = match comp.get("path").and_then(|v| v.as_str()) {
                Some(p) => PathBuf::from(p),
                None => continue,
            };
            let comp_role = match comp_role_str {
                "TextEncoder" => ModelRole::TextEncoder,
                "TextEncoder2" => ModelRole::TextEncoder,
                "UNet" => ModelRole::DiffusionModel,
                "VAEDecoder" => ModelRole::Vae,
                role if role == role_str => role_from_str(role),
                _ => continue,
            };
            sources.push(
                ResolvedInferenceModelSource::new(
                    ModelSourceKind::SplitComponent,
                    comp_role,
                    comp_path,
                    ModelFormat::SafeTensors,
                ),
            );
        }
        if sources.is_empty() {
            resolved.to_checkpoint_bundle_source_set()
        } else {
            ResolvedInferenceModelSourceSet::from_sources(sources)
        }
    } else {
        resolved.to_checkpoint_bundle_source_set()
    };

    let resolved_with_set = resolved.with_source_set(source_set);

    let token = tokens.next();
    let request = LoadBundleRequest::new(
        resolved_with_set,
        RunId::new(format!("wrk:{token}")),
        WorkflowId::new("burn-worker"),
        WorkflowVersion::new(1),
        NodeId::new("op"),
    );

    let response = match rt.block_on(backend.load_bundle(request)) {
        Ok(r) => r,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "backend_error".to_string(),
                message: e.to_string(),
                retryable: false,
            });
        }
    };

    let model_token = response.model().payload_key().as_str().to_string();
    let clip_token = response.clip().payload_key().as_str().to_string();
    let vae_token = response.vae().payload_key().as_str().to_string();

    MappingResult::Success(serde_json::json!({
        "model_token": model_token,
        "clip_token": clip_token,
        "vae_token": vae_token,
    }))
}

fn role_from_str(s: &str) -> ModelRole {
    match s {
        "CheckpointBundle" => ModelRole::CheckpointBundle,
        "DiffusionModel" => ModelRole::DiffusionModel,
        "TextEncoder" => ModelRole::TextEncoder,
        "Vae" => ModelRole::Vae,
        "Scheduler" => ModelRole::Scheduler,
        "Lora" => ModelRole::Lora,
        "ControlNet" => ModelRole::ControlNet,
        "Upscaler" => ModelRole::Upscaler,
        _ => ModelRole::CheckpointBundle,
    }
}

// ---------------------------------------------------------------------------
// text_encode
// ---------------------------------------------------------------------------

/// Handle `text_encode` — tokenize the prompt text and run the CLIP
/// text encoder forward pass.
fn dispatch_text_encode(
    rt: &tokio::runtime::Runtime,
    backend: &BurnBackend,
    tokens: &TokenGenerator,
    payload: &serde_json::Value,
) -> MappingResult {
    let clip_token = match extract_string(payload, "clip_token") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };
    let prompt_text = match extract_string(payload, "prompt_text") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };

    let token = tokens.next();

    // Reconstruct the clip handle from its worker-side payload key.
    // The backend stores handles in its model cache — we can derive
    // model_id from the bundle lookup in the backend, but for the
    // wire mapping we use a deterministic model_id based on the
    // clip_token string. The TextEncodeRequest accepts a RuntimeClipHandle
    // and expects it to carry the correct backend metadata.
    let backend_kind = backend.backend_kind().clone();
    let backend_instance = backend.backend_instance();
    let clip_handle = reimagine_inference::RuntimeClipHandle::with_instance(
        ModelId::new(format!("clip:{clip_token}")),
        backend_kind,
        backend_instance,
        clip_token,
    );

    let request = TextEncodeRequest::new(
        clip_handle,
        std::sync::Arc::new(reimagine_inference::ExecutionValue::Param(
            ParamValue::String(prompt_text),
        )),
        RunId::new(format!("wrk:{token}")),
        WorkflowId::new("burn-worker"),
        WorkflowVersion::new(1),
        NodeId::new("op"),
    );

    let response = match rt.block_on(backend.text_encode(request)) {
        Ok(r) => r,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "backend_error".to_string(),
                message: e.to_string(),
                retryable: false,
            });
        }
    };

    let conditioning = response.into_conditioning();
    let conditioning_token = conditioning.text_embedding().payload_key().as_str().to_string();

    let mut result = serde_json::json!({
        "conditioning_token": conditioning_token,
    });

    // Include pooled embedding token if available
    if let Some(pooled) = conditioning.pooled_embedding() {
        result["pooled_token"] = serde_json::Value::String(pooled.payload_key().as_str().to_string());
    }

    MappingResult::Success(result)
}

// ---------------------------------------------------------------------------
// diffusion_sample
// ---------------------------------------------------------------------------

/// Handle `diffusion_sample` — run the denoising diffusion sampling loop.
fn dispatch_diffusion_sample(
    rt: &tokio::runtime::Runtime,
    backend: &BurnBackend,
    tokens: &TokenGenerator,
    payload: &serde_json::Value,
) -> MappingResult {
    let model_token = match extract_string(payload, "model_token") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };
    let pos_cond_token = match extract_string(payload, "pos_cond_token") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };
    let neg_cond_token = match extract_string(payload, "neg_cond_token") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };
    let latent_token = match extract_string(payload, "latent_token") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };
    let seed = match extract_u32(payload, "seed") {
        Ok(v) => v as u64,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };
    let steps = match extract_u32(payload, "steps") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };
    let cfg = match extract_f32(payload, "cfg") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };
    let denoise = match extract_f32(payload, "denoise") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };
    let sampler = match extract_string(payload, "sampler") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };
    let scheduler = match extract_string(payload, "scheduler") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };

    let token = tokens.next();

    // Reconstruct handles from worker tokens.
    let backend_kind = backend.backend_kind().clone();
    let backend_instance = backend.backend_instance();

    // Model handle: derive model_id from the model_cache via lookup,
    // but for the wire protocol we construct it from the token string.
    let model_handle = reimagine_inference::RuntimeModelHandle::with_instance(
        ModelId::new(format!("model:{model_token}")),
        ModelRole::DiffusionModel,
        backend_kind.clone(),
        backend_instance.clone(),
        model_token,
    );

    // Conditioning handles
    // For DiffusionSampleRequest, we use positive conditioning for
    // the text embedding; negative conditioning is passed via the
    // `negative` field.
    let pos_text = reimagine_inference::BackendTensorHandle::with_instance(
        backend_kind.clone(),
        backend_instance.clone(),
        pos_cond_token.as_str(),
        reimagine_core::model::TensorDType::F32,
        reimagine_core::model::TensorShape::new(vec![1, 77, 2048]),
        backend.device_label(),
    );
    let pos_exec_cond = ExecutionConditioning::new(
        pos_text,
        reimagine_inference::ConditioningMetadata::new(1024, 1024),
    );

    let neg_text = reimagine_inference::BackendTensorHandle::with_instance(
        backend_kind.clone(),
        backend_instance.clone(),
        neg_cond_token.as_str(),
        reimagine_core::model::TensorDType::F32,
        reimagine_core::model::TensorShape::new(vec![1, 77, 2048]),
        backend.device_label(),
    );
    let neg_exec_cond = ExecutionConditioning::new(
        neg_text,
        reimagine_inference::ConditioningMetadata::new(1024, 1024),
    );

    // Latent handle
    let latent_payload = reimagine_inference::BackendTensorHandle::with_instance(
        backend_kind.clone(),
        backend_instance.clone(),
        latent_token.as_str(),
        reimagine_core::model::TensorDType::F32,
        reimagine_core::model::TensorShape::new(vec![1, 4, 128, 128]),
        backend.device_label(),
    );
    let latent = reimagine_inference::RuntimeLatent::with_sdxl_base(
        latent_payload,
        1024,
        1024,
        1,
        4,
    );

    let sampler_name = reimagine_inference::SamplerName::from_standard_name(&sampler);
    let scheduler_name = reimagine_inference::SchedulerName::from_standard_name(&scheduler);

    let request = DiffusionSampleRequest::new(
        model_handle,
        pos_exec_cond,
        neg_exec_cond,
        latent,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler_name,
        denoise,
        RunId::new(format!("wrk:{token}")),
        WorkflowId::new("burn-worker"),
        WorkflowVersion::new(1),
        NodeId::new("op"),
    );

    let response = match rt.block_on(backend.diffusion_sample(request)) {
        Ok(r) => r,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "backend_error".to_string(),
                message: e.to_string(),
                retryable: false,
            });
        }
    };

    let sampled_latent = response.into_latent();
    let result_latent_token = sampled_latent.payload().payload_key().as_str().to_string();

    MappingResult::Success(serde_json::json!({
        "latent_token": result_latent_token,
    }))
}

// Build an ExecutionConditioning from a worker token. This creates a
// lightweight handle referencing the backend store; the actual tensor
// data lives in the BurnStore on the worker side.
// _build_conditioning_from_token is kept as a helper for future operation
// implementations that need to reconstruct conditioning handles from wire tokens.
#[allow(dead_code)]
fn build_conditioning_from_token(
    backend: &BurnBackend,
    token: &str,
) -> ExecutionConditioning {
    // V1 stub: the conditioning data is already in the burn store.
    // The handle's payload_key lets the backend retrieve it.
    ExecutionConditioning::new(
        reimagine_inference::BackendTensorHandle::new(
            backend.backend_kind().clone(),
            reimagine_inference::BackendPayloadKey::new(token),
            reimagine_core::model::TensorDType::F32,
            reimagine_core::model::TensorShape::new(vec![1, 77, 2048]),
            backend.device_label(),
        ),
        reimagine_inference::ConditioningMetadata::new(1024, 1024),
    )
}

// ---------------------------------------------------------------------------
// latent_decode
// ---------------------------------------------------------------------------

/// Handle `latent_decode` — decode a latent tensor into an RGB image.
fn dispatch_latent_decode(
    rt: &tokio::runtime::Runtime,
    backend: &BurnBackend,
    tokens: &TokenGenerator,
    payload: &serde_json::Value,
) -> MappingResult {
    let vae_token = match extract_string(payload, "vae_token") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };
    let latent_token = match extract_string(payload, "latent_token") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };

    let token = tokens.next();
    let backend_kind = backend.backend_kind().clone();
    let backend_instance = backend.backend_instance();

    // VAE handle
    let vae_handle = reimagine_inference::RuntimeVaeHandle::with_instance(
        ModelId::new(format!("vae:{vae_token}")),
        backend_kind.clone(),
        backend_instance.clone(),
        vae_token,
    );

    // Latent handle — reconstruct from the token stored in the backend store
    let latent_payload = reimagine_inference::BackendTensorHandle::with_instance(
        backend_kind,
        backend_instance,
        latent_token.as_str(),
        reimagine_core::model::TensorDType::F32,
        reimagine_core::model::TensorShape::new(vec![1, 4, 128, 128]),
        backend.device_label(),
    );
    let latent = reimagine_inference::RuntimeLatent::with_sdxl_base(
        latent_payload,
        1024,
        1024,
        1,
        4,
    );

    let request = LatentDecodeRequest::new(
        vae_handle,
        latent,
        RunId::new(format!("wrk:{token}")),
        WorkflowId::new("burn-worker"),
        WorkflowVersion::new(1),
        NodeId::new("op"),
    );

    let response = match rt.block_on(backend.latent_decode(request)) {
        Ok(r) => r,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "backend_error".to_string(),
                message: e.to_string(),
                retryable: false,
            });
        }
    };

    let image = response.into_image();
    let image_token = image.payload().payload_key().as_str().to_string();

    MappingResult::Success(serde_json::json!({
        "image_token": image_token,
        "width": image.width(),
        "height": image.height(),
    }))
}

// ---------------------------------------------------------------------------
// image_save
// ---------------------------------------------------------------------------

/// Handle `image_save` — write a decoded image tensor to disk as PNG.
fn dispatch_image_save(
    rt: &tokio::runtime::Runtime,
    backend: &BurnBackend,
    tokens: &TokenGenerator,
    payload: &serde_json::Value,
) -> MappingResult {
    let image_token = match extract_string(payload, "image_token") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };
    let filename_prefix = match payload.get("filename_prefix").and_then(|v| v.as_str()) {
        Some(p) => p.to_owned(),
        None => "reimagine".to_owned(),
    };

    let token = tokens.next();
    let backend_kind = backend.backend_kind().clone();
    let backend_instance = backend.backend_instance();

    let image_payload = reimagine_inference::BackendTensorHandle::with_instance(
        backend_kind,
        backend_instance,
        image_token.as_str(),
        reimagine_core::model::TensorDType::F32,
        reimagine_core::model::TensorShape::new(vec![1, 3, 1024, 1024]),
        backend.device_label(),
    );
    let image = reimagine_inference::RuntimeImage::new(image_payload, 1024, 1024, 1, "rgb");

    let request = ImageSaveRequest::new(
        image,
        RunId::new(format!("wrk:{token}")),
        WorkflowId::new("burn-worker"),
        WorkflowVersion::new(1),
        NodeId::new("op"),
    )
    .with_filename_prefix(filename_prefix);

    let response = match rt.block_on(backend.image_save(request)) {
        Ok(r) => r,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "backend_error".to_string(),
                message: e.to_string(),
                retryable: false,
            });
        }
    };

    let artifact = response.into_artifact();
    let artifact_path = artifact.as_str().to_string();

    MappingResult::Success(serde_json::json!({
        "artifact_path": artifact_path,
    }))
}

// ---------------------------------------------------------------------------
// image_preview
// ---------------------------------------------------------------------------

/// Handle `image_preview` — write a preview PNG of the decoded image.
fn dispatch_image_preview(
    rt: &tokio::runtime::Runtime,
    backend: &BurnBackend,
    tokens: &TokenGenerator,
    payload: &serde_json::Value,
) -> MappingResult {
    let image_token = match extract_string(payload, "image_token") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };

    let token = tokens.next();
    let backend_kind = backend.backend_kind().clone();
    let backend_instance = backend.backend_instance();

    let image_payload = reimagine_inference::BackendTensorHandle::with_instance(
        backend_kind,
        backend_instance,
        image_token.as_str(),
        reimagine_core::model::TensorDType::F32,
        reimagine_core::model::TensorShape::new(vec![1, 3, 1024, 1024]),
        backend.device_label(),
    );
    let image = reimagine_inference::RuntimeImage::new(image_payload, 1024, 1024, 1, "rgb");

    let request = ImagePreviewRequest::new(
        image,
        RunId::new(format!("wrk:{token}")),
        WorkflowId::new("burn-worker"),
        WorkflowVersion::new(1),
        NodeId::new("op"),
    );

    let response = match rt.block_on(backend.image_preview(request)) {
        Ok(r) => r,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "backend_error".to_string(),
                message: e.to_string(),
                retryable: false,
            });
        }
    };

    let artifact = response.into_artifact();
    let artifact_path = artifact.as_str().to_string();

    MappingResult::Success(serde_json::json!({
        "artifact_path": artifact_path,
    }))
}