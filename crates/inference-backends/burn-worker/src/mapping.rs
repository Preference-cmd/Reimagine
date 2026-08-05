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
use reimagine_backend_worker_protocol::request_operation;
use reimagine_core::model::{
    ModelId, ModelRole, ModelSeries, ModelVariant, NodeId, ParamValue, RunId, WorkflowId,
    WorkflowVersion,
};
use reimagine_inference::{
    CreateEmptyLatentRequest, DiffusionSampleRequest, ExecutionConditioning, ImageImportRequest,
    ImagePreviewRequest, ImageSaveRequest, InferenceBackend, LatentDecodeRequest,
    LatentEncodeRequest, LoadBundleRequest, ModelFormat, ModelSourceKind, ResolvedInferenceModel,
    ResolvedInferenceModelSource, ResolvedInferenceModelSourceSet, ResourceHintSink, ResourceHints,
    TextEncodeRequest, VramBudget,
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
#[derive(Debug)]
#[allow(dead_code)]
pub enum MappingResult {
    Success(serde_json::Value),
    Cancelled,
    BackendError(BackendExecutionError),
    NotImplemented,
}

/// Map a backend error into a [`MappingResult`], preserving
/// cancellation so the serve loop can emit `TerminalOutcome::Cancelled`.
fn backend_error(error: reimagine_inference::InferenceError) -> MappingResult {
    match error {
        reimagine_inference::InferenceError::Cancelled => MappingResult::Cancelled,
        other => MappingResult::BackendError(BackendExecutionError {
            code: "backend_error".to_string(),
            message: other.to_string(),
            retryable: false,
        }),
    }
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
        "latent.create_empty" => dispatch_create_empty_latent(rt, backend, tokens, payload),
        "model.load_bundle" => dispatch_load_bundle(rt, backend, tokens, payload),
        "text.encode" => dispatch_text_encode(rt, backend, tokens, payload),
        "diffusion.sample" => dispatch_diffusion_sample(rt, backend, tokens, payload),
        "latent.decode" => dispatch_latent_decode(rt, backend, tokens, payload),
        "latent.encode" => dispatch_latent_encode(rt, backend, tokens, payload),
        "image.import" => dispatch_image_import(rt, backend, tokens, payload),
        "image.save" => dispatch_image_save(rt, backend, tokens, payload),
        "image.preview" => dispatch_image_preview(rt, backend, tokens, payload),
        request_operation::APPLY_RESOURCE_HINTS => {
            dispatch_apply_resource_hints(rt, backend, payload)
        }
        request_operation::SET_MODEL_CACHE_BUDGET => {
            dispatch_set_model_cache_budget(backend, payload)
        }
        #[cfg(test)]
        "test.spin" => dispatch_test_spin(backend, payload),
        other => MappingResult::BackendError(BackendExecutionError {
            code: "unknown_operation".to_string(),
            message: format!("unknown operation: {other}"),
            retryable: false,
        }),
    }
}

/// Handle `resource.apply_hints` — forward resource-management hints
/// (VRAM budget, prefetch intent, component lifecycle) to the backend.
///
/// The operation is advisory and cheap: it deserializes the wire
/// payload into [`ResourceHints`] and forwards it to the backend's
/// [`ResourceHintSink`] implementation.
fn dispatch_apply_resource_hints(
    rt: &tokio::runtime::Runtime,
    backend: &BurnBackend,
    payload: &serde_json::Value,
) -> MappingResult {
    let hints = match serde_json::from_value::<ResourceHints>(payload.clone()) {
        Ok(hints) => hints,
        Err(error) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_owned(),
                message: format!("invalid resource hints payload: {error}"),
                retryable: false,
            });
        }
    };
    match rt.block_on(ResourceHintSink::apply_resource_hints(backend, hints)) {
        Ok(()) => MappingResult::Success(serde_json::json!({ "applied": true })),
        Err(error) => backend_error(error),
    }
}

/// Handle `model.set_cache_budget` — apply a VRAM budget to the
/// cross-run model cache.
///
/// The operation deserializes the wire payload into [`VramBudget`]
/// and forwards it to `BurnModelCache::apply_vram_budget`, which
/// evicts least-recently-used bundles to fit the budget. It is
/// synchronous and cheap (no GPU work), so it does not need the
/// tokio runtime.
fn dispatch_set_model_cache_budget(
    backend: &BurnBackend,
    payload: &serde_json::Value,
) -> MappingResult {
    let budget = match serde_json::from_value::<VramBudget>(payload.clone()) {
        Ok(budget) => budget,
        Err(error) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_owned(),
                message: format!("invalid vram budget payload: {error}"),
                retryable: false,
            });
        }
    };
    backend.model_cache().apply_vram_budget(budget);
    MappingResult::Success(serde_json::json!({ "applied": true }))
}

// ---------------------------------------------------------------------------
// Helper: extract a u32 from a JSON field
// ---------------------------------------------------------------------------

fn extract_u32(value: &serde_json::Value, field: &str) -> Result<u32, String> {
    value
        .get(field)
        .and_then(|v| v.as_u64())
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| format!("missing or invalid field `{field}`"))
}

fn request_context(
    payload: &serde_json::Value,
    fallback: u64,
) -> (RunId, WorkflowId, WorkflowVersion, NodeId) {
    let run_id = payload
        .get("run_id")
        .and_then(serde_json::Value::as_str)
        .map_or_else(|| format!("wrk:{fallback}"), str::to_owned);
    let workflow_id = payload
        .get("workflow_id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("burn-worker");
    let workflow_version = payload
        .get("workflow_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(1);
    let node_id = payload
        .get("node_id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("op");
    (
        RunId::new(run_id),
        WorkflowId::new(workflow_id),
        WorkflowVersion::new(workflow_version),
        NodeId::new(node_id),
    )
}

fn validate_model_path(backend: &BurnBackend, path: &std::path::Path) -> Result<PathBuf, String> {
    let root = backend
        .config()
        .models_dir()
        .canonicalize()
        .map_err(|error| format!("models root cannot be canonicalized: {error}"))?;
    let canonical = path.canonicalize().map_err(|error| {
        format!(
            "model source `{}` cannot be canonicalized: {error}",
            path.display()
        )
    })?;
    if !canonical.starts_with(&root) {
        return Err(format!(
            "model source `{}` is outside authorized root `{}`",
            canonical.display(),
            root.display()
        ));
    }
    Ok(canonical)
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

fn invalid_request(message: String) -> MappingResult {
    MappingResult::BackendError(BackendExecutionError {
        code: "invalid_request".to_owned(),
        message,
        retryable: false,
    })
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
    let (run_id, workflow_id, workflow_version, node_id) = request_context(payload, token);
    let request = CreateEmptyLatentRequest::new(
        width,
        height,
        batch_size,
        run_id,
        workflow_id,
        workflow_version,
        node_id,
    );

    let response = match rt.block_on(backend.create_empty_latent(request)) {
        Ok(r) => r,
        Err(e) => return backend_error(e),
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
    let source_path = match extract_string(payload, "source_path")
        .and_then(|value| validate_model_path(backend, std::path::Path::new(&value)))
    {
        Ok(v) => v,
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
    let source_set = if let Some(components) = payload.get("components").and_then(|v| v.as_array())
    {
        let mut sources = Vec::new();
        for comp in components {
            let parsed = (|| -> Result<ResolvedInferenceModelSource, String> {
                let kind = serde_json::from_value::<ModelSourceKind>(
                    comp.get("kind").cloned().ok_or("component omitted kind")?,
                )
                .map_err(|error| format!("invalid component kind: {error}"))?;
                let role = serde_json::from_value::<ModelRole>(
                    comp.get("role").cloned().ok_or("component omitted role")?,
                )
                .map_err(|error| format!("invalid component role: {error}"))?;
                let format = serde_json::from_value::<ModelFormat>(
                    comp.get("format")
                        .cloned()
                        .ok_or("component omitted format")?,
                )
                .map_err(|error| format!("invalid component format: {error}"))?;
                let raw_path = comp
                    .get("path")
                    .and_then(serde_json::Value::as_str)
                    .ok_or("component omitted path")?;
                let path = validate_model_path(backend, std::path::Path::new(raw_path))?;
                let mut source = ResolvedInferenceModelSource::new(kind, role, path, format);
                if let Some(metadata) = comp.get("metadata").and_then(serde_json::Value::as_str) {
                    source = source.with_metadata(metadata);
                }
                Ok(source)
            })();
            match parsed {
                Ok(source) => sources.push(source),
                Err(message) => {
                    return MappingResult::BackendError(BackendExecutionError {
                        code: "invalid_request".to_owned(),
                        message,
                        retryable: false,
                    });
                }
            }
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
    let (run_id, workflow_id, workflow_version, node_id) = request_context(payload, token);
    let request = LoadBundleRequest::new(
        resolved_with_set,
        run_id,
        workflow_id,
        workflow_version,
        node_id,
    );

    let response = match rt.block_on(backend.load_bundle(request)) {
        Ok(r) => r,
        Err(e) => return backend_error(e),
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
    let model_id = match extract_string(payload, "model_id") {
        Ok(value) => ModelId::new(value),
        Err(message) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_owned(),
                message,
                retryable: false,
            });
        }
    };

    let token = tokens.next();
    let (run_id, workflow_id, workflow_version, node_id) = request_context(payload, token);

    // Reconstruct the clip handle from its worker-side payload key.
    // The backend stores handles in its model cache — we can derive
    // model_id from the bundle lookup in the backend, but for the
    // wire mapping we use a deterministic model_id based on the
    // clip_token string. The TextEncodeRequest accepts a RuntimeClipHandle
    // and expects it to carry the correct backend metadata.
    let backend_kind = backend.backend_kind().clone();
    let backend_instance = backend.backend_instance();
    let clip_handle = reimagine_inference::RuntimeClipHandle::with_instance(
        model_id,
        backend_kind,
        backend_instance,
        clip_token,
    );

    let request = TextEncodeRequest::new(
        clip_handle,
        std::sync::Arc::new(reimagine_inference::ExecutionValue::Param(
            ParamValue::String(prompt_text),
        )),
        run_id,
        workflow_id,
        workflow_version,
        node_id,
    );

    let response = match rt.block_on(backend.text_encode(request)) {
        Ok(r) => r,
        Err(e) => return backend_error(e),
    };

    let conditioning = response.into_conditioning();
    let conditioning_token = conditioning
        .text_embedding()
        .payload_key()
        .as_str()
        .to_string();

    let mut result = serde_json::json!({
        "conditioning_token": conditioning_token,
    });

    // Include pooled embedding token if available
    if let Some(pooled) = conditioning.pooled_embedding() {
        result["pooled_token"] =
            serde_json::Value::String(pooled.payload_key().as_str().to_string());
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
    let pos_pooled_token = payload
        .get("pos_pooled_token")
        .and_then(serde_json::Value::as_str);
    let neg_pooled_token = payload
        .get("neg_pooled_token")
        .and_then(serde_json::Value::as_str);
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
    let model_id = match extract_string(payload, "model_id") {
        Ok(value) => ModelId::new(value),
        Err(message) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_owned(),
                message,
                retryable: false,
            });
        }
    };
    let width = match extract_u32(payload, "width") {
        Ok(value) => value,
        Err(message) => return invalid_request(message),
    };
    let height = match extract_u32(payload, "height") {
        Ok(value) => value,
        Err(message) => return invalid_request(message),
    };
    let batch_size = match extract_u32(payload, "batch_size") {
        Ok(value) => value,
        Err(message) => return invalid_request(message),
    };
    let channels = match extract_u32(payload, "channels") {
        Ok(value) => value,
        Err(message) => return invalid_request(message),
    };

    let token = tokens.next();
    let (run_id, workflow_id, workflow_version, node_id) = request_context(payload, token);

    // Reconstruct handles from worker tokens.
    let backend_kind = backend.backend_kind().clone();
    let backend_instance = backend.backend_instance();

    // Model handle: derive model_id from the model_cache via lookup,
    // but for the wire protocol we construct it from the token string.
    let model_handle = reimagine_inference::RuntimeModelHandle::with_instance(
        model_id,
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
        reimagine_core::model::TensorShape::new(vec![batch_size as usize, 77, 2048]),
        backend.device_label(),
    );
    let mut pos_exec_cond = ExecutionConditioning::new(
        pos_text,
        reimagine_inference::ConditioningMetadata::new(width, height),
    );
    if let Some(token) = pos_pooled_token {
        pos_exec_cond = pos_exec_cond.with_pooled_embedding(
            reimagine_inference::BackendTensorHandle::with_instance(
                backend_kind.clone(),
                backend_instance.clone(),
                token,
                reimagine_core::model::TensorDType::F32,
                reimagine_core::model::TensorShape::new(vec![batch_size as usize, 1280]),
                backend.device_label(),
            ),
        );
    }

    let neg_text = reimagine_inference::BackendTensorHandle::with_instance(
        backend_kind.clone(),
        backend_instance.clone(),
        neg_cond_token.as_str(),
        reimagine_core::model::TensorDType::F32,
        reimagine_core::model::TensorShape::new(vec![batch_size as usize, 77, 2048]),
        backend.device_label(),
    );
    let mut neg_exec_cond = ExecutionConditioning::new(
        neg_text,
        reimagine_inference::ConditioningMetadata::new(width, height),
    );
    if let Some(token) = neg_pooled_token {
        neg_exec_cond = neg_exec_cond.with_pooled_embedding(
            reimagine_inference::BackendTensorHandle::with_instance(
                backend_kind.clone(),
                backend_instance.clone(),
                token,
                reimagine_core::model::TensorDType::F32,
                reimagine_core::model::TensorShape::new(vec![batch_size as usize, 1280]),
                backend.device_label(),
            ),
        );
    }

    // Latent handle
    let latent_payload = reimagine_inference::BackendTensorHandle::with_instance(
        backend_kind.clone(),
        backend_instance.clone(),
        latent_token.as_str(),
        reimagine_core::model::TensorDType::F32,
        reimagine_core::model::TensorShape::new(vec![
            batch_size as usize,
            channels as usize,
            (height / 8) as usize,
            (width / 8) as usize,
        ]),
        backend.device_label(),
    );
    let latent = reimagine_inference::RuntimeLatent::with_sdxl_base(
        latent_payload,
        width,
        height,
        batch_size,
        channels,
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
        run_id,
        workflow_id,
        workflow_version,
        node_id,
    );

    let response = match rt.block_on(backend.diffusion_sample(request)) {
        Ok(r) => r,
        Err(e) => return backend_error(e),
    };

    let sampled_latent = response.into_latent();
    let result_latent_token = sampled_latent.payload().payload_key().as_str().to_string();

    MappingResult::Success(serde_json::json!({
        "latent_token": result_latent_token,
        "width": sampled_latent.width(),
        "height": sampled_latent.height(),
        "batch_size": sampled_latent.batch(),
        "channels": sampled_latent.channels(),
    }))
}

// Build an ExecutionConditioning from a worker token. This creates a
// lightweight handle referencing the backend store; the actual tensor
// data lives in the BurnStore on the worker side.
// _build_conditioning_from_token is kept as a helper for future operation
// implementations that need to reconstruct conditioning handles from wire tokens.
#[allow(dead_code)]
fn build_conditioning_from_token(backend: &BurnBackend, token: &str) -> ExecutionConditioning {
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
    let model_id = match extract_string(payload, "model_id") {
        Ok(value) => ModelId::new(value),
        Err(message) => return invalid_request(message),
    };
    let width = match extract_u32(payload, "width") {
        Ok(value) => value,
        Err(message) => return invalid_request(message),
    };
    let height = match extract_u32(payload, "height") {
        Ok(value) => value,
        Err(message) => return invalid_request(message),
    };
    let batch_size = match extract_u32(payload, "batch_size") {
        Ok(value) => value,
        Err(message) => return invalid_request(message),
    };
    let channels = match extract_u32(payload, "channels") {
        Ok(value) => value,
        Err(message) => return invalid_request(message),
    };

    let token = tokens.next();
    let (run_id, workflow_id, workflow_version, node_id) = request_context(payload, token);
    let backend_kind = backend.backend_kind().clone();
    let backend_instance = backend.backend_instance();

    // VAE handle
    let vae_handle = reimagine_inference::RuntimeVaeHandle::with_instance(
        model_id,
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
        reimagine_core::model::TensorShape::new(vec![
            batch_size as usize,
            channels as usize,
            (height / 8) as usize,
            (width / 8) as usize,
        ]),
        backend.device_label(),
    );
    let latent = reimagine_inference::RuntimeLatent::with_sdxl_base(
        latent_payload,
        width,
        height,
        batch_size,
        channels,
    )
    .with_content(reimagine_inference::LatentContent::Sampled);

    let request = LatentDecodeRequest::new(
        vae_handle,
        latent,
        run_id,
        workflow_id,
        workflow_version,
        node_id,
    );

    let response = match rt.block_on(backend.latent_decode(request)) {
        Ok(r) => r,
        Err(e) => return backend_error(e),
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
// latent_encode
// ---------------------------------------------------------------------------

/// Handle `latent_encode` — run the Burn VAE encoder on a stored image.
fn dispatch_latent_encode(
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
    let model_id = match extract_string(payload, "model_id") {
        Ok(value) => ModelId::new(value),
        Err(message) => return invalid_request(message),
    };
    let width = match extract_u32(payload, "width") {
        Ok(value) => value,
        Err(message) => return invalid_request(message),
    };
    let height = match extract_u32(payload, "height") {
        Ok(value) => value,
        Err(message) => return invalid_request(message),
    };
    let batch_size = match extract_u32(payload, "batch_size") {
        Ok(value) => value,
        Err(message) => return invalid_request(message),
    };

    let token = tokens.next();
    let (run_id, workflow_id, workflow_version, node_id) = request_context(payload, token);
    let backend_kind = backend.backend_kind().clone();
    let backend_instance = backend.backend_instance();

    // VAE handle
    let vae_handle = reimagine_inference::RuntimeVaeHandle::with_instance(
        model_id,
        backend_kind.clone(),
        backend_instance.clone(),
        vae_token,
    );

    // Image handle — reconstruct from the token stored in the backend store
    let image_payload = reimagine_inference::BackendTensorHandle::with_instance(
        backend_kind,
        backend_instance,
        image_token.as_str(),
        reimagine_core::model::TensorDType::F32,
        reimagine_core::model::TensorShape::new(vec![
            batch_size as usize,
            3,
            height as usize,
            width as usize,
        ]),
        backend.device_label(),
    );
    let image =
        reimagine_inference::RuntimeImage::new(image_payload, width, height, batch_size, "rgb");

    let request = LatentEncodeRequest::new(
        vae_handle,
        image,
        run_id,
        workflow_id,
        workflow_version,
        node_id,
    );

    let response = match rt.block_on(backend.latent_encode(request)) {
        Ok(r) => r,
        Err(e) => return backend_error(e),
    };

    let latent = response.into_latent();
    let latent_token = latent.payload().payload_key().as_str().to_string();

    MappingResult::Success(serde_json::json!({
        "latent_token": latent_token,
        "width": latent.width(),
        "height": latent.height(),
        "batch_size": latent.batch(),
        "channels": latent.channels(),
    }))
}

// ---------------------------------------------------------------------------
// image_import
// ---------------------------------------------------------------------------

/// Handle `image_import` — decode an image file into a stored image payload.
fn dispatch_image_import(
    rt: &tokio::runtime::Runtime,
    backend: &BurnBackend,
    tokens: &TokenGenerator,
    payload: &serde_json::Value,
) -> MappingResult {
    let source_path = match extract_string(payload, "source_path") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };
    let media_type = match extract_string(payload, "media_type") {
        Ok(v) => v,
        Err(e) => {
            return MappingResult::BackendError(BackendExecutionError {
                code: "invalid_request".to_string(),
                message: e,
                retryable: false,
            });
        }
    };
    let display_name = payload
        .get("display_name")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);

    let token = tokens.next();
    let (run_id, workflow_id, workflow_version, node_id) = request_context(payload, token);

    let source =
        reimagine_inference::ResolvedImageSource::new(source_path, media_type, display_name);
    let request = ImageImportRequest::new(source, run_id, workflow_id, workflow_version, node_id);

    let response = match rt.block_on(backend.image_import(request)) {
        Ok(r) => r,
        Err(e) => return backend_error(e),
    };

    let image = response.into_image();
    let image_token = image.payload().payload_key().as_str().to_string();

    MappingResult::Success(serde_json::json!({
        "image_token": image_token,
        "width": image.width(),
        "height": image.height(),
        "batch_size": image.batch(),
        "color_space": image.color_space(),
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
    let width = match extract_u32(payload, "width") {
        Ok(value) => value,
        Err(message) => return invalid_request(message),
    };
    let height = match extract_u32(payload, "height") {
        Ok(value) => value,
        Err(message) => return invalid_request(message),
    };
    let batch_size = match extract_u32(payload, "batch_size") {
        Ok(value) => value,
        Err(message) => return invalid_request(message),
    };
    let color_space = match extract_string(payload, "color_space") {
        Ok(value) => value,
        Err(message) => return invalid_request(message),
    };

    let token = tokens.next();
    let (run_id, workflow_id, workflow_version, node_id) = request_context(payload, token);
    let backend_kind = backend.backend_kind().clone();
    let backend_instance = backend.backend_instance();

    let image_payload = reimagine_inference::BackendTensorHandle::with_instance(
        backend_kind,
        backend_instance,
        image_token.as_str(),
        reimagine_core::model::TensorDType::F32,
        reimagine_core::model::TensorShape::new(vec![
            batch_size as usize,
            3,
            height as usize,
            width as usize,
        ]),
        backend.device_label(),
    );
    let image = reimagine_inference::RuntimeImage::new(
        image_payload,
        width,
        height,
        batch_size,
        color_space,
    );

    let request = ImageSaveRequest::new(image, run_id, workflow_id, workflow_version, node_id)
        .with_filename_prefix(filename_prefix);

    let response = match rt.block_on(backend.image_save(request)) {
        Ok(r) => r,
        Err(e) => return backend_error(e),
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
    let width = match extract_u32(payload, "width") {
        Ok(value) => value,
        Err(message) => return invalid_request(message),
    };
    let height = match extract_u32(payload, "height") {
        Ok(value) => value,
        Err(message) => return invalid_request(message),
    };
    let batch_size = match extract_u32(payload, "batch_size") {
        Ok(value) => value,
        Err(message) => return invalid_request(message),
    };
    let color_space = match extract_string(payload, "color_space") {
        Ok(value) => value,
        Err(message) => return invalid_request(message),
    };

    let token = tokens.next();
    let (run_id, workflow_id, workflow_version, node_id) = request_context(payload, token);
    let backend_kind = backend.backend_kind().clone();
    let backend_instance = backend.backend_instance();

    let image_payload = reimagine_inference::BackendTensorHandle::with_instance(
        backend_kind,
        backend_instance,
        image_token.as_str(),
        reimagine_core::model::TensorDType::F32,
        reimagine_core::model::TensorShape::new(vec![
            batch_size as usize,
            3,
            height as usize,
            width as usize,
        ]),
        backend.device_label(),
    );
    let image = reimagine_inference::RuntimeImage::new(
        image_payload,
        width,
        height,
        batch_size,
        color_space,
    );

    let request = ImagePreviewRequest::new(image, run_id, workflow_id, workflow_version, node_id);

    let response = match rt.block_on(backend.image_preview(request)) {
        Ok(r) => r,
        Err(e) => return backend_error(e),
    };

    let artifact = response.into_artifact();
    let artifact_path = artifact.as_str().to_string();

    MappingResult::Success(serde_json::json!({
        "artifact_path": artifact_path,
    }))
}

/// Test-only `test.spin` operation: blocks for `duration_ms`
/// (default 60s) while polling the request cancellation token, so
/// serve-loop tests can hold a request in flight deterministically
/// without model weights or GPU work.
#[cfg(test)]
fn dispatch_test_spin(backend: &BurnBackend, payload: &serde_json::Value) -> MappingResult {
    let duration_ms = payload
        .get("duration_ms")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(60_000);
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
    while !backend.is_cancelled() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    if backend.is_cancelled() {
        MappingResult::Cancelled
    } else {
        MappingResult::Success(serde_json::json!({ "spinner": "done" }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_u32_rejects_overflow() {
        let payload = serde_json::json!({ "value": u64::from(u32::MAX) + 1 });
        assert!(extract_u32(&payload, "value").is_err());
    }

    #[test]
    fn model_source_outside_configured_root_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        let backend = BurnBackend::new(reimagine_inference_burn::BurnBackendConfig::new(
            root.path(),
            root.path(),
        ))
        .unwrap();
        assert!(validate_model_path(&backend, outside.path()).is_err());
    }

    #[test]
    fn test_spin_completes_without_cancellation() {
        let backend = BurnBackend::new(reimagine_inference_burn::BurnBackendConfig::new(
            "/tmp/models",
            "/tmp/output",
        ))
        .unwrap();
        let result = dispatch_test_spin(&backend, &serde_json::json!({ "duration_ms": 10 }));
        assert!(matches!(result, MappingResult::Success(_)));
    }

    #[test]
    fn test_spin_returns_cancelled_when_token_cancelled() {
        let backend = BurnBackend::new(reimagine_inference_burn::BurnBackendConfig::new(
            "/tmp/models",
            "/tmp/output",
        ))
        .unwrap();
        backend.cancellation().cancel();
        let result = dispatch_test_spin(&backend, &serde_json::json!({}));
        assert!(matches!(result, MappingResult::Cancelled));
    }

    #[test]
    fn apply_resource_hints_dispatch_applies_vram_budget() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let backend = BurnBackend::new(reimagine_inference_burn::BurnBackendConfig::new(
            "/tmp/models",
            "/tmp/output",
        ))
        .unwrap();
        let hints = reimagine_inference::ResourceHints::new(RunId::new("run-hints"))
            .with_vram_budget(
                reimagine_inference::VramBudget::unlimited().with_total_bytes(1 << 30),
            );
        let payload = serde_json::to_value(&hints).unwrap();
        let result = dispatch(
            &rt,
            &backend,
            &TokenGenerator::new(),
            request_operation::APPLY_RESOURCE_HINTS,
            &payload,
        );
        assert!(
            matches!(result, MappingResult::Success(_)),
            "hint application should succeed, got {result:?}"
        );
    }

    #[test]
    fn apply_resource_hints_rejects_malformed_payload() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let backend = BurnBackend::new(reimagine_inference_burn::BurnBackendConfig::new(
            "/tmp/models",
            "/tmp/output",
        ))
        .unwrap();
        let result = dispatch(
            &rt,
            &backend,
            &TokenGenerator::new(),
            request_operation::APPLY_RESOURCE_HINTS,
            &serde_json::json!({ "run_id": 42 }),
        );
        assert!(
            matches!(result, MappingResult::BackendError(_)),
            "malformed hints payload should be rejected, got {result:?}"
        );
    }

    #[test]
    fn set_model_cache_budget_dispatch_applies_budget_to_cache() {
        let backend = BurnBackend::new(reimagine_inference_burn::BurnBackendConfig::new(
            "/tmp/models",
            "/tmp/output",
        ))
        .unwrap();
        let budget = VramBudget::unlimited().with_total_bytes(1 << 30);
        let payload = serde_json::to_value(budget).unwrap();
        let result = dispatch(
            &tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap(),
            &backend,
            &TokenGenerator::new(),
            request_operation::SET_MODEL_CACHE_BUDGET,
            &payload,
        );
        assert!(
            matches!(result, MappingResult::Success(_)),
            "budget application should succeed, got {result:?}"
        );
    }

    #[test]
    fn set_model_cache_budget_rejects_malformed_payload() {
        let backend = BurnBackend::new(reimagine_inference_burn::BurnBackendConfig::new(
            "/tmp/models",
            "/tmp/output",
        ))
        .unwrap();
        let result = dispatch(
            &tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap(),
            &backend,
            &TokenGenerator::new(),
            request_operation::SET_MODEL_CACHE_BUDGET,
            &serde_json::json!({ "total_bytes": "not-a-number" }),
        );
        assert!(
            matches!(result, MappingResult::BackendError(_)),
            "malformed budget payload should be rejected, got {result:?}"
        );
    }

    #[test]
    #[cfg_attr(
        any(feature = "cuda", feature = "wgpu"),
        ignore = "requires a GPU device; runs on flex (CPU) builds"
    )]
    fn image_import_dispatch_decodes_png_into_worker_token() {
        let temp = tempfile::tempdir().unwrap();
        let png_path = temp.path().join("input.png");
        let mut rgb = Vec::with_capacity(64 * 64 * 3);
        rgb.resize(64 * 64 * 3, 128);
        let png_bytes = {
            let mut out = Vec::new();
            let encoder = ::image::codecs::png::PngEncoder::new(&mut out);
            ::image::ImageEncoder::write_image(
                encoder,
                &rgb,
                64,
                64,
                ::image::ColorType::Rgb8.into(),
            )
            .expect("encode fixture PNG");
            out
        };
        std::fs::write(&png_path, &png_bytes).unwrap();

        let backend = BurnBackend::new(reimagine_inference_burn::BurnBackendConfig::new(
            "/tmp/models",
            "/tmp/output",
        ))
        .unwrap();
        let result = dispatch(
            &tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap(),
            &backend,
            &TokenGenerator::new(),
            "image.import",
            &serde_json::json!({
                "source_path": png_path.to_string_lossy(),
                "media_type": "image/png",
                "display_name": "input.png",
            }),
        );
        match result {
            MappingResult::Success(output) => {
                assert!(output.get("image_token").is_some());
                assert_eq!(
                    output.get("width").and_then(serde_json::Value::as_u64),
                    Some(64)
                );
                assert_eq!(
                    output.get("height").and_then(serde_json::Value::as_u64),
                    Some(64)
                );
            }
            other => panic!("expected import success, got {other:?}"),
        }
    }

    #[test]
    fn image_import_dispatch_rejects_missing_source_path() {
        let backend = BurnBackend::new(reimagine_inference_burn::BurnBackendConfig::new(
            "/tmp/models",
            "/tmp/output",
        ))
        .unwrap();
        let result = dispatch(
            &tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap(),
            &backend,
            &TokenGenerator::new(),
            "image.import",
            &serde_json::json!({ "media_type": "image/png" }),
        );
        assert!(
            matches!(result, MappingResult::BackendError(_)),
            "malformed import payload should be rejected, got {result:?}"
        );
    }

    #[test]
    fn latent_encode_dispatch_rejects_missing_fields() {
        let backend = BurnBackend::new(reimagine_inference_burn::BurnBackendConfig::new(
            "/tmp/models",
            "/tmp/output",
        ))
        .unwrap();
        let result = dispatch(
            &tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap(),
            &backend,
            &TokenGenerator::new(),
            "latent.encode",
            &serde_json::json!({ "vae_token": "vae-1" }),
        );
        assert!(
            matches!(result, MappingResult::BackendError(_)),
            "malformed encode payload should be rejected, got {result:?}"
        );
    }

    #[test]
    #[cfg_attr(
        any(feature = "cuda", feature = "wgpu"),
        ignore = "requires a GPU device; runs on flex (CPU) builds"
    )]
    fn latent_encode_dispatch_reports_missing_bundle() {
        let temp = tempfile::tempdir().unwrap();
        let png_path = temp.path().join("input.png");
        let mut rgb = Vec::with_capacity(8 * 8 * 3);
        rgb.resize(8 * 8 * 3, 128);
        let png_bytes = {
            let mut out = Vec::new();
            let encoder = ::image::codecs::png::PngEncoder::new(&mut out);
            ::image::ImageEncoder::write_image(
                encoder,
                &rgb,
                8,
                8,
                ::image::ColorType::Rgb8.into(),
            )
            .expect("encode fixture PNG");
            out
        };
        std::fs::write(&png_path, &png_bytes).unwrap();

        let backend = BurnBackend::new(reimagine_inference_burn::BurnBackendConfig::new(
            "/tmp/models",
            "/tmp/output",
        ))
        .unwrap();
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let tokens = TokenGenerator::new();

        // Import a real image through the public dispatch path so a
        // payload exists in the store.
        let imported = dispatch(
            &rt,
            &backend,
            &tokens,
            "image.import",
            &serde_json::json!({
                "source_path": png_path.to_string_lossy(),
                "media_type": "image/png",
            }),
        );
        let image_token = match imported {
            MappingResult::Success(output) => output["image_token"].as_str().unwrap().to_owned(),
            other => panic!("expected import success, got {other:?}"),
        };

        // The bundle lookup must fail with a precise diagnostic
        // before any VAE work.
        let result = dispatch(
            &rt,
            &backend,
            &tokens,
            "latent.encode",
            &serde_json::json!({
                "vae_token": "vae-1",
                "model_id": "missing-model",
                "image_token": image_token,
                "width": 8,
                "height": 8,
                "batch_size": 1,
            }),
        );
        match result {
            MappingResult::BackendError(error) => {
                assert!(
                    error.message.contains("load_bundle first"),
                    "unexpected error: {error:?}"
                );
            }
            other => panic!("expected missing-bundle error, got {other:?}"),
        }
    }
}
