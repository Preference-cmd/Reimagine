//! `diffusion.sample` operation for the Burn backend.
//!
//! Implements the first Burn diffusion sampling path behind the existing
//! model-neutral `diffusion.sample` capability. V1 is scoped to SDXL
//! euler/normal txt2img full denoise (denoise=1.0), batch=1, with
//! `LatentContent::EmptyGeometry` input.
//!
//! The operation layer is model-neutral; SDXL-specific sampling mechanics
//! live under `models/stable_diffusion/sdxl/diffusion/`.

use reimagine_core::model::{NodeId, RunId, TensorShape};
use reimagine_inference::{
    Backend, BackendPayloadKey, BackendTensorHandle, DiffusionSampleRequest,
    DiffusionSampleResponse, ExecutionConditioning, InferenceBackend, LatentContent,
    LatentSpaceMetadata, RuntimeLatent,
};

use crate::backend::BurnBackend;
use crate::error::BurnBackendError;
use crate::profile::BACKEND_LABEL;
use crate::store::BurnLatentPayload;

/// Deterministic payload key for a sampled latent stored by
/// `diffusion.sample`.
fn sampled_latent_key(run_id: &RunId, node_id: &NodeId) -> BackendPayloadKey {
    BackendPayloadKey::new(format!(
        "diffusion:{}:{}",
        run_id.as_str(),
        node_id.as_str()
    ))
}

/// `diffusion.sample` entry point for the Burn backend.
///
/// V1 scope: SDXL euler/normal txt2img full denoise, batch=1,
/// `LatentContent::EmptyGeometry` input, real pooled conditioning.
pub fn execute_diffusion_sample(
    backend: &BurnBackend,
    request: DiffusionSampleRequest,
) -> Result<DiffusionSampleResponse, BurnBackendError> {
    // 1. Validate handles and loaded bundle
    let model_handle = request.model();
    validate_backend(
        model_handle.backend(),
        model_handle.backend_instance(),
        backend,
    )?;

    let bundle = backend
        .model_cache()
        .get_bundle(model_handle.model_id())
        .ok_or_else(|| {
            BurnBackendError::InvalidRequest(format!(
                "diffusion.sample requires loaded bundle for model `{}`; call load_bundle first",
                model_handle.model_id()
            ))
        })?;

    // 2. Validate conditioning handles and payloads
    let positive_cond = request.positive();
    let negative_cond = request.negative();

    validate_conditioning_backend(positive_cond, backend)?;
    validate_conditioning_backend(negative_cond, backend)?;

    let positive_payload = backend
        .store()
        .get_conditioning(positive_cond.text_embedding().payload_key())?;
    let negative_payload = backend
        .store()
        .get_conditioning(negative_cond.text_embedding().payload_key())?;

    // Verify pooled embedding handles exist
    let positive_pooled = positive_cond.pooled_embedding().ok_or_else(|| {
        BurnBackendError::InvalidRequest(
            "diffusion.sample requires positive pooled embedding handle".to_owned(),
        )
    })?;
    let negative_pooled = negative_cond.pooled_embedding().ok_or_else(|| {
        BurnBackendError::InvalidRequest(
            "diffusion.sample requires negative pooled embedding handle".to_owned(),
        )
    })?;

    validate_store_payload(backend, positive_pooled.payload_key())?;
    validate_store_payload(backend, negative_pooled.payload_key())?;

    // Verify conditioning metadata matches stable_diffusion/sdxl
    for payload in [&positive_payload, &negative_payload] {
        let meta = payload.metadata();
        if meta.series() != "stable_diffusion" || meta.variant() != "sdxl" {
            return Err(BurnBackendError::InvalidRequest(format!(
                "diffusion.sample conditioning payload produced for {}/{}; expected stable_diffusion/sdxl",
                meta.series(),
                meta.variant()
            )));
        }
        if meta.sequence_length() != 77 {
            return Err(BurnBackendError::InvalidRequest(format!(
                "diffusion.sample conditioning sequence length is {}; expected 77",
                meta.sequence_length()
            )));
        }
    }

    // 3. Validate latent handle and payload
    let latent_handle = request.latent();
    validate_backend(
        latent_handle.payload().backend(),
        latent_handle.payload().backend_instance(),
        backend,
    )?;

    let latent_payload = backend
        .store()
        .get_latent(latent_handle.payload().payload_key())?;

    // V1 only supports SDXL base latent space
    if !latent_handle
        .latent_space()
        .is_compatible(&LatentSpaceMetadata::sdxl_base())
    {
        return Err(BurnBackendError::InvalidRequest(format!(
            "diffusion.sample latent space `{}` is not compatible with SDXL base",
            latent_handle.latent_space().id()
        )));
    }

    // V1 only supports EmptyGeometry input
    if latent_handle.content() != LatentContent::EmptyGeometry {
        return Err(BurnBackendError::InvalidRequest(format!(
            "diffusion.sample V1 only accepts EmptyGeometry input, got {:?}",
            latent_handle.content()
        )));
    }

    // V1 only supports denoise = 1.0
    let denoise = request.denoise();
    if (denoise - 1.0).abs() > f32::EPSILON {
        return Err(BurnBackendError::InvalidRequest(format!(
            "diffusion.sample V1 only supports denoise=1.0, got {denoise}"
        )));
    }

    // V1 only supports euler/normal
    let sampler = request.sampler().as_str();
    let scheduler = request.scheduler().as_str();
    if sampler != "euler" || scheduler != "normal" {
        return Err(BurnBackendError::InvalidRequest(format!(
            "diffusion.sample V1 only supports sampler=euler scheduler=normal, got {sampler}/{scheduler}"
        )));
    }

    // V1 only supports batch=1
    let batch = latent_handle.batch();
    if batch != 1 {
        return Err(BurnBackendError::InvalidRequest(format!(
            "diffusion.sample V1 only supports batch=1, got {batch}"
        )));
    }

    // 4. Validate steps and cfg
    let steps = request.steps();
    if steps == 0 {
        return Err(BurnBackendError::InvalidRequest(
            "diffusion.sample steps must be positive".to_owned(),
        ));
    }
    let cfg = request.cfg();
    if !cfg.is_finite() {
        return Err(BurnBackendError::InvalidRequest(
            "diffusion.sample cfg must be finite".to_owned(),
        ));
    }

    // 5. Apply seed/noise — for V1, seed is deterministic from request
    let seed = request.seed();

    // 6. Run euler/normal denoise loop (SDXL-specific)
    let latent_tensor = latent_payload.into_active_tensor()?;
    let mut wgpu_guard = crate::wgpu_guard::WgpuErrorGuard::new();
    let sampled = crate::models::stable_diffusion::sdxl::diffusion::sample_sdxl(
        &bundle,
        latent_tensor,
        &positive_payload,
        &negative_payload,
        positive_cond.metadata(),
        negative_cond.metadata(),
        steps,
        cfg,
        seed,
        backend,
    )?;
    wgpu_guard.check().map_err(|_| {
        BurnBackendError::InvalidRequest(
            "WGPU validation error during diffusion sample; GPU commands may not have executed correctly".to_string(),
        )
    })?;

    // 7. Store sampled latent
    let output_key = sampled_latent_key(request.run_id(), request.node_id());
    let output_payload = BurnLatentPayload::new_active(
        sampled,
        LatentSpaceMetadata::sdxl_base(),
        latent_handle.width(),
        latent_handle.height(),
        batch,
    );
    backend
        .store()
        .insert_latent(request.run_id().clone(), output_key.clone(), output_payload);

    // 8. Build response
    let latent_h = (latent_handle.height() as f64 / 8.0) as usize;
    let latent_w = (latent_handle.width() as f64 / 8.0) as usize;

    let latent = RuntimeLatent::new(
        BackendTensorHandle::with_instance(
            backend.backend_kind().clone(),
            backend.backend_instance(),
            output_key,
            reimagine_core::model::TensorDType::F32,
            TensorShape::new(vec![batch as usize, 4, latent_h, latent_w]),
            backend.device_label(),
        ),
        latent_handle.width(),
        latent_handle.height(),
        batch,
        4,
        LatentSpaceMetadata::sdxl_base(),
        LatentContent::Sampled,
    );

    Ok(DiffusionSampleResponse::new(latent))
}

fn validate_backend(
    handle_backend: &Backend,
    handle_instance: &reimagine_inference::BackendInstance,
    backend: &BurnBackend,
) -> Result<(), BurnBackendError> {
    if handle_backend.as_str() != BACKEND_LABEL {
        return Err(BurnBackendError::InvalidRequest(format!(
            "handle belongs to backend `{}`, expected `{BACKEND_LABEL}`",
            handle_backend.as_str()
        )));
    }
    if handle_instance != &backend.backend_instance() {
        return Err(BurnBackendError::InvalidRequest(format!(
            "handle belongs to backend instance `{}`, expected `{}`",
            handle_instance,
            backend.backend_instance()
        )));
    }
    Ok(())
}

fn validate_conditioning_backend(
    cond: &ExecutionConditioning,
    backend: &BurnBackend,
) -> Result<(), BurnBackendError> {
    let text = cond.text_embedding();
    if text.backend().as_str() != BACKEND_LABEL {
        return Err(BurnBackendError::InvalidRequest(format!(
            "conditioning handle belongs to backend `{}`, expected `{BACKEND_LABEL}`",
            text.backend().as_str()
        )));
    }
    if text.backend_instance() != &backend.backend_instance() {
        return Err(BurnBackendError::InvalidRequest(format!(
            "conditioning handle belongs to backend instance `{}`, expected `{}`",
            text.backend_instance(),
            backend.backend_instance()
        )));
    }
    Ok(())
}

fn validate_store_payload(
    backend: &BurnBackend,
    key: &BackendPayloadKey,
) -> Result<(), BurnBackendError> {
    if !backend.store().contains_payload(key) {
        return Err(BurnBackendError::InvalidRequest(format!(
            "payload `{}` not found in store",
            key.as_str()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::active_backend::{ActiveBurnBackend, active_device};
    use crate::config::BurnBackendConfig;
    use crate::models::stable_diffusion::sdxl::{
        BurnLoadedModelBundle, BurnLoadedSdxlBundle, BurnSdxlTokenizedPrompt,
        BurnSdxlTokenizedPromptPair,
    };
    use crate::profile::BACKEND_LABEL;
    use crate::store::{BurnConditioningMetadata, BurnConditioningPayload, ClipOutputs};
    use burn_tensor::Tensor;
    use reimagine_core::model::{ModelId, NodeId, RunId, WorkflowId, WorkflowVersion};
    use reimagine_inference::{
        Backend, BackendInstance, BackendPayloadKey, ExecutionConditioning, RuntimeClipHandle,
        RuntimeLatent,
    };
    use std::sync::Arc;

    fn test_backend() -> BurnBackend {
        BurnBackend::new(BurnBackendConfig::new("/models", "/output")).expect("test backend")
    }

    fn seed_bundle(backend: &BurnBackend, model_id: &str) {
        let bundle = BurnLoadedSdxlBundle::for_test_only(
            ModelId::new(model_id),
            BackendPayloadKey::new(format!("burn:model:{model_id}:clip")),
        );
        backend.model_cache().insert_bundle(
            ModelId::new(model_id),
            Arc::new(BurnLoadedModelBundle::StableDiffusionSdxl(Arc::new(bundle))),
        );
    }

    fn burn_clip(backend: &BurnBackend, model_id: &str) -> RuntimeClipHandle {
        RuntimeClipHandle::with_instance(
            ModelId::new(model_id),
            Backend::new(BACKEND_LABEL),
            backend.backend_instance(),
            BackendPayloadKey::new(format!("burn:model:{model_id}:clip")),
        )
        .with_device(backend.device_label())
    }

    fn create_conditioning(backend: &BurnBackend, model_id: &str) -> ExecutionConditioning {
        let clip = burn_clip(backend, model_id);
        let request = reimagine_inference::TextEncodeRequest::new(
            clip,
            Arc::new(reimagine_inference::ExecutionValue::Param(
                reimagine_core::model::ParamValue::String("hello".to_owned()),
            )),
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-encode"),
        );
        let response =
            crate::operation::text::execute_text_encode(backend, request).expect("text.encode");
        response.into_conditioning()
    }

    fn create_latent(backend: &BurnBackend) -> RuntimeLatent {
        let request = reimagine_inference::CreateEmptyLatentRequest::new(
            64,
            64,
            1,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-latent"),
        );
        crate::operation::latent::execute_latent_create_empty(backend, request)
            .expect("create_empty")
            .into_latent()
    }

    fn build_request(backend: &BurnBackend, model_id: &str) -> DiffusionSampleRequest {
        let model_handle = reimagine_inference::RuntimeModelHandle::with_instance(
            ModelId::new(model_id),
            reimagine_core::model::ModelRole::CheckpointBundle,
            Backend::new(BACKEND_LABEL),
            backend.backend_instance(),
            BackendPayloadKey::new(format!("burn:model:{model_id}:diffusion")),
        );

        let conditioning = create_conditioning(backend, model_id);
        let latent = create_latent(backend);

        DiffusionSampleRequest::new(
            model_handle,
            conditioning.clone(),
            conditioning,
            latent,
            0,
            50,
            7.5,
            reimagine_inference::SamplerName::from_standard_name("euler"),
            reimagine_inference::SchedulerName::from_standard_name("normal"),
            1.0,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-sample"),
        )
    }

    fn conditioning_payload_without_embeddings(model_id: &str) -> BurnConditioningPayload {
        let metadata = BurnConditioningMetadata::test_only(
            ModelId::new(model_id),
            77,
            "primary://test".to_owned(),
            "secondary://test".to_owned(),
        );
        let tokenized = BurnSdxlTokenizedPromptPair {
            clip_l: BurnSdxlTokenizedPrompt {
                token_ids: vec![0; 77],
                attention_mask: vec![1; 77],
            },
            clip_g: BurnSdxlTokenizedPrompt {
                token_ids: vec![0; 77],
                attention_mask: vec![1; 77],
            },
        };
        BurnConditioningPayload::test_only(metadata, tokenized)
    }

    fn conditioning_payload_without_pooled_embeddings(model_id: &str) -> BurnConditioningPayload {
        let config = BurnBackendConfig::new("/models", "/output");
        let device = active_device(config.device());
        let text = Tensor::<ActiveBurnBackend, 3>::zeros([1, 77, 2048], &device);
        conditioning_payload_without_embeddings(model_id)
            .with_embeddings(ClipOutputs::active(text, None))
    }

    #[test]
    fn diffusion_sample_rejects_non_burn_model_handle() {
        let backend = test_backend();
        let model = reimagine_inference::RuntimeModelHandle::with_instance(
            ModelId::new("sdxl-base"),
            reimagine_core::model::ModelRole::CheckpointBundle,
            Backend::new("candle"),
            BackendInstance::new("candle:cpu"),
            BackendPayloadKey::new("candle:model:sdxl-base:diffusion"),
        );
        let conditioning = ExecutionConditioning::new(
            BackendTensorHandle::new(
                Backend::new(BACKEND_LABEL),
                BackendPayloadKey::new("cond:test"),
                reimagine_core::model::TensorDType::F32,
                reimagine_core::model::TensorShape::new(vec![1, 77, 2048]),
                "cpu",
            ),
            reimagine_inference::ConditioningMetadata::new(512, 512),
        );
        let latent = RuntimeLatent::with_sdxl_base(
            BackendTensorHandle::new(
                Backend::new(BACKEND_LABEL),
                BackendPayloadKey::new("latent:test"),
                reimagine_core::model::TensorDType::F32,
                reimagine_core::model::TensorShape::new(vec![1, 4, 8, 8]),
                "cpu",
            ),
            64,
            64,
            1,
            4,
        );

        let request = DiffusionSampleRequest::new(
            model,
            conditioning.clone(),
            conditioning,
            latent,
            0,
            50,
            7.5,
            reimagine_inference::SamplerName::from_standard_name("euler"),
            reimagine_inference::SchedulerName::from_standard_name("normal"),
            1.0,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-sample"),
        );

        let err = execute_diffusion_sample(&backend, request).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("candle"), "msg: {msg}");
    }

    #[test]
    fn diffusion_sample_rejects_missing_bundle() {
        let backend = test_backend();
        let model = reimagine_inference::RuntimeModelHandle::with_instance(
            ModelId::new("missing-model"),
            reimagine_core::model::ModelRole::CheckpointBundle,
            Backend::new(BACKEND_LABEL),
            backend.backend_instance(),
            BackendPayloadKey::new("burn:model:missing-model:diffusion"),
        );
        let conditioning = ExecutionConditioning::new(
            BackendTensorHandle::new(
                Backend::new(BACKEND_LABEL),
                BackendPayloadKey::new("cond:test"),
                reimagine_core::model::TensorDType::F32,
                reimagine_core::model::TensorShape::new(vec![1, 77, 2048]),
                "cpu",
            ),
            reimagine_inference::ConditioningMetadata::new(512, 512),
        );
        let latent = RuntimeLatent::with_sdxl_base(
            BackendTensorHandle::new(
                Backend::new(BACKEND_LABEL),
                BackendPayloadKey::new("latent:test"),
                reimagine_core::model::TensorDType::F32,
                reimagine_core::model::TensorShape::new(vec![1, 4, 8, 8]),
                "cpu",
            ),
            64,
            64,
            1,
            4,
        );

        let request = DiffusionSampleRequest::new(
            model,
            conditioning.clone(),
            conditioning,
            latent,
            0,
            50,
            7.5,
            reimagine_inference::SamplerName::from_standard_name("euler"),
            reimagine_inference::SchedulerName::from_standard_name("normal"),
            1.0,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-sample"),
        );

        let err = execute_diffusion_sample(&backend, request).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("missing-model"), "msg: {msg}");
        assert!(msg.contains("load_bundle"), "msg: {msg}");
    }

    #[test]
    fn diffusion_sample_rejects_unsupported_denoise() {
        let backend = test_backend();
        seed_bundle(&backend, "sdxl-base");

        let model = reimagine_inference::RuntimeModelHandle::with_instance(
            ModelId::new("sdxl-base"),
            reimagine_core::model::ModelRole::CheckpointBundle,
            Backend::new(BACKEND_LABEL),
            backend.backend_instance(),
            BackendPayloadKey::new("burn:model:sdxl-base:diffusion"),
        );
        let conditioning = create_conditioning(&backend, "sdxl-base");
        let latent = create_latent(&backend);

        let request = DiffusionSampleRequest::new(
            model,
            conditioning.clone(),
            conditioning,
            latent,
            0,
            50,
            7.5,
            reimagine_inference::SamplerName::from_standard_name("euler"),
            reimagine_inference::SchedulerName::from_standard_name("normal"),
            0.8,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-sample"),
        );

        let err = execute_diffusion_sample(&backend, request).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("denoise"), "msg: {msg}");
        assert!(msg.contains("1.0"), "msg: {msg}");
    }

    #[test]
    fn diffusion_sample_rejects_unsupported_sampler() {
        let backend = test_backend();
        seed_bundle(&backend, "sdxl-base");

        let model = reimagine_inference::RuntimeModelHandle::with_instance(
            ModelId::new("sdxl-base"),
            reimagine_core::model::ModelRole::CheckpointBundle,
            Backend::new(BACKEND_LABEL),
            backend.backend_instance(),
            BackendPayloadKey::new("burn:model:sdxl-base:diffusion"),
        );
        let conditioning = create_conditioning(&backend, "sdxl-base");
        let latent = create_latent(&backend);

        let request = DiffusionSampleRequest::new(
            model,
            conditioning.clone(),
            conditioning,
            latent,
            0,
            50,
            7.5,
            reimagine_inference::SamplerName::from_standard_name("ddpm"),
            reimagine_inference::SchedulerName::from_standard_name("normal"),
            1.0,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-sample"),
        );

        let err = execute_diffusion_sample(&backend, request).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("ddpm"), "msg: {msg}");
    }

    #[test]
    fn diffusion_sample_succeeds_and_stores_sampled_latent() {
        let backend = test_backend();
        seed_bundle(&backend, "sdxl-base");

        let request = build_request(&backend, "sdxl-base");
        let response = execute_diffusion_sample(&backend, request).expect("diffusion.sample");
        let latent = response.into_latent();

        assert_eq!(latent.content(), LatentContent::Sampled);
        assert_eq!(latent.batch(), 1);
        assert_eq!(latent.width(), 64);
        assert_eq!(latent.height(), 64);
        assert_eq!(latent.latent_space(), &LatentSpaceMetadata::sdxl_base());
        assert_eq!(latent.payload().backend().as_str(), BACKEND_LABEL);

        // Verify stored in store
        let stored = backend
            .store()
            .get_latent(latent.payload().payload_key())
            .expect("stored sampled latent");
        assert!(stored.is_active_backend());
        assert_eq!(stored.dims(), [1, 4, 8, 8]);
    }

    #[test]
    fn diffusion_sample_rejects_conditioning_without_embeddings() {
        let backend = test_backend();
        seed_bundle(&backend, "sdxl-base");
        let request = build_request(&backend, "sdxl-base");
        let key = request.positive().text_embedding().payload_key().clone();
        backend.store().insert_conditioning(
            RunId::new("run-test"),
            key,
            conditioning_payload_without_embeddings("sdxl-base"),
        );

        let err = execute_diffusion_sample(&backend, request).unwrap_err();
        let msg = err.to_string();

        assert!(msg.contains("stored text encoder embeddings"), "msg: {msg}");
    }

    #[test]
    fn diffusion_sample_rejects_conditioning_without_pooled_embeddings_before_store_mutation() {
        let backend = test_backend();
        seed_bundle(&backend, "sdxl-base");
        let request = build_request(&backend, "sdxl-base");
        let key = request.positive().text_embedding().payload_key().clone();
        backend.store().insert_conditioning(
            RunId::new("run-test"),
            key,
            conditioning_payload_without_pooled_embeddings("sdxl-base"),
        );
        let before_payload_count = backend.store().payload_count();

        let err = execute_diffusion_sample(&backend, request).unwrap_err();
        let msg = err.to_string();

        assert!(msg.contains("pooled"), "msg: {msg}");
        assert_eq!(backend.store().payload_count(), before_payload_count);
        assert!(
            !backend
                .store()
                .contains_payload(&BackendPayloadKey::new("diffusion:run-test:node-sample"))
        );
    }

    #[test]
    fn diffusion_sample_produces_different_latent_for_different_seed() {
        let backend = test_backend();
        seed_bundle(&backend, "sdxl-base");

        // First sample with seed=0
        // First sample with seed=0
        let model_h = reimagine_inference::RuntimeModelHandle::with_instance(
            ModelId::new("sdxl-base"),
            reimagine_core::model::ModelRole::CheckpointBundle,
            Backend::new(BACKEND_LABEL),
            backend.backend_instance(),
            BackendPayloadKey::new("burn:model:sdxl-base:diffusion"),
        );
        let cond = create_conditioning(&backend, "sdxl-base");
        let lat = create_latent(&backend);
        let req1 = DiffusionSampleRequest::new(
            model_h.clone(),
            cond.clone(),
            cond.clone(),
            lat.clone(),
            0,
            50,
            7.5,
            reimagine_inference::SamplerName::from_standard_name("euler"),
            reimagine_inference::SchedulerName::from_standard_name("normal"),
            1.0,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-sample"),
        );
        let resp1 = execute_diffusion_sample(&backend, req1).expect("sample 1");
        let latent1 = resp1.into_latent();
        let stored1 = backend
            .store()
            .get_latent(latent1.payload().payload_key())
            .expect("stored 1");
        let data1 = stored1.to_data();

        // Second sample with seed=1 (different node_id to avoid overwrite)
        let req2 = DiffusionSampleRequest::new(
            model_h,
            cond.clone(),
            cond.clone(),
            lat,
            1,
            50,
            7.5,
            reimagine_inference::SamplerName::from_standard_name("euler"),
            reimagine_inference::SchedulerName::from_standard_name("normal"),
            1.0,
            RunId::new("run-test-2"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-sample-2"),
        );
        let resp2 = execute_diffusion_sample(&backend, req2).expect("sample 2");
        let latent2 = resp2.into_latent();
        let stored2 = backend
            .store()
            .get_latent(latent2.payload().payload_key())
            .expect("stored 2");
        let data2 = stored2.to_data();

        // Different seeds should produce different outputs
        let vals1: Vec<f32> = data1.to_vec::<f32>().unwrap();
        let vals2: Vec<f32> = data2.to_vec::<f32>().unwrap();
        assert!(
            vals1
                .iter()
                .zip(vals2.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "different seeds should produce different output tensors"
        );
    }
}
