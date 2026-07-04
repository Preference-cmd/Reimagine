//! `text.encode` operation for the Burn backend.
//!
//! burn/08a owned the preflight pipeline (validation + tokenization);
//! burn/08f replaces it with the real `execute_text_encode` entry
//! point that runs the same preflight, persists the conditioning
//! payload, and returns backend-affine handles matching the expected
//! SDXL output shapes (`[1, 77, 2048]` text, `[1, 1280]` pooled).

use reimagine_inference::{
    BackendInstance, ConditioningMetadata, ExecutionConditioning, RuntimeClipHandle,
    TextEncodeRequest, TextEncodeResponse,
};

use crate::backend::BurnBackend;
use crate::error::BurnBackendError;
use crate::models::stable_diffusion::sdxl::text_conditioning::forward::clip_forward;
use crate::models::stable_diffusion::sdxl::text_conditioning::loading;
use crate::models::stable_diffusion::sdxl::{
    BurnSdxlTextEncoderResources, BurnSdxlTokenizedPromptPair,
};
use crate::profile::{BACKEND_LABEL, BurnProfileProvider};
use crate::store::{BurnConditioningMetadata, BurnConditioningPayload, ClipOutputs};

/// `text.encode` entry point for the Burn backend (burn/08f).
///
/// The function runs the full V1 preflight pipeline (validation +
/// tokenization), persists the conditioning payload, and returns a
/// `TextEncodeResponse` with backend-affine handles matching the
/// expected SDXL output shapes (`[1, 77, 2048]` text embedding,
/// `[1, 1280]` pooled embedding). The actual CLIP tensor forward
/// pass is a follow-up deepening; V1 advertises `TextEncode` so the
/// router can select the Burn backend for text.encode tasks and the
/// stored payload is the preconditioned tokenization record.
pub fn execute_text_encode(
    backend: &BurnBackend,
    request: TextEncodeRequest,
) -> Result<TextEncodeResponse, BurnBackendError> {
    let run_id = request.run_id().clone();
    let preflight = build_preflight(backend, request)?;
    let model_id = preflight.metadata().model_id().to_string();

    // burn/14c: run the real CLIP-L and OpenCLIP-G forward passes
    // using weights loaded from the bundle's component files. The
    // tokenized prompts were already validated and produced by
    // `build_preflight`.
    let bundle = backend
        .model_cache()
        .get_bundle(preflight.metadata().model_id())
        .ok_or_else(|| {
            BurnBackendError::InvalidRequest(format!(
                "text.encode could not resolve the loaded bundle for model `{}`; \
                 the bundle may have been evicted between validation and lookup",
                preflight.metadata().model_id()
            ))
        })?;
    let sdxl_bundle = match bundle.as_ref() {
        crate::models::stable_diffusion::sdxl::BurnLoadedModelBundle::StableDiffusionSdxl(b) => {
            b.as_ref()
        }
    };

    // For test-only bundles that have no component paths (e.g. those built
    // via `for_test_only`), the CLIP weights are unavailable. Production
    // bundles built via `from_resolved` always populate component paths,
    // so this branch is purely a test seam.
    let embeddings = if let (Some(_), Some(_)) = (
        sdxl_bundle.components().iter().find(|c| {
            c.component_role
                == crate::models::stable_diffusion::sdxl::BurnSdxlComponentRole::TextEncoder
        }),
        sdxl_bundle.components().iter().find(|c| {
            c.component_role
                == crate::models::stable_diffusion::sdxl::BurnSdxlComponentRole::TextEncoder2
        }),
    ) {
        let clip_l_weights = loading::load_clip_l(sdxl_bundle)?;
        let clip_g_weights = loading::load_clip_g(sdxl_bundle)?;

        let profile_l = crate::text_encoder::clip::ClipTextEncoderProfile::clip_l();
        let profile_g = crate::text_encoder::clip::ClipTextEncoderProfile::open_clip_g();

        let clip_l_out = clip_forward(
            &preflight.tokenized_prompts().clip_l.token_ids,
            &clip_l_weights,
            &profile_l,
        )?;
        let clip_g_out = clip_forward(
            &preflight.tokenized_prompts().clip_g.token_ids,
            &clip_g_weights,
            &profile_g,
        )?;

        // Concatenate last hidden states along channel axis: [1,77,768] + [1,77,1280] → [1,77,2048]
        let text_emb = concat_channel(clip_l_out.text_embeddings, clip_g_out.text_embeddings)?;
        let pooled_emb = clip_g_out.pooled.ok_or_else(|| {
            BurnBackendError::InvalidRequest(
                "OpenCLIP-G forward did not produce a pooled embedding".to_string(),
            )
        })?;
        ClipOutputs {
            text_embeddings: text_emb,
            pooled_embeddings: Some(pooled_emb),
        }
    } else {
        // Test-only bundle without components: fall back to synthetic
        // shape-correct placeholders so unit tests can still exercise
        // the surrounding store + handle plumbing.
        synthetic_clip_outputs(preflight.tokenized_prompts())?
    };

    let payload = BurnConditioningPayload::from_tokenized(
        preflight.metadata().clone(),
        preflight.tokenized_prompts().clone(),
    )
    .with_embeddings(embeddings);

    let payload_key =
        reimagine_inference::BackendPayloadKey::new(format!("conditioning:{model_id}"));
    backend
        .store()
        .insert_conditioning(run_id, payload_key.clone(), payload);

    // Build response handles with correct shape metadata.
    let text_handle = reimagine_inference::BackendTensorHandle::with_instance(
        BurnProfileProvider::backend_kind(),
        backend.backend_instance(),
        payload_key.clone(),
        reimagine_core::model::TensorDType::F32,
        reimagine_core::model::TensorShape::new(vec![1, 77, 2048]),
        backend.device_label(),
    );
    let pooled_handle = reimagine_inference::BackendTensorHandle::with_instance(
        BurnProfileProvider::backend_kind(),
        backend.backend_instance(),
        payload_key,
        reimagine_core::model::TensorDType::F32,
        reimagine_core::model::TensorShape::new(vec![1, 1280]),
        backend.device_label(),
    );

    let conditioning = ExecutionConditioning::new(text_handle, ConditioningMetadata::new(512, 512))
        .with_pooled_embedding(pooled_handle);

    Ok(TextEncodeResponse::new(conditioning))
}

/// Build synthetic `ClipOutputs` for tests where the bundle has no
/// component paths (constructed via `BurnLoadedSdxlBundle::for_test_only`).
/// The placeholders carry the correct shapes so surrounding store and
/// handle logic can be exercised. Production bundles always have real
/// CLIP weights and never take this branch.
fn synthetic_clip_outputs(
    tokenized: &crate::models::stable_diffusion::sdxl::BurnSdxlTokenizedPromptPair,
) -> Result<ClipOutputs, BurnBackendError> {
    use burn_ndarray::NdArray;
    let seq_len = tokenized.clip_l.token_ids.len();
    let total_text = seq_len * 2048;
    let text_data = vec![0.0f32; total_text];
    let text_tensor = burn_tensor::Tensor::<NdArray, 3>::from_data(
        burn_tensor::TensorData::new(text_data, burn_tensor::Shape::new([1, seq_len, 2048])),
        &burn_ndarray::NdArrayDevice::Cpu,
    );
    let pooled_tensor = burn_tensor::Tensor::<NdArray, 2>::from_data(
        burn_tensor::TensorData::new(vec![0.0f32; 1280], burn_tensor::Shape::new([1, 1280])),
        &burn_ndarray::NdArrayDevice::Cpu,
    );
    Ok(ClipOutputs {
        text_embeddings: crate::tensor::BurnTensor::Ndarray(text_tensor),
        pooled_embeddings: Some(crate::tensor::BurnTensor::Ndarray(pooled_tensor)),
    })
}

/// Concatenate two `[batch, seq, w1]` and `[batch, seq, w2]` tensors along
/// the channel axis to produce `[batch, seq, w1 + w2]`.
fn concat_channel(
    a: crate::tensor::BurnTensor<3>,
    b: crate::tensor::BurnTensor<3>,
) -> Result<crate::tensor::BurnTensor<3>, BurnBackendError> {
    use burn_ndarray::NdArray;
    let (ta, tb) = match (a, b) {
        (crate::tensor::BurnTensor::Ndarray(ta), crate::tensor::BurnTensor::Ndarray(tb)) => {
            (ta, tb)
        }
    };
    let shape_a = ta.dims();
    let shape_b = tb.dims();
    if shape_a[0] != shape_b[0] || shape_a[1] != shape_b[1] {
        return Err(BurnBackendError::InvalidRequest(format!(
            "concat_channel: shape mismatch (a={shape_a:?}, b={shape_b:?})"
        )));
    }
    let batch = shape_a[0];
    let seq_len = shape_a[1];
    let w1 = shape_a[2];
    let w2 = shape_b[2];
    let w_combined = w1 + w2;

    let a_data = ta.to_data();
    let b_data = tb.to_data();
    let a_slice = a_data
        .to_vec::<f32>()
        .map_err(|e| BurnBackendError::InvalidRequest(format!("concat_channel a data: {e}")))?;
    let b_slice = b_data
        .to_vec::<f32>()
        .map_err(|e| BurnBackendError::InvalidRequest(format!("concat_channel b data: {e}")))?;

    // Plan: for each (b, s) row, output [a_row; b_row]
    let mut combined = Vec::with_capacity(batch * seq_len * w_combined);
    for idx in 0..(batch * seq_len) {
        let a_start = idx * w1;
        let b_start = idx * w2;
        combined.extend_from_slice(&a_slice[a_start..a_start + w1]);
        combined.extend_from_slice(&b_slice[b_start..b_start + w2]);
    }

    let tensor = burn_tensor::Tensor::<NdArray, 3>::from_data(
        burn_tensor::TensorData::new(
            combined,
            burn_tensor::Shape::new([batch, seq_len, w_combined]),
        ),
        &burn_ndarray::NdArrayDevice::Cpu,
    );
    Ok(crate::tensor::BurnTensor::Ndarray(tensor))
}

/// Result of a `text.encode` preflight: the validated inputs and
/// the deterministic tokenization outputs needed by future
/// execution slices. The struct stays Burn-private; callers that
/// want a real `TextEncodeResponse` must wait for burn/08f.
#[derive(Debug, Clone)]
pub struct BurnTextEncodePreflight {
    clip: RuntimeClipHandle,
    prompt: String,
    tokenized: BurnSdxlTokenizedPromptPair,
    metadata: BurnConditioningMetadata,
}

#[allow(dead_code)] // burn/08f consumes the accessors; production text.encode discards the record.
impl BurnTextEncodePreflight {
    pub fn clip(&self) -> &RuntimeClipHandle {
        &self.clip
    }

    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    pub fn tokenized_prompts(&self) -> &BurnSdxlTokenizedPromptPair {
        &self.tokenized
    }

    pub fn metadata(&self) -> &BurnConditioningMetadata {
        &self.metadata
    }

    /// Build the Burn-private conditioning payload that the future
    /// burn/08f slice would insert into the shared store. The
    /// preflight only constructs the payload; it does not insert
    /// it, so production `text.encode` never claims a successful
    /// forward pass.
    pub fn into_conditioning_payload(self) -> BurnConditioningPayload {
        BurnConditioningPayload::from_tokenized(self.metadata, self.tokenized)
    }
}

/// Construct a fully validated [`BurnTextEncodePreflight`].
///
/// This helper is split out from `execute_text_encode` so the
/// validation + tokenization path can be reused without going
/// through the production `text.encode` insertion boundary.
pub fn build_preflight(
    backend: &BurnBackend,
    request: TextEncodeRequest,
) -> Result<BurnTextEncodePreflight, BurnBackendError> {
    let clip = request.clip().clone();
    validate_clip_backend(&clip, &backend.backend_instance())?;
    validate_bundle_loaded(backend, &clip)?;
    let prompt = extract_prompt(&request)?;
    let (tokenized, metadata) = tokenize_and_capture_metadata(backend, &clip, &prompt)?;
    Ok(BurnTextEncodePreflight {
        clip,
        prompt,
        tokenized,
        metadata,
    })
}

fn validate_clip_backend(
    clip: &RuntimeClipHandle,
    current_instance: &BackendInstance,
) -> Result<(), BurnBackendError> {
    if clip.backend().as_str() != BACKEND_LABEL {
        return Err(BurnBackendError::InvalidRequest(format!(
            "text.encode preflight requires a burn clip handle, got backend `{}`",
            clip.backend().as_str()
        )));
    }
    if clip.backend_instance() != current_instance {
        return Err(BurnBackendError::InvalidRequest(format!(
            "text.encode preflight received clip handle for backend instance `{}` but this Burn backend is `{}`",
            clip.backend_instance(),
            current_instance
        )));
    }
    Ok(())
}

fn validate_bundle_loaded(
    backend: &BurnBackend,
    clip: &RuntimeClipHandle,
) -> Result<(), BurnBackendError> {
    if !backend.model_cache().contains(clip.model_id()) {
        return Err(BurnBackendError::InvalidRequest(format!(
            "text.encode preflight requires the burn clip bundle for model `{}` to be loaded; call load_bundle first",
            clip.model_id()
        )));
    }
    Ok(())
}

fn extract_prompt(request: &TextEncodeRequest) -> Result<String, BurnBackendError> {
    request.prompt_string().ok_or_else(|| {
        BurnBackendError::InvalidRequest(format!(
            "text.encode preflight requires a `Param(String)` or `Param(Text)` prompt; got execution value of type `{:?}`",
            request.text().as_ref()
        ))
    })
}

fn tokenize_and_capture_metadata(
    backend: &BurnBackend,
    clip: &RuntimeClipHandle,
    prompt: &str,
) -> Result<(BurnSdxlTokenizedPromptPair, BurnConditioningMetadata), BurnBackendError> {
    let text_resources = BurnSdxlTextEncoderResources::load(backend.config())?;
    let tokenized = text_resources.tokenize_pair(prompt)?;
    // The preflight already validated the bundle is loaded via
    // `validate_bundle_loaded`, but that check and this lookup
    // take the cache lock separately, so a future `remove_bundle`
    // could evict the entry in between. We surface that as a
    // precise `InvalidRequest` instead of panicking — the burn/08f
    // path will reuse this helper.
    let bundle = backend
        .model_cache()
        .get_bundle(clip.model_id())
        .ok_or_else(|| {
            BurnBackendError::InvalidRequest(format!(
                "text.encode preflight could not resolve the loaded bundle for model `{}`; \
                 the bundle may have been evicted between validation and lookup",
                clip.model_id()
            ))
        })?;
    let metadata = BurnConditioningMetadata::from_bundle(
        &bundle,
        text_resources.sequence_length() as u32,
        text_resources
            .tokenizer()
            .resources()
            .primary_path()
            .display()
            .to_string(),
        text_resources
            .tokenizer()
            .resources()
            .secondary_path()
            .display()
            .to_string(),
    );
    Ok((tokenized, metadata))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BurnBackendConfig;
    use reimagine_core::model::{ModelId, NodeId, RunId, WorkflowId, WorkflowVersion};
    use reimagine_inference::{Backend, BackendInstance, BackendPayloadKey, RuntimeClipHandle};
    use std::sync::Arc;

    fn backend() -> BurnBackend {
        BurnBackend::new(BurnBackendConfig::new("/models", "/output")).expect("burn backend")
    }

    fn build_request(
        _backend: &BurnBackend,
        clip: RuntimeClipHandle,
        text: reimagine_inference::ExecutionValue,
    ) -> TextEncodeRequest {
        TextEncodeRequest::new(
            clip,
            Arc::new(text),
            RunId::new("run-text"),
            WorkflowId::new("wf-text"),
            WorkflowVersion::new(1),
            NodeId::new("node-text"),
        )
    }

    fn burn_clip(
        backend: &BurnBackend,
        model_id: &str,
        instance: BackendInstance,
    ) -> RuntimeClipHandle {
        RuntimeClipHandle::with_instance(
            ModelId::new(model_id),
            Backend::new(BACKEND_LABEL),
            instance,
            BackendPayloadKey::new(format!("burn:model:{model_id}:clip")),
        )
        .with_device(backend.device_label())
    }

    fn not_burn_clip(model_id: &str) -> RuntimeClipHandle {
        RuntimeClipHandle::with_instance(
            ModelId::new(model_id),
            Backend::new("candle"),
            BackendInstance::new("candle:cpu"),
            BackendPayloadKey::new(format!("candle:model:{model_id}:clip")),
        )
    }

    fn seed_bundle(backend: &BurnBackend, model_id: &str) {
        use crate::models::stable_diffusion::sdxl::{BurnLoadedModelBundle, BurnLoadedSdxlBundle};
        let bundle = BurnLoadedSdxlBundle::for_test_only(
            ModelId::new(model_id),
            BackendPayloadKey::new(format!("burn:model:{model_id}:clip")),
        );
        backend.model_cache().insert_bundle(
            ModelId::new(model_id),
            Arc::new(BurnLoadedModelBundle::StableDiffusionSdxl(Arc::new(bundle))),
        );
    }

    #[test]
    fn preflight_rejects_non_burn_clip_handle() {
        let backend = backend();
        let clip = not_burn_clip("sdxl");
        let request = build_request(
            &backend,
            clip.clone(),
            reimagine_inference::ExecutionValue::Param(reimagine_core::model::ParamValue::String(
                "hello".to_owned(),
            )),
        );

        let err = execute_text_encode(&backend, request).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("burn clip handle"), "msg: {msg}");
        assert!(msg.contains("candle"), "msg: {msg}");
    }

    #[test]
    fn preflight_rejects_clip_handle_for_different_burn_instance() {
        let backend = backend();
        let foreign_instance = BackendInstance::new("burn:wgpu");
        let clip = burn_clip(&backend, "sdxl-base", foreign_instance);
        let request = build_request(
            &backend,
            clip,
            reimagine_inference::ExecutionValue::Param(reimagine_core::model::ParamValue::String(
                "hello".to_owned(),
            )),
        );

        let err = execute_text_encode(&backend, request).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("backend instance"), "msg: {msg}");
        assert!(msg.contains("burn:wgpu"), "msg: {msg}");
        // burn/13: the default instance label is `burn:cpu`
        // under `wgpu` (or neither), or `burn:flex:cpu` under
        // `flex`.
        let expected_self = if cfg!(all(not(feature = "wgpu"), feature = "flex")) {
            "burn:flex:cpu"
        } else {
            "burn:cpu"
        };
        assert!(msg.contains(expected_self), "msg: {msg}");
    }

    #[test]
    fn preflight_rejects_missing_bundle() {
        let backend = backend();
        let clip = burn_clip(&backend, "missing-model", backend.backend_instance());
        let request = build_request(
            &backend,
            clip,
            reimagine_inference::ExecutionValue::Param(reimagine_core::model::ParamValue::String(
                "hello".to_owned(),
            )),
        );

        let err = execute_text_encode(&backend, request).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("missing-model"), "msg: {msg}");
        assert!(msg.contains("loaded"), "msg: {msg}");
    }

    #[test]
    fn preflight_rejects_non_param_string_or_text_prompt() {
        let backend = backend();
        seed_bundle(&backend, "sdxl-base");
        let clip = burn_clip(&backend, "sdxl-base", backend.backend_instance());
        let request = build_request(
            &backend,
            clip,
            reimagine_inference::ExecutionValue::Param(reimagine_core::model::ParamValue::Integer(
                42,
            )),
        );

        let err = execute_text_encode(&backend, request).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Param(String)"), "msg: {msg}");
        assert!(msg.contains("Param(Text)"), "msg: {msg}");
    }

    #[test]
    fn preflight_rejects_orphan_latent_as_prompt() {
        let backend = backend();
        seed_bundle(&backend, "sdxl-base");
        let clip = burn_clip(&backend, "sdxl-base", backend.backend_instance());
        let request = build_request(
            &backend,
            clip,
            reimagine_inference::ExecutionValue::Latent(
                reimagine_inference::RuntimeLatent::with_sdxl_base(
                    reimagine_inference::BackendTensorHandle::new(
                        Backend::new(BACKEND_LABEL),
                        BackendPayloadKey::new("latent:run-text:node-text"),
                        reimagine_core::model::TensorDType::F32,
                        reimagine_core::model::TensorShape::new(vec![1, 4, 8, 8]),
                        "cpu",
                    ),
                    64,
                    64,
                    1,
                    4,
                ),
            ),
        );

        let err = execute_text_encode(&backend, request).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Param(String)"), "msg: {msg}");
    }

    #[test]
    fn preflight_succeeds_with_param_string_prompt_and_inserts_conditioning() {
        let backend = backend();
        seed_bundle(&backend, "sdxl-base");
        let clip = burn_clip(&backend, "sdxl-base", backend.backend_instance());
        let request = build_request(
            &backend,
            clip,
            reimagine_inference::ExecutionValue::Param(reimagine_core::model::ParamValue::String(
                "hello".to_owned(),
            )),
        );

        // burn/08f: production text.encode runs the full preflight,
        // persists the preconditioned tokenization as a conditioning
        // payload, and returns backend-affine handles with the
        // expected SDXL output shapes. Real CLIP tensor forward is
        // a follow-up deepening.
        let response = execute_text_encode(&backend, request).expect("text.encode succeeds");
        let conditioning = response.conditioning();
        assert_eq!(conditioning.text_embedding().backend().as_str(), "burn");
        assert_eq!(
            conditioning.text_embedding().shape().dims(),
            &[1_usize, 77, 2048]
        );
        let pooled = conditioning
            .pooled_embedding()
            .expect("pooled handle present");
        assert_eq!(pooled.shape().dims(), &[1_usize, 1280]);
        // Production text.encode must insert the preconditioned
        // record into the store.
        assert!(
            backend.store().payload_count() > 0,
            "conditioning payload stored"
        );
    }

    #[test]
    fn preflight_succeeds_with_param_text_prompt_and_inserts_conditioning() {
        let backend = backend();
        seed_bundle(&backend, "sdxl-base");
        let clip = burn_clip(&backend, "sdxl-base", backend.backend_instance());
        let request = build_request(
            &backend,
            clip,
            reimagine_inference::ExecutionValue::Param(reimagine_core::model::ParamValue::Text(
                "hello world".to_owned(),
            )),
        );

        let response = execute_text_encode(&backend, request).expect("text.encode succeeds");
        assert_eq!(
            response.conditioning().text_embedding().shape().dims(),
            &[1_usize, 77, 2048]
        );
        assert!(
            backend.store().payload_count() > 0,
            "conditioning payload stored"
        );
    }

    #[test]
    fn preflight_tokenizes_with_both_tokenizers_via_build_helper() {
        // `build_preflight` is the seam future burn/08f will
        // reuse to obtain the preconditioned record before
        // running the CLIP forward pass. The preflight itself
        // does not expose the record through the production
        // text.encode entry point, so we exercise it directly
        // here.
        let backend = backend();
        seed_bundle(&backend, "sdxl-base");
        let clip = burn_clip(&backend, "sdxl-base", backend.backend_instance());
        let request = build_request(
            &backend,
            clip.clone(),
            reimagine_inference::ExecutionValue::Param(reimagine_core::model::ParamValue::String(
                "hello".to_owned(),
            )),
        );

        let preflight = build_preflight(&backend, request).expect("preflight");
        assert_eq!(preflight.prompt(), "hello");
        assert_eq!(preflight.tokenized_prompts().clip_l.token_ids.len(), 77);
        assert_eq!(preflight.tokenized_prompts().clip_g.token_ids.len(), 77);
        assert_eq!(preflight.metadata().model_id().as_str(), "sdxl-base");
        assert_eq!(preflight.metadata().sequence_length(), 77);
        assert!(preflight.metadata().pooled_embedding_available());
        assert!(
            preflight
                .metadata()
                .primary_tokenizer_id()
                .contains("tokenizer")
        );
        assert!(
            preflight
                .metadata()
                .secondary_tokenizer_id()
                .contains("tokenizer")
        );
        // Building the conditioning payload from the preflight
        // works, but production text.encode never calls it.
        let _ = preflight.into_conditioning_payload();
        assert_eq!(backend.store().payload_count(), 0);
    }
}
