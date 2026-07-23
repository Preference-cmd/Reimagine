//! `latent.create_empty` and `latent.decode` operations.
//!
//! ## `latent.create_empty`
//!
//! Allocates a zero-initialized latent tensor for the request's
//! latent space in the backend store and returns a
//! [`CreateEmptyLatentResponse`] carrying a lightweight
//! `RuntimeLatent` handle. The handle's metadata is the request's
//! [`LatentSpaceMetadata`], so the returned latent agrees with
//! what the caller requested even if a future workflow layer
//! derives the latent space from a connected model handle.
//!
//! ## `latent.decode`
//!
//! Consumes a VAE handle and a sampled latent, validates that the
//! latent's metadata matches the loaded VAE's expected latent
//! space, dispatches to the loaded model family's decoder through
//! the graph facade, stores the resulting image payload in the
//! backend store, and returns a [`LatentDecodeResponse`] carrying
//! a lightweight `RuntimeImage` handle.
//!
//! The operation is model-family-neutral at the protocol level.
//! SDXL-specific VAE decoding lives behind the facade in
//! `models/stable_diffusion/sdxl/vae.rs`.
//!
//! V1 real decode is limited to `batch=1`. Latents with `batch != 1`
//! are rejected with a precise unsupported-batch error before any
//! decode work is performed, so save/preview cannot silently drop
//! or mishandle outputs.
//!
//! The VAE handle and latent handle must point at the same
//! `BackendInstance`. Mismatched backend instances produce a
//! precise affinity error.

use candle_core::{DType, Tensor};
use reimagine_core::model::{TensorDType, TensorShape};
use reimagine_inference::latent_space::validate_pixel_dimensions_against;
use reimagine_inference::{
    BackendPayloadKey, BackendTensorHandle, CreateEmptyLatentRequest, CreateEmptyLatentResponse,
    InferenceBackend, LatentDecodeRequest, LatentDecodeResponse, LatentEncodeRequest,
    LatentEncodeResponse, LatentSpaceError, LatentSpaceMetadata, RuntimeImage, RuntimeLatent,
    RuntimeVaeHandle,
};

use crate::backend::CandleBackend;
use crate::error::CandleBackendError;
use crate::graph::{LatentDecodeInput, LatentDecodeResult, LatentEncodeInput, LatentEncodeResult};

fn map_latent_space_error(label: &str, err: LatentSpaceError) -> CandleBackendError {
    CandleBackendError::InvalidRequest(format!("{label} {err}"))
}

fn tensor_dtype_for(metadata: &LatentSpaceMetadata) -> Result<DType, CandleBackendError> {
    match metadata.dtype() {
        TensorDType::F32 => Ok(DType::F32),
        TensorDType::F16 => Ok(DType::F16),
        TensorDType::BF16 => Ok(DType::BF16),
        TensorDType::I64 => Ok(DType::I64),
        TensorDType::U8 => Ok(DType::U8),
    }
}

fn validate_batch(batch_size: u32) -> Result<(), CandleBackendError> {
    if batch_size == 0 {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent batch_size must be positive (got {batch_size})"
        )));
    }
    Ok(())
}

pub fn execute_latent_create_empty(
    backend: &CandleBackend,
    request: CreateEmptyLatentRequest,
) -> Result<CreateEmptyLatentResponse, CandleBackendError> {
    let width = request.width();
    let height = request.height();
    let batch_size = request.batch_size();
    let latent_space = request.latent_space().clone();

    // V1 Candle only supports the SDXL base latent space. Any
    // other latent space — even one that is otherwise well-formed
    // — must be rejected here with a precise error so the
    // downstream operations do not silently materialize a tensor
    // that no V1 model can consume.
    let supported = LatentSpaceMetadata::sdxl_base();
    if !latent_space.is_compatible(&supported) {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.create_empty latent space `{}` (channels={}, scale={}) is not supported by backend `candle`; only `{}` is supported in V1",
            latent_space.id(),
            latent_space.channels(),
            latent_space.spatial_scale_factor(),
            supported.id(),
        )));
    }

    validate_batch(batch_size)?;
    validate_pixel_dimensions_against(width, height, &latent_space)
        .map_err(|err| map_latent_space_error("latent.create_empty", err))?;

    let channels: u32 = latent_space.channels();
    let scale = latent_space.spatial_scale_factor();
    let latent_width = width / scale;
    let latent_height = height / scale;
    let shape: Vec<usize> = vec![
        batch_size as usize,
        channels as usize,
        latent_height as usize,
        latent_width as usize,
    ];

    let tensor = Tensor::zeros(
        shape.clone(),
        tensor_dtype_for(&latent_space)?,
        backend.device().as_ref(),
    )
    .map_err(|err| {
        CandleBackendError::InvalidRequest(format!(
            "failed to allocate latent tensor for shape {shape:?}: {err}"
        ))
    })?;

    let payload_key = BackendPayloadKey::new(format!(
        "latent:{}:{}",
        request.run_id().as_str(),
        request.node_id().as_str()
    ));

    backend.store().insert_latent(
        request.run_id().clone(),
        payload_key.clone(),
        tensor,
        latent_space.clone(),
    );

    let latent = RuntimeLatent::new(
        BackendTensorHandle::with_instance(
            backend.backend_kind().clone(),
            backend.backend_instance(),
            payload_key,
            latent_space.dtype(),
            TensorShape::new(shape),
            backend.device_label(),
        ),
        width,
        height,
        batch_size,
        channels,
        latent_space,
        reimagine_inference::LatentContent::EmptyGeometry,
    );
    Ok(CreateEmptyLatentResponse::new(latent))
}

pub fn execute_latent_decode(
    request: LatentDecodeRequest,
    backend: &CandleBackend,
) -> Result<LatentDecodeResponse, CandleBackendError> {
    let vae_handle = request.vae().clone();
    let latent_handle = request.latent().clone();

    // Reject `LatentContent::EmptyGeometry` at the operation
    // boundary. txt2img geometry is a placeholder tensor, not a
    // real latent payload; the VAE decoder must not consume it.
    // The check fires before any affinity / latent-space
    // validation so the diagnostic is precise and easy to attribute
    // to the upstream node producing empty geometry.
    if matches!(
        latent_handle.content(),
        reimagine_inference::LatentContent::EmptyGeometry
    ) {
        return Err(CandleBackendError::InvalidRequest(
            "latent.decode rejects latent content `empty_geometry`; \
             the input latent was produced by `latent.create_empty` and is not a real starting latent. \
             Use `latent.encode` (image → latent) or `diffusion.sample` (geometry → sampled) before decoding."
                .to_string(),
        ));
    }

    if vae_handle.backend() != backend.backend_kind() {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.decode `vae` handle belongs to backend `{}`, expected `{}`",
            vae_handle.backend().as_str(),
            backend.backend_kind()
        )));
    }
    if latent_handle.payload().backend() != backend.backend_kind() {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.decode `latent` handle belongs to backend `{}`, expected `{}`",
            latent_handle.payload().backend().as_str(),
            backend.backend_kind()
        )));
    }

    // Validate that VAE and latent handles share the same backend
    // instance. Cross-instance decode is unsupported in V1; this
    // must be surfaced as a precise affinity error rather than
    // silently coercing tensors across instances. The check is
    // gated on the handle having a non-default instance token so
    // callers using the legacy `Backend::new("candle")` shortcut
    // continue to work; the model-bundle affinity check below
    // still rejects cross-instance decodes for explicit handles.
    let expected_instance = backend.backend_instance();
    if vae_handle.backend_instance().as_str() != expected_instance.as_str()
        && vae_handle.backend_instance().as_str() != backend.backend_kind().as_str()
    {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.decode `vae` handle belongs to backend instance `{}`, expected `{}`",
            vae_handle.backend_instance().as_str(),
            expected_instance.as_str()
        )));
    }
    if latent_handle.payload().backend_instance().as_str() != expected_instance.as_str()
        && latent_handle.payload().backend_instance().as_str() != backend.backend_kind().as_str()
    {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.decode `latent` handle belongs to backend instance `{}`, expected `{}`",
            latent_handle.payload().backend_instance().as_str(),
            expected_instance.as_str()
        )));
    }

    // Reject batch != 1 at the operation boundary so save/preview
    // cannot silently drop or mishandle outputs. V1 real decode is
    // limited to a single latent sample.
    if latent_handle.batch() != 1 {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.decode V1 supports only batch=1; got batch={} for latent handle `{}`. Multi-image decode is not supported in this issue",
            latent_handle.batch(),
            latent_handle.payload().payload_key().as_str()
        )));
    }

    let bundle = require_bundle_for_vae(&vae_handle, backend)?;
    bundle.validate_vae_handle(&vae_handle)?;

    // Validate the incoming latent space against the VAE bundle's
    // expected latent space. This is a precise request-time
    // rejection: if a different latent space reaches the decoder,
    // the call fails before the decoder tensor ops can blow up
    // obscurely.
    let expected = bundle.expected_latent_space();
    if !latent_handle.latent_space().is_compatible(&expected) {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.decode input latent space `{}` (channels={}, scale={}) is incompatible with loaded {} VAE expected latent space `{}` (channels={}, scale={})",
            latent_handle.latent_space().id(),
            latent_handle.latent_space().channels(),
            latent_handle.latent_space().spatial_scale_factor(),
            bundle.family_label(),
            expected.id(),
            expected.channels(),
            expected.spatial_scale_factor(),
        )));
    }

    let input_latent = backend
        .store()
        .get_latent(latent_handle.payload().payload_key())?;

    // The stored payload and the handle must agree on latent space.
    // This protects against stale/aliased keys that landed in the
    // store with a different metadata record than the caller used.
    if !input_latent
        .latent_space()
        .is_compatible(latent_handle.latent_space())
    {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.decode stored latent payload `{}` latent space `{}` disagrees with latent handle latent space `{}`",
            latent_handle.payload().payload_key().as_str(),
            input_latent.latent_space().id(),
            latent_handle.latent_space().id(),
        )));
    }
    if !input_latent.latent_space().is_compatible(&expected) {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.decode stored latent payload `{}` latent space `{}` disagrees with loaded {} VAE expected latent space `{}`",
            latent_handle.payload().payload_key().as_str(),
            input_latent.latent_space().id(),
            bundle.family_label(),
            expected.id(),
        )));
    }

    let LatentDecodeResult { image } = bundle.decode_latent(
        LatentDecodeInput {
            latent: input_latent,
        },
        backend.device().as_ref(),
    )?;

    let payload_key = BackendPayloadKey::new(format!(
        "image:{}:{}",
        request.run_id().as_str(),
        request.node_id().as_str()
    ));

    backend
        .store()
        .insert_image(request.run_id().clone(), payload_key.clone(), image.clone());

    let image_value = RuntimeImage::new(
        BackendTensorHandle::with_instance(
            backend.backend_kind().clone(),
            backend.backend_instance(),
            payload_key,
            TensorDType::F32,
            TensorShape::new(image.dims()),
            backend.device_label(),
        ),
        image.width(),
        image.height(),
        image.batch(),
        image.color_space().to_string(),
    );

    Ok(LatentDecodeResponse::new(image_value))
}

pub fn execute_latent_encode(
    backend: &CandleBackend,
    request: LatentEncodeRequest,
) -> Result<LatentEncodeResponse, CandleBackendError> {
    let vae_handle = request.vae().clone();
    let image_handle = request.image().clone();

    if vae_handle.backend() != backend.backend_kind() {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.encode `vae` handle belongs to backend `{}`, expected `{}`",
            vae_handle.backend().as_str(),
            backend.backend_kind()
        )));
    }
    if image_handle.payload().backend() != backend.backend_kind() {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.encode `image` handle belongs to backend `{}`, expected `{}`",
            image_handle.payload().backend().as_str(),
            backend.backend_kind()
        )));
    }

    let expected_instance = backend.backend_instance();
    if vae_handle.backend_instance().as_str() != expected_instance.as_str()
        && vae_handle.backend_instance().as_str() != backend.backend_kind().as_str()
    {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.encode `vae` handle belongs to backend instance `{}`, expected `{}`",
            vae_handle.backend_instance().as_str(),
            expected_instance.as_str()
        )));
    }
    if image_handle.payload().backend_instance().as_str() != expected_instance.as_str()
        && image_handle.payload().backend_instance().as_str() != backend.backend_kind().as_str()
    {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.encode `image` handle belongs to backend instance `{}`, expected `{}`",
            image_handle.payload().backend_instance().as_str(),
            expected_instance.as_str()
        )));
    }

    let bundle = require_bundle_for_vae(&vae_handle, backend)?;
    bundle.validate_vae_handle(&vae_handle)?;
    let expected = bundle.expected_latent_space();

    let image = backend
        .store()
        .get_image(image_handle.payload().payload_key())?;
    validate_image_handle_matches_store(&image_handle, &image)?;
    validate_image_for_latent_encode(&image_handle, &image, &expected)?;

    let LatentEncodeResult { latent } =
        bundle.encode_image(LatentEncodeInput { image }, backend.device().as_ref())?;

    let payload_key = BackendPayloadKey::new(format!(
        "latent:{}:{}",
        request.run_id().as_str(),
        request.node_id().as_str()
    ));

    backend.store().insert_latent(
        request.run_id().clone(),
        payload_key.clone(),
        latent.tensor().clone(),
        latent.latent_space().clone(),
    );

    let dims = latent.dims();
    if dims.len() != 4 {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.encode encoded latent payload `{}` expected 4D shape, got {:?}",
            payload_key.as_str(),
            dims
        )));
    }
    let latent_value = RuntimeLatent::new(
        BackendTensorHandle::with_instance(
            backend.backend_kind().clone(),
            backend.backend_instance(),
            payload_key,
            latent.latent_space().dtype(),
            TensorShape::new(dims),
            backend.device_label(),
        ),
        image_handle.width(),
        image_handle.height(),
        image_handle.batch(),
        latent.latent_space().channels(),
        latent.latent_space().clone(),
        reimagine_inference::LatentContent::EncodedImage,
    );

    Ok(LatentEncodeResponse::new(latent_value))
}

fn validate_image_handle_matches_store(
    handle: &RuntimeImage,
    stored: &crate::store::CandleImage,
) -> Result<(), CandleBackendError> {
    if handle.payload().dtype() != TensorDType::F32 {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.encode image handle `{}` dtype {:?} is unsupported; expected F32",
            handle.payload().payload_key().as_str(),
            handle.payload().dtype()
        )));
    }
    if stored.dtype() != DType::F32 {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.encode stored image payload `{}` dtype {:?} is unsupported; expected F32",
            handle.payload().payload_key().as_str(),
            stored.dtype()
        )));
    }

    let handle_shape = handle.payload().shape();
    if handle_shape.dims() != stored.dims().as_slice() {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.encode image handle shape {:?} disagrees with stored image payload shape {:?} for key `{}`",
            handle_shape.dims(),
            stored.dims(),
            handle.payload().payload_key().as_str()
        )));
    }
    if handle.width() != stored.width() {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.encode image handle width {} disagrees with stored image payload width {} for key `{}`",
            handle.width(),
            stored.width(),
            handle.payload().payload_key().as_str()
        )));
    }
    if handle.height() != stored.height() {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.encode image handle height {} disagrees with stored image payload height {} for key `{}`",
            handle.height(),
            stored.height(),
            handle.payload().payload_key().as_str()
        )));
    }
    if handle.batch() != stored.batch() {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.encode image handle batch {} disagrees with stored image payload batch {} for key `{}`",
            handle.batch(),
            stored.batch(),
            handle.payload().payload_key().as_str()
        )));
    }
    if handle.color_space() != stored.color_space() {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.encode image handle color space `{}` disagrees with stored image payload color space `{}` for key `{}`",
            handle.color_space(),
            stored.color_space(),
            handle.payload().payload_key().as_str()
        )));
    }
    Ok(())
}

fn validate_image_for_latent_encode(
    handle: &RuntimeImage,
    stored: &crate::store::CandleImage,
    latent_space: &LatentSpaceMetadata,
) -> Result<(), CandleBackendError> {
    let dims = stored.dims();
    if dims.len() != 4 {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.encode expected stored image payload `{}` to be 4D [batch, channels, height, width], got {}-D shape {:?}",
            handle.payload().payload_key().as_str(),
            dims.len(),
            dims
        )));
    }
    let batch = dims[0];
    let channels = dims[1];
    let height = dims[2];
    let width = dims[3];
    if stored.batch() as usize != batch
        || stored.height() as usize != height
        || stored.width() as usize != width
    {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.encode stored image payload `{}` metadata disagrees with tensor shape {:?}: metadata batch={}, width={}, height={}",
            handle.payload().payload_key().as_str(),
            dims,
            stored.batch(),
            stored.width(),
            stored.height()
        )));
    }
    if batch != 1 || stored.batch() != 1 || handle.batch() != 1 {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.encode V1 supports only batch=1; got handle batch={}, stored batch={}, tensor batch={} for image `{}`",
            handle.batch(),
            stored.batch(),
            batch,
            handle.payload().payload_key().as_str()
        )));
    }
    if channels != 3 {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.encode expected 3-channel RGB image payload `{}`, got channels={channels}",
            handle.payload().payload_key().as_str()
        )));
    }
    if stored.color_space() != "rgb" {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.encode expected RGB image payload `{}`, got color space `{}`",
            handle.payload().payload_key().as_str(),
            stored.color_space()
        )));
    }
    if width == 0 || height == 0 || stored.width() == 0 || stored.height() == 0 {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.encode expected positive image dimensions for payload `{}`, got width={}, height={}",
            handle.payload().payload_key().as_str(),
            stored.width(),
            stored.height()
        )));
    }
    validate_pixel_dimensions_against(stored.width(), stored.height(), latent_space)
        .map_err(|err| map_latent_space_error("latent.encode", err))?;
    Ok(())
}

fn require_bundle_for_vae(
    vae: &RuntimeVaeHandle,
    backend: &CandleBackend,
) -> Result<std::sync::Arc<crate::models::LoadedModelBundle>, CandleBackendError> {
    backend.model_cache().get_bundle(vae.model_id()).ok_or_else(|| {
        CandleBackendError::InvalidRequest(format!(
            "no loaded model bundle found for model `{}`; load the model first via `model.load_bundle`",
            vae.model_id().as_str()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_core::model::TensorDType;
    use reimagine_inference::{LatentSpaceId, TensorLayout};

    fn custom_latent_space(id: &str, channels: u32, scale: u32) -> LatentSpaceMetadata {
        LatentSpaceMetadata::new(
            LatentSpaceId::new(id),
            channels,
            scale,
            TensorDType::F32,
            TensorLayout::Nchw,
        )
    }

    fn build_request_with_latent_space(
        latent_space: LatentSpaceMetadata,
    ) -> CreateEmptyLatentRequest {
        CreateEmptyLatentRequest::new(
            64,
            64,
            1,
            reimagine_core::model::RunId::new("run-test"),
            reimagine_core::model::WorkflowId::new("wf-test"),
            reimagine_core::model::WorkflowVersion::new(1),
            reimagine_core::model::NodeId::new("node-test"),
        )
        .with_latent_space(latent_space)
    }

    #[test]
    fn create_empty_returns_sdxl_base_latent_metadata_by_default() {
        let backend = test_backend();
        let request = CreateEmptyLatentRequest::new(
            64,
            64,
            1,
            reimagine_core::model::RunId::new("run-test"),
            reimagine_core::model::WorkflowId::new("wf-test"),
            reimagine_core::model::WorkflowVersion::new(1),
            reimagine_core::model::NodeId::new("node-test"),
        );

        let response = execute_latent_create_empty(&backend, request).expect("create");
        let latent = response.into_latent();
        assert_eq!(latent.latent_space(), &LatentSpaceMetadata::sdxl_base());
        assert_eq!(latent.channels(), 4);
    }

    #[test]
    fn create_empty_propagates_custom_latent_space_metadata() {
        let backend = test_backend();
        let custom = custom_latent_space("custom/v1", 8, 4);
        let request = build_request_with_latent_space(custom.clone());

        // V1 Candle only supports the SDXL base latent space; a
        // well-formed but non-SDXL space is rejected with a
        // precise diagnostic before any tensor allocation.
        let err = execute_latent_create_empty(&backend, request).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("not supported"), "msg: {msg}");
        assert!(msg.contains("custom/v1"), "msg: {msg}");
        assert!(msg.contains("stable_diffusion/sdxl/base"), "msg: {msg}");
    }

    #[test]
    fn create_empty_rejects_pixel_dimensions_not_divisible_by_latent_scale() {
        let backend = test_backend();
        // 63 is not divisible by 8 (SDXL base scale)
        let request = CreateEmptyLatentRequest::new(
            63,
            64,
            1,
            reimagine_core::model::RunId::new("run-test"),
            reimagine_core::model::WorkflowId::new("wf-test"),
            reimagine_core::model::WorkflowVersion::new(1),
            reimagine_core::model::NodeId::new("node-test"),
        );

        let err = execute_latent_create_empty(&backend, request).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("divisible by latent-space"), "got {msg}");
        assert!(msg.contains("width=63"), "got {msg}");
    }

    #[test]
    fn create_empty_rejects_zero_batch_size() {
        let backend = test_backend();
        let request = CreateEmptyLatentRequest::new(
            64,
            64,
            0,
            reimagine_core::model::RunId::new("run-test"),
            reimagine_core::model::WorkflowId::new("wf-test"),
            reimagine_core::model::WorkflowVersion::new(1),
            reimagine_core::model::NodeId::new("node-test"),
        );

        let err = execute_latent_create_empty(&backend, request).unwrap_err();
        assert!(err.to_string().contains("batch_size"), "got {}", err);
    }

    fn test_backend() -> CandleBackend {
        // Unit tests for `create_empty` exercise the operation
        // boundary and the store, not the loaded-bundle path. The
        // existing public `CandleBackend::new` constructor with a
        // CPU config is sufficient.
        CandleBackend::new(crate::config::CandleBackendConfig::new(
            "/models", "/output",
        ))
        .expect("test backend")
    }
}
