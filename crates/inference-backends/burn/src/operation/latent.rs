//! `latent.create_empty` for the Burn backend.
//!
//! ## What this module owns
//!
//! The narrow V1 capability required by `burn/09`: build a real
//! Burn-private latent payload (a zero burn-ndarray tensor) sized
//! for the requested pixel dimensions, store it in the shared
//! [`BurnStore`](crate::store::BurnStore), and return a
//! [`RuntimeLatent`](reimagine_inference::RuntimeLatent) with
//! `LatentContent::EmptyGeometry`.
//!
//! ## V1 rules (fixed by the issue)
//!
//! - Only [`LatentSpaceMetadata::sdxl_base()`] is supported; any
//!   other latent space fails with a precise diagnostic before
//!   any tensor work.
//! - Only `batch_size == 1` is supported; `0` and `>1` fail with
//!   precise diagnostics.
//! - The latent payload is a zero tensor. Real sampling lives in
//!   `diffusion.sample` (burn/10), not here.
//! - Allocation is the only state mutation; no model cache, no
//!   runtime hooks.

use burn_ndarray::NdArray;
use burn_tensor::Tensor;
use reimagine_core::model::{NodeId, RunId, TensorShape};
use reimagine_inference::latent_space::validate_pixel_dimensions_against;
use reimagine_inference::{
    BackendPayloadKey, BackendTensorHandle, CreateEmptyLatentRequest, CreateEmptyLatentResponse,
    InferenceBackend, InferenceError, LatentContent, LatentDecodeRequest, LatentDecodeResponse,
    LatentSpaceError, LatentSpaceMetadata, RuntimeImage, RuntimeLatent,
};

use crate::backend::BurnBackend;
use crate::error::BurnBackendError;
use crate::store::{BurnImagePayload, BurnLatentPayload};
use crate::tensor::BurnTensor;

fn map_latent_space_error(label: &'static str, err: LatentSpaceError) -> BurnBackendError {
    BurnBackendError::InvalidRequest(format!("{label} {err}"))
}

fn validate_batch(batch_size: u32) -> Result<(), BurnBackendError> {
    if batch_size == 0 {
        return Err(BurnBackendError::InvalidRequest(format!(
            "latent batch_size must be positive (got {batch_size})"
        )));
    }
    if batch_size != 1 {
        return Err(BurnBackendError::InvalidRequest(format!(
            "latent.create_empty V1 supports only batch_size=1; got {batch_size}. \
             Multi-image empty latents are not supported in this issue"
        )));
    }
    Ok(())
}

/// Build the deterministic per-request payload key for a latent
/// created by `latent.create_empty`.
///
/// The key must be stable per `(run_id, node_id)` so later Burn
/// operations can resolve the payload via [`BurnStore::get_latent`].
fn latent_payload_key(run_id: &RunId, node_id: &NodeId) -> BackendPayloadKey {
    BackendPayloadKey::new(format!("latent:{}:{}", run_id.as_str(), node_id.as_str()))
}

/// Bundle of validated request values used to allocate a latent
/// payload. Internal helper — keeping the values together stops
/// `allocate_zero_latent` from sprawling its parameter list.
struct LatentAllocationSpec {
    batch: u32,
    channels: u32,
    latent_height: u32,
    latent_width: u32,
    width: u32,
    height: u32,
    latent_space: LatentSpaceMetadata,
}

/// Allocate a real Burn-private latent payload (zero burn-ndarray
/// tensor) for the requested shape.
///
/// The tensor is allocated on the backend's concrete device (V1 is
/// always CPU via burn-ndarray). The function never panics; Burn
/// tensor construction is infallible for valid shapes.
fn allocate_zero_latent(backend: &BurnBackend, spec: &LatentAllocationSpec) -> BurnLatentPayload {
    let tensor = Tensor::<NdArray, 4>::zeros(
        [
            spec.batch as usize,
            spec.channels as usize,
            spec.latent_height as usize,
            spec.latent_width as usize,
        ],
        &backend.ndarray_device(),
    );
    BurnLatentPayload::new_burn(
        BurnTensor::Ndarray(tensor),
        spec.latent_space.clone(),
        spec.width,
        spec.height,
        spec.batch,
    )
}

/// `latent.create_empty` entry point for the Burn backend.
///
/// V1 only supports [`LatentSpaceMetadata::sdxl_base()`] and
/// `batch_size == 1`. Anything else fails with a precise
/// diagnostic before any tensor allocation. On success the
/// returned handle carries a real Burn-private latent payload
/// stored in the shared [`BurnStore`](crate::store::BurnStore)
/// under a deterministic per-(run, node) key.
pub fn execute_latent_create_empty(
    backend: &BurnBackend,
    request: CreateEmptyLatentRequest,
) -> Result<CreateEmptyLatentResponse, BurnBackendError> {
    let width = request.width();
    let height = request.height();
    let batch_size = request.batch_size();
    let latent_space = request.latent_space().clone();

    // V1 Burn only supports the SDXL base latent space. Any
    // other latent space — even one that is otherwise well-formed
    // — must be rejected here with a precise error so the
    // downstream operations do not silently materialize a tensor
    // that no V1 model can consume.
    let supported = LatentSpaceMetadata::sdxl_base();
    if !latent_space.is_compatible(&supported) {
        return Err(BurnBackendError::InvalidRequest(format!(
            "latent.create_empty latent space `{}` (channels={}, scale={}) is not supported by backend `burn`; only `{}` is supported in V1",
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
    let shape = vec![
        batch_size as usize,
        channels as usize,
        latent_height as usize,
        latent_width as usize,
    ];

    let spec = LatentAllocationSpec {
        batch: batch_size,
        channels,
        latent_height,
        latent_width,
        width,
        height,
        latent_space: latent_space.clone(),
    };
    let payload = allocate_zero_latent(backend, &spec);

    let payload_key = latent_payload_key(request.run_id(), request.node_id());

    backend
        .store()
        .insert_latent(request.run_id().clone(), payload_key.clone(), payload);

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
        LatentContent::EmptyGeometry,
    );

    Ok(CreateEmptyLatentResponse::new(latent))
}

/// Map a [`BurnBackendError`] into the inference-layer error type.
pub fn map_to_inference_error(err: BurnBackendError) -> InferenceError {
    InferenceError::BackendExecutionFailed {
        message: err.to_string(),
    }
}

/// `latent.decode` entry point for the Burn backend.
///
/// V1 accepts real latent contents (`Sampled`, `EncodedImage`, `Imported`)
/// and rejects `EmptyGeometry`. Runs Burn VAE decode and stores the
/// resulting image payload in the shared BurnStore.
pub fn execute_latent_decode(
    backend: &BurnBackend,
    request: LatentDecodeRequest,
) -> Result<LatentDecodeResponse, BurnBackendError> {
    let latent_handle = request.latent();

    // Validate backend affinity
    let handle_backend = latent_handle.payload().backend();
    if handle_backend.as_str() != crate::profile::BACKEND_LABEL {
        return Err(BurnBackendError::InvalidRequest(format!(
            "latent.decode handle belongs to backend `{}`, expected `{}`",
            handle_backend.as_str(),
            crate::profile::BACKEND_LABEL
        )));
    }
    if latent_handle.payload().backend_instance() != &backend.backend_instance() {
        return Err(BurnBackendError::InvalidRequest(format!(
            "latent.decode handle belongs to backend instance `{}`, expected `{}`",
            latent_handle.payload().backend_instance(),
            backend.backend_instance()
        )));
    }

    // Reject EmptyGeometry
    if latent_handle.content() == LatentContent::EmptyGeometry {
        return Err(BurnBackendError::InvalidRequest(
            "latent.decode cannot decode EmptyGeometry; use diffusion.sample first".to_owned(),
        ));
    }

    // Validate latent space is SDXL base
    if !latent_handle
        .latent_space()
        .is_compatible(&LatentSpaceMetadata::sdxl_base())
    {
        return Err(BurnBackendError::InvalidRequest(format!(
            "latent.decode latent space `{}` is not compatible with SDXL base",
            latent_handle.latent_space().id()
        )));
    }

    // batch=1 only
    if latent_handle.batch() != 1 {
        return Err(BurnBackendError::InvalidRequest(format!(
            "latent.decode V1 only supports batch=1, got {}",
            latent_handle.batch()
        )));
    }

    // Get latent payload from store
    let latent_payload = backend
        .store()
        .get_latent(latent_handle.payload().payload_key())?;

    // Run VAE decode
    let vae = request.vae();
    let bundle = backend
        .model_cache()
        .get_bundle(vae.model_id())
        .ok_or_else(|| {
            BurnBackendError::InvalidRequest(
                "latent.decode requires loaded bundle; call load_bundle first".to_string(),
            )
        })?;

    let decoded = crate::models::stable_diffusion::sdxl::vae::decode_latent(
        &bundle,
        latent_payload,
        backend,
    )?;

    // Store image payload
    let dims: [usize; 4] = decoded.dims();
    let height = dims[2] as u32;
    let width = dims[3] as u32;
    let image_payload = BurnImagePayload::new(decoded, width, height, 1, "rgb");

    let image_key = reimagine_inference::BackendPayloadKey::new(format!(
        "image:{}:{}",
        request.run_id().as_str(),
        request.node_id().as_str()
    ));
    backend
        .store()
        .insert_image(request.run_id().clone(), image_key.clone(), image_payload);

    // Build RuntimeImage response
    let runtime_image = RuntimeImage::new(
        BackendTensorHandle::with_instance(
            backend.backend_kind().clone(),
            backend.backend_instance(),
            image_key,
            reimagine_core::model::TensorDType::F32,
            TensorShape::new(vec![1, 3, height as usize, width as usize]),
            backend.device_label(),
        ),
        width,
        height,
        1,
        "rgb",
    );

    Ok(LatentDecodeResponse::new(runtime_image))
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};
    use reimagine_inference::{LatentSpaceId, TensorLayout};

    fn test_backend() -> BurnBackend {
        // Unit tests exercise the operation boundary and the
        // shared store; a CPU burn backend is sufficient.
        BurnBackend::new(crate::config::BurnBackendConfig::new("/models", "/output"))
            .expect("test backend")
    }

    fn build_request(batch_size: u32) -> CreateEmptyLatentRequest {
        CreateEmptyLatentRequest::new(
            64,
            64,
            batch_size,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-test"),
        )
    }

    fn custom_latent_space(id: &str, channels: u32, scale: u32) -> LatentSpaceMetadata {
        LatentSpaceMetadata::new(
            LatentSpaceId::new(id),
            channels,
            scale,
            reimagine_core::model::TensorDType::F32,
            TensorLayout::Nchw,
        )
    }

    #[test]
    fn create_empty_returns_sdxl_base_latent_metadata_by_default() {
        let backend = test_backend();
        let response = execute_latent_create_empty(&backend, build_request(1)).expect("create");
        let latent = response.into_latent();
        assert_eq!(latent.latent_space(), &LatentSpaceMetadata::sdxl_base());
        assert_eq!(latent.channels(), 4);
        assert_eq!(latent.batch(), 1);
        assert_eq!(latent.width(), 64);
        assert_eq!(latent.height(), 64);
        // scale=8 → 64/8 = 8 → shape [1, 4, 8, 8]
        assert_eq!(
            latent.payload().shape().dims(),
            vec![1_usize, 4, 8, 8].as_slice()
        );
    }

    #[test]
    fn create_empty_returns_empty_geometry_content() {
        let backend = test_backend();
        let response = execute_latent_create_empty(&backend, build_request(1)).expect("create");
        let latent = response.into_latent();
        assert_eq!(latent.content(), LatentContent::EmptyGeometry);
    }

    #[test]
    fn create_empty_payload_carries_burn_backend_and_instance() {
        let backend = test_backend();
        let response = execute_latent_create_empty(&backend, build_request(1)).expect("create");
        let latent = response.into_latent();
        assert_eq!(latent.payload().backend().as_str(), "burn");
        // burn/13: under the `flex` feature the default backend
        // instance is `burn:flex:cpu`; under wgpu (or neither)
        // it's `burn:cpu`.
        let expected_instance = if cfg!(all(not(feature = "wgpu"), feature = "flex")) {
            "burn:flex:cpu"
        } else {
            "burn:cpu"
        };
        assert_eq!(
            latent.payload().backend_instance().as_str(),
            expected_instance
        );
    }

    #[test]
    fn create_empty_rejects_unsupported_latent_space() {
        let backend = test_backend();
        let request = build_request(1).with_latent_space(custom_latent_space("custom/v1", 8, 4));

        let err = execute_latent_create_empty(&backend, request).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("not supported"), "msg: {msg}");
        assert!(msg.contains("custom/v1"), "msg: {msg}");
        assert!(msg.contains("stable_diffusion/sdxl/base"), "msg: {msg}");
    }

    #[test]
    fn create_empty_rejects_zero_batch_size() {
        let backend = test_backend();
        let err = execute_latent_create_empty(&backend, build_request(0)).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("batch_size"), "msg: {msg}");
        assert!(msg.contains("0"), "msg: {msg}");
    }

    #[test]
    fn create_empty_rejects_multi_batch_size() {
        let backend = test_backend();
        let err = execute_latent_create_empty(&backend, build_request(2)).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("batch_size=1"), "msg: {msg}");
        assert!(msg.contains("got 2"), "msg: {msg}");
    }

    #[test]
    fn create_empty_rejects_pixel_dimensions_not_divisible_by_latent_scale() {
        let backend = test_backend();
        // 63 is not divisible by 8 (SDXL base scale).
        let request = CreateEmptyLatentRequest::new(
            63,
            64,
            1,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-test"),
        );

        let err = execute_latent_create_empty(&backend, request).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("divisible by latent-space"), "msg: {msg}");
        assert!(msg.contains("width=63"), "msg: {msg}");
    }

    #[test]
    fn create_empty_stores_payload_in_shared_store() {
        let backend = test_backend();
        let request = build_request(1);

        let response = execute_latent_create_empty(&backend, request.clone()).expect("create");
        let latent = response.into_latent();

        // The shared Burn store must hold the latent under the
        // deterministic (run, node) key, and the stored payload
        // shape must match the handle's metadata.
        let stored = backend
            .store()
            .get_latent(latent.payload().payload_key())
            .expect("stored latent");
        assert_eq!(stored.dims(), [1, 4, 8, 8]);
        assert_eq!(stored.width(), latent.width());
        assert_eq!(stored.height(), latent.height());
        assert_eq!(stored.batch(), latent.batch());
        assert_eq!(stored.latent_space(), latent.latent_space());
    }

    #[test]
    fn create_empty_increments_store_payload_count_per_unique_run_node() {
        let backend = test_backend();
        assert_eq!(backend.store().payload_count(), 0);

        execute_latent_create_empty(&backend, build_request(1)).expect("first");
        assert_eq!(backend.store().payload_count(), 1);

        // A different run_id, even with the same node_id, lands a
        // new payload — the (run_id, node_id) tuple is the
        // deterministic payload key.
        let second = CreateEmptyLatentRequest::new(
            64,
            64,
            1,
            RunId::new("run-other"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-test"),
        );
        execute_latent_create_empty(&backend, second).expect("second");
        assert_eq!(backend.store().payload_count(), 2);
    }

    #[test]
    fn create_empty_with_same_run_node_overwrites_existing_payload() {
        let backend = test_backend();
        execute_latent_create_empty(&backend, build_request(1)).expect("first");
        assert_eq!(backend.store().payload_count(), 1);
        // Same (run_id, node_id) → same deterministic key. The
        // store keeps one payload per key rather than fan out
        // duplicates; downstream operations resolve a single
        // payload per key.
        execute_latent_create_empty(&backend, build_request(1)).expect("second");
        assert_eq!(backend.store().payload_count(), 1);
    }

    #[test]
    fn create_empty_payload_key_is_deterministic_per_run_and_node() {
        let backend = test_backend();
        let response = execute_latent_create_empty(&backend, build_request(1)).expect("create");
        let latent = response.into_latent();
        assert_eq!(
            latent.payload().payload_key().as_str(),
            "latent:run-test:node-test"
        );
    }
}
