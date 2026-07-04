//! `image.save` and `image.preview` operations for the Burn backend.
//!
//! Retrieves a `BurnImagePayload` from the shared store, converts the
//! float32 NCHW tensor to an interleaved RGB8 buffer, encodes it as
//! PNG, and writes to the filesystem (save) or returns base64-encoded
//! bytes (preview).

use reimagine_core::model::{ArtifactRef, RunId};
use reimagine_inference::{
    FilenamePrefix, ImagePreviewRequest, ImagePreviewResponse, ImageSaveRequest, ImageSaveResponse,
    InferenceBackend,
};

use crate::backend::BurnBackend;
use crate::error::BurnBackendError;

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

pub fn execute_image_save(
    request: ImageSaveRequest,
    backend: &BurnBackend,
) -> Result<ImageSaveResponse, BurnBackendError> {
    let prefix = match request.filename_prefix() {
        FilenamePrefix::Default => "reimagine".to_string(),
        FilenamePrefix::Custom(s) => s.clone(),
    };
    let run_id = request.run_id().clone();
    let node_id = request.node_id().clone();
    let artifact = persist_image(request.into_image(), &prefix, &run_id, &node_id, backend)?;
    Ok(ImageSaveResponse::new(artifact))
}

pub fn execute_image_preview(
    request: ImagePreviewRequest,
    backend: &BurnBackend,
) -> Result<ImagePreviewResponse, BurnBackendError> {
    let run_id = request.run_id().clone();
    let node_id = request.node_id().clone();
    let artifact = persist_image(request.into_image(), "preview", &run_id, &node_id, backend)?;
    Ok(ImagePreviewResponse::new(artifact))
}

// ---------------------------------------------------------------------------
// Persistence helpers
// ---------------------------------------------------------------------------

fn persist_image(
    image_value: reimagine_inference::RuntimeImage,
    prefix: &str,
    run_id: &RunId,
    node_id: &reimagine_core::model::NodeId,
    backend: &BurnBackend,
) -> Result<ArtifactRef, BurnBackendError> {
    // Validate backend affinity.
    if image_value.payload().backend() != backend.backend_kind() {
        return Err(BurnBackendError::InvalidRequest(format!(
            "image.save `image` handle belongs to backend `{}`, expected `{}`",
            image_value.payload().backend().as_str(),
            backend.backend_kind()
        )));
    }

    let payload_key = image_value.payload().payload_key();
    if !backend.store().contains_payload(payload_key) {
        return Err(BurnBackendError::InvalidRequest(format!(
            "image.save image payload `{}` not found in store",
            payload_key.as_str()
        )));
    }

    let burn_image = backend.store().get_image(payload_key)?;

    let filename = build_safe_filename(prefix, run_id.as_str(), node_id.as_str());
    let output_path = backend.config().output_dir().join(&filename);

    std::fs::create_dir_all(backend.config().output_dir()).map_err(|e| {
        BurnBackendError::InvalidRequest(format!(
            "image.save failed to create output directory: {}",
            e
        ))
    })?;

    validate_path_inside_output_dir(&output_path, backend.config().output_dir())?;

    let png_bytes = encode_image_to_png(&burn_image)?;
    std::fs::write(&output_path, &png_bytes).map_err(|e| {
        BurnBackendError::InvalidRequest(format!("image.save failed to write PNG file: {e}"))
    })?;

    Ok(artifact_ref_for_output_path(
        &output_path,
        backend.config().output_dir(),
    ))
}

fn artifact_ref_for_output_path(
    output_path: &std::path::Path,
    output_dir: &std::path::Path,
) -> ArtifactRef {
    let reference = output_path
        .strip_prefix(output_dir)
        .ok()
        .map(|relative| format!("output/{}", relative.to_string_lossy()))
        .unwrap_or_else(|| output_path.to_string_lossy().to_string());
    ArtifactRef::new(reference)
}

fn build_safe_filename(prefix: &str, run_id: &str, node_id: &str) -> String {
    let safe_prefix = sanitize_component(prefix, 64);
    let safe_run_id = sanitize_component(run_id, 128);
    let safe_node_id = sanitize_component(node_id, 128);
    format!(
        "{}_{}_{}_{}.png",
        safe_prefix,
        safe_run_id,
        safe_node_id,
        next_seq()
    )
}

fn next_seq() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    SEQ.fetch_add(1, Ordering::Relaxed)
}

fn sanitize_component(input: &str, max_len: usize) -> String {
    let result: String = input
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();

    if result.is_empty() {
        return "_unnamed".to_string();
    }

    if result.chars().count() > max_len {
        result.chars().take(max_len).collect()
    } else {
        result
    }
}

fn validate_path_inside_output_dir(
    path: &std::path::Path,
    output_dir: &std::path::Path,
) -> Result<(), BurnBackendError> {
    let output_dir_canon = std::fs::canonicalize(output_dir).map_err(|_| {
        BurnBackendError::InvalidRequest(format!(
            "image.save output directory path does not exist or is inaccessible: {}",
            output_dir.display()
        ))
    })?;

    let parent = path.parent().unwrap_or(path);
    let parent_canon = std::fs::canonicalize(parent).map_err(|_| {
        BurnBackendError::InvalidRequest(format!(
            "image.save target path parent directory is inaccessible: {}",
            parent.display()
        ))
    })?;

    if !parent_canon.starts_with(&output_dir_canon) {
        return Err(BurnBackendError::InvalidRequest(format!(
            "image.save target path escapes workspace output directory: {}",
            path.display()
        )));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tensor → PNG encoding
// ---------------------------------------------------------------------------

fn encode_image_to_png(
    image: &crate::store::BurnImagePayload,
) -> Result<Vec<u8>, BurnBackendError> {
    let dims = image.dims();
    let batch = dims[0];
    let channels = dims[1];
    let height = dims[2];
    let width = dims[3];

    if batch != 1 {
        return Err(BurnBackendError::InvalidRequest(format!(
            "image.save only supports batch=1 (got batch={batch})"
        )));
    }
    if channels != 3 {
        return Err(BurnBackendError::InvalidRequest(format!(
            "image.save only supports 3-channel RGB images (got channels={channels})"
        )));
    }

    // Extract raw f32 data from the BurnTensor.
    let data = image.tensor().to_data();
    let f32_data: Vec<f32> = data.to_vec::<f32>().map_err(|e| {
        BurnBackendError::InvalidRequest(format!("image.save failed to extract tensor data: {e}"))
    })?;

    let rgb = nchw_f32_to_interleaved_rgb8(&f32_data, width, height)?;
    encode_rgb8_to_png(&rgb, width, height)
}

fn nchw_f32_to_interleaved_rgb8(
    data: &[f32],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, BurnBackendError> {
    let plane_len = width.checked_mul(height).ok_or_else(|| {
        BurnBackendError::InvalidRequest(format!(
            "image.save image dimensions overflow: {width}x{height}"
        ))
    })?;
    let expected_len = plane_len.checked_mul(3).ok_or_else(|| {
        BurnBackendError::InvalidRequest(format!(
            "image.save RGB image byte length overflow: {width}x{height}"
        ))
    })?;
    if data.len() != expected_len {
        return Err(BurnBackendError::InvalidRequest(format!(
            "image.save tensor data length {} does not match RGB shape [3, {height}, {width}]",
            data.len()
        )));
    }

    let mut rgb = Vec::with_capacity(expected_len);
    for idx in 0..plane_len {
        for channel in 0..3 {
            let value = data[channel * plane_len + idx].clamp(0.0, 1.0);
            rgb.push((value * 255.0).round() as u8);
        }
    }
    Ok(rgb)
}

fn encode_rgb8_to_png(
    rgb: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, BurnBackendError> {
    let width_u32 = u32::try_from(width).map_err(|_| {
        BurnBackendError::InvalidRequest(format!("image.save width too large for PNG: {width}"))
    })?;
    let height_u32 = u32::try_from(height).map_err(|_| {
        BurnBackendError::InvalidRequest(format!("image.save height too large for PNG: {height}"))
    })?;

    let mut out = Vec::new();
    let encoder = ::image::codecs::png::PngEncoder::new(&mut out);
    ::image::ImageEncoder::write_image(
        encoder,
        rgb,
        width_u32,
        height_u32,
        ::image::ColorType::Rgb8.into(),
    )
    .map_err(|e| {
        BurnBackendError::InvalidRequest(format!("image.save failed to encode PNG: {e}"))
    })?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaves_nchw_rgb_planes() {
        let data = vec![
            1.0, 0.0, 0.0, 1.0, // R
            0.0, 1.0, 0.0, 1.0, // G
            0.0, 0.0, 1.0, 1.0, // B
        ];
        let rgb = nchw_f32_to_interleaved_rgb8(&data, 2, 2).unwrap();
        assert_eq!(
            rgb,
            vec![
                255, 0, 0, // pixel 0
                0, 255, 0, // pixel 1
                0, 0, 255, // pixel 2
                255, 255, 255, // pixel 3
            ]
        );
    }

    #[test]
    fn encodes_valid_png_with_expected_dimensions() {
        let rgb = vec![0u8; 32 * 16 * 3];
        let png = encode_rgb8_to_png(&rgb, 32, 16).unwrap();

        assert_eq!(
            &png[0..8],
            &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]
        );

        let decoded = ::image::load_from_memory_with_format(&png, ::image::ImageFormat::Png)
            .expect("encoded PNG should decode");
        assert_eq!(decoded.width(), 32);
        assert_eq!(decoded.height(), 16);
    }

    #[test]
    fn sanitize_allows_alphanumeric_dash_underscore() {
        assert_eq!(sanitize_component("abc-123_xyz", 64), "abc-123_xyz");
    }

    #[test]
    fn sanitize_replaces_invalid_chars() {
        assert_eq!(sanitize_component("a/b:c*d", 64), "a_b_c_d");
    }

    #[test]
    fn sanitize_empty_becomes_unnamed() {
        assert_eq!(sanitize_component("", 64), "_unnamed");
    }

    #[test]
    fn sanitize_truncates_long_input() {
        let long = "a".repeat(200);
        let result = sanitize_component(&long, 64);
        assert_eq!(result.chars().count(), 64);
        assert!(result.chars().all(|c| c == 'a'));
    }

    #[test]
    fn build_safe_filename_format() {
        let filename = build_safe_filename("reimagine", "run-test", "node-a");
        assert!(filename.contains("reimagine"));
        assert!(filename.contains("run-test"));
        assert!(filename.contains("node-a"));
        assert!(filename.ends_with(".png"));
    }

    #[test]
    fn build_safe_filename_sanitizes_path_traversal() {
        let filename = build_safe_filename("../../etc", "run", "node");
        assert!(!filename.contains(".."));
        assert!(filename.contains("_"));
    }

    #[test]
    fn nchw_f32_to_interleaved_rgb8_clamps_values() {
        let data = vec![
            -0.5, 1.5, 0.0, 0.0, // R
            0.0, 0.0, 0.0, 0.0, // G
            0.0, 0.0, 0.0, 0.0, // B
        ];
        let rgb = nchw_f32_to_interleaved_rgb8(&data, 2, 2).unwrap();
        // -0.5 → 0, 1.5 → 255
        assert_eq!(rgb[0], 0);
        assert_eq!(rgb[3], 255);
    }
}
