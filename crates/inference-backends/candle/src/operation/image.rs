//! `image.save` and `image.preview` operations.
//!
//! Both operations consume a `RuntimeImage`, encode the image tensor
//! to PNG format, write it to the workspace output directory, and
//! return a typed response carrying an [`ArtifactRef`] so the
//! inference executor can record the artifact via
//! `NodeArtifactCapability`.
//!
//! ## Sanitization policy
//!
//! Filename components (`prefix`, `run_id`, `node_id`) are sanitized:
//! - Allowed: ASCII alphanumeric (`a-zA-Z0-9`), `-`, `_`
//! - Everything else: replaced with `_`
//! - Empty after sanitization: replaced with `_unnamed`
//! - Result is bounded to avoid OS path limits (prefix: 64 chars,
//!   run_id/node_id: 128 chars)

use candle_core::{Device, Tensor};
use reimagine_core::model::{ArtifactRef, TensorDType, TensorShape};
use reimagine_inference::{
    BackendPayloadKey, FilenamePrefix, ImageImportRequest, ImageImportResponse, ImagePreviewRequest,
    ImagePreviewResponse, ImageSaveRequest, ImageSaveResponse, InferenceBackend, RuntimeImage,
};

use crate::backend::CandleBackend;
use crate::error::CandleBackendError;
use crate::store::CandleImage;

fn is_cpu_device(device: &Device) -> bool {
    matches!(device, Device::Cpu)
}

pub fn execute_image_save(
    request: ImageSaveRequest,
    backend: &CandleBackend,
) -> Result<ImageSaveResponse, CandleBackendError> {
    let prefix = match request.filename_prefix() {
        FilenamePrefix::Default => "reimagine".to_string(),
        FilenamePrefix::Custom(s) => s.clone(),
    };
    let run_id = request.run_id().clone();
    let node_id = request.node_id().clone();
    let response_artifact =
        persist_image(request.into_image(), &prefix, &run_id, &node_id, backend)?;
    Ok(ImageSaveResponse::new(response_artifact))
}

pub fn execute_image_preview(
    request: ImagePreviewRequest,
    backend: &CandleBackend,
) -> Result<ImagePreviewResponse, CandleBackendError> {
    let run_id = request.run_id().clone();
    let node_id = request.node_id().clone();
    let response_artifact =
        persist_image(request.into_image(), "preview", &run_id, &node_id, backend)?;
    Ok(ImagePreviewResponse::new(response_artifact))
}

const SUPPORTED_MEDIA_TYPES: &[&str] = &["image/png", "image/jpeg", "image/webp"];

pub fn execute_image_import(
    backend: &CandleBackend,
    request: ImageImportRequest,
) -> Result<ImageImportResponse, CandleBackendError> {
    let source = request.source();
    let media_type = source.media_type();

    if !SUPPORTED_MEDIA_TYPES.contains(&media_type) {
        return Err(CandleBackendError::InvalidRequest(format!(
            "image.import unsupported media type `{media_type}`; supported: png, jpeg, webp"
        )));
    }

    let path = source.path();
    let bytes = std::fs::read(path).map_err(|e| {
        CandleBackendError::InvalidRequest(format!(
            "image.import failed to read source file `{}`: {e}",
            path.display()
        ))
    })?;

    let dynamic = image::load_from_memory(&bytes).map_err(|e| {
        CandleBackendError::InvalidRequest(format!(
            "image.import failed to decode image from `{}`: {e}",
            path.display()
        ))
    })?;

    let width = dynamic.width();
    let height = dynamic.height();

    if width == 0 || height == 0 {
        return Err(CandleBackendError::InvalidRequest(format!(
            "image.import rejected zero-dimension image: {width}x{height}"
        )));
    }

    let rgb8 = dynamic.to_rgb8();
    let raw = rgb8.as_raw();

    let plane = (width * height) as usize;
    let mut nchw = vec![0.0f32; plane * 3];
    for (i, pixel) in raw.chunks_exact(3).enumerate() {
        nchw[i] = pixel[0] as f32 / 255.0;
        nchw[plane + i] = pixel[1] as f32 / 255.0;
        nchw[2 * plane + i] = pixel[2] as f32 / 255.0;
    }

    let tensor = Tensor::from_vec(nchw, (1, 3, height as usize, width as usize), backend.device())
        .map_err(|e| {
            CandleBackendError::InvalidRequest(format!(
                "image.import failed to create tensor: {e}"
            ))
        })?;

    let payload_key = BackendPayloadKey::new(format!(
        "image:{}:{}",
        request.run_id().as_str(),
        request.node_id().as_str()
    ));

    let candle_image = CandleImage::new(
        tensor,
        width,
        height,
        1,
        "rgb".to_string(),
    );

    backend
        .store()
        .insert_image(request.run_id().clone(), payload_key.clone(), candle_image);

    let device_label = backend.device_label().to_string();
    let tensor_handle = BackendPayloadKey::new(payload_key.as_str());
    let runtime_image = RuntimeImage::new(
        reimagine_inference::BackendTensorHandle::with_instance(
            backend.backend_kind().clone(),
            backend.backend_instance(),
            tensor_handle,
            TensorDType::F32,
            TensorShape::new(vec![1, 3, height as usize, width as usize]),
            device_label,
        ),
        width,
        height,
        1,
        "rgb",
    );

    Ok(ImageImportResponse::new(runtime_image))
}

fn persist_image(
    image_value: RuntimeImage,
    prefix: &str,
    run_id: &reimagine_core::model::RunId,
    node_id: &reimagine_core::model::NodeId,
    backend: &CandleBackend,
) -> Result<ArtifactRef, CandleBackendError> {
    if image_value.payload().backend() != backend.backend_kind() {
        return Err(CandleBackendError::InvalidRequest(format!(
            "image.save `image` handle belongs to backend `{}`, expected `{}`",
            image_value.payload().backend().as_str(),
            backend.backend_kind()
        )));
    }

    let payload_key = image_value.payload().payload_key();
    if !backend.store().contains_payload(payload_key) {
        return Err(CandleBackendError::InvalidRequest(format!(
            "image.save image payload `{}` not found in store",
            payload_key.as_str()
        )));
    }

    // Fan-out friendly read: a single decoded image must remain
    // available for downstream artifact nodes (e.g. image.save and
    // image.preview sharing the same source). The store retains the
    // payload; run-scoped cleanup drops it when the run finishes.
    let candle_image = backend.store().get_image(payload_key)?;

    let filename = build_safe_filename(
        prefix,
        run_id.as_str(),
        node_id.as_str(),
        backend.next_image_seq(),
    );

    let output_path = backend.output_dir().join(&filename);

    std::fs::create_dir_all(backend.output_dir()).map_err(|e| {
        CandleBackendError::InvalidRequest(format!(
            "image.save failed to create output directory: {e}"
        ))
    })?;

    validate_path_inside_output_dir(&output_path, backend.output_dir())?;

    let png_bytes = encode_tensor_to_png(&candle_image, backend.device().as_ref())?;

    std::fs::write(&output_path, &png_bytes).map_err(|e| {
        CandleBackendError::InvalidRequest(format!("image.save failed to write PNG file: {e}"))
    })?;

    let reference = output_path
        .strip_prefix(backend.output_dir())
        .ok()
        .map(|relative| format!("output/{}", relative.to_string_lossy()))
        .unwrap_or_else(|| output_path.to_string_lossy().to_string());
    Ok(ArtifactRef::new(reference))
}

fn build_safe_filename(prefix: &str, run_id: &str, node_id: &str, seq: u64) -> String {
    let safe_prefix = sanitize_component(prefix, 64);
    let safe_run_id = sanitize_component(run_id, 128);
    let safe_node_id = sanitize_component(node_id, 128);
    format!(
        "{}_{}_{}_{}.png",
        safe_prefix, safe_run_id, safe_node_id, seq
    )
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
) -> Result<(), CandleBackendError> {
    let output_dir_canon = std::fs::canonicalize(output_dir).map_err(|_| {
        CandleBackendError::InvalidRequest(format!(
            "image.save output directory path does not exist or is inaccessible: {}",
            output_dir.display()
        ))
    })?;

    let parent = path.parent().unwrap_or(path);

    let parent_canon = std::fs::canonicalize(parent).map_err(|_| {
        CandleBackendError::InvalidRequest(format!(
            "image.save target path parent directory is inaccessible: {}",
            parent.display()
        ))
    })?;

    if !parent_canon.starts_with(&output_dir_canon) {
        return Err(CandleBackendError::InvalidRequest(format!(
            "image.save target path escapes workspace output directory: {}",
            path.display()
        )));
    }

    Ok(())
}

fn encode_tensor_to_png(
    image: &crate::store::CandleImage,
    _device: &Device,
) -> Result<Vec<u8>, CandleBackendError> {
    let tensor = image.tensor();

    let tensor_on_cpu = if is_cpu_device(tensor.device()) {
        tensor.clone()
    } else {
        tensor.to_device(&Device::Cpu).map_err(|e| {
            CandleBackendError::InvalidRequest(format!(
                "image.save failed to move tensor to CPU: {e}"
            ))
        })?
    };

    let dims = tensor_on_cpu.shape().dims();
    if dims.len() != 4 {
        return Err(CandleBackendError::InvalidRequest(format!(
            "image.save expected 4D tensor [batch, channels, height, width], got {}-D shape {:?}",
            dims.len(),
            dims
        )));
    }

    let batch = dims[0];
    let channels = dims[1];
    let height = dims[2];
    let width = dims[3];

    if batch != 1 {
        return Err(CandleBackendError::InvalidRequest(format!(
            "image.save only supports batch=1 (got batch={})",
            batch
        )));
    }

    if channels != 3 {
        return Err(CandleBackendError::InvalidRequest(format!(
            "image.save only supports 3-channel RGB images (got channels={})",
            channels
        )));
    }

    let float_vec = tensor_on_cpu
        .flatten_all()
        .map_err(|e| {
            CandleBackendError::InvalidRequest(format!("image.save failed to flatten tensor: {e}"))
        })?
        .to_vec1::<f32>()
        .map_err(|e| {
            CandleBackendError::InvalidRequest(format!(
                "image.save failed to convert tensor data: {e}"
            ))
        })?;

    let rgb = nchw_f32_to_interleaved_rgb8(&float_vec, width, height);

    Ok(encode_rgb8_to_png(&rgb, width, height))
}

fn nchw_f32_to_interleaved_rgb8(data: &[f32], width: usize, height: usize) -> Vec<u8> {
    let plane_len = width * height;
    let mut rgb = Vec::with_capacity(plane_len * 3);
    for idx in 0..plane_len {
        for channel in 0..3 {
            let value = data[channel * plane_len + idx].clamp(0.0, 1.0);
            rgb.push((value * 255.0).round() as u8);
        }
    }
    rgb
}

fn encode_rgb8_to_png(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(width * height * 3 + 1024);

    out.extend_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);

    let mut ihdr_data = Vec::with_capacity(13);
    ihdr_data.extend_from_slice(&(width as u32).to_be_bytes());
    ihdr_data.extend_from_slice(&(height as u32).to_be_bytes());
    ihdr_data.push(8);
    ihdr_data.push(2);
    ihdr_data.push(0);
    ihdr_data.push(0);
    ihdr_data.push(0);
    write_chunk(&mut out, b"IHDR", &ihdr_data);

    let filtered = add_filter_bytes(rgb, width, height);
    let compressed = encode_zlib_stored(&filtered);
    write_chunk(&mut out, b"IDAT", &compressed);

    write_chunk(&mut out, b"IEND", &[]);

    out
}

fn write_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    let len = data.len() as u32;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);

    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(chunk_type);
    crc_input.extend_from_slice(data);
    let crc = crc32(&crc_input);
    out.extend_from_slice(&crc.to_be_bytes());
}

fn crc32(data: &[u8]) -> u32 {
    let table = make_crc32_table();
    let mut crc = 0xFFFFFFFF_u32;
    for &byte in data {
        crc = table[((crc ^ byte as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFFFFFF_u32
}

fn make_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    for i in 0..256 {
        let mut c = i as u32;
        for _ in 0..8 {
            if c & 1 != 0 {
                c = 0xEDB88320_u32 ^ (c >> 1);
            } else {
                c = c >> 1;
            }
        }
        table[i] = c;
    }
    table
}

fn add_filter_bytes(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut filtered = Vec::with_capacity(width * height * 3 + height);
    for y in 0..height {
        filtered.push(0);
        let row_start = y * width * 3;
        let row_end = row_start + width * 3;
        filtered.extend_from_slice(&rgb[row_start..row_end]);
    }
    filtered
}

fn encode_zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 64);

    out.push(0x78);
    out.push(0x01);

    let mut pos = 0;
    while pos < data.len() {
        let remaining = data.len() - pos;
        let block_len = remaining.min(65535);
        let is_final = pos + block_len >= data.len();

        let header = if is_final { 0x01 } else { 0x00 };
        out.push(header);

        let len = block_len as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!(len)).to_le_bytes());

        out.extend_from_slice(&data[pos..pos + block_len]);
        pos += block_len;
    }

    let adler = adler32(data);
    out.extend_from_slice(&adler.to_be_bytes());

    out
}

fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let filename = build_safe_filename("reimagine", "run-test", "node-a", 5);
        assert_eq!(filename, "reimagine_run-test_node-a_5.png");
    }

    #[test]
    fn build_safe_filename_sanitizes_path_traversal() {
        let filename = build_safe_filename("../../etc", "run", "node", 0);
        assert!(!filename.contains(".."));
        assert!(filename.contains("_"));
    }

    #[test]
    fn png_signature_bytes() {
        let rgb = vec![0u8; 64 * 64 * 3];
        let png = encode_rgb8_to_png(&rgb, 64, 64);
        assert_eq!(
            &png[0..8],
            &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        );
    }

    #[test]
    fn interleaves_nchw_rgb_planes() {
        let data = vec![
            1.0, 0.0, 0.0, 1.0, // R
            0.0, 1.0, 0.0, 1.0, // G
            0.0, 0.0, 1.0, 1.0, // B
        ];
        let rgb = nchw_f32_to_interleaved_rgb8(&data, 2, 2);
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
    fn png_zlib_stored_blocks_use_little_endian_lengths() {
        let filtered = add_filter_bytes(&[255, 0, 0], 1, 1);
        let zlib = encode_zlib_stored(&filtered);

        assert_eq!(&zlib[0..2], &[0x78, 0x01]);
        assert_eq!(zlib[2], 0x01, "single block should be final");

        let len = u16::from_le_bytes([zlib[3], zlib[4]]);
        let nlen = u16::from_le_bytes([zlib[5], zlib[6]]);
        assert_eq!(len as usize, filtered.len());
        assert_eq!(nlen, !len);
        assert_eq!(&zlib[7..7 + filtered.len()], filtered.as_slice());
    }

    #[test]
    fn png_ihdr_contains_correct_dimensions() {
        let rgb = vec![0u8; 32 * 16 * 3];
        let png = encode_rgb8_to_png(&rgb, 32, 16);

        let sig_len = 8;
        let ihdr_offset = sig_len + 4 + 4;
        let width = u32::from_be_bytes([
            png[ihdr_offset],
            png[ihdr_offset + 1],
            png[ihdr_offset + 2],
            png[ihdr_offset + 3],
        ]);
        let height = u32::from_be_bytes([
            png[ihdr_offset + 4],
            png[ihdr_offset + 5],
            png[ihdr_offset + 6],
            png[ihdr_offset + 7],
        ]);

        assert_eq!(width, 32);
        assert_eq!(height, 16);
        assert_eq!(png[ihdr_offset + 10], 0);
        assert_eq!(png[ihdr_offset + 11], 0);
        assert_eq!(png[ihdr_offset + 12], 0);
    }

    #[test]
    fn png_ihdr_has_thirteen_byte_payload() {
        let rgb = vec![0u8; 4 * 4 * 3];
        let png = encode_rgb8_to_png(&rgb, 4, 4);

        let sig_len = 8;
        let len_offset = sig_len;
        let ihdr_len = u32::from_be_bytes([
            png[len_offset],
            png[len_offset + 1],
            png[len_offset + 2],
            png[len_offset + 3],
        ]);
        assert_eq!(ihdr_len, 13, "IHDR chunk length field must be 13");

        let ihdr_data_len = ihdr_len as usize;
        assert_eq!(
            ihdr_data_len, 13,
            "IHDR data section must be exactly 13 bytes"
        );
    }

    #[test]
    fn png_ihdr_ends_with_compression_filter_interlace_zero() {
        let rgb = vec![0u8; 4 * 4 * 3];
        let png = encode_rgb8_to_png(&rgb, 4, 4);

        let sig_len = 8;
        let ihdr_data_offset = sig_len + 4 + 4;
        assert_eq!(
            png[ihdr_data_offset + 10],
            0,
            "compression method must be 0"
        );
        assert_eq!(png[ihdr_data_offset + 11], 0, "filter method must be 0");
        assert_eq!(png[ihdr_data_offset + 12], 0, "interlace method must be 0");
    }

    #[test]
    fn adler32_is_deterministic() {
        let data = b"hello world";
        let a = adler32(data);
        let b = adler32(data);
        assert_eq!(a, b);
    }

    #[test]
    fn adler32_differs_for_different_data() {
        let a = adler32(b"hello");
        let b = adler32(b"world");
        assert_ne!(a, b);
    }

    #[test]
    fn image_import_png_produces_correct_runtime_image() {
        use crate::config::CandleBackendConfig;
        use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};

        let tmp = std::env::temp_dir().join(format!(
            "reimagine-img-import-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();

        let img = image::RgbImage::from_fn(8, 6, |x, y| {
            image::Rgb([(x * 32) as u8, (y * 40) as u8, 200])
        });
        let path = tmp.join("test.png");
        img.save(&path).unwrap();

        let backend = CandleBackend::new(CandleBackendConfig::new(
            tmp.join("models"),
            tmp.join("output"),
        ))
        .unwrap();

        let source = reimagine_inference::ResolvedImageSource::new(
            &path,
            "image/png",
            Some("test.png".to_string()),
        );
        let request = ImageImportRequest::new(
            source,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-import"),
        );

        let response = execute_image_import(&backend, request).unwrap();
        let image = response.image();
        assert_eq!(image.width(), 8);
        assert_eq!(image.height(), 6);
        assert_eq!(image.batch(), 1);
        assert_eq!(image.color_space(), "rgb");
        assert_eq!(image.payload().dtype(), TensorDType::F32);
        assert_eq!(image.payload().shape().dims(), &[1, 3, 6, 8]);

        let stored = backend
            .store()
            .get_image(image.payload().payload_key())
            .expect("image should be in store");
        assert_eq!(stored.width(), 8);
        assert_eq!(stored.height(), 6);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn image_import_missing_file_returns_error() {
        use crate::config::CandleBackendConfig;
        use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};

        let backend = CandleBackend::new(CandleBackendConfig::new(
            "/tmp/reimagine-img-import-missing",
            "/tmp/reimagine-img-import-missing-output",
        ))
        .unwrap();

        let source = reimagine_inference::ResolvedImageSource::new(
            "/tmp/reimagine-img-import-missing/no-such-file.png",
            "image/png",
            None,
        );
        let request = ImageImportRequest::new(
            source,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-import"),
        );

        let err = execute_image_import(&backend, request).unwrap_err();
        let msg = match err {
            CandleBackendError::InvalidRequest(msg) => msg,
            other => panic!("expected InvalidRequest, got {other:?}"),
        };
        assert!(msg.contains("failed to read source file"), "msg: {msg}");
    }

    #[test]
    fn image_import_unsupported_media_type_returns_error() {
        use crate::config::CandleBackendConfig;
        use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};

        let tmp = std::env::temp_dir().join(format!(
            "reimagine-img-import-media-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("test.bmp"), b"not a real bmp").unwrap();

        let backend = CandleBackend::new(CandleBackendConfig::new(
            tmp.join("models"),
            tmp.join("output"),
        ))
        .unwrap();

        let source = reimagine_inference::ResolvedImageSource::new(
            tmp.join("test.bmp"),
            "image/bmp",
            None,
        );
        let request = ImageImportRequest::new(
            source,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-import"),
        );

        let err = execute_image_import(&backend, request).unwrap_err();
        let msg = match err {
            CandleBackendError::InvalidRequest(msg) => msg,
            other => panic!("expected InvalidRequest, got {other:?}"),
        };
        assert!(msg.contains("unsupported media type"), "msg: {msg}");
        assert!(msg.contains("image/bmp"), "msg: {msg}");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn image_import_corrupt_image_returns_decoder_error() {
        use crate::config::CandleBackendConfig;
        use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};

        let tmp = std::env::temp_dir().join(format!(
            "reimagine-img-import-corrupt-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("corrupt.png"), b"this is not a valid PNG file").unwrap();

        let backend = CandleBackend::new(CandleBackendConfig::new(
            tmp.join("models"),
            tmp.join("output"),
        ))
        .unwrap();

        let source = reimagine_inference::ResolvedImageSource::new(
            tmp.join("corrupt.png"),
            "image/png",
            None,
        );
        let request = ImageImportRequest::new(
            source,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-import"),
        );

        let err = execute_image_import(&backend, request).unwrap_err();
        let msg = match err {
            CandleBackendError::InvalidRequest(msg) => msg,
            other => panic!("expected InvalidRequest, got {other:?}"),
        };
        assert!(msg.contains("failed to decode image"), "msg: {msg}");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn image_import_normalizes_to_unit_range() {
        use crate::config::CandleBackendConfig;
        use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};

        let tmp = std::env::temp_dir().join(format!(
            "reimagine-img-import-norm-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();

        // Create a 2x1 image: pixel 0 = pure white (255,255,255), pixel 1 = pure black (0,0,0)
        let img = image::RgbImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                image::Rgb([255u8, 255, 255])
            } else {
                image::Rgb([0u8, 0, 0])
            }
        });
        let path = tmp.join("white_black.png");
        img.save(&path).unwrap();

        let backend = CandleBackend::new(CandleBackendConfig::new(
            tmp.join("models"),
            tmp.join("output"),
        ))
        .unwrap();

        let source = reimagine_inference::ResolvedImageSource::new(&path, "image/png", None);
        let request = ImageImportRequest::new(
            source,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-import"),
        );

        let response = execute_image_import(&backend, request).unwrap();
        let payload_key = response.image().payload().payload_key().clone();
        let stored = backend.store().get_image(&payload_key).unwrap();
        let values = stored
            .tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // NCHW layout: [1, 3, 1, 2]
        // R plane: [1.0, 0.0]
        // G plane: [1.0, 0.0]
        // B plane: [1.0, 0.0]
        assert!((values[0] - 1.0).abs() < 1e-6, "white R should be 1.0");
        assert!((values[1] - 0.0).abs() < 1e-6, "black R should be 0.0");
        assert!((values[2] - 1.0).abs() < 1e-6, "white G should be 1.0");
        assert!((values[3] - 0.0).abs() < 1e-6, "black G should be 0.0");
        assert!((values[4] - 1.0).abs() < 1e-6, "white B should be 1.0");
        assert!((values[5] - 0.0).abs() < 1e-6, "black B should be 0.0");

        // All values should be in [0, 1]
        for v in &values {
            assert!(*v >= 0.0 && *v <= 1.0, "value {v} outside [0, 1]");
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
