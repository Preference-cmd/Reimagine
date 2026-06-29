//! Image import decoding and backend-store insertion.

use candle_core::Tensor;
use reimagine_core::model::{TensorDType, TensorShape};
use reimagine_inference::{
    BackendPayloadKey, BackendTensorHandle, ImageImportRequest, ImageImportResponse,
    InferenceBackend, RuntimeImage,
};

use crate::backend::CandleBackend;
use crate::error::CandleBackendError;
use crate::store::CandleImage;

const SUPPORTED_MEDIA_TYPES: &[&str] = &["image/png", "image/jpeg", "image/webp"];
const MAX_IMPORT_BYTES: u64 = 256 * 1024 * 1024; // 256 MiB

pub fn execute_image_import(
    request: ImageImportRequest,
    backend: &CandleBackend,
) -> Result<ImageImportResponse, CandleBackendError> {
    let source = request.source();
    let media_type = source.media_type();

    let format = supported_format(media_type)?;
    let bytes = read_bounded_source(source.path())?;
    let candle_image = decode_image_bytes(&bytes, format, source.path(), backend.device())?;

    let payload_key = BackendPayloadKey::new(format!(
        "image:{}:{}",
        request.run_id().as_str(),
        request.node_id().as_str()
    ));
    let runtime_image = runtime_image_from_payload(backend, &payload_key, &candle_image);

    backend
        .store()
        .insert_image(request.run_id().clone(), payload_key.clone(), candle_image);

    Ok(ImageImportResponse::new(runtime_image))
}

fn supported_format(media_type: &str) -> Result<::image::ImageFormat, CandleBackendError> {
    if !SUPPORTED_MEDIA_TYPES.contains(&media_type) {
        return Err(unsupported_media_type(media_type));
    }

    ::image::ImageFormat::from_mime_type(media_type)
        .ok_or_else(|| unsupported_media_type(media_type))
}

fn unsupported_media_type(media_type: &str) -> CandleBackendError {
    CandleBackendError::InvalidRequest(format!(
        "image.import unsupported media type `{media_type}`; supported: png, jpeg, webp"
    ))
}

fn read_bounded_source(path: &std::path::Path) -> Result<Vec<u8>, CandleBackendError> {
    let meta = std::fs::metadata(path).map_err(|e| {
        CandleBackendError::InvalidRequest(format!(
            "image.import failed to read source file `{}`: {e}",
            path.display()
        ))
    })?;

    if meta.len() > MAX_IMPORT_BYTES {
        return Err(CandleBackendError::InvalidRequest(format!(
            "image.import file too large ({} bytes, max {} bytes)",
            meta.len(),
            MAX_IMPORT_BYTES
        )));
    }

    std::fs::read(path).map_err(|e| {
        CandleBackendError::InvalidRequest(format!(
            "image.import failed to read source file `{}`: {e}",
            path.display()
        ))
    })
}

fn decode_image_bytes(
    bytes: &[u8],
    format: ::image::ImageFormat,
    path: &std::path::Path,
    device: &candle_core::Device,
) -> Result<CandleImage, CandleBackendError> {
    let dynamic = ::image::load_from_memory_with_format(bytes, format).map_err(|e| {
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
    let plane = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| {
            CandleBackendError::InvalidRequest(format!(
                "image.import image dimensions overflow: {width}x{height}"
            ))
        })?;
    let value_count = plane.checked_mul(3).ok_or_else(|| {
        CandleBackendError::InvalidRequest(format!(
            "image.import RGB tensor size overflow: {width}x{height}"
        ))
    })?;
    let mut nchw = vec![0.0f32; value_count];
    for (i, pixel) in raw.chunks_exact(3).enumerate() {
        nchw[i] = pixel[0] as f32 / 255.0;
        nchw[plane + i] = pixel[1] as f32 / 255.0;
        nchw[2 * plane + i] = pixel[2] as f32 / 255.0;
    }

    let tensor =
        Tensor::from_vec(nchw, (1, 3, height as usize, width as usize), device).map_err(|e| {
            CandleBackendError::InvalidRequest(format!("image.import failed to create tensor: {e}"))
        })?;

    Ok(CandleImage::new(
        tensor,
        width,
        height,
        1,
        "rgb".to_string(),
    ))
}

fn runtime_image_from_payload(
    backend: &CandleBackend,
    payload_key: &BackendPayloadKey,
    candle_image: &CandleImage,
) -> RuntimeImage {
    let tensor_handle = BackendPayloadKey::new(payload_key.as_str());
    RuntimeImage::new(
        BackendTensorHandle::with_instance(
            backend.backend_kind().clone(),
            backend.backend_instance(),
            tensor_handle,
            TensorDType::F32,
            TensorShape::new(vec![
                1,
                3,
                candle_image.height() as usize,
                candle_image.width() as usize,
            ]),
            backend.device_label().to_string(),
        ),
        candle_image.width(),
        candle_image.height(),
        candle_image.batch(),
        candle_image.color_space(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CandleBackendConfig;
    use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};

    fn backend(root: &std::path::Path) -> CandleBackend {
        CandleBackend::new(CandleBackendConfig::new(
            root.join("models"),
            root.join("output"),
        ))
        .unwrap()
    }

    fn import_request(path: &std::path::Path, media_type: &str) -> ImageImportRequest {
        let source = reimagine_inference::ResolvedImageSource::new(path, media_type, None);
        ImageImportRequest::new(
            source,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-import"),
        )
    }

    #[test]
    fn image_import_png_produces_correct_runtime_image() {
        let tmp = tempfile::tempdir().unwrap();
        let img = ::image::RgbImage::from_fn(8, 6, |x, y| {
            ::image::Rgb([(x * 32) as u8, (y * 40) as u8, 200])
        });
        let path = tmp.path().join("test.png");
        img.save(&path).unwrap();

        let backend = backend(tmp.path());
        let response = execute_image_import(import_request(&path, "image/png"), &backend).unwrap();
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
    }

    #[test]
    fn image_import_missing_file_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let backend = backend(tmp.path());
        let path = tmp.path().join("no-such-file.png");

        let err = execute_image_import(import_request(&path, "image/png"), &backend).unwrap_err();
        let msg = match err {
            CandleBackendError::InvalidRequest(msg) => msg,
            other => panic!("expected InvalidRequest, got {other:?}"),
        };
        assert!(msg.contains("failed to read source file"), "msg: {msg}");
    }

    #[test]
    fn image_import_unsupported_media_type_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.bmp");
        std::fs::write(&path, b"not a real bmp").unwrap();
        let backend = backend(tmp.path());

        let err = execute_image_import(import_request(&path, "image/bmp"), &backend).unwrap_err();
        let msg = match err {
            CandleBackendError::InvalidRequest(msg) => msg,
            other => panic!("expected InvalidRequest, got {other:?}"),
        };
        assert!(msg.contains("unsupported media type"), "msg: {msg}");
        assert!(msg.contains("image/bmp"), "msg: {msg}");
    }

    #[test]
    fn image_import_corrupt_image_returns_decoder_error() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("corrupt.png");
        std::fs::write(&path, b"this is not a valid PNG file").unwrap();
        let backend = backend(tmp.path());

        let err = execute_image_import(import_request(&path, "image/png"), &backend).unwrap_err();
        let msg = match err {
            CandleBackendError::InvalidRequest(msg) => msg,
            other => panic!("expected InvalidRequest, got {other:?}"),
        };
        assert!(msg.contains("failed to decode image"), "msg: {msg}");
    }

    #[test]
    fn image_import_normalizes_to_unit_range() {
        let tmp = tempfile::tempdir().unwrap();
        let img = ::image::RgbImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                ::image::Rgb([255u8, 255, 255])
            } else {
                ::image::Rgb([0u8, 0, 0])
            }
        });
        let path = tmp.path().join("white_black.png");
        img.save(&path).unwrap();
        let backend = backend(tmp.path());

        let response = execute_image_import(import_request(&path, "image/png"), &backend).unwrap();
        let stored = backend
            .store()
            .get_image(response.image().payload().payload_key())
            .unwrap();
        let values = stored
            .tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert!((values[0] - 1.0).abs() < 1e-6, "white R should be 1.0");
        assert!((values[1] - 0.0).abs() < 1e-6, "black R should be 0.0");
        assert!((values[2] - 1.0).abs() < 1e-6, "white G should be 1.0");
        assert!((values[3] - 0.0).abs() < 1e-6, "black G should be 0.0");
        assert!((values[4] - 1.0).abs() < 1e-6, "white B should be 1.0");
        assert!((values[5] - 0.0).abs() < 1e-6, "black B should be 0.0");
        assert!(values.iter().all(|v| *v >= 0.0 && *v <= 1.0));
    }
}
