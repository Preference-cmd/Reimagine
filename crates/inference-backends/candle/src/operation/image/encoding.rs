//! Backend image tensor to PNG encoding.

use candle_core::Device;

use crate::error::CandleBackendError;
use crate::store::CandleImage;

pub(super) fn encode_tensor_to_png(image: &CandleImage) -> Result<Vec<u8>, CandleBackendError> {
    let tensor = image.tensor();
    let tensor_on_cpu = if matches!(tensor.device(), Device::Cpu) {
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

    let rgb = nchw_f32_to_interleaved_rgb8(&float_vec, width, height)?;
    encode_rgb8_to_png(&rgb, width, height)
}

fn nchw_f32_to_interleaved_rgb8(
    data: &[f32],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, CandleBackendError> {
    let plane_len = width.checked_mul(height).ok_or_else(|| {
        CandleBackendError::InvalidRequest(format!(
            "image.save image dimensions overflow: {width}x{height}"
        ))
    })?;
    let expected_len = plane_len.checked_mul(3).ok_or_else(|| {
        CandleBackendError::InvalidRequest(format!(
            "image.save RGB image byte length overflow: {width}x{height}"
        ))
    })?;
    if data.len() != expected_len {
        return Err(CandleBackendError::InvalidRequest(format!(
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
) -> Result<Vec<u8>, CandleBackendError> {
    let width_u32 = u32::try_from(width).map_err(|_| {
        CandleBackendError::InvalidRequest(format!("image.save width too large for PNG: {width}"))
    })?;
    let height_u32 = u32::try_from(height).map_err(|_| {
        CandleBackendError::InvalidRequest(format!("image.save height too large for PNG: {height}"))
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
        CandleBackendError::InvalidRequest(format!("image.save failed to encode PNG: {e}"))
    })?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Tensor};

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
    fn encode_tensor_rejects_non_rgb_tensor() {
        let tensor = Tensor::zeros((1, 4, 8, 8), DType::F32, &Device::Cpu).unwrap();
        let image = CandleImage::new(tensor, 8, 8, 1, "rgb".to_string());

        let err = encode_tensor_to_png(&image).unwrap_err();
        let msg = match err {
            CandleBackendError::InvalidRequest(msg) => msg,
            other => panic!("expected InvalidRequest, got {other:?}"),
        };
        assert!(msg.contains("3-channel RGB"), "msg: {msg}");
    }
}
