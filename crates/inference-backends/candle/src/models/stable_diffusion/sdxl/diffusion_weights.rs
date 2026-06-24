use std::io::Read;
use std::path::Path;

use crate::error::CandleBackendError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SdxlDiffusionWeightLayout {
    OriginalCheckpoint,
    DiffusersUnet,
}

pub(crate) fn detect_diffusion_weight_layout_from_names<'a>(
    names: impl IntoIterator<Item = &'a str>,
) -> Result<SdxlDiffusionWeightLayout, CandleBackendError> {
    let mut saw_original = false;
    let mut saw_diffusers = false;
    let mut inspected = 0usize;

    for name in names {
        inspected += 1;
        if name.starts_with("model.diffusion_model.") {
            saw_original = true;
        }
        if name.starts_with("unet.")
            || name.starts_with("diffusion_model.")
            || name == "conv_in.weight"
            || name.starts_with("down_blocks.")
            || name.starts_with("up_blocks.")
            || name.starts_with("mid_block.")
        {
            saw_diffusers = true;
        }
    }

    match (saw_original, saw_diffusers) {
        (true, false) => Ok(SdxlDiffusionWeightLayout::OriginalCheckpoint),
        (false, true) => Ok(SdxlDiffusionWeightLayout::DiffusersUnet),
        (true, true) => Err(CandleBackendError::InvalidRequest(
            "SDXL diffusion weights contain both original checkpoint and diffusers UNet prefixes; refusing ambiguous layout".to_string(),
        )),
        (false, false) => Err(CandleBackendError::InvalidRequest(format!(
            "unsupported SDXL diffusion weight layout: inspected {inspected} tensors but found no `model.diffusion_model.*`, `unet.*`, `diffusion_model.*`, or root diffusers UNet keys"
        ))),
    }
}

#[allow(dead_code)]
pub(crate) fn detect_diffusion_weight_layout_from_file(
    path: &Path,
) -> Result<SdxlDiffusionWeightLayout, CandleBackendError> {
    let names = read_safetensors_header_names(path)?;
    detect_diffusion_weight_layout_from_names(names.iter().map(String::as_str))
}

fn read_safetensors_header_names(path: &Path) -> Result<Vec<String>, CandleBackendError> {
    let mut file = std::fs::File::open(path).map_err(|err| {
        CandleBackendError::InvalidRequest(format!(
            "failed to open SDXL diffusion safetensors `{}` for header inspection: {err}",
            path.display()
        ))
    })?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes).map_err(|err| {
        CandleBackendError::InvalidRequest(format!(
            "failed to read SDXL diffusion safetensors header length at `{}`: {err}",
            path.display()
        ))
    })?;
    let header_len = u64::from_le_bytes(len_bytes);
    let file_len = file
        .metadata()
        .map_err(|err| {
            CandleBackendError::InvalidRequest(format!(
                "failed to inspect SDXL diffusion safetensors metadata at `{}`: {err}",
                path.display()
            ))
        })?
        .len();
    if header_len > file_len.saturating_sub(8) {
        return Err(CandleBackendError::InvalidRequest(format!(
            "invalid SDXL diffusion safetensors header at `{}`: header length {header_len} exceeds file size {file_len}",
            path.display()
        )));
    }
    if header_len > 64 * 1024 * 1024 {
        return Err(CandleBackendError::InvalidRequest(format!(
            "SDXL diffusion safetensors header at `{}` is too large to inspect safely: {header_len} bytes",
            path.display()
        )));
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header).map_err(|err| {
        CandleBackendError::InvalidRequest(format!(
            "failed to read SDXL diffusion safetensors header at `{}`: {err}",
            path.display()
        ))
    })?;
    let value: serde_json::Value = serde_json::from_slice(&header).map_err(|err| {
        CandleBackendError::InvalidRequest(format!(
            "failed to parse SDXL diffusion safetensors header at `{}`: {err}",
            path.display()
        ))
    })?;
    let object = value.as_object().ok_or_else(|| {
        CandleBackendError::InvalidRequest(format!(
            "SDXL diffusion safetensors header at `{}` is not a JSON object",
            path.display()
        ))
    })?;
    Ok(object
        .keys()
        .filter(|name| name.as_str() != "__metadata__")
        .cloned()
        .collect())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;
    use std::path::{Path, PathBuf};

    use candle_core::{DType, Device, Tensor};

    use super::{
        SdxlDiffusionWeightLayout, detect_diffusion_weight_layout_from_file,
        detect_diffusion_weight_layout_from_names,
    };

    fn unique_temp_dir() -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "reimagine-sdxl-diffusion-weights-{}-{nonce}",
            std::process::id()
        ))
    }

    fn write_safetensors(path: &Path, name: &str) {
        let tensor = Tensor::zeros((1,), DType::F32, &Device::Cpu).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert(name, tensor);
        candle_core::safetensors::save(&tensors, path).unwrap();
    }

    #[test]
    fn detects_original_checkpoint_diffusion_prefix() {
        let layout = detect_diffusion_weight_layout_from_names([
            "model.diffusion_model.input_blocks.0.0.weight",
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
        ])
        .unwrap();

        assert_eq!(layout, SdxlDiffusionWeightLayout::OriginalCheckpoint);
    }

    #[test]
    fn detects_diffusers_unet_prefix() {
        let layout = detect_diffusion_weight_layout_from_names([
            "unet.down_blocks.0.resnets.0.conv1.weight",
        ])
        .unwrap();

        assert_eq!(layout, SdxlDiffusionWeightLayout::DiffusersUnet);
    }

    #[test]
    fn detects_root_diffusers_unet_keys() {
        let layout = detect_diffusion_weight_layout_from_names([
            "conv_in.weight",
            "down_blocks.0.resnets.0.conv1.weight",
        ])
        .unwrap();

        assert_eq!(layout, SdxlDiffusionWeightLayout::DiffusersUnet);
    }

    #[test]
    fn rejects_missing_diffusion_prefix_from_tiny_safetensors() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("unrelated.safetensors");
        write_safetensors(&path, "text_model.embeddings.token_embedding.weight");

        let err = detect_diffusion_weight_layout_from_file(&path).unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("unsupported SDXL diffusion weight layout"));
        assert!(msg.contains("model.diffusion_model"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_invalid_safetensors_header_length_before_layout_detection() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("placeholder.safetensors");
        fs::write(&path, b"placeholder").unwrap();

        let err = detect_diffusion_weight_layout_from_file(&path).unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("invalid SDXL diffusion safetensors header"));
        assert!(msg.contains("exceeds file size"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_ambiguous_original_and_diffusers_prefixes() {
        let err = detect_diffusion_weight_layout_from_names([
            "model.diffusion_model.input_blocks.0.0.weight",
            "unet.down_blocks.0.resnets.0.conv1.weight",
        ])
        .unwrap_err();

        assert!(err.to_string().contains("ambiguous"));
    }

    #[test]
    fn detects_layout_from_header_without_loading_tensor_payload() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("header-only.safetensors");
        let header = br#"{"model.diffusion_model.input_blocks.0.0.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        fs::write(&path, bytes).unwrap();

        let layout = detect_diffusion_weight_layout_from_file(&path).unwrap();

        assert_eq!(layout, SdxlDiffusionWeightLayout::OriginalCheckpoint);
        let _ = fs::remove_dir_all(&dir);
    }
}
