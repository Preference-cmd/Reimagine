//! Burn-native SDXL checkpoint writer.
//!
//! Reads a single safetensors checkpoint file, remaps tensors to Burn Module
//! snapshot keys, groups by component role, and writes separate safetensors
//! files in the ComfyUI-style layout directly.
//!
//! Does NOT reuse the legacy `write_synthetic_sdxl_components` pipeline because
//! the output layout is fundamentally different (top-level flat files per role
//! vs nested `<role>/model.safetensors` under a fingerprint directory).

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

use safetensors::tensor::{Dtype, SafeTensors, View, serialize_to_file};

use super::checkpoint_projection::{
    BurnCheckpointRole, BurnCheckpointProjection, TextEncoderCount, text_encoder_count,
};
use super::conversion::{
    BurnSdxlConversionError, BurnSdxlConversionReport, BurnSdxlOutputComponentReport,
};
use super::source_mapping::{DIFFUSION_MAPPINGS, VAE_MAPPINGS};
use crate::models::stable_diffusion::sdxl::metadata::metadata_keys;

/// Output layout directories relative to the model root.
pub(crate) const DIFFUSION_DIR: &str = "diffusion_model";
pub(crate) const VAE_DIR: &str = "vae";
pub(crate) const CLIP_DIR: &str = "clip";

/// In-memory tensor view for safetensors serialization.
#[derive(Debug, Clone)]
struct OwnedTensorView {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for OwnedTensorView {
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// Write component files from a single safetensors checkpoint.
pub(crate) fn write_real_checkpoint_components(
    source_path: &Path,
    model_id: &str,
    model_root: &Path,
    projection: &BurnCheckpointProjection,
) -> Result<BurnSdxlConversionReport, BurnSdxlConversionError> {
    let encoder_count = text_encoder_count(projection);

    let bytes = fs::read(source_path).map_err(|err| BurnSdxlConversionError::Io {
        path: source_path.to_path_buf(),
        source: err,
    })?;
    let safetensors = SafeTensors::deserialize(&bytes).map_err(|err| {
        BurnSdxlConversionError::SafetensorsReadBack {
            path: source_path.to_path_buf(),
            source: err,
        }
    })?;

    let mut out_components: Vec<BurnSdxlOutputComponentReport> = Vec::new();
    let mut mapped_count: usize = 0;
    let mut ignored: Vec<String> = Vec::new();

    // Group tensors by role: (target_key, shape, data)
    let mut role_tensors: BTreeMap<BurnCheckpointRole, Vec<(String, Vec<usize>, Vec<u8>)>> =
        BTreeMap::new();

    for key in safetensors.names() {
        if key == "__metadata__" {
            continue;
        }
        match classify_checkpoint_tensor(key, encoder_count) {
            Some(Classification { role, target_key }) => {
                let tensor = safetensors.tensor(key).map_err(|err| {
                    BurnSdxlConversionError::SafetensorsReadBack {
                        path: source_path.to_path_buf(),
                        source: err,
                    }
                })?;
                let shape = tensor.shape().to_vec();
                let data = tensor.data().to_vec();
                role_tensors
                    .entry(role)
                    .or_default()
                    .push((target_key, shape, data));
                mapped_count += 1;
            }
            None => {
                if key.starts_with("model_ema.") {
                    ignored.push(format!("ignored:{key}"));
                } else {
                    let prefix = unknown_family_prefix(key);
                    ignored.push(format!("unknown:{prefix}:{key}"));
                }
            }
        }
    }

    // Write diffusion
    if let Some(tensors) = role_tensors.remove(&BurnCheckpointRole::Diffusion) {
        if !tensors.is_empty() {
            let dir = model_root.join(DIFFUSION_DIR);
            fs::create_dir_all(&dir).map_err(|e| BurnSdxlConversionError::Io {
                path: dir.clone(),
                source: e,
            })?;
            let path = dir.join(format!("{model_id}.safetensors"));
            let count = tensors.len();
            write_component_file(&path, "diffusion", &tensors)?;
            out_components.push(BurnSdxlOutputComponentReport {
                role: super::component::BurnSdxlComponentRole::Diffusion,
                path: format!("{DIFFUSION_DIR}/{model_id}.safetensors"),
                tensor_count: count,
                validated_required_tensor_count: count,
            });
        }
    }

    // Write VAE
    if let Some(tensors) = role_tensors.remove(&BurnCheckpointRole::Vae) {
        if !tensors.is_empty() {
            let dir = model_root.join(VAE_DIR);
            fs::create_dir_all(&dir).map_err(|e| BurnSdxlConversionError::Io {
                path: dir.clone(),
                source: e,
            })?;
            let path = dir.join(format!("{model_id}.safetensors"));
            let count = tensors.len();
            write_component_file(&path, "vae", &tensors)?;
            out_components.push(BurnSdxlOutputComponentReport {
                role: super::component::BurnSdxlComponentRole::Vae,
                path: format!("{VAE_DIR}/{model_id}.safetensors"),
                tensor_count: count,
                validated_required_tensor_count: count,
            });
        }
    }

    // Write clip
    let text_tensors = role_tensors.remove(&BurnCheckpointRole::TextEncoder);
    let text2_tensors = role_tensors.remove(&BurnCheckpointRole::TextEncoder2);

    match encoder_count {
        TextEncoderCount::Two => {
            let clip_dir = model_root.join(CLIP_DIR).join(model_id);
            fs::create_dir_all(&clip_dir).map_err(|e| BurnSdxlConversionError::Io {
                path: clip_dir.clone(),
                source: e,
            })?;

            if let Some(tensors) = &text_tensors {
                let path = clip_dir.join("clip-l.safetensors");
                let count = tensors.len();
                write_component_file(&path, "clip-l", tensors)?;
                out_components.push(BurnSdxlOutputComponentReport {
                    role: super::component::BurnSdxlComponentRole::TextEncoder,
                    path: format!("{CLIP_DIR}/{model_id}/clip-l.safetensors"),
                    tensor_count: count,
                    validated_required_tensor_count: count,
                });
            }
            if let Some(tensors) = &text2_tensors {
                let path = clip_dir.join("clip-g.safetensors");
                let count = tensors.len();
                write_component_file(&path, "clip-g", tensors)?;
                out_components.push(BurnSdxlOutputComponentReport {
                    role: super::component::BurnSdxlComponentRole::TextEncoder2,
                    path: format!("{CLIP_DIR}/{model_id}/clip-g.safetensors"),
                    tensor_count: count,
                    validated_required_tensor_count: count,
                });
            }
        }
        TextEncoderCount::One => {
            if let Some(tensors) = &text_tensors {
                let clip_dir = model_root.join(CLIP_DIR);
                fs::create_dir_all(&clip_dir).map_err(|e| BurnSdxlConversionError::Io {
                    path: clip_dir.clone(),
                    source: e,
                })?;
                let path = clip_dir.join(format!("{model_id}.safetensors"));
                let count = tensors.len();
                write_component_file(&path, "clip", tensors)?;
                out_components.push(BurnSdxlOutputComponentReport {
                    role: super::component::BurnSdxlComponentRole::TextEncoder,
                    path: format!("{CLIP_DIR}/{model_id}.safetensors"),
                    tensor_count: count,
                    validated_required_tensor_count: count,
                });
            }
        }
    }

    // Build report
    let source_identity = source_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| model_id.to_string());

    Ok(BurnSdxlConversionReport {
        source_identity,
        source_layout: "original_sdxl_checkpoint".to_string(),
        target_contract_version: 1,
        output_components: out_components,
        mapped_tensor_count: mapped_count,
        ignored_tensor_families: ignored,
        diagnostics: Vec::new(),
        package: None,
    })
}

// ---------------------------------------------------------------------------
// Single-file component writer (uses OwnedTensorView + serialize_to_file)
// ---------------------------------------------------------------------------

fn write_component_file(
    output_path: &Path,
    component_label: &str,
    tensors: &[(String, Vec<usize>, Vec<u8>)],
) -> Result<(), BurnSdxlConversionError> {
    let mut metadata = BTreeMap::new();
    metadata.insert(
        metadata_keys::CONTRACT.to_owned(),
        "burn.component".to_string(),
    );
    metadata.insert(metadata_keys::CONTRACT_VERSION.to_owned(), "1".to_string());
    metadata.insert(metadata_keys::BACKEND.to_owned(), "burn".to_string());
    metadata.insert(
        metadata_keys::MODEL_SERIES.to_owned(),
        "stable_diffusion".to_string(),
    );
    metadata.insert(metadata_keys::VARIANT.to_owned(), "sdxl".to_string());
    metadata.insert(
        metadata_keys::COMPONENT_ROLE.to_owned(),
        component_label.to_string(),
    );
    metadata.insert(
        metadata_keys::TENSOR_LAYOUT.to_owned(),
        "burn-module-snapshot".to_string(),
    );

    let mut tensor_map: BTreeMap<String, OwnedTensorView> = BTreeMap::new();
    for (key, shape, data) in tensors {
        tensor_map.insert(
            key.clone(),
            OwnedTensorView {
                dtype: Dtype::F32,
                shape: shape.clone(),
                data: data.clone(),
            },
        );
    }

    let metadata_map: HashMap<String, String> = metadata.into_iter().collect();

    serialize_to_file(tensor_map, Some(metadata_map), output_path).map_err(|source| {
        BurnSdxlConversionError::SafetensorsWrite {
            path: output_path.to_path_buf(),
            source,
        }
    })?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tensor classification
// ---------------------------------------------------------------------------

struct Classification {
    role: BurnCheckpointRole,
    target_key: String,
}

fn classify_checkpoint_tensor(
    key: &str,
    #[allow(unused_variables)] encoder_count: TextEncoderCount,
) -> Option<Classification> {
    // Diffusion: model.diffusion_model.* → strip → match DIFFUSION_MAPPINGS
    if let Some(inner) = key.strip_prefix("model.diffusion_model.") {
        // Try both `model.diffusion.{inner}` (legacy prefix) and the
        // raw inner key (diffusers-format names that are already Burn-target).
        let legacy_source = format!("model.diffusion.{inner}");
        if let Some(m) = DIFFUSION_MAPPINGS
            .iter()
            .find(|m| m.source_key == legacy_source.as_str())
        {
            return Some(Classification {
                role: BurnCheckpointRole::Diffusion,
                target_key: m.target_key.to_string(),
            });
        }
        // Pass-through: many LDM keys (e.g. input_blocks.0.0.weight) don't
        // have an explicit mapping but their target name is the same.
        return Some(Classification {
            role: BurnCheckpointRole::Diffusion,
            target_key: inner.to_string(),
        });
    }
    // Diffusion: model.diffusion.* (legacy converter prefix used by some LDM checkpoints)
    if let Some(inner) = key.strip_prefix("model.diffusion.") {
        if let Some(m) = DIFFUSION_MAPPINGS.iter().find(|m| m.source_key == key) {
            return Some(Classification {
                role: BurnCheckpointRole::Diffusion,
                target_key: m.target_key.to_string(),
            });
        }
        return Some(Classification {
            role: BurnCheckpointRole::Diffusion,
            target_key: inner.to_string(),
        });
    }
    // Diffusion: diffusers-format single file (direct match)
    if let Some(m) = DIFFUSION_MAPPINGS.iter().find(|m| m.source_key == key) {
        return Some(Classification {
            role: BurnCheckpointRole::Diffusion,
            target_key: m.target_key.to_string(),
        });
    }

    // Text encoder: CLIP-L
    if let Some(inner) = key.strip_prefix("conditioner.embedders.0.transformer.text_model.") {
        return Some(Classification {
            role: BurnCheckpointRole::TextEncoder,
            target_key: format!("model.text_encoder.transformer.text_model.{inner}"),
        });
    }
    if key.starts_with("conditioner.embedders.0.text_projection.") {
        let inner = key.strip_prefix("conditioner.embedders.0.").unwrap();
        return Some(Classification {
            role: BurnCheckpointRole::TextEncoder,
            target_key: format!("model.text_encoder.{inner}"),
        });
    }

    // Text encoder 2: OpenCLIP-G
    if let Some(inner) = key.strip_prefix("conditioner.embedders.1.model.transformer.text_model.") {
        return Some(Classification {
            role: BurnCheckpointRole::TextEncoder2,
            target_key: format!("model.text_encoder_2.transformer.text_model.{inner}"),
        });
    }
    if let Some(inner) = key.strip_prefix("conditioner.embedders.1.model.") {
        if inner.starts_with("text_projection.") {
            return Some(Classification {
                role: BurnCheckpointRole::TextEncoder2,
                target_key: format!("model.text_encoder_2.{inner}"),
            });
        }
    }
    if let Some(inner) = key.strip_prefix("conditioner.embedders.1.transformer.text_model.") {
        return Some(Classification {
            role: BurnCheckpointRole::TextEncoder2,
            target_key: format!("model.text_encoder_2.transformer.text_model.{inner}"),
        });
    }

    // VAE
    if let Some(inner) = key.strip_prefix("first_stage_model.") {
        if let Some(m) = VAE_MAPPINGS.iter().find(|m| m.source_key == inner) {
            return Some(Classification {
                role: BurnCheckpointRole::Vae,
                target_key: m.target_key.to_string(),
            });
        }
    }

    None
}

fn unknown_family_prefix(name: &str) -> String {
    let mut segments = name.split('.');
    let first = segments.next().unwrap_or(name);
    let second = segments.next();
    match second {
        Some(sec) if segments.next().is_some() => format!("{first}.{sec}."),
        Some(_) | None => format!("{first}."),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::stable_diffusion::sdxl::checkpoint_inventory::BurnCheckpointInventory;
    use crate::models::stable_diffusion::sdxl::checkpoint_projection::project_from_inventory;
    use crate::models::stable_diffusion::sdxl::component::BurnSdxlComponentRole;
    use crate::models::stable_diffusion::sdxl::writer::inspect_component_safetensors;
    use std::fs;
    use std::path::PathBuf;

    fn temp_dir() -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("burn-ckpt-writer-{}-{nonce}", std::process::id()))
    }

    /// Build a minimal-length safetensors file with the given tensor names
    /// (each 1-element F32 = 4 bytes).
    fn make_ckpt(path: &Path, names: &[&str]) {
        use std::collections::BTreeMap;
        let mut map: BTreeMap<String, OwnedTensorView> = BTreeMap::new();
        for name in names {
            map.insert(
                name.to_string(),
                OwnedTensorView {
                    dtype: Dtype::F32,
                    shape: vec![1],
                    data: vec![0u8; 4],
                },
            );
        }
        serialize_to_file(map, None, path).unwrap();
    }

    #[test]
    fn detect_ldm_diffusion_key() {
        let r = classify_checkpoint_tensor(
            "model.diffusion_model.time_embed.0.weight",
            TextEncoderCount::One,
        );
        assert!(r.is_some());
        assert_eq!(r.unwrap().role, BurnCheckpointRole::Diffusion);
    }

    #[test]
    fn detect_ldm_vae_key() {
        let r = classify_checkpoint_tensor(
            "first_stage_model.decoder.conv_in.weight",
            TextEncoderCount::One,
        );
        assert!(r.is_some());
        assert_eq!(r.unwrap().role, BurnCheckpointRole::Vae);
    }

    #[test]
    fn write_and_read_single_clip() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let ckpt = dir.join("model.safetensors");
        make_ckpt(
            &ckpt,
            &[
                "model.diffusion_model.time_embed.0.weight",
                "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
                "first_stage_model.decoder.conv_in.weight",
            ],
        );

        let inv = BurnCheckpointInventory::from_path(&ckpt).unwrap();
        let projection = project_from_inventory(&inv).unwrap();
        let model_root = dir.join("models");
        let report =
            write_real_checkpoint_components(&ckpt, "test-model", &model_root, &projection)
                .unwrap();

        assert!(model_root.join("diffusion_model/test-model.safetensors").is_file());
        assert!(model_root.join("vae/test-model.safetensors").is_file());
        assert!(model_root.join("clip/test-model.safetensors").is_file());
        assert_eq!(report.output_components.len(), 3);

        // AC6: output files must pass inspect_component_safetensors() read-back
        // and satisfy the Burn component contract.
        for (role, file) in [
            (BurnSdxlComponentRole::Diffusion, "diffusion_model/test-model.safetensors"),
            (BurnSdxlComponentRole::Vae, "vae/test-model.safetensors"),
            (BurnSdxlComponentRole::TextEncoder, "clip/test-model.safetensors"),
        ] {
            let path = model_root.join(file);
            let inspected = inspect_component_safetensors(&path).unwrap();
            assert!(
                !inspected.inventory.is_empty(),
                "inventory must not be empty for {role:?}",
            );
            assert!(
                inspected
                    .metadata
                    .values()
                    .all(|v| !v.is_empty()),
                "metadata must not contain empty values for {role:?}",
            );
        }

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn write_and_read_dual_clip() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let ckpt = dir.join("sdxl.safetensors");
        make_ckpt(
            &ckpt,
            &[
                "model.diffusion_model.input_blocks.0.0.weight",
                "model.diffusion_model.out.2.weight",
                "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
                "conditioner.embedders.1.model.text_projection.weight",
                "first_stage_model.decoder.conv_in.weight",
            ],
        );

        let inv = BurnCheckpointInventory::from_path(&ckpt).unwrap();
        let projection = project_from_inventory(&inv).unwrap();
        let model_root = dir.join("models");
        let report =
            write_real_checkpoint_components(&ckpt, "sdxl-test", &model_root, &projection)
                .unwrap();

        assert!(model_root.join("diffusion_model/sdxl-test.safetensors").is_file());
        assert!(model_root.join("vae/sdxl-test.safetensors").is_file());
        assert!(model_root.join("clip/sdxl-test/clip-l.safetensors").is_file());
        assert!(model_root.join("clip/sdxl-test/clip-g.safetensors").is_file());
        assert_eq!(report.output_components.len(), 4);

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn ema_keys_are_ignored() {
        let r = classify_checkpoint_tensor("model_ema.decay", TextEncoderCount::One);
        assert!(r.is_none());
    }
}
