use std::path::{Path, PathBuf};

use crate::hf::model_index::ComponentMapping;

/// Role of a resolved component within a Stable Diffusion pipeline.
///
/// These map directly to the role names the Burn backend expects
/// when loading split components.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComponentRole {
    /// UNet diffusion model (maps to `unet/` directory).
    Diffusion,
    /// VAE decoder (maps to `vae/` directory).
    Vae,
    /// Primary text encoder, e.g. CLIP-L (maps to `text_encoder/`).
    TextEncoder,
    /// Secondary text encoder, e.g. CLIP-G for SDXL (maps to `text_encoder_2/`).
    TextEncoder2,
}

impl ComponentRole {
    /// Canonical string representation used as the `component` metadata
    /// key in `ModelComponentSource` and Burn component metadata.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Diffusion => "diffusion",
            Self::Vae => "vae",
            Self::TextEncoder => "text_encoder",
            Self::TextEncoder2 => "text_encoder_2",
        }
    }
}

/// A single resolved component with absolute path and pipeline role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedComponent {
    /// Pipeline role of this component.
    pub role: ComponentRole,
    /// Absolute path to the safetensors weights file.
    pub path: PathBuf,
}

/// Resolve component mapping entries to absolute paths.
///
/// Takes a `ComponentMapping` (parsed from `model_index.json`) and a
/// `base_dir` (the downloaded model directory) and produces one
/// `ResolvedComponent` per available component. Missing optional
/// components (e.g. `text_encoder_2` for SD 1.5) are silently skipped.
///
/// The mapping from diffusers component names to Burn role names:
/// - `unet` → `ComponentRole::Diffusion`
/// - `vae` → `ComponentRole::Vae`
/// - `text_encoder` → `ComponentRole::TextEncoder`
/// - `text_encoder_2` → `ComponentRole::TextEncoder2`
/// - `tokenizer`, `tokenizer_2`, `scheduler` → skipped (not weight files)
///
/// Returns a `Vec` sorted by role name for deterministic output.
pub fn resolve_component_paths(
    mapping: &ComponentMapping,
    base_dir: &Path,
) -> Vec<ResolvedComponent> {
    let mut components = Vec::new();

    if let Some(ref unet_path) = mapping.unet {
        components.push(ResolvedComponent {
            role: ComponentRole::Diffusion,
            path: base_dir.join(unet_path),
        });
    }

    if let Some(ref vae_path) = mapping.vae {
        components.push(ResolvedComponent {
            role: ComponentRole::Vae,
            path: base_dir.join(vae_path),
        });
    }

    if let Some(ref te_path) = mapping.text_encoder {
        components.push(ResolvedComponent {
            role: ComponentRole::TextEncoder,
            path: base_dir.join(te_path),
        });
    }

    if let Some(ref te2_path) = mapping.text_encoder_2 {
        components.push(ResolvedComponent {
            role: ComponentRole::TextEncoder2,
            path: base_dir.join(te2_path),
        });
    }

    // Sort by role name for deterministic ordering.
    components.sort_by(|a, b| a.role.as_str().cmp(b.role.as_str()));
    components
}

/// Resolve component mapping and verify all resolved paths exist on disk.
///
/// Like [`resolve_component_paths`] but also checks that each resolved
/// path points to an existing file. Returns only components whose
/// files are present, logging diagnostics for missing ones.
pub fn resolve_component_paths_verified(
    mapping: &ComponentMapping,
    base_dir: &Path,
) -> Vec<ResolvedComponent> {
    let all = resolve_component_paths(mapping, base_dir);
    all.into_iter()
        .filter(|component| component.path.exists())
        .collect()
}

/// Try to parse `model_index.json` from a directory and resolve its
/// component mapping to absolute paths.
///
/// Returns `None` if `model_index.json` does not exist, is not valid
/// JSON, or contains no recognizable components.
pub fn resolve_from_model_index(
    base_dir: &Path,
) -> Option<Vec<ResolvedComponent>> {
    let index_path = base_dir.join("model_index.json");
    if !index_path.exists() {
        return None;
    }

    let content = std::fs::read_to_string(&index_path).ok()?;
    let json_value: serde_json::Value = serde_json::from_str(&content).ok()?;
    let model_index = crate::hf::model_index::ModelIndex::from_json(json_value).ok()?;
    let mapping = model_index.to_component_mapping();

    let components = resolve_component_paths(&mapping, base_dir);
    if components.is_empty() {
        return None;
    }

    Some(components)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf::model_index::ComponentMapping;
    use std::fs;

    fn make_temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn test_resolve_component_paths_sdxl() {
        let dir = make_temp_dir();
        let mapping = ComponentMapping {
            unet: Some("unet/diffusion_pytorch_model.safetensors".to_string()),
            vae: Some("vae/diffusion_pytorch_model.safetensors".to_string()),
            text_encoder: Some("text_encoder/model.safetensors".to_string()),
            text_encoder_2: Some("text_encoder_2/model.safetensors".to_string()),
            tokenizer: Some("tokenizer".to_string()),
            tokenizer_2: Some("tokenizer_2".to_string()),
            scheduler: Some("scheduler_config.json".to_string()),
        };

        let components = resolve_component_paths(&mapping, dir.path());
        assert_eq!(components.len(), 4);

        // Sorted by role name: diffusion, text_encoder, text_encoder_2, vae
        assert_eq!(components[0].role, ComponentRole::Diffusion);
        assert_eq!(
            components[0].path,
            dir.path().join("unet/diffusion_pytorch_model.safetensors")
        );

        assert_eq!(components[1].role, ComponentRole::TextEncoder);
        assert_eq!(
            components[1].path,
            dir.path().join("text_encoder/model.safetensors")
        );

        assert_eq!(components[2].role, ComponentRole::TextEncoder2);
        assert_eq!(
            components[2].path,
            dir.path().join("text_encoder_2/model.safetensors")
        );

        assert_eq!(components[3].role, ComponentRole::Vae);
        assert_eq!(
            components[3].path,
            dir.path().join("vae/diffusion_pytorch_model.safetensors")
        );
    }

    #[test]
    fn test_resolve_component_paths_sd15() {
        let dir = make_temp_dir();
        let mapping = ComponentMapping {
            unet: Some("unet/diffusion_pytorch_model.safetensors".to_string()),
            vae: Some("vae/diffusion_pytorch_model.safetensors".to_string()),
            text_encoder: Some("text_encoder/model.safetensors".to_string()),
            text_encoder_2: None, // SD 1.5 has no second text encoder
            tokenizer: Some("tokenizer".to_string()),
            tokenizer_2: None,
            scheduler: None,
        };

        let components = resolve_component_paths(&mapping, dir.path());
        assert_eq!(components.len(), 3);

        assert_eq!(components[0].role, ComponentRole::Diffusion);
        assert_eq!(components[1].role, ComponentRole::TextEncoder);
        assert_eq!(components[2].role, ComponentRole::Vae);
    }

    #[test]
    fn test_resolve_component_paths_empty_mapping() {
        let dir = make_temp_dir();
        let mapping = ComponentMapping::default();

        let components = resolve_component_paths(&mapping, dir.path());
        assert!(components.is_empty());
    }

    #[test]
    fn test_resolve_component_paths_verified_filters_missing() {
        let dir = make_temp_dir();

        // Create only the unet file
        fs::create_dir_all(dir.path().join("unet")).unwrap();
        fs::write(
            dir.path().join("unet/diffusion_pytorch_model.safetensors"),
            "fake weights",
        )
        .unwrap();

        let mapping = ComponentMapping {
            unet: Some("unet/diffusion_pytorch_model.safetensors".to_string()),
            vae: Some("vae/diffusion_pytorch_model.safetensors".to_string()),
            text_encoder: None,
            text_encoder_2: None,
            tokenizer: None,
            tokenizer_2: None,
            scheduler: None,
        };

        let components = resolve_component_paths_verified(&mapping, dir.path());
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].role, ComponentRole::Diffusion);
    }

    #[test]
    fn test_resolve_from_model_index_success() {
        let dir = make_temp_dir();
        let model_index = serde_json::json!({
            "_class_name": "StableDiffusionXLPipeline",
            "unet": {
                "_class_name": "UNet2DConditionModel",
                "path": "unet/model.safetensors"
            },
            "vae": {
                "_class_name": "AutoencoderKL",
                "path": "vae/model.safetensors"
            },
            "text_encoder": {
                "_class_name": "CLIPTextModel",
                "path": "text_encoder/model.safetensors"
            },
            "text_encoder_2": {
                "_class_name": "CLIPTextModel",
                "path": "text_encoder_2/model.safetensors"
            }
        });

        fs::write(
            dir.path().join("model_index.json"),
            serde_json::to_string_pretty(&model_index).unwrap(),
        )
        .unwrap();

        let components = resolve_from_model_index(dir.path()).unwrap();
        assert_eq!(components.len(), 4);
    }

    #[test]
    fn test_resolve_from_model_index_missing_file() {
        let dir = make_temp_dir();
        assert!(resolve_from_model_index(dir.path()).is_none());
    }

    #[test]
    fn test_resolve_from_model_index_invalid_json() {
        let dir = make_temp_dir();
        fs::write(dir.path().join("model_index.json"), "not json").unwrap();
        assert!(resolve_from_model_index(dir.path()).is_none());
    }

    #[test]
    fn test_component_role_as_str() {
        assert_eq!(ComponentRole::Diffusion.as_str(), "diffusion");
        assert_eq!(ComponentRole::Vae.as_str(), "vae");
        assert_eq!(ComponentRole::TextEncoder.as_str(), "text_encoder");
        assert_eq!(ComponentRole::TextEncoder2.as_str(), "text_encoder_2");
    }

    #[test]
    fn test_resolve_component_paths_sorting() {
        let dir = make_temp_dir();
        // Provide components in reverse order to verify sorting
        let mapping = ComponentMapping {
            unet: None,
            vae: Some("vae/model.safetensors".to_string()),
            text_encoder: Some("te/model.safetensors".to_string()),
            text_encoder_2: Some("te2/model.safetensors".to_string()),
            tokenizer: None,
            tokenizer_2: None,
            scheduler: None,
        };

        let components = resolve_component_paths(&mapping, dir.path());
        assert_eq!(components.len(), 3);
        // Sorted: diffusion (absent), text_encoder, text_encoder_2, vae
        assert_eq!(components[0].role, ComponentRole::TextEncoder);
        assert_eq!(components[1].role, ComponentRole::TextEncoder2);
        assert_eq!(components[2].role, ComponentRole::Vae);
    }
}
