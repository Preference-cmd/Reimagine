use std::collections::BTreeMap;

use crate::error::ModelAcquisitionError;

/// A component entry from model_index.json.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ComponentEntry {
    /// The class name for this component (e.g., "UNet2DConditionModel").
    pub class_name: Option<String>,
    /// The path to the component's weights file (e.g., "unet/diffusion_pytorch_model.safetensors").
    pub path: Option<String>,
}

/// Parsed model_index.json content.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelIndex {
    /// The top-level class name (e.g., "StableDiffusionPipeline").
    pub class_name: Option<String>,
    /// Mapping of component names to their entries.
    pub components: BTreeMap<String, ComponentEntry>,
}

/// Typed component mapping for common Stable Diffusion pipeline components.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ComponentMapping {
    /// UNet model path.
    pub unet: Option<String>,
    /// Primary text encoder path (e.g., CLIPTextModel).
    pub text_encoder: Option<String>,
    /// Secondary text encoder path (e.g., CLIPTextModel_2 for SDXL).
    pub text_encoder_2: Option<String>,
    /// VAE model path.
    pub vae: Option<String>,
    /// Primary tokenizer path.
    pub tokenizer: Option<String>,
    /// Secondary tokenizer path (for SDXL).
    pub tokenizer_2: Option<String>,
    /// Scheduler/noise scheduler path.
    pub scheduler: Option<String>,
}

impl ModelIndex {
    /// Parse model_index.json from a serde_json::Value.
    pub fn from_json(value: serde_json::Value) -> Result<Self, ModelAcquisitionError> {
        let obj = value
            .as_object()
            .ok_or_else(|| ModelAcquisitionError::Json {
                path: Some("model_index.json".into()),
                message: "expected a JSON object".to_string(),
            })?;

        let class_name = obj
            .get("_class_name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let mut components = BTreeMap::new();

        // Parse component entries - they are top-level keys with _class_name and _diffusers_version
        for (key, value) in obj {
            // Skip metadata keys
            if key == "_class_name" || key == "_diffusers_version" {
                continue;
            }

            if let Some(entry_obj) = value.as_object() {
                let entry_class_name = entry_obj
                    .get("_class_name")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                // The path can be either a string directly or an object with "path" key
                let path = if let Some(path_str) = value.as_str() {
                    Some(path_str.to_string())
                } else if let Some(path_val) = entry_obj.get("path") {
                    path_val.as_str().map(|s| s.to_string())
                } else {
                    None
                };

                components.insert(
                    key.clone(),
                    ComponentEntry {
                        class_name: entry_class_name,
                        path,
                    },
                );
            }
        }

        Ok(Self {
            class_name,
            components,
        })
    }

    /// Convert to a typed ComponentMapping for common Stable Diffusion components.
    pub fn to_component_mapping(&self) -> ComponentMapping {
        let mut mapping = ComponentMapping::default();

        // Map common component names
        for (key, entry) in &self.components {
            let path = entry.path.as_deref();
            match key.as_str() {
                "unet" => mapping.unet = path.map(|s| s.to_string()),
                "text_encoder" => mapping.text_encoder = path.map(|s| s.to_string()),
                "text_encoder_2" => mapping.text_encoder_2 = path.map(|s| s.to_string()),
                "vae" => mapping.vae = path.map(|s| s.to_string()),
                "tokenizer" => mapping.tokenizer = path.map(|s| s.to_string()),
                "tokenizer_2" => mapping.tokenizer_2 = path.map(|s| s.to_string()),
                "scheduler" => mapping.scheduler = path.map(|s| s.to_string()),
                _ => {} // Ignore unknown components
            }
        }

        mapping
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sdxl_model_index() {
        // Real SDXL model_index.json structure
        let json = serde_json::json!({
            "_class_name": "StableDiffusionXLPipeline",
            "_diffusers_version": "0.21.4",
            "text_encoder": {
                "_class_name": "CLIPTextModel",
                "path": "text_encoder/model.safetensors"
            },
            "text_encoder_2": {
                "_class_name": "CLIPTextModel",
                "path": "text_encoder_2/model.safetensors"
            },
            "tokenizer": {
                "_class_name": "CLIPTokenizer",
                "path": "tokenizer"
            },
            "tokenizer_2": {
                "_class_name": "CLIPTokenizer",
                "path": "tokenizer_2"
            },
            "unet": {
                "_class_name": "UNet2DConditionModel",
                "path": "unet/diffusion_pytorch_model.safetensors"
            },
            "vae": {
                "_class_name": "AutoencoderKL",
                "path": "vae/diffusion_pytorch_model.safetensors"
            },
            "scheduler": {
                "_class_name": "DPMSolverMultistepScheduler",
                "path": null
            }
        });

        let model_index = ModelIndex::from_json(json).unwrap();

        assert_eq!(
            model_index.class_name.as_deref(),
            Some("StableDiffusionXLPipeline")
        );
        assert_eq!(model_index.components.len(), 7);

        // Check text_encoder_2 exists
        let text_encoder_2 = model_index.components.get("text_encoder_2").unwrap();
        assert_eq!(text_encoder_2.class_name.as_deref(), Some("CLIPTextModel"));
        assert_eq!(
            text_encoder_2.path.as_deref(),
            Some("text_encoder_2/model.safetensors")
        );
    }

    #[test]
    fn test_parse_sd15_model_index() {
        // SD 1.5 model_index.json structure
        let json = serde_json::json!({
            "_class_name": "StableDiffusionPipeline",
            "_diffusers_version": "0.21.4",
            "text_encoder": {
                "_class_name": "CLIPTextModel",
                "path": "text_encoder/model.safetensors"
            },
            "tokenizer": {
                "_class_name": "CLIPTokenizer",
                "path": "tokenizer"
            },
            "unet": {
                "_class_name": "UNet2DConditionModel",
                "path": "unet/diffusion_pytorch_model.safetensors"
            },
            "vae": {
                "_class_name": "AutoencoderKL",
                "path": "vae/diffusion_pytorch_model.safetensors"
            },
            "scheduler": {
                "_class_name": "PNDMScheduler",
                "path": null
            }
        });

        let model_index = ModelIndex::from_json(json).unwrap();

        assert_eq!(
            model_index.class_name.as_deref(),
            Some("StableDiffusionPipeline")
        );
        assert_eq!(model_index.components.len(), 5);

        // Check text_encoder_2 does not exist
        assert!(!model_index.components.contains_key("text_encoder_2"));
    }

    #[test]
    fn test_component_mapping_conversion() {
        let json = serde_json::json!({
            "_class_name": "StableDiffusionPipeline",
            "unet": {
                "_class_name": "UNet2DConditionModel",
                "path": "unet/diffusion_pytorch_model.safetensors"
            },
            "text_encoder": {
                "_class_name": "CLIPTextModel",
                "path": "text_encoder/model.safetensors"
            },
            "vae": {
                "_class_name": "AutoencoderKL",
                "path": "vae/diffusion_pytorch_model.safetensors"
            }
        });

        let model_index = ModelIndex::from_json(json).unwrap();
        let mapping = model_index.to_component_mapping();

        assert_eq!(
            mapping.unet.as_deref(),
            Some("unet/diffusion_pytorch_model.safetensors")
        );
        assert_eq!(
            mapping.text_encoder.as_deref(),
            Some("text_encoder/model.safetensors")
        );
        assert_eq!(
            mapping.vae.as_deref(),
            Some("vae/diffusion_pytorch_model.safetensors")
        );
        assert!(mapping.text_encoder_2.is_none());
    }

    #[test]
    fn test_parse_empty_json() {
        let json = serde_json::json!({});

        let model_index = ModelIndex::from_json(json).unwrap();

        assert!(model_index.class_name.is_none());
        assert!(model_index.components.is_empty());
    }

    #[test]
    fn test_parse_invalid_json() {
        let json = serde_json::json!("not an object");

        let result = ModelIndex::from_json(json);

        assert!(result.is_err());
    }

    #[test]
    fn test_component_mapping_default() {
        let mapping = ComponentMapping::default();

        assert!(mapping.unet.is_none());
        assert!(mapping.text_encoder.is_none());
        assert!(mapping.text_encoder_2.is_none());
        assert!(mapping.vae.is_none());
        assert!(mapping.tokenizer.is_none());
        assert!(mapping.tokenizer_2.is_none());
        assert!(mapping.scheduler.is_none());
    }

    #[test]
    fn test_model_index_serde() {
        let model_index = ModelIndex {
            class_name: Some("StableDiffusionPipeline".to_string()),
            components: BTreeMap::from([
                (
                    "unet".to_string(),
                    ComponentEntry {
                        class_name: Some("UNet2DConditionModel".to_string()),
                        path: Some("unet/model.safetensors".to_string()),
                    },
                ),
                (
                    "vae".to_string(),
                    ComponentEntry {
                        class_name: Some("AutoencoderKL".to_string()),
                        path: Some("vae/model.safetensors".to_string()),
                    },
                ),
            ]),
        };

        let json = serde_json::to_string(&model_index).unwrap();
        let parsed: ModelIndex = serde_json::from_str(&json).unwrap();

        assert_eq!(model_index, parsed);
    }
}
