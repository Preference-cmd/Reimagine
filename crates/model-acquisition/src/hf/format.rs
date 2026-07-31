use crate::hf::metadata::HfRepoMetadata;

/// Format of a HuggingFace model repository.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ModelRepoFormat {
    /// Diffusers format: has `model_index.json` with component mapping.
    Diffusers,
    /// CompVis/LDM format: has `.ckpt` files, no `model_index.json`.
    CompVis,
    /// Single-file safetensors: only root-level `.safetensors` files, no subdirectories.
    SingleFileSafetensors,
    /// Unknown format: cannot determine the format from the metadata.
    Unknown,
}

/// Detect the model format from repository metadata.
///
/// Detection logic:
/// - Has sibling with `rfilename == "model_index.json"` → `Diffusers`
/// - Has sibling ending with `.ckpt` and no `model_index.json` → `CompVis`
/// - Has only `*.safetensors` in root (no `/` in rfilename) and no subdirectories → `SingleFileSafetensors`
/// - Otherwise → `Unknown`
pub fn detect_format(metadata: &HfRepoMetadata) -> ModelRepoFormat {
    let has_model_index = metadata
        .siblings
        .iter()
        .any(|s| s.rfilename == "model_index.json");

    if has_model_index {
        return ModelRepoFormat::Diffusers;
    }

    let has_ckpt = metadata
        .siblings
        .iter()
        .any(|s| s.rfilename.ends_with(".ckpt"));

    if has_ckpt {
        return ModelRepoFormat::CompVis;
    }

    // Check if we have only root-level safetensors files
    let has_safetensors = metadata
        .siblings
        .iter()
        .any(|s| s.rfilename.ends_with(".safetensors"));

    if has_safetensors {
        // Check if all safetensors files are in the root directory (no `/` in path)
        let all_root_level = metadata
            .siblings
            .iter()
            .filter(|s| s.rfilename.ends_with(".safetensors"))
            .all(|s| !s.rfilename.contains('/'));

        // Check if there are any subdirectories (files with `/` in path)
        let has_subdirectories = metadata.siblings.iter().any(|s| s.rfilename.contains('/'));

        if all_root_level && !has_subdirectories {
            return ModelRepoFormat::SingleFileSafetensors;
        }
    }

    ModelRepoFormat::Unknown
}

/// Build download patterns for diffusers format based on actual sibling paths.
///
/// Returns a list of glob patterns that can be used to filter files for download.
/// The patterns are based on the actual file paths in the repository.
pub fn diffusers_download_patterns(metadata: &HfRepoMetadata) -> Vec<String> {
    let mut patterns = Vec::new();

    for sibling in &metadata.siblings {
        let path = &sibling.rfilename;

        // Include all files except .gitattributes and .huggingface
        if path == ".gitattributes" || path == ".huggingface" {
            continue;
        }

        // Include safetensors, JSON config, tokenizer, and other important files
        if path.ends_with(".safetensors")
            || path.ends_with(".json")
            || path.contains("tokenizer")
            || path.contains("vocab")
            || path.contains("merges")
            || path.ends_with(".txt")
            || path.ends_with(".py")
            || path.ends_with(".onnx")
        {
            patterns.push(path.clone());
        }
    }

    // If no specific patterns were found, return a wildcard pattern
    if patterns.is_empty() {
        patterns.push("*".to_string());
    }

    patterns
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf::metadata::{HfLfsInfo, HfSibling};

    fn make_metadata(siblings: Vec<HfSibling>) -> HfRepoMetadata {
        HfRepoMetadata {
            repo_id: "test/model".to_string(),
            revision: "main".to_string(),
            siblings,
            tags: vec![],
            pipeline_tag: None,
            library_name: None,
        }
    }

    #[test]
    fn test_detect_diffusers_format() {
        let metadata = make_metadata(vec![
            HfSibling {
                rfilename: "model_index.json".to_string(),
                size: Some(100),
                lfs: None,
            },
            HfSibling {
                rfilename: "unet/diffusion_pytorch_model.safetensors".to_string(),
                size: Some(1_000_000),
                lfs: Some(HfLfsInfo {
                    size: 1_000_000,
                    sha256: None,
                }),
            },
            HfSibling {
                rfilename: "vae/diffusion_pytorch_model.safetensors".to_string(),
                size: Some(500_000),
                lfs: Some(HfLfsInfo {
                    size: 500_000,
                    sha256: None,
                }),
            },
        ]);

        assert_eq!(detect_format(&metadata), ModelRepoFormat::Diffusers);
    }

    #[test]
    fn test_detect_compvis_format() {
        let metadata = make_metadata(vec![
            HfSibling {
                rfilename: "model.ckpt".to_string(),
                size: Some(4_000_000_000),
                lfs: Some(HfLfsInfo {
                    size: 4_000_000_000,
                    sha256: None,
                }),
            },
            HfSibling {
                rfilename: "config.yaml".to_string(),
                size: Some(100),
                lfs: None,
            },
        ]);

        assert_eq!(detect_format(&metadata), ModelRepoFormat::CompVis);
    }

    #[test]
    fn test_detect_single_file_safetensors() {
        let metadata = make_metadata(vec![
            HfSibling {
                rfilename: "model.safetensors".to_string(),
                size: Some(4_000_000_000),
                lfs: Some(HfLfsInfo {
                    size: 4_000_000_000,
                    sha256: None,
                }),
            },
            HfSibling {
                rfilename: "config.json".to_string(),
                size: Some(500),
                lfs: None,
            },
        ]);

        assert_eq!(
            detect_format(&metadata),
            ModelRepoFormat::SingleFileSafetensors
        );
    }

    #[test]
    fn test_detect_unknown_format() {
        let metadata = make_metadata(vec![HfSibling {
            rfilename: "README.md".to_string(),
            size: Some(1000),
            lfs: None,
        }]);

        assert_eq!(detect_format(&metadata), ModelRepoFormat::Unknown);
    }

    #[test]
    fn test_detect_diffusers_takes_priority_over_compvis() {
        // If both model_index.json and .ckpt exist, diffusers takes priority
        let metadata = make_metadata(vec![
            HfSibling {
                rfilename: "model_index.json".to_string(),
                size: Some(100),
                lfs: None,
            },
            HfSibling {
                rfilename: "model.ckpt".to_string(),
                size: Some(4_000_000_000),
                lfs: Some(HfLfsInfo {
                    size: 4_000_000_000,
                    sha256: None,
                }),
            },
        ]);

        assert_eq!(detect_format(&metadata), ModelRepoFormat::Diffusers);
    }

    #[test]
    fn test_diffusers_download_patterns() {
        let metadata = make_metadata(vec![
            HfSibling {
                rfilename: "model_index.json".to_string(),
                size: Some(100),
                lfs: None,
            },
            HfSibling {
                rfilename: "unet/diffusion_pytorch_model.safetensors".to_string(),
                size: Some(1_000_000),
                lfs: Some(HfLfsInfo {
                    size: 1_000_000,
                    sha256: None,
                }),
            },
            HfSibling {
                rfilename: "tokenizer/tokenizer.json".to_string(),
                size: Some(10_000),
                lfs: None,
            },
            HfSibling {
                rfilename: ".gitattributes".to_string(),
                size: Some(100),
                lfs: None,
            },
        ]);

        let patterns = diffusers_download_patterns(&metadata);

        assert!(patterns.contains(&"model_index.json".to_string()));
        assert!(patterns.contains(&"unet/diffusion_pytorch_model.safetensors".to_string()));
        assert!(patterns.contains(&"tokenizer/tokenizer.json".to_string()));
        assert!(!patterns.contains(&".gitattributes".to_string()));
    }

    #[test]
    fn test_diffusers_download_patterns_empty() {
        let metadata = make_metadata(vec![]);

        let patterns = diffusers_download_patterns(&metadata);

        assert_eq!(patterns, vec!["*".to_string()]);
    }

    #[test]
    fn test_format_serde() {
        let format = ModelRepoFormat::Diffusers;
        let json = serde_json::to_string(&format).unwrap();
        let parsed: ModelRepoFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(format, parsed);
    }
}
