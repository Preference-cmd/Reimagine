use crate::error::ModelAcquisitionError;
use crate::hf::format::{ModelRepoFormat, detect_format, diffusers_download_patterns};
use crate::hf::metadata::HfRepoMetadata;
use crate::hf::model_index::{ComponentMapping, ModelIndex};
use crate::request::AllowPatterns;

/// Result of resolving download patterns for a repository.
#[derive(Debug, Clone)]
pub struct ResolvedPatterns {
    /// The allow patterns to use for the download.
    pub patterns: AllowPatterns,
    /// The detected repository format.
    pub format: ModelRepoFormat,
    /// Typed component mapping (only populated for Diffusers format with a valid model_index.json).
    pub component_mapping: Option<ComponentMapping>,
}

/// Resolve the optimal download patterns for a HuggingFace model repository.
///
/// If `explicit_patterns` is non-empty, those patterns are returned as-is with
/// `ModelRepoFormat::Unknown` (no metadata fetch needed).
///
/// If `explicit_patterns` is empty, this fetches repository metadata, detects the
/// format, and builds optimal patterns via [`diffusers_download_patterns`] for
/// diffusers-format repos.
pub async fn resolve_download_patterns(
    client: &hf_hub::HFClient,
    repo_id: &str,
    revision: &str,
    explicit_patterns: &AllowPatterns,
) -> Result<ResolvedPatterns, ModelAcquisitionError> {
    if !explicit_patterns.is_empty() {
        return Ok(ResolvedPatterns {
            patterns: explicit_patterns.clone(),
            format: ModelRepoFormat::Unknown,
            component_mapping: None,
        });
    }

    let metadata = HfRepoMetadata::fetch(client, repo_id, revision).await?;

    let format = detect_format(&metadata);

    let (patterns, component_mapping) = match format {
        ModelRepoFormat::Diffusers => {
            let model_index = if metadata
                .siblings
                .iter()
                .any(|s| s.rfilename == "model_index.json")
            {
                fetch_model_index(client, repo_id, revision).await.ok()
            } else {
                None
            };

            let component_mapping = model_index.as_ref().map(|mi| mi.to_component_mapping());

            let patterns = diffusers_download_patterns(&metadata);
            (patterns, component_mapping)
        }
        ModelRepoFormat::SingleFileSafetensors => {
            // For single-file safetensors, download the safetensors file plus any config.
            let patterns = build_single_file_patterns(&metadata);
            (patterns, None)
        }
        ModelRepoFormat::CompVis => {
            // For CompVis, download the .ckpt file plus config files.
            let patterns = build_compvis_patterns(&metadata);
            (patterns, None)
        }
        ModelRepoFormat::Unknown => {
            // Fallback: download everything.
            let patterns = build_unknown_patterns(&metadata);
            (patterns, None)
        }
    };

    Ok(ResolvedPatterns {
        patterns: AllowPatterns::new(patterns),
        format,
        component_mapping,
    })
}

/// Fetch and parse model_index.json from a diffusers-format repository.
pub(crate) async fn fetch_model_index(
    client: &hf_hub::HFClient,
    repo_id: &str,
    revision: &str,
) -> Result<ModelIndex, ModelAcquisitionError> {
    let (owner, name) = repo_id
        .split_once('/')
        .ok_or_else(|| ModelAcquisitionError::Hub {
            repo: repo_id.to_string(),
            message: "invalid repo_id format, expected 'owner/name'".to_string(),
        })?;

    let repo = client.model(owner, name);

    let bytes = repo
        .download_file_to_bytes()
        .filename("model_index.json")
        .revision(revision)
        .send()
        .await
        .map_err(|e| ModelAcquisitionError::Hub {
            repo: repo_id.to_string(),
            message: format!("failed to download model_index.json: {e}"),
        })?;

    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| ModelAcquisitionError::Json {
            path: Some("model_index.json".into()),
            message: e.to_string(),
        })?;

    ModelIndex::from_json(value)
}

/// Build download patterns for single-file safetensors format.
fn build_single_file_patterns(metadata: &HfRepoMetadata) -> Vec<String> {
    let mut patterns = Vec::new();
    for sibling in &metadata.siblings {
        let path = &sibling.rfilename;
        if path == ".gitattributes" || path == ".huggingface" {
            continue;
        }
        // Include safetensors and config files
        if path.ends_with(".safetensors")
            || path.ends_with(".json")
            || path.ends_with(".txt")
            || path.ends_with(".yaml")
            || path.ends_with(".yml")
        {
            patterns.push(path.clone());
        }
    }
    if patterns.is_empty() {
        patterns.push("*".to_string());
    }
    patterns
}

/// Build download patterns for CompVis format.
fn build_compvis_patterns(metadata: &HfRepoMetadata) -> Vec<String> {
    let mut patterns = Vec::new();
    for sibling in &metadata.siblings {
        let path = &sibling.rfilename;
        if path == ".gitattributes" || path == ".huggingface" {
            continue;
        }
        // Include .ckpt files and config files
        if path.ends_with(".ckpt")
            || path.ends_with(".json")
            || path.ends_with(".yaml")
            || path.ends_with(".yml")
            || path.ends_with(".txt")
        {
            patterns.push(path.clone());
        }
    }
    if patterns.is_empty() {
        patterns.push("*".to_string());
    }
    patterns
}

/// Build download patterns for unknown format (download everything except metadata files).
fn build_unknown_patterns(metadata: &HfRepoMetadata) -> Vec<String> {
    let mut patterns = Vec::new();
    for sibling in &metadata.siblings {
        let path = &sibling.rfilename;
        if path == ".gitattributes" || path == ".huggingface" {
            continue;
        }
        patterns.push(path.clone());
    }
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
    fn test_build_single_file_patterns() {
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
            HfSibling {
                rfilename: ".gitattributes".to_string(),
                size: Some(100),
                lfs: None,
            },
        ]);

        let patterns = build_single_file_patterns(&metadata);
        assert!(patterns.contains(&"model.safetensors".to_string()));
        assert!(patterns.contains(&"config.json".to_string()));
        assert!(!patterns.contains(&".gitattributes".to_string()));
    }

    #[test]
    fn test_build_compvis_patterns() {
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

        let patterns = build_compvis_patterns(&metadata);
        assert!(patterns.contains(&"model.ckpt".to_string()));
        assert!(patterns.contains(&"config.yaml".to_string()));
    }

    #[test]
    fn test_build_unknown_patterns() {
        let metadata = make_metadata(vec![
            HfSibling {
                rfilename: "README.md".to_string(),
                size: Some(1000),
                lfs: None,
            },
            HfSibling {
                rfilename: "weights.bin".to_string(),
                size: Some(1_000_000),
                lfs: None,
            },
        ]);

        let patterns = build_unknown_patterns(&metadata);
        assert!(patterns.contains(&"README.md".to_string()));
        assert!(patterns.contains(&"weights.bin".to_string()));
    }

    #[test]
    fn test_build_unknown_patterns_empty_metadata() {
        let metadata = make_metadata(vec![]);
        let patterns = build_unknown_patterns(&metadata);
        assert_eq!(patterns, vec!["*".to_string()]);
    }

    #[tokio::test]
    async fn test_resolve_explicit_patterns_passthrough() {
        let client = hf_hub::HFClientBuilder::new()
            .build()
            .expect("failed to build client");
        let explicit = AllowPatterns::new(vec!["unet/*.safetensors".to_string()]);

        let result = resolve_download_patterns(&client, "test/model", "main", &explicit)
            .await
            .unwrap();

        assert_eq!(result.format, ModelRepoFormat::Unknown);
        assert!(result.component_mapping.is_none());
        assert_eq!(result.patterns.as_slice(), &["unet/*.safetensors"]);
    }
}
