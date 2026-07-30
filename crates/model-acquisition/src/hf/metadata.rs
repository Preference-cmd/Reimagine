use crate::error::ModelAcquisitionError;

/// LFS metadata for a file stored with Git LFS.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HfLfsInfo {
    /// Original file size in bytes.
    pub size: u64,
    /// LFS object SHA-256 hash.
    pub sha256: Option<String>,
}

/// A single file entry in a repository's file listing.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HfSibling {
    /// File path relative to the repository root.
    pub rfilename: String,
    /// File size in bytes (populated when file metadata was requested).
    pub size: Option<u64>,
    /// LFS metadata for the file (populated when the file is stored with Git LFS).
    pub lfs: Option<HfLfsInfo>,
}

/// Metadata for a HuggingFace model repository.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HfRepoMetadata {
    /// Repository ID in `owner/name` form.
    pub repo_id: String,
    /// Git revision (branch, tag, or commit SHA).
    pub revision: String,
    /// Files in the repository.
    pub siblings: Vec<HfSibling>,
    /// Hub tags.
    pub tags: Vec<String>,
    /// Primary task tag (e.g., "text-generation", "image-to-text").
    pub pipeline_tag: Option<String>,
    /// Library name (e.g., "diffusers", "transformers").
    pub library_name: Option<String>,
}

impl HfRepoMetadata {
    /// Fetch repository metadata from HuggingFace Hub.
    ///
    /// Uses `client.model(repo_id).info()` to fetch the full file list,
    /// revision, and tags.
    pub async fn fetch(
        client: &hf_hub::HFClient,
        repo_id: &str,
        revision: &str,
    ) -> Result<Self, ModelAcquisitionError> {
        let (owner, name) = match repo_id.split_once('/') {
            Some((owner, name)) => (owner, name),
            None => {
                return Err(ModelAcquisitionError::Hub {
                    repo: repo_id.to_string(),
                    message: "invalid repo_id format, expected 'owner/name'".to_string(),
                });
            }
        };

        let repo = client.model(owner, name);

        let model_info = repo
            .info()
            .revision(revision)
            .expand(vec!["tags".to_string(), "pipeline_tag".to_string()])
            .send()
            .await
            .map_err(|e| ModelAcquisitionError::Hub {
                repo: repo_id.to_string(),
                message: e.to_string(),
            })?;

        let siblings = model_info
            .siblings
            .unwrap_or_default()
            .into_iter()
            .map(|s| HfSibling {
                rfilename: s.rfilename,
                size: s.size,
                lfs: s.lfs.map(|lfs| HfLfsInfo {
                    size: lfs.size.unwrap_or(0),
                    sha256: lfs.sha256,
                }),
            })
            .collect();

        let tags = model_info.tags.unwrap_or_default();

        // Get revision from the sha field if available, otherwise use the requested revision
        let revision_str = model_info.sha.unwrap_or_else(|| revision.to_string());

        Ok(Self {
            repo_id: repo_id.to_string(),
            revision: revision_str,
            siblings,
            tags,
            pipeline_tag: model_info.pipeline_tag,
            library_name: model_info.library_name,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_sibling_construction() {
        let sibling = HfSibling {
            rfilename: "model.safetensors".to_string(),
            size: Some(1_000_000),
            lfs: Some(HfLfsInfo {
                size: 1_000_000,
                sha256: Some("abc123".to_string()),
            }),
        };

        assert_eq!(sibling.rfilename, "model.safetensors");
        assert_eq!(sibling.size, Some(1_000_000));
        assert!(sibling.lfs.is_some());
    }

    #[test]
    fn test_hf_repo_metadata_construction() {
        let metadata = HfRepoMetadata {
            repo_id: "runwayml/stable-diffusion-v1-5".to_string(),
            revision: "abc123".to_string(),
            siblings: vec![
                HfSibling {
                    rfilename: "model_index.json".to_string(),
                    size: None,
                    lfs: None,
                },
                HfSibling {
                    rfilename: "unet/diffusion_pytorch_model.safetensors".to_string(),
                    size: Some(3_440_000_000),
                    lfs: Some(HfLfsInfo {
                        size: 3_440_000_000,
                        sha256: None,
                    }),
                },
            ],
            tags: vec!["diffusers".to_string(), "safetensors".to_string()],
            pipeline_tag: Some("text-to-image".to_string()),
            library_name: Some("diffusers".to_string()),
        };

        assert_eq!(metadata.repo_id, "runwayml/stable-diffusion-v1-5");
        assert_eq!(metadata.siblings.len(), 2);
        assert_eq!(metadata.tags.len(), 2);
        assert_eq!(metadata.pipeline_tag.as_deref(), Some("text-to-image"));
        assert_eq!(metadata.library_name.as_deref(), Some("diffusers"));
    }

    #[test]
    fn test_hf_lfs_info_serde() {
        let lfs = HfLfsInfo {
            size: 12345,
            sha256: Some("def456".to_string()),
        };

        let json = serde_json::to_string(&lfs).unwrap();
        let parsed: HfLfsInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(lfs, parsed);
    }

    #[test]
    fn test_hf_sibling_serde() {
        let sibling = HfSibling {
            rfilename: "config.json".to_string(),
            size: Some(500),
            lfs: None,
        };

        let json = serde_json::to_string(&sibling).unwrap();
        let parsed: HfSibling = serde_json::from_str(&json).unwrap();

        assert_eq!(sibling, parsed);
    }

    #[test]
    fn test_hf_repo_metadata_serde() {
        let metadata = HfRepoMetadata {
            repo_id: "owner/model".to_string(),
            revision: "main".to_string(),
            siblings: vec![],
            tags: vec!["tag1".to_string()],
            pipeline_tag: None,
            library_name: None,
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let parsed: HfRepoMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(metadata, parsed);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use hf_hub::HFClient;

    #[ignore = "requires network access to HuggingFace Hub"]
    #[tokio::test]
    async fn test_fetch_real_metadata() {
        let client = HFClient::builder()
            .build()
            .expect("failed to build HFClient");

        let metadata = HfRepoMetadata::fetch(
            &client,
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            "main",
        )
        .await
        .expect("failed to fetch metadata");

        assert_eq!(
            metadata.repo_id,
            "hf-internal-testing/tiny-stable-diffusion-pipe"
        );
        assert!(!metadata.siblings.is_empty());
        assert!(!metadata.tags.is_empty());
    }
}
