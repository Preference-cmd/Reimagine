use std::collections::HashMap;

use crate::catalog::compatibility::{CompatibilityFilter, TargetCustomMetadata};
use crate::catalog::tuf;
use crate::catalog::{CatalogError, CatalogResult, CatalogTarget};

/// A TUF catalog client that fetches metadata, verifies the TUF chain,
/// and returns verified compatible targets.
///
/// The client is stateless and thread-safe: all mutable verification
/// state is passed explicitly through method parameters.
#[derive(Clone, Debug)]
pub struct CatalogClient {
    base_url: String,
    http_client: reqwest::Client,
    filter: CompatibilityFilter,
}

#[derive(Clone, Debug)]
pub struct VerifiedCatalog {
    pub targets: Vec<CatalogTarget>,
    pub timestamp_version: u64,
}

impl CatalogClient {
    /// Create a new catalog client.
    #[must_use]
    pub fn new(base_url: String, filter: CompatibilityFilter) -> Self {
        Self {
            base_url,
            http_client: reqwest::Client::new(),
            filter,
        }
    }

    /// Fetch and verify the full TUF chain, returning compatible targets.
    ///
    /// This performs the following steps:
    /// 1. Fetch and verify timestamp metadata (against stored root keys)
    /// 2. Fetch and verify snapshot metadata (against timestamp linkage)
    /// 3. Fetch and verify targets metadata (against snapshot linkage)
    /// 4. Filter targets by compatibility
    /// 5. Return verified, compatible targets
    pub async fn fetch_catalog(
        &self,
        trusted_root: &tuf::RootMetadata,
        stored_root_keys: &HashMap<String, tuf::TufKey>,
        stored_timestamp_version: u64,
    ) -> CatalogResult<VerifiedCatalog> {
        // 1. Root verification (bootstrap trust)
        let root_keys = tuf::verify_root(trusted_root, None)?;
        if !stored_root_keys.is_empty()
            && stored_root_keys
                .keys()
                .any(|id| !root_keys.contains_key(id))
        {
            return Err(CatalogError::RootLoad {
                message: "embedded root keys do not contain the stored trusted keys".to_owned(),
            });
        }

        // 2. Fetch and verify timestamp
        let timestamp_url = format!("{}/timestamp.json", self.base_url);
        let timestamp_bytes = self.fetch_url(&timestamp_url).await?;
        let timestamp: tuf::TimestampMetadata =
            serde_json::from_slice(&timestamp_bytes).map_err(|e| CatalogError::Json {
                path: None,
                message: format!("timestamp parse: {e}"),
            })?;
        tuf::verify_timestamp(&timestamp, trusted_root, stored_timestamp_version)?;

        // Extract snapshot version from timestamp
        let snapshot_meta =
            timestamp
                .signed
                .meta
                .get("snapshot.json")
                .ok_or_else(|| CatalogError::Json {
                    path: None,
                    message: "timestamp missing snapshot.json meta entry".to_string(),
                })?;

        // 3. Fetch and verify snapshot
        let snapshot_url = self.meta_url("snapshot.json", snapshot_meta.version);
        let snapshot_bytes = self.fetch_url(&snapshot_url).await?;
        let snapshot: tuf::SnapshotMetadata =
            serde_json::from_slice(&snapshot_bytes).map_err(|e| CatalogError::Json {
                path: None,
                message: format!("snapshot parse: {e}"),
            })?;
        tuf::verify_snapshot(&snapshot, &snapshot_bytes, trusted_root, snapshot_meta)?;

        // Extract targets version from snapshot
        let targets_meta =
            snapshot
                .signed
                .meta
                .get("targets.json")
                .ok_or_else(|| CatalogError::Json {
                    path: None,
                    message: "snapshot missing targets.json meta entry".to_string(),
                })?;

        // 4. Fetch and verify targets
        let targets_url = self.meta_url("targets.json", targets_meta.version);
        let targets_bytes = self.fetch_url(&targets_url).await?;
        let targets: tuf::TargetsMetadata =
            serde_json::from_slice(&targets_bytes).map_err(|e| CatalogError::Json {
                path: None,
                message: format!("targets parse: {e}"),
            })?;
        tuf::verify_targets(&targets, &targets_bytes, trusted_root, targets_meta)?;

        // 5. Build catalog targets and filter by compatibility
        let catalog_targets = self.build_targets(&targets, &root_keys)?;

        Ok(VerifiedCatalog {
            targets: catalog_targets,
            timestamp_version: timestamp.signed.version,
        })
    }

    /// Download a target's content bytes. The target must have come
    /// from a verified `fetch_catalog` call.
    pub async fn download_target(&self, target: &CatalogTarget) -> CatalogResult<Vec<u8>> {
        let data = self.fetch_url(&target.download_url).await?;

        // Verify hash and length match the TUF target description
        let hash_hex = {
            use sha2::Digest;
            hex::encode(sha2::Sha256::digest(&data))
        };
        if hash_hex != target.sha256 {
            return Err(CatalogError::TargetHashMismatch {
                target: target.path.clone(),
                algorithm: "sha256".to_string(),
            });
        }
        if data.len() as u64 != target.length {
            return Err(CatalogError::TargetLengthMismatch {
                target: target.path.clone(),
                expected: target.length,
                actual: data.len() as u64,
            });
        }

        Ok(data)
    }

    // ── Internal helpers ──────────────────────────────────────────

    async fn fetch_url(&self, url: &str) -> CatalogResult<Vec<u8>> {
        let response =
            self.http_client
                .get(url)
                .send()
                .await
                .map_err(|e| CatalogError::Network {
                    url: url.to_string(),
                    message: e.to_string(),
                })?;

        if !response.status().is_success() {
            return Err(CatalogError::Network {
                url: url.to_string(),
                message: format!("HTTP {}", response.status()),
            });
        }

        response
            .bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| CatalogError::Network {
                url: url.to_string(),
                message: e.to_string(),
            })
    }

    /// Build a metadata URL, optionally with version prefix for
    /// consistent snapshots.
    fn meta_url(&self, name: &str, version: u64) -> String {
        format!("{}/{}.{}", self.base_url, version, name)
    }

    fn build_targets(
        &self,
        targets: &tuf::TargetsMetadata,
        _root_keys: &HashMap<String, tuf::TufKey>,
    ) -> CatalogResult<Vec<CatalogTarget>> {
        let mut catalog_targets = Vec::new();

        for (path, desc) in &targets.signed.targets {
            // Extract custom metadata
            let custom = match &desc.custom {
                Some(value) => serde_json::from_value::<TargetCustomMetadata>(value.clone())
                    .map_err(|e| CatalogError::Json {
                        path: Some(std::path::PathBuf::from(path)),
                        message: format!("custom metadata parse: {e}"),
                    })?,
                None => continue, // No custom metadata = not installable
            };

            // Filter by compatibility
            if !self.filter.is_compatible(&custom) {
                continue;
            }

            // Get SHA-256 hash
            let sha256 = desc
                .hashes
                .get("sha256")
                .filter(|digest| {
                    digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
                })
                .cloned()
                .ok_or_else(|| CatalogError::Json {
                    path: Some(std::path::PathBuf::from(path)),
                    message: "target must contain a 64-character SHA-256 digest".to_owned(),
                })?;

            let download_url = format!("{}/{}", self.base_url, path);

            catalog_targets.push(CatalogTarget {
                path: path.clone(),
                sha256,
                length: desc.length,
                custom,
                download_url,
            });
        }

        Ok(catalog_targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::compatibility::HostInfo;
    use crate::catalog::tuf::TargetDesc;

    #[test]
    fn build_targets_filters_by_compatibility() {
        let filter = CompatibilityFilter::new(HostInfo {
            os: "darwin".to_string(),
            arch: "aarch64".to_string(),
            supported_protocol_range: (1, 3),
        });
        let client = CatalogClient::new("https://example.com".to_string(), filter);

        let mut targets_map = HashMap::new();
        targets_map.insert(
            "burn-worker-darwin-aarch64.tar.gz".to_string(),
            TargetDesc {
                length: 1000,
                hashes: {
                    let mut m = HashMap::new();
                    m.insert("sha256".to_string(), "ab".repeat(32));
                    m
                },
                custom: Some(serde_json::json!({
                    "version": "1.0.0",
                    "installation_id": "burn-wgpu-v1",
                    "backend_instance_id": "burn:wgpu:default",
                    "os": "darwin",
                    "arch": "aarch64",
                    "worker_kind": "burn",
                    "protocol_version_min": 1,
                    "protocol_version_max": 3,
                    "package_format": "tar.gz",
                    "min_runtime_version": null,
                    "target": "aarch64-apple-darwin",
                    "manifest_digest": "digest",
                })),
            },
        );
        targets_map.insert(
            "burn-worker-linux-x86_64.tar.gz".to_string(),
            TargetDesc {
                length: 2000,
                hashes: {
                    let mut m = HashMap::new();
                    m.insert("sha256".to_string(), "def".to_string());
                    m
                },
                custom: Some(serde_json::json!({
                    "version": "1.0.0",
                    "installation_id": "burn-wgpu-linux-v1",
                    "backend_instance_id": "burn:wgpu:default",
                    "os": "linux",
                    "arch": "x86_64",
                    "worker_kind": "burn",
                    "protocol_version_min": 1,
                    "protocol_version_max": 3,
                    "package_format": "tar.gz",
                    "min_runtime_version": null,
                    "target": "x86_64-unknown-linux-gnu",
                    "manifest_digest": "digest",
                })),
            },
        );

        let metadata = tuf::TargetsMetadata {
            signed: tuf::TargetsSigned {
                kind: "targets".to_string(),
                version: 1,
                expires: "2999-01-01T00:00:00Z".to_string(),
                targets: targets_map,
            },
            signatures: Vec::new(),
        };

        let result = client.build_targets(&metadata, &HashMap::new()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].path, "burn-worker-darwin-aarch64.tar.gz");
    }

    #[test]
    fn build_targets_rejects_missing_sha256() {
        let filter = CompatibilityFilter::new(HostInfo {
            os: "darwin".to_owned(),
            arch: "aarch64".to_owned(),
            supported_protocol_range: (1, 3),
        });
        let client = CatalogClient::new("https://example.com".to_owned(), filter);
        let mut targets_map = HashMap::new();
        targets_map.insert(
            "worker.tar.gz".to_owned(),
            TargetDesc {
                length: 1,
                hashes: HashMap::new(),
                custom: Some(serde_json::json!({
                    "version": "1.0.0",
                    "installation_id": "burn-wgpu-v1",
                    "backend_instance_id": "burn:wgpu:default",
                    "os": "darwin",
                    "arch": "aarch64",
                    "worker_kind": "burn",
                    "protocol_version_min": 1,
                    "protocol_version_max": 3,
                    "package_format": "tar.gz",
                    "min_runtime_version": null,
                    "target": "aarch64-apple-darwin",
                    "manifest_digest": "digest"
                })),
            },
        );
        let metadata = tuf::TargetsMetadata {
            signed: tuf::TargetsSigned {
                kind: "targets".to_owned(),
                version: 1,
                expires: "2035-01-01T00:00:00Z".to_owned(),
                targets: targets_map,
            },
            signatures: Vec::new(),
        };

        assert!(client.build_targets(&metadata, &HashMap::new()).is_err());
    }
}
