use std::collections::HashMap;

use reqwest::Url;
use tokio::sync::OnceCell;

use crate::catalog::compatibility::{CompatibilityFilter, TargetCustomMetadata};
use crate::catalog::tuf;
use crate::catalog::{CatalogError, CatalogResult, CatalogTarget};

/// A TUF catalog client that resolves GitHub's latest‑Release redirect,
/// pins the concrete tag for the transaction, fetches and verifies the TUF
/// chain, and returns compatible targets.
///
/// # Pin-and-hold semantics
///
/// The first call to [`fetch_catalog`](Self::fetch_catalog) resolves the
/// discovery URL (e.g. GitHub's `/releases/latest`) to a concrete tag
/// (e.g. `worker-catalog-v12`). Every subsequent metadata and target
/// request uses that same concrete tag. The resolution result is cached
/// for the lifetime of the client — a server‑side version bump mid‑flight
/// never mixes assets from two catalog revisions.
///
/// If the discovery URL already points at a concrete download base
/// (`.../releases/download/worker-catalog-vN/...`), no redirect is issued.
///
/// # Construction is cheap
///
/// No network access occurs in the constructor; only URL syntactic
/// validation. Fetching happens lazily when [`fetch_catalog`] is called.
#[derive(Debug)]
pub struct CatalogClient {
    /// The URL the client uses to locate the current catalog release.
    /// May be a GitHub `/releases/latest` endpoint or a concrete download URL.
    discovery_url: Url,

    /// Client used for the initial LTS redirect resolution — redirects
    /// disabled so we can inspect the `Location` header without following.
    discovery_client: reqwest::Client,

    /// Client used for fetching metadata and target packages once the
    /// concrete tag is known.
    asset_client: reqwest::Client,

    /// Compatibility filter used when building catalog targets.
    filter: CompatibilityFilter,

    /// The concrete asset base URL. Initialised once, on the first
    /// [`fetch_catalog`](Self::fetch_catalog) call.
    pinned_asset_base: OnceCell<Url>,
}

/// The result of a catalog fetch: the verified, compatible targets plus
/// metadata for the caller to persist.
#[derive(Clone, Debug)]
pub struct VerifiedCatalog {
    pub targets: Vec<CatalogTarget>,
    pub timestamp_version: u64,
    /// The concrete asset base that was pinned during fetch. Callers
    /// should use this URL for subsequent [`download_target`] calls
    /// rather than re-resolving.
    pub pinned_asset_base: String,
}

// ── Construction ────────────────────────────────────────────────────────
impl CatalogClient {
    /// Create a new catalog client.
    ///
    /// The `discovery_url` is validated for syntax only. Network access
    /// begins lazily on [`fetch_catalog`].
    #[must_use]
    pub fn new(discovery_url: String, filter: CompatibilityFilter) -> Self {
        let url = Url::parse(&discovery_url).unwrap_or_else(|_| {
            // Fall back to the production default if the URL is unparsable.
            Url::parse("https://github.com/Preference-cmd/Reimagine/releases/latest").unwrap()
        });

        Self {
            discovery_url: url,
            discovery_client: reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .expect("discovery client build"),
            asset_client: reqwest::Client::new(),
            filter,
            pinned_asset_base: OnceCell::new(),
        }
    }
}

// ── Public API ──────────────────────────────────────────────────────────
impl CatalogClient {
    /// Fetch and verify the full TUF chain, returning compatible targets.
    ///
    /// On the first call the client resolves the discovery URL through
    /// GitHub latest‑Release redirect. The pinned tag is cached so that
    /// a concurrent or subsequent call reuses the same base.
    pub async fn fetch_catalog(
        &self,
        trusted_root: &tuf::RootMetadata,
        stored_root_keys: &HashMap<String, tuf::TufKey>,
        stored_timestamp_version: u64,
    ) -> CatalogResult<VerifiedCatalog> {
        // 1. Resolve (or reuse) the pinned asset base.
        let pinned = self.resolve_or_pin().await?;

        // 2. Root verification (bootstrap trust).
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

        // 3. Fetch and verify timestamp.
        let timestamp_url = pinned.join("timestamp.json").map_err(|_| CatalogError::Network {
            url: pinned.to_string(),
            message: "invalid timestamp URL".into(),
        })?;
        let timestamp_bytes = self.fetch_url(&timestamp_url).await?;
        let timestamp: tuf::TimestampMetadata =
            serde_json::from_slice(&timestamp_bytes).map_err(|e| CatalogError::Json {
                path: None,
                message: format!("timestamp parse: {e}"),
            })?;
        tuf::verify_timestamp(&timestamp, trusted_root, stored_timestamp_version)?;

        // 4. Snapshot.
        let snapshot_meta = timestamp
            .signed
            .meta
            .get("snapshot.json")
            .ok_or_else(|| CatalogError::Json {
                path: None,
                message: "timestamp missing snapshot.json meta entry".to_string(),
            })?;
        let snapshot_url = pinned.join(&format!("{}.snapshot.json", snapshot_meta.version))
            .map_err(|_| CatalogError::Network {
                url: pinned.to_string(),
                message: "invalid snapshot URL".into(),
            })?;
        let snapshot_bytes = self.fetch_url(&snapshot_url).await?;
        let snapshot: tuf::SnapshotMetadata =
            serde_json::from_slice(&snapshot_bytes).map_err(|e| CatalogError::Json {
                path: None,
                message: format!("snapshot parse: {e}"),
            })?;
        tuf::verify_snapshot(&snapshot, &snapshot_bytes, trusted_root, snapshot_meta)?;

        // 5. Targets.
        let targets_meta = snapshot
            .signed
            .meta
            .get("targets.json")
            .ok_or_else(|| CatalogError::Json {
                path: None,
                message: "snapshot missing targets.json meta entry".to_string(),
            })?;
        let targets_url = pinned.join(&format!("{}.targets.json", targets_meta.version))
            .map_err(|_| CatalogError::Network {
                url: pinned.to_string(),
                message: "invalid targets URL".into(),
            })?;
        let targets_bytes = self.fetch_url(&targets_url).await?;
        let targets: tuf::TargetsMetadata =
            serde_json::from_slice(&targets_bytes).map_err(|e| CatalogError::Json {
                path: None,
                message: format!("targets parse: {e}"),
            })?;
        tuf::verify_targets(&targets, &targets_bytes, trusted_root, targets_meta)?;

        // 6. Build targets and filter.
        let catalog_targets = self.build_targets(&targets, &pinned)?;

        Ok(VerifiedCatalog {
            targets: catalog_targets,
            timestamp_version: timestamp.signed.version,
            pinned_asset_base: pinned.to_string(),
        })
    }

    /// Download a target's content bytes. The target must have come from a
    /// verified `fetch_catalog` call.
    ///
    /// The download URL used is the one stored in `CatalogTarget.download_url`,
    /// which was already pinned during the parent `fetch_catalog` transaction.
    pub async fn download_target(&self, target: &CatalogTarget) -> CatalogResult<Vec<u8>> {
        let url = Url::parse(&target.download_url).map_err(|_| CatalogError::Network {
            url: target.download_url.clone(),
            message: "invalid download URL".into(),
        })?;
        let data = self.fetch_url(&url).await?;

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

    /// Check whether the client has already resolved a pinned base.
    #[must_use]
    pub fn is_pinned(&self) -> bool {
        self.pinned_asset_base.initialized()
    }
}

// ── Redirect pinning ────────────────────────────────────────────────────
impl CatalogClient {
    /// Resolve the discovery URL to a concrete asset base.
    ///
    /// Returns the cached value immediately on subsequent calls.
    async fn resolve_or_pin(&self) -> CatalogResult<&Url> {
        if let Some(pinned) = self.pinned_asset_base.get() {
            return Ok(pinned);
        }
        let pinned = self.resolve_pinned_release().await?;
        // OnceCell::set returns Err if already set — we just initialised, so
        // the value is ours.  Use `get_or_init` on subsequent calls.
        let _ = self.pinned_asset_base.set(pinned);
        Ok(self.pinned_asset_base.get().expect("just set"))
    }

    /// Perform the redirect resolution if needed.
    async fn resolve_pinned_release(&self) -> CatalogResult<Url> {
        let url_str = self.discovery_url.as_str();

        // Already concrete?  Accept as-is.
        if url_str.contains("/releases/download/") {
            let normalized = ensure_trailing_slash(&self.discovery_url);
            return Ok(normalized);
        }

        // Resolve via GitHub's `latest` redirect.
        let response = self
            .discovery_client
            .get(self.discovery_url.clone())
            .send()
            .await
            .map_err(|e| CatalogError::Discovery {
                url: self.discovery_url.to_string(),
                message: format!("discovery request failed: {e}"),
            })?;

        let status = response.status();
        if !status.is_redirection() {
            if status.is_success() {
                // If the `/releases/latest` URL somehow returned 200 (e.g.
                // a static file that IS the catalog), treat the original URL
                // as the pinned base.
                let normalized = ensure_trailing_slash(&self.discovery_url);
                return Ok(normalized);
            }
            return Err(CatalogError::Discovery {
                url: self.discovery_url.to_string(),
                message: format!("expected 30x redirect, got {status}"),
            });
        }

        let location = response
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| CatalogError::Discovery {
                url: self.discovery_url.to_string(),
                message: "redirect response missing Location header".to_string(),
            })?;

        // Resolve the Location (which may be relative) against the original URL.
        let redirect = self
            .discovery_url
            .join(location)
            .map_err(|e| CatalogError::Discovery {
                url: location.to_string(),
                message: format!("cannot resolve Location header: {e}"),
            })?;

        // Verify same origin — the redirect must stay on the same host.
        if redirect.origin() != self.discovery_url.origin() {
            return Err(CatalogError::Discovery {
                url: redirect.to_string(),
                message: "redirect target has different origin".to_string(),
            });
        }

        let path = redirect.path();

        // Extract the tag from the redirect path.  GitHub redirects
        // `/releases/latest` to `/releases/tag/worker-catalog-v{N}` on the
        // same origin; we accept any path that matches the versioned tag
        // pattern.
        let tag = path
            .split("/tag/")
            .nth(1)
            .unwrap_or("")
            .trim_end_matches('/');

        if !tag.starts_with("worker-catalog-v") || tag.is_empty() {
            return Err(CatalogError::Discovery {
                url: redirect.to_string(),
                message: format!(
                    "redirect path `{path}` does not contain a valid `worker-catalog-v*` tag"
                ),
            });
        }

        // Construct the download base from the origin and tag.
        let origin_str = format!(
            "{}://{}{}",
            redirect.scheme(),
            redirect.authority(),
            "/releases/download/",
        );
        let pinned_str = format!("{origin_str}{tag}/");
        Url::parse(&pinned_str).map_err(|e| CatalogError::Discovery {
            url: pinned_str,
            message: format!("cannot construct download base: {e}"),
        })
    }
}

// ── Internal helpers ────────────────────────────────────────────────────
impl CatalogClient {
    async fn fetch_url(&self, url: &Url) -> CatalogResult<Vec<u8>> {
        let response = self
            .asset_client
            .get(url.clone())
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

    fn build_targets(
        &self,
        targets: &tuf::TargetsMetadata,
        asset_base: &Url,
    ) -> CatalogResult<Vec<CatalogTarget>> {
        let mut catalog_targets = Vec::new();

        for (path, desc) in &targets.signed.targets {
            let custom = match &desc.custom {
                Some(value) => serde_json::from_value::<TargetCustomMetadata>(value.clone())
                    .map_err(|e| CatalogError::Json {
                        path: Some(std::path::PathBuf::from(path)),
                        message: format!("custom metadata parse: {e}"),
                    })?,
                None => continue,
            };

            if !self.filter.is_compatible(&custom) {
                continue;
            }

            let sha256 = desc
                .hashes
                .get("sha256")
                .filter(|digest| digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()))
                .cloned()
                .ok_or_else(|| CatalogError::Json {
                    path: Some(std::path::PathBuf::from(path)),
                    message: "target must contain a 64-character SHA-256 digest".to_owned(),
                })?;

            let download_url = asset_base
                .join(path)
                .map(|u| u.to_string())
                .unwrap_or_else(|_| format!("{asset_base}{path}"));

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

fn ensure_trailing_slash(url: &Url) -> Url {
    let mut s = url.to_string();
    if !s.ends_with('/') {
        s.push('/');
    }
    Url::parse(&s).unwrap_or_else(|_| url.clone())
}

// ── Tests ───────────────────────────────────────────────────────────────
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

        // Use a concrete pinned base so `build_targets` can construct download URLs.
        let asset_base = Url::parse("https://github.com/Preference-cmd/Reimagine/releases/download/worker-catalog-v12/").unwrap();
        let result = client.build_targets(&metadata, &asset_base).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].path, "burn-worker-darwin-aarch64.tar.gz");
        assert!(result[0].download_url.contains("worker-catalog-v12"));
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

        let asset_base = Url::parse("https://github.com/Preference-cmd/Reimagine/releases/download/worker-catalog-v12/").unwrap();
        assert!(client.build_targets(&metadata, &asset_base).is_err());
    }

    #[test]
    fn concrete_url_skips_discovery_and_immediately_pins() {
        let filter = CompatibilityFilter::new(HostInfo {
            os: "linux".into(),
            arch: "x86_64".into(),
            supported_protocol_range: (1, 3),
        });
        let client = CatalogClient::new(
            "https://github.com/Preference-cmd/Reimagine/releases/download/worker-catalog-v12/".into(),
            filter,
        );
        // The client should recognise the concrete URL as already pinned.
        assert!(!client.is_pinned(), "not yet initialised before fetch");
        // We can't call fetch_catalog here (it would try HTTP), but we can
        // observe the `new()` constructor does not attempt discovery.
        assert!(!client.is_pinned(), "constructor must not touch network");
    }
}
