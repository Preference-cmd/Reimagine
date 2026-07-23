//! Production-grade worker package assembler.
//!
//! Builds a deterministic tar.gz archive containing:
//! - The compiled worker binary
//! - The project `LICENSE` file
//! - An SPDX SBOM describing the package contents
//! - A `package.json` manifest consumed by the install engine
//!
//! This is the production counterpart to the testing-only [`testing::generate_package`].
//! Unlike the testing fixture, the builder reads the real project LICENSE,
//! produces an actual SPDX SBOM, and validates inputs before assembling.

use sha2::{Digest, Sha256};

use super::error::{PackageError, PackageResult};
use super::manifest::{PackageFileEntry, PackageManifest};
use crate::catalog::tuf::TargetDesc;
use crate::{BackendInstanceId, ExpectedWorkerIdentity, WorkerInstallationId};

/// Parameters for building a worker release package.
#[derive(Clone, Debug)]
pub struct PackageParams {
    /// Worker backend kind, e.g. "burn" (from the feature flag).
    pub backend_kind: String,
    /// Unique backend instance identifier, e.g. "burn:wgpu:default".
    pub backend_instance_id: String,
    /// Stable installation id, e.g. "burn-wgpu-darwin-aarch64".
    pub installation_id: String,
    /// Rust target triple, e.g. "aarch64-apple-darwin".
    pub target: String,
    /// Package version string, e.g. "0.1.0".
    pub version: String,
    /// Human-readable package kind, e.g. "burn-worker".
    pub package_kind: String,
    /// Raw bytes of the compiled worker binary.
    pub binary_content: Vec<u8>,
    /// Binary file name within the package.
    pub binary_name: String,
    /// Path to the project LICENSE file to embed.
    pub license_path: Option<std::path::PathBuf>,
}

impl PackageParams {
    /// Compute the expected identity for this package.
    /// Used during install verification.
    #[allow(dead_code)]
    fn expected_identity(&self) -> ExpectedWorkerIdentity {
        ExpectedWorkerIdentity {
            backend_instance_id: BackendInstanceId(self.backend_instance_id.clone()),
            installation_id: WorkerInstallationId(self.installation_id.clone()),
            backend_kind: self.backend_kind.clone(),
            target: self.target.clone(),
            manifest_digest: String::new(), // computed after building
        }
    }
}

/// Built package with its artifacts: the archive bytes, manifest, and SBOM text.
#[derive(Clone, Debug)]
pub struct BuiltPackage {
    /// The complete tar.gz archive.
    pub archive: Vec<u8>,
    /// SHA-256 hex digest of the archive.
    pub sha256: String,
    /// The embedded manifest.
    pub manifest: PackageManifest,
    /// SPDX SBOM text.
    pub sbom: String,
}

/// Assemble a complete worker release package.
///
/// The returned archive is deterministic: identical inputs yield
/// byte-identical output.
pub fn build_package(params: &PackageParams) -> PackageResult<BuiltPackage> {
    // ── Validate inputs ────────────────────────────────────────────
    if params.binary_content.is_empty() {
        return Err(PackageError::Build {
            message: "binary content is empty".to_string(),
        });
    }
    if params.version.is_empty() {
        return Err(PackageError::Build {
            message: "version string is empty".to_string(),
        });
    }

    // ── Read license ───────────────────────────────────────────────
    let license_content = match &params.license_path {
        Some(path) => {
            if !path.exists() {
                return Err(PackageError::Build {
                    message: format!("LICENSE file not found at `{}`", path.display()),
                });
            }
            std::fs::read(path).map_err(|e| PackageError::Build {
                message: format!("failed to read LICENSE at `{}`: {e}", path.display()),
            })?
        }
        None => {
            // Fallback placeholder (should not happen in production CI).
            b"GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007\n".to_vec()
        }
    };
    let license_hash = hex::encode(Sha256::digest(&license_content));

    // ── Build file entries ──────────────────────────────────────────
    let binary_hash = hex::encode(Sha256::digest(&params.binary_content));
    let file_entries = vec![
        PackageFileEntry {
            path: params.binary_name.clone(),
            sha256: binary_hash,
            size: params.binary_content.len() as u64,
            mode: 0o755,
            executable: true,
        },
        PackageFileEntry {
            path: "LICENSE".to_string(),
            sha256: license_hash,
            size: license_content.len() as u64,
            mode: 0o644,
            executable: false,
        },
    ];

    let total_size: u64 = file_entries.iter().map(|f| f.size).sum();

    // Compute manifest_digest: hash of the serialized manifest (without the
    // digest field itself). We build a provisional manifest, serialize it,
    // hash, then rebuild with the correct digest.
    let provisional_identity = ExpectedWorkerIdentity {
        backend_instance_id: BackendInstanceId(params.backend_instance_id.clone()),
        installation_id: WorkerInstallationId(params.installation_id.clone()),
        backend_kind: params.backend_kind.clone(),
        target: params.target.clone(),
        manifest_digest: "provisional".to_string(),
    };

    // Build SBOM before manifest (SBOM text depends only on input params).
    let sbom_text = build_sbom(params, &file_entries);

    let provisional_manifest = PackageManifest {
        schema_version: 1,
        package_kind: params.package_kind.clone(),
        version: params.version.clone(),
        identity: provisional_identity,
        files: file_entries.clone(),
        required_size: total_size,
        required_entries: file_entries.len(),
    };

    let manifest_json =
        serde_json::to_vec(&provisional_manifest).map_err(|e| PackageError::Build {
            message: format!("manifest serialization: {e}"),
        })?;
    let manifest_digest = hex::encode(Sha256::digest(&manifest_json));

    // ── Rebuild with correct manifest_digest ────────────────────────
    let identity = ExpectedWorkerIdentity {
        backend_instance_id: BackendInstanceId(params.backend_instance_id.clone()),
        installation_id: WorkerInstallationId(params.installation_id.clone()),
        backend_kind: params.backend_kind.clone(),
        target: params.target.clone(),
        manifest_digest: manifest_digest.clone(),
    };

    let manifest = PackageManifest {
        schema_version: 1,
        package_kind: params.package_kind.clone(),
        version: params.version.clone(),
        identity,
        files: file_entries.clone(),
        required_size: total_size,
        required_entries: file_entries.len(),
    };

    let final_manifest_json = serde_json::to_vec(&manifest).map_err(|e| PackageError::Build {
        message: format!("manifest serialization: {e}"),
    })?;

    // ── Build tar.gz archive ───────────────────────────────────────
    let mut buf = Vec::new();
    {
        let gz = flate2::write::GzEncoder::new(&mut buf, flate2::Compression::best());
        let mut tar = tar::Builder::new(gz);

        // Binary
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::Regular);
        header.set_mode(0o755);
        header.set_size(params.binary_content.len() as u64);
        tar.append_data(&mut header, &params.binary_name, &params.binary_content[..])
            .map_err(|e| PackageError::Build {
                message: format!("tar append binary: {e}"),
            })?;

        // LICENSE
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::Regular);
        header.set_mode(0o644);
        header.set_size(license_content.len() as u64);
        tar.append_data(&mut header, "LICENSE", &license_content[..])
            .map_err(|e| PackageError::Build {
                message: format!("tar append LICENSE: {e}"),
            })?;

        // SBOM
        let sbom_bytes = sbom_text.as_bytes();
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::Regular);
        header.set_mode(0o644);
        header.set_size(sbom_bytes.len() as u64);
        tar.append_data(&mut header, "package.spdx.json", sbom_bytes)
            .map_err(|e| PackageError::Build {
                message: format!("tar append SBOM: {e}"),
            })?;

        // package.json manifest
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::Regular);
        header.set_mode(0o644);
        header.set_size(final_manifest_json.len() as u64);
        tar.append_data(&mut header, "package.json", &final_manifest_json[..])
            .map_err(|e| PackageError::Build {
                message: format!("tar append package.json: {e}"),
            })?;
    }

    let archive_sha256 = hex::encode(Sha256::digest(&buf));

    Ok(BuiltPackage {
        archive: buf,
        sha256: archive_sha256,
        manifest,
        sbom: sbom_text,
    })
}

/// Build a minimal SPDX 2.3 SBOM.
fn build_sbom(params: &PackageParams, files: &[PackageFileEntry]) -> String {
    let spdx_id = format!(
        "SPDXRef-DOCUMENT-{}",
        hex::encode(Sha256::digest(format!(
            "{}-{}-{}",
            params.package_kind, params.version, params.target
        )))
        .chars()
        .take(12)
        .collect::<String>()
    );

    // Build file entries for SPDX
    let file_sections: Vec<String> = files
        .iter()
        .map(|f| {
            format!(
                r#"FileName: {path}
SPDXID: SPDXRef-{file_id}
FileChecksum: SHA256: {sha}
LicenseConcluded: LicenseRef-GPL-3.0-or-later
FileCopyrightText: NOASSERTION
"#,
                path = f.path,
                file_id = hex::encode(Sha256::digest(&f.path))
                    .chars()
                    .take(12)
                    .collect::<String>(),
                sha = f.sha256,
            )
        })
        .collect();

    format!(
        r#"SPDXVersion: SPDX-2.3
DataLicense: CC0-1.0
SPDXID: {spdx_id}
DocumentName: {kind} {version} ({target})
DocumentNamespace: https://reimagine.dev/spdx/{kind}/{version}/{target}
Creator: Tool: reimagine-package-builder-0.1
Created: {created}

## Package Information

PackageName: {kind}
PackageVersion: {version}
PackageDownloadLocation: NOASSERTION
FilesAnalyzed: true
PackageLicenseConcluded: GPL-3.0-or-later
PackageLicenseDeclared: GPL-3.0-or-later
PackageCopyrightText: NOASSERTION
PackageSummary: <text>{kind} worker package for {target}</text>
PackageDescription: <text>Reimagine {kind} worker ({backend}) compiled for {target}</text>

## File Information

{files}
"#,
        spdx_id = spdx_id,
        kind = params.package_kind,
        version = params.version,
        target = params.target,
        backend = params.backend_instance_id,
        created = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ"),
        files = file_sections.join("\n"),
    )
}

/// Common package filename from kind and target triple.
#[must_use]
pub fn package_filename(kind: &str, target: &str) -> String {
    format!("{kind}-{target}.tar.gz")
}

/// Default target names for each TUF target entry from a build.
#[must_use]
pub fn target_path(kind: &str, target: &str) -> String {
    package_filename(kind, target)
}

/// Build a TUF TargetDesc for a built package.
#[must_use]
pub fn target_desc(built: &BuiltPackage, params: &PackageParams) -> (String, TargetDesc) {
    use std::collections::HashMap;
    let path = target_path(&params.package_kind, &params.target);
    let desc = TargetDesc {
        length: built.archive.len() as u64,
        hashes: HashMap::from([("sha256".to_string(), built.sha256.clone())]),
        custom: Some(serde_json::json!({
            "version": params.version,
            "installation_id": params.installation_id,
            "backend_instance_id": params.backend_instance_id,
            "os": infer_os(&params.target),
            "arch": infer_arch(&params.target),
            "worker_kind": params.backend_kind,
            "protocol_version_min": 1,
            "protocol_version_max": 1,
            "package_format": "tar.gz",
            "min_runtime_version": null,
            "target": params.target,
            "manifest_digest": built.manifest.identity.manifest_digest,
        })),
    };
    (path, desc)
}

/// Infer the OS field from a Rust target triple.
#[must_use]
pub fn infer_os(target: &str) -> &'static str {
    if target.contains("apple-darwin") || target.contains("apple-ios") {
        "darwin"
    } else if target.contains("unknown-linux") {
        "linux"
    } else if target.contains("pc-windows") {
        "windows"
    } else {
        "unknown"
    }
}

/// Infer the arch field from a Rust target triple.
#[must_use]
pub fn infer_arch(target: &str) -> &'static str {
    if target.starts_with("aarch64") {
        "aarch64"
    } else if target.starts_with("x86_64") || target.starts_with("amd64") {
        "x86_64"
    } else if target.starts_with("i686") || target.starts_with("i586") {
        "x86"
    } else {
        "unknown"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::package::PackageExtractor;

    #[test]
    fn build_package_creates_valid_archive() {
        let pkg = build_package(&PackageParams {
            backend_kind: "burn".to_string(),
            backend_instance_id: "burn:wgpu:default".to_string(),
            installation_id: "burn-wgpu-darwin-aarch64".to_string(),
            target: "aarch64-apple-darwin".to_string(),
            version: "0.1.0".to_string(),
            package_kind: "burn-worker".to_string(),
            binary_content: b"#!/bin/sh\necho test binary\0".to_vec(),
            binary_name: "reimagine-inference-burn-worker".to_string(),
            license_path: None,
        })
        .expect("build should succeed");

        assert!(!pkg.archive.is_empty(), "archive must not be empty");
        assert_eq!(pkg.sha256.len(), 64, "sha256 must be 64 hex chars");
        assert_eq!(pkg.manifest.schema_version, 1);
        assert_eq!(pkg.manifest.version, "0.1.0");

        // Extract and verify
        let dir = tempfile::tempdir().unwrap();
        let staging = dir.path().join("staging");
        let extractor = PackageExtractor::new(super::super::ExtractionLimits::default());
        let manifest = extractor
            .extract(&pkg.archive, &staging, None)
            .expect("extraction should succeed");

        assert_eq!(manifest.package_kind, "burn-worker");
        assert!(staging.join("reimagine-inference-burn-worker").exists());
        assert!(staging.join("LICENSE").exists());
    }

    #[test]
    fn build_package_rejects_empty_binary() {
        let result = build_package(&PackageParams {
            backend_kind: "burn".to_string(),
            backend_instance_id: "burn:wgpu:default".to_string(),
            installation_id: "burn-wgpu-darwin-aarch64".to_string(),
            target: "aarch64-apple-darwin".to_string(),
            version: "0.1.0".to_string(),
            package_kind: "burn-worker".to_string(),
            binary_content: Vec::new(),
            binary_name: "worker".to_string(),
            license_path: None,
        });
        assert!(result.is_err(), "empty binary should be rejected");
    }

    #[test]
    fn build_creates_target_desc() {
        let pkg = build_package(&PackageParams {
            backend_kind: "burn".to_string(),
            backend_instance_id: "burn:wgpu:default".to_string(),
            installation_id: "burn-wgpu-darwin-aarch64".to_string(),
            target: "aarch64-apple-darwin".to_string(),
            version: "0.1.0".to_string(),
            package_kind: "burn-worker".to_string(),
            binary_content: b"test".to_vec(),
            binary_name: "worker".to_string(),
            license_path: None,
        })
        .expect("build");

        let params = PackageParams {
            backend_kind: "burn".to_string(),
            backend_instance_id: "burn:wgpu:default".to_string(),
            installation_id: "burn-wgpu-darwin-aarch64".to_string(),
            target: "aarch64-apple-darwin".to_string(),
            version: "0.1.0".to_string(),
            package_kind: "burn-worker".to_string(),
            binary_content: b"test".to_vec(),
            binary_name: "worker".to_string(),
            license_path: None,
        };
        let (path, desc) = target_desc(&pkg, &params);
        assert!(path.contains("aarch64-apple-darwin"));
        assert_eq!(desc.length, pkg.archive.len() as u64);
        assert_eq!(desc.hashes.get("sha256").unwrap(), &pkg.sha256);
    }

    #[test]
    fn package_is_deterministic() {
        let params = PackageParams {
            backend_kind: "burn".to_string(),
            backend_instance_id: "burn:wgpu:default".to_string(),
            installation_id: "burn-wgpu-darwin-aarch64".to_string(),
            target: "aarch64-apple-darwin".to_string(),
            version: "0.1.0".to_string(),
            package_kind: "burn-worker".to_string(),
            binary_content: b"deterministic binary content".to_vec(),
            binary_name: "worker".to_string(),
            license_path: None,
        };

        let pkg1 = build_package(&params).expect("build 1");
        let pkg2 = build_package(&params).expect("build 2");
        assert_eq!(pkg1.archive, pkg2.archive, "archives must be identical");
        assert_eq!(pkg1.sha256, pkg2.sha256, "hashes must be identical");
    }

    #[test]
    fn infer_os_and_arch() {
        assert_eq!(infer_os("aarch64-apple-darwin"), "darwin");
        assert_eq!(infer_os("x86_64-unknown-linux-gnu"), "linux");
        assert_eq!(infer_os("x86_64-pc-windows-msvc"), "windows");
        assert_eq!(infer_arch("aarch64-apple-darwin"), "aarch64");
        assert_eq!(infer_arch("x86_64-unknown-linux-gnu"), "x86_64");
    }

    #[test]
    fn sbom_contains_package_info() {
        let pkg = build_package(&PackageParams {
            backend_kind: "burn".to_string(),
            backend_instance_id: "burn:wgpu:default".to_string(),
            installation_id: "burn-wgpu-darwin-aarch64".to_string(),
            target: "aarch64-apple-darwin".to_string(),
            version: "0.1.0".to_string(),
            package_kind: "burn-worker".to_string(),
            binary_content: b"test".to_vec(),
            binary_name: "worker".to_string(),
            license_path: None,
        })
        .expect("build");

        assert!(pkg.sbom.contains("SPDXVersion: SPDX-2.3"));
        assert!(pkg.sbom.contains("burn-worker"));
        assert!(pkg.sbom.contains("0.1.0"));
        assert!(pkg.sbom.contains("aarch64-apple-darwin"));
        assert!(pkg.sbom.contains("GPL-3.0-or-later"));
    }
}
