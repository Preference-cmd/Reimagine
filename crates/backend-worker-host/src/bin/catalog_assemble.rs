//! CI helper: assemble a TUF catalog from a set of release packages and verify it.
//!
//! Usage:
//! ```text
//! reimagine-catalog-assemble \
//!   --output-dir /path/to/catalog \
//!   --package "burn-worker-aarch64-apple-darwin.tar.gz@burn:wgpu:darwin@burn-wgpu-darwin@aarch64-apple-darwin@darwin" \
//!   --package "burn-worker-x86_64-unknown-linux-gnu.tar.gz@burn:wgpu:linux@burn-wgpu-linux@x86_64-unknown-linux-gnu@linux" ...
//! ```
//!
//! Each `--package` argument uses `@`-delimited fields:
//!   {filename}@{backend_instance_id}@{installation_id}@{target}@{os}
//!
//! The tool:
//! 1. Reads each package archive and computes its hash
//! 2. Builds a complete TUF metadata chain with deterministic test keys
//! 3. Verifies the chain
//! 4. Writes the catalog to the output directory
//! 5. Prints a human-readable summary
//!
//! Exit code is non-zero if verification fails.

use std::collections::HashMap;
use std::path::PathBuf;

use clap::Parser;

use reimagine_backend_worker_host::catalog::builder::{
    CatalogParams, TestSigningKey, build_catalog, verify_catalog, write_catalog,
};
use reimagine_backend_worker_host::catalog::tuf::TargetDesc;
use sha2::{Digest, Sha256};

/// A parsed `--package` argument.
#[derive(Debug)]
struct PackageEntry {
    path: String,
    backend_instance_id: String,
    installation_id: String,
    target: String,
    os: String,
    arch: String,
}

fn parse_package(s: &str) -> PackageEntry {
    let parts: Vec<&str> = s.split('@').collect();
    if parts.len() < 5 {
        eprintln!("error: --package must have at least 5 @-delimited fields: got {parts:?}");
        std::process::exit(1);
    }
    let arch = if parts[3].starts_with("aarch64") {
        "aarch64"
    } else if parts[3].starts_with("x86_64") {
        "x86_64"
    } else {
        "unknown"
    };
    PackageEntry {
        path: parts[0].to_string(),
        backend_instance_id: parts[1].to_string(),
        installation_id: parts[2].to_string(),
        target: parts[3].to_string(),
        os: parts[4].to_string(),
        arch: arch.to_string(),
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "reimagine-catalog-assemble",
    about = "Assemble and verify a TUF catalog from worker packages"
)]
struct Args {
    /// Output directory for the catalog metadata and package files.
    #[arg(long)]
    output_dir: PathBuf,

    /// A package to include in the catalog.
    /// Format: {filename}@{backend_instance_id}@{installation_id}@{target}@{os}
    #[arg(long, required = true)]
    package: Vec<String>,

    /// Root metadata version (default: 1).
    #[arg(long, default_value = "1")]
    root_version: u64,

    /// Targets metadata version (default: 1).
    #[arg(long, default_value = "1")]
    targets_version: u64,

    /// Snapshot metadata version (default: 1).
    #[arg(long, default_value = "1")]
    snapshot_version: u64,

    /// Timestamp metadata version (default: 1).
    #[arg(long, default_value = "1")]
    timestamp_version: u64,

    /// Expiry date (ISO 8601).
    #[arg(long, default_value = "2999-12-31T23:59:59Z")]
    expires: String,

    /// Input directory containing the package files.
    #[arg(long)]
    input_dir: PathBuf,

    /// Package kind string (default: burn-worker).
    #[arg(long, default_value = "burn-worker")]
    package_kind: String,

    /// Package version string (default: 0.1.0).
    #[arg(long, default_value = "0.1.0")]
    version: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let packages: Vec<PackageEntry> = args.package.iter().map(|s| parse_package(s)).collect();

    if packages.is_empty() {
        eprintln!("error: at least one --package is required");
        std::process::exit(1);
    }

    // ── Read and hash each package ─────────────────────────────────
    let mut target_entries: Vec<(String, TargetDesc)> = Vec::new();
    let mut archive_data: Vec<(String, Vec<u8>)> = Vec::new();

    for pkg in &packages {
        let full_path = args.input_dir.join(&pkg.path);
        let data = std::fs::read(&full_path)
            .map_err(|e| format!("failed to read package `{}`: {e}", full_path.display()))?;

        if data.is_empty() {
            eprintln!("error: package `{}` is empty", full_path.display());
            std::process::exit(1);
        }

        let sha256 = hex::encode(Sha256::digest(&data));

        let custom = serde_json::json!({
            "version": args.version,
            "installation_id": pkg.installation_id,
            "backend_instance_id": pkg.backend_instance_id,
            "os": pkg.os,
            "arch": pkg.arch,
            "worker_kind": "burn",
            "protocol_version_min": 1,
            "protocol_version_max": 1,
            "package_format": "tar.gz",
            "min_runtime_version": null,
            "target": pkg.target,
            "manifest_digest": "ci-dry-run",
        });

        let desc = TargetDesc {
            length: data.len() as u64,
            hashes: HashMap::from([("sha256".to_string(), sha256)]),
            custom: Some(custom),
        };

        target_entries.push((pkg.path.clone(), desc));
        archive_data.push((pkg.path.clone(), data));
    }

    // ── Build catalog ──────────────────────────────────────────────
    let online = TestSigningKey::new();
    let root_key = TestSigningKey::new();

    let params = CatalogParams {
        root: None,
        root_version: args.root_version,
        targets_version: args.targets_version,
        snapshot_version: args.snapshot_version,
        timestamp_version: args.timestamp_version,
        expires: args.expires,
        online_provider: Box::new(online),
        root_provider: Box::new(root_key),
    };

    let bundle = build_catalog(&params, &target_entries);

    // ── Verify ─────────────────────────────────────────────────────
    verify_catalog(&bundle)?;

    // ── Write ──────────────────────────────────────────────────────
    let refs: Vec<(&str, &[u8])> = archive_data
        .iter()
        .map(|(path, data)| (path.as_str(), data.as_slice()))
        .collect();
    write_catalog(&bundle, &args.output_dir, &refs)?;

    // ── Summary ────────────────────────────────────────────────────
    println!("=== Catalog Assembly Summary ===");
    println!("Packages: {}", packages.len());
    println!("Output: {}", args.output_dir.display());
    println!("Root metadata version: {}", bundle.root.signed.version);
    println!(
        "Targets metadata version: {}",
        bundle.targets.signed.version
    );
    println!("Targets count: {}", bundle.targets.signed.targets.len());
    for (path, target) in &bundle.targets.signed.targets {
        let sha = target
            .hashes
            .get("sha256")
            .map(|s| &s[..16])
            .unwrap_or("???");
        println!("  {} ({} bytes, sha256: {}...)", path, target.length, sha);
    }
    println!(
        "Snapshot metadata version: {}",
        bundle.snapshot.signed.version
    );
    println!(
        "Timestamp metadata version: {}",
        bundle.timestamp.signed.version
    );
    println!("Verify: PASSED");

    Ok(())
}
