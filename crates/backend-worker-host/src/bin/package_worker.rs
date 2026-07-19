//! CI helper: read a compiled worker binary and wrap it into a release package.
//!
//! Usage:
//! ```text
//! reimagine-package-worker \
//!   --binary /path/to/reimagine-inference-burn-worker \
//!   --binary-name reimagine-inference-burn-worker \
//!   --backend-kind burn \
//!   --backend-instance-id "burn:wgpu:default" \
//!   --installation-id "burn-wgpu-darwin-aarch64" \
//!   --target aarch64-apple-darwin \
//!   --version 0.1.0 \
//!   --output /path/to/output.tar.gz
//! ```
//!
//! The tool reads the binary, packages it into a tar.gz archive with
//! manifest, LICENSE, and SBOM, and writes the archive to the output path.
//! It also writes a sidecar `.sha256` file with the archive digest.

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "reimagine-package-worker", about = "Package a compiled worker binary into a release archive")]
struct Args {
    /// Path to the compiled worker binary.
    #[arg(long)]
    binary: PathBuf,

    /// File name for the binary inside the package.
    #[arg(long)]
    binary_name: String,

    /// Worker backend kind (e.g., "burn").
    #[arg(long)]
    backend_kind: String,

    /// Backend instance ID (e.g., "burn:wgpu:default").
    #[arg(long)]
    backend_instance_id: String,

    /// Installation ID (e.g., "burn-wgpu-darwin-aarch64").
    #[arg(long)]
    installation_id: String,

    /// Rust target triple (e.g., "aarch64-apple-darwin").
    #[arg(long)]
    target: String,

    /// Package version (e.g., "0.1.0").
    #[arg(long)]
    version: String,

    /// Output tar.gz path.
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let binary_content = std::fs::read(&args.binary)
        .map_err(|e| format!("failed to read binary `{}`: {e}", args.binary.display()))?;

    let license_path = {
        // Walk up from the binary to find the workspace LICENSE
        let mut candidate = std::env::current_dir()?;
        loop {
            let license = candidate.join("LICENSE");
            if license.exists() {
                break Some(license);
            }
            if !candidate.pop() {
                break None;
            }
        }
    };

    let params = reimagine_backend_worker_host::package::builder::PackageParams {
        backend_kind: args.backend_kind,
        backend_instance_id: args.backend_instance_id,
        installation_id: args.installation_id,
        target: args.target,
        version: args.version,
        package_kind: "burn-worker".to_string(),
        binary_content,
        binary_name: args.binary_name,
        license_path,
    };

    let built = reimagine_backend_worker_host::package::builder::build_package(&params)?;

    // Write the archive
    std::fs::write(&args.output, &built.archive)
        .map_err(|e| format!("failed to write package `{}`: {e}", args.output.display()))?;

    // Write sidecar SHA-256
    let sha_path = {
        let mut p = args.output.clone().into_os_string();
        p.push(".sha256");
        std::path::PathBuf::from(p)
    };
    std::fs::write(&sha_path, &built.sha256)?;

    eprintln!(
        "Packaged worker: {} ({} bytes, sha256: {})",
        args.output.display(),
        built.archive.len(),
        built.sha256,
    );

    Ok(())
}
