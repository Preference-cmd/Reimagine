//! Standalone dev binary to convert an original SDXL checkpoint into
//! Candle-example-compatible split components.
//!
//! Usage:
//!   cargo run --bin import-sdxl -- /path/to/sd_xl_base_1.0.safetensors /tmp/split-out/
//!
//! Output layout:
//!   <output-root>/
//!     <source-model-id>/<source-fingerprint>/
//!       unet/model.safetensors
//!       text_encoder/model.safetensors
//!       text_encoder_2/model.safetensors
//!       vae/model.safetensors
//!       tokenizer/tokenizer.json
//!       tokenizer_2/tokenizer.json
//!       conversion.json

use std::path::{Path, PathBuf};

use reimagine_inference_candle::{
    SdxlCheckpointImportRequest, SdxlConvertedComponent,
    import_sdxl_checkpoint_to_candle_example_split,
};

fn usage() -> ! {
    eprintln!("Usage: import-sdxl <checkpoint.safetensors> <output-root>");
    eprintln!();
    eprintln!(
        "Writes <output-root>/<source-model-id>/<source-fingerprint>/<component>/model.safetensors"
    );
    std::process::exit(1);
}

fn source_model_id_from_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("sdxl-checkpoint")
        .to_owned()
}

fn source_fingerprint_from_path(path: &Path) -> String {
    // Stable enough for a dev tool: two checkpoints with the same byte size
    // but different content would share a cache directory. Users can delete
    // the output dir if they collide. We intentionally avoid a slow SHA-256
    // pass over multi-gigabyte files here.
    let metadata = std::fs::metadata(path).unwrap_or_else(|error| {
        eprintln!(
            "failed to read checkpoint metadata for {}: {error}",
            path.display()
        );
        std::process::exit(1);
    });
    format!("size-{}", metadata.len())
}

fn main() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap_or_else(|error| {
            eprintln!("failed to create tokio runtime: {error}");
            std::process::exit(1);
        });
    runtime.block_on(async_main());
}

async fn async_main() {
    let mut args = std::env::args_os().skip(1);
    let checkpoint = args.next().map(PathBuf::from).unwrap_or_else(|| usage());
    let output_dir = args.next().map(PathBuf::from).unwrap_or_else(|| usage());
    if args.next().is_some() {
        usage();
    }

    if !checkpoint.exists() {
        eprintln!("checkpoint does not exist: {}", checkpoint.display());
        std::process::exit(1);
    }

    let source_model_id = source_model_id_from_path(&checkpoint);
    let source_fingerprint = source_fingerprint_from_path(&checkpoint);
    let source_format = checkpoint
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("safetensors")
        .to_ascii_lowercase();

    let request = SdxlCheckpointImportRequest::new(
        &source_model_id,
        &checkpoint,
        &source_fingerprint,
        &source_format,
        &output_dir,
    );

    match import_sdxl_checkpoint_to_candle_example_split(request).await {
        Ok(result) => {
            if result.reused_existing() {
                println!(
                    "Reused existing conversion at {}",
                    result.conversion_dir().display()
                );
            } else {
                println!("Wrote conversion to {}", result.conversion_dir().display());
            }
            for component in SdxlConvertedComponent::all() {
                let path = result.component_path(component);
                println!("  {component:?}: {}", path.display());
            }
            println!(
                "  manifest: {}",
                result.conversion_manifest_path().display()
            );
        }
        Err(error) => {
            eprintln!("import failed: {error}");
            std::process::exit(1);
        }
    }
}
