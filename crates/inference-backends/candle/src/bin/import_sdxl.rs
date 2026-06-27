use clap::Parser;
use reimagine_inference_candle::{
    SdxlCheckpointImportRequest, SdxlConvertedComponent,
    import_sdxl_checkpoint_to_candle_example_split,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "import-sdxl",
    about = "Import an original SDXL checkpoint to Candle-compatible split components"
)]
struct Cli {
    /// Path to the original SDXL safetensors checkpoint.
    checkpoint: PathBuf,

    /// Output directory for the bundle.
    #[arg(long, short)]
    output: PathBuf,

    /// Bundle name (used as output subdirectory).
    #[arg(long, short = 'n', default_value = "sdxl-base-1.0")]
    bundle: String,

    /// Checkpoint fingerprint (used for cache invalidation).
    #[arg(long, default_value = "auto")]
    fingerprint: String,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let output_dir = cli.output.join(&cli.bundle);
    let fingerprint = if cli.fingerprint == "auto" {
        sha256_hex(&cli.checkpoint)
    } else {
        cli.fingerprint
    };

    eprintln!("Importing {} ...", cli.checkpoint.display());
    eprintln!("  Output:  {}", output_dir.display());
    eprintln!("  Bundle:  {}", cli.bundle);
    eprintln!("  Fingerprint: {}", fingerprint);
    eprintln!();

    let request = SdxlCheckpointImportRequest::new(
        &cli.bundle,
        &cli.checkpoint,
        &fingerprint,
        "safetensors",
        &output_dir,
    );

    match import_sdxl_checkpoint_to_candle_example_split(request).await {
        Ok(result) => {
            println!("IMPORT SUCCESS");
            for component in SdxlConvertedComponent::all() {
                let path = result.component_path(component);
                if path.is_file() {
                    let size = std::fs::metadata(&path).unwrap().len();
                    let mb = size as f64 / 1_000_000.0;
                    println!(
                        "  {:<20} {:>8.2} MB  {}",
                        component.manifest_key(),
                        mb,
                        path.display()
                    );
                } else {
                    eprintln!("  ERROR: missing component at {}", path.display());
                }
            }
        }
        Err(e) => {
            eprintln!("IMPORT FAILED: {e}");
            std::process::exit(1);
        }
    }
}

fn sha256_hex(path: &std::path::Path) -> String {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path).expect("cannot open checkpoint for hashing");
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer).expect("read error during hashing");
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    format!("sha256:{:x}", hasher.finalize())
}
