//! Build-time enforcement of MB03 worker feature contract.
//!
//! The worker binary compiles with exactly one compute backend:
//! `wgpu` (GPU via wgpu/Metal/Vulkan/CubeCL) or `flex` (CPU via SIMD+rayon).
//!
//! Zero or dual features are rejected with an intentional diagnostic.

fn main() {
    let wgpu = std::env::var("CARGO_FEATURE_WGPU").is_ok();
    let flex = std::env::var("CARGO_FEATURE_FLEX").is_ok();

    match (wgpu, flex) {
        (false, false) => {
            println!(
                "cargo:warning=MB03 worker: no compute backend feature selected."
            );
            eprintln!(
                "error: reimagine-inference-burn-worker requires exactly one of \
                 `--features wgpu` or `--features flex`. A zero-feature build \
                 produces no usable worker."
            );
            std::process::exit(1);
        }
        (true, true) => {
            println!(
                "cargo:warning=MB03 worker: both wgpu and flex features selected."
            );
            eprintln!(
                "error: reimagine-inference-burn-worker must not enable both \
                 `wgpu` and `flex`. Select exactly one compute backend."
            );
            std::process::exit(1);
        }
        (true, false) | (false, true) => {
            // Valid: exactly one feature.
        }
    }
}