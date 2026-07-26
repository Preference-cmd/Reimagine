//! Build-time enforcement of MB03/MB09 worker feature contract.
//!
//! The worker binary compiles with exactly one compute backend:
//! `wgpu` (GPU via wgpu/Metal/Vulkan/CubeCL), `flex` (CPU via SIMD+rayon),
//! or `cuda` (GPU via CubeCL + cudarc).
//!
//! Zero or multiple features are rejected with an intentional diagnostic.

fn main() {
    let wgpu = std::env::var("CARGO_FEATURE_WGPU").is_ok();
    let flex = std::env::var("CARGO_FEATURE_FLEX").is_ok();
    let cuda = std::env::var("CARGO_FEATURE_CUDA").is_ok();

    let active_features = [wgpu, flex, cuda].iter().filter(|&&f| f).count();

    match active_features {
        0 => {
            println!("cargo:warning=MB03 worker: no compute backend feature selected.");
            eprintln!(
                "error: reimagine-inference-burn-worker requires exactly one of \
                 `--features wgpu`, `--features flex`, or `--features cuda`. \
                 A zero-feature build produces no usable worker."
            );
            std::process::exit(1);
        }
        1 => {
            // Valid: exactly one feature.
        }
        _ => {
            println!("cargo:warning=MB03 worker: multiple features selected.");
            eprintln!(
                "error: reimagine-inference-burn-worker must not enable multiple \
                 compute backend features. Select exactly one: \
                 `wgpu`, `flex`, or `cuda`."
            );
            std::process::exit(1);
        }
    }
}
