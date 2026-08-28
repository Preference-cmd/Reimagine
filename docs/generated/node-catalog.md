# Node Catalog — Generated

> Generated from `crates/nodes/src/builtins/**.rs` on 2026-08-20. Do not edit by hand — run `node scripts/gen-node-catalog.mjs`.
> Freshness-gated by `node scripts/gen-node-catalog.mjs --check` (CI lane `static`).

| Type ID | Display | Category | Inputs | Outputs | File |
|---------|---------|----------|--------|---------|------|
| `builtin.checkpoint_loader` | Checkpoint Loader | Model | 1 | 3 | `model.rs` |
| `builtin.clip_text_encode` | CLIP Text Encode | Conditioning | 2 | 1 | `conditioning.rs` |
| `builtin.empty_latent_image` | Empty Latent Image | Latent | 3 | 1 | `latent.rs` |
| `builtin.ksampler` | KSampler | Sampling | 6 | 0 | `sampling.rs` |
| `builtin.load_image` | Load Image | Input | 1 | 1 | `inputs.rs` |
| `builtin.preview_image` | Preview Image | Image | 1 | 0 | `image.rs` |
| `builtin.save_image` | Save Image | Image | 3 | 0 | `image.rs` |
| `builtin.string` | String | Input | 1 | 1 | `inputs.rs` |
| `builtin.vae_decode` | VAE Decode | Image | 2 | 1 | `image.rs` |
| `builtin.vae_encode` | VAE Encode | Latent | 2 | 1 | `latent.rs` |

## Stats

- Total builtins: 10
- By category: Model=1, Conditioning=1, Latent=2, Sampling=1, Input=2, Image=3

## Source

- All definitions via `all_builtin_defs()` in `crates/nodes/src/builtins.rs` (10 nodes)
