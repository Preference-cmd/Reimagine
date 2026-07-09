### Now

| Issue | Status | Notes |
| --- | --- | --- |
| [15g: Burn real SDXL package smoke to image artifact](inference-backends/burn/issues/15g-burn-real-sdxl-package-smoke-to-image.md) | ready-for-agent | **Decision:** package+Module keyspace = **diffusers** (VAE done this branch; UNet Module topology still stage-shaped). VAE mid uses `attentions.0`/`to_out.0`, up uses `upsamplers.0`. See `.scratch/inference-backends/burn/package-dialect-diffusers.md`. Remaining: UNet diffusers topology, converter cleanup, real image smoke. |

### Next

| Issue | Status | Notes |
| --- | --- | --- |
| [15d9: Burn full-topology sampler parity](inference-backends/burn/issues/15d9-burn-full-topology-sampler-parity.md) | blocked | Wait for 15g before claiming numeric sampler parity against a trusted reference. |

### Design Gates

| Issue | Status | Notes |
| --- | --- | --- |
| [16a: Burn LoRA and training readiness design gate](inference-backends/burn/issues/16a-burn-lora-training-readiness-design-gate.md) | blocked | Wait for 15g and stable UNet/VAE Module naming before adapter attachment points are final. |

### Recently Landed

| Issue | Status | Notes |
| --- | --- | --- |
| [15g tranche: UNet up-block skip fidelity + VAE loader remapper](inference-backends/burn/issues/15g-burn-real-sdxl-package-smoke-to-image.md) | partial | Landed on main (not full 15g). Full-profile skip stack balances `conv_in` + residual + downsample pushes against per-residual up pops; VAE load remaps diffusers plural key dialect into Burn Module names. Real image artifact still open. |
| [15h: Burn align Module snapshot keys to diffusers](inference-backends/burn/issues/15h-burn-align-module-snapshot-keys-to-diffusers.md) | done | Merged to main (`c766f3d`). UNet/VAE Module fields renamed to match diffusers/Candle target key space. Package converter version `burn-sdxl-package-15h-v1`. |
| [15f: Burn full SDXL VAE decoder fidelity](inference-backends/burn/issues/15f-burn-full-sdxl-vae-decoder-fidelity.md) | done | Full-profile VAE decoder Module residual/upsample path; real-package artifact proof remains 15g. |
| [15e: Burn full SDXL UNet executable topology](inference-backends/burn/issues/15e-burn-full-sdxl-unet-executable-topology.md) | done | Full-profile UNet Module graph guard open; sampler parity remains 15d9. |
| [15e0: Burn full SDXL execution architecture](inference-backends/burn/issues/15e0-burn-full-sdxl-execution-architecture.md) | done | Accepted design in `docs/architecture/modules/burn-full-sdxl-execution.md`. |
| [15d8: Burn WGPU/Flex performance envelope](inference-backends/burn/issues/15d8-burn-wgpu-flex-performance-envelope.md) | done | Landed on main (`654d4bc`). |
| [15d7: Burn sampler fidelity audit](inference-backends/burn/issues/15d7-burn-sampler-fidelity-audit.md) | done | Sampler-to-UNet forward evidence; full-topology numeric parity split to 15d9. |
| [15d6: Burn VAE decoder topology and key-space](inference-backends/burn/issues/15d6-burn-vae-decoder-topology-and-keyspace.md) | done | First VAE key-space tranche; full decoder fidelity deferred then covered by 15f. |
| [15d5b: Burn UNet attention and stage fidelity tranche](inference-backends/burn/issues/15d5b-burn-unet-attention-stage-fidelity.md) | done | First down-block attention projection tranche. |
| [15d3a: Burn SDXL added-conditioning UNet injection](inference-backends/burn/issues/15d3a-burn-sdxl-added-conditioning-unet-injection.md) | done | Landed on main (`200b412`). |
| [15d5a: Burn UNet source mapping and loader alignment](inference-backends/burn/issues/15d5a-burn-unet-source-mapping-alignment.md) | done | Landed on main (`dba094b`). |
| [15d5: Burn UNet block fidelity tranches](inference-backends/burn/issues/15d5-burn-unet-block-fidelity-tranches.md) | done | First full-profile UNet tranche. |