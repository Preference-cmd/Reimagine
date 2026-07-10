### Now

| Issue | Status | Notes |
| --- | --- | --- |
| [15i: Burn capability and compute-profile truth alignment](inference-backends/burn/issues/15i-burn-capability-and-compute-profile-truth.md) | ready-for-agent | Align Burn runtime/profile capabilities, remove false image.import advertisement, and prove explicit Burn selection has no Candle fallback route. |
| [15j: Burn WGPU validation error elimination and propagation](inference-backends/burn/issues/15j-burn-wgpu-validation-error-elimination-and-propagation.md) | ready-for-agent | Eliminate the real-smoke 8-byte/16-byte binding error and make asynchronous WGPU validation fail the operation/run/test. |

### Next

| Issue | Status | Notes |
| --- | --- | --- |
| [05a: Burn package workspace bootstrap and smoke workflow](e2e-workflow/issues/05a-burn-package-workspace-and-smoke-workflow.md) | blocked | Starts after 15i. Import an existing Burn package through ModelService, select Burn WGPU, and open the 256x256 one-step example through Axum. |
| [05b: Burn Axum HTTP workflow-to-PNG E2E](e2e-workflow/issues/05b-burn-axum-http-workflow-to-png-e2e.md) | blocked | Starts after 05a and 15j. HTTP open/run/poll/events/artifact must return a valid 256x256 PNG with no WGPU validation failure. |

### After Burn Axum E2E

| Issue | Status | Notes |
| --- | --- | --- |
| [15d9: Burn full-topology sampler parity](inference-backends/burn/issues/15d9-burn-full-topology-sampler-parity.md) | in_progress | Preserve current parity work, but remove it from the HTTP-to-PNG critical path. Resume numeric tolerance/reference closure after e2e-workflow/05. |

### Design Gates

| Issue | Status | Notes |
| --- | --- | --- |
| [16a: Burn LoRA and training readiness design gate](inference-backends/burn/issues/16a-burn-lora-training-readiness-design-gate.md) | blocked | Wait for Burn Axum E2E plus stable full-topology numeric evidence (15d9) before freezing adapter attachment points. |

### Recently Landed

| Issue | Status | Notes |
| --- | --- | --- |
| [15g: Burn real SDXL package smoke to image artifact](inference-backends/burn/issues/15g-burn-real-sdxl-package-smoke-to-image.md) | done | Backend-direct only: real `15h-v1` package bound UNet 1676/1676 and wrote 256x256 one-step PNGs. The observed WGPU validation error is a false-green blocker tracked by 15j. |
| [15g tranche: UNet up-block skip fidelity + VAE loader remapper](inference-backends/burn/issues/15g-burn-real-sdxl-package-smoke-to-image.md) | done | Absorbed into full 15g close-out. |
| [15h: Burn align Module snapshot keys to diffusers](inference-backends/burn/issues/15h-burn-align-module-snapshot-keys-to-diffusers.md) | done | Merged to main (`c766f3d`). UNet/VAE Module fields renamed to match diffusers/Candle target key space. Package converter version `burn-sdxl-package-15h-v1`. |
| [15f: Burn full SDXL VAE decoder fidelity](inference-backends/burn/issues/15f-burn-full-sdxl-vae-decoder-fidelity.md) | done | Full-profile VAE decoder Module residual/upsample path. |
| [15e: Burn full SDXL UNet executable topology](inference-backends/burn/issues/15e-burn-full-sdxl-unet-executable-topology.md) | done | Full-profile UNet Module graph guard open; sampler parity remains 15d9. |
| [15e0: Burn full SDXL execution architecture](inference-backends/burn/issues/15e0-burn-full-sdxl-execution-architecture.md) | done | Accepted design in `docs/architecture/modules/burn-full-sdxl-execution.md`. |
| [15d8: Burn WGPU/Flex performance envelope](inference-backends/burn/issues/15d8-burn-wgpu-flex-performance-envelope.md) | done | Landed on main (`654d4bc`). |
| [15d7: Burn sampler fidelity audit](inference-backends/burn/issues/15d7-burn-sampler-fidelity-audit.md) | done | Sampler-to-UNet forward evidence; full-topology numeric parity split to 15d9. |
| [15d6: Burn VAE decoder topology and key-space](inference-backends/burn/issues/15d6-burn-vae-decoder-topology-and-keyspace.md) | done | First VAE key-space tranche; full decoder fidelity deferred then covered by 15f. |
| [15d5b: Burn UNet attention and stage fidelity tranche](inference-backends/burn/issues/15d5b-burn-unet-attention-stage-fidelity.md) | done | First down-block attention projection tranche. |
| [15d3a: Burn SDXL added-conditioning UNet injection](inference-backends/burn/issues/15d3a-burn-sdxl-added-conditioning-unet-injection.md) | done | Landed on main (`200b412`). |
| [15d5a: Burn UNet source mapping and loader alignment](inference-backends/burn/issues/15d5a-burn-unet-source-mapping-alignment.md) | done | Landed on main (`dba094b`). |
| [15d5: Burn UNet block fidelity tranches](inference-backends/burn/issues/15d5-burn-unet-block-fidelity-tranches.md) | done | First full-profile UNet tranche. |
