### Now

| Issue | Status | Notes |
| --- | --- | --- |
| [05b: Burn Axum HTTP workflow-to-PNG E2E](e2e-workflow/issues/05b-burn-axum-http-workflow-to-png-e2e.md) | ready-for-agent | 05a done. Run smoke workflow end-to-end through Axum, validate 256×256 PNG artifact. |

### Next (blocked)

| Issue | Status | Blocked by |
| --- | --- | --- |
| (none) | | |

### After Burn Axum E2E

| Issue | Status | Notes |
| --- | --- | --- |
| [15d9: Burn full-topology sampler parity](inference-backends/burn/issues/15d9-burn-full-topology-sampler-parity.md) | in_progress | Resume after e2e-workflow/05. Deterministic evidence done, WGPU/Flex tolerances pending. |
| [16a: Burn LoRA/training design gate](inference-backends/burn/issues/16a-burn-lora-training-readiness-design-gate.md) | blocked | Wait for Axum E2E + 15d9. |

### Recently Landed

| Issue | Notes |
| --- | --- |
| 05a: Burn package workspace bootstrap | opt-in Axum test imports via ModelService, selects burn:wgpu:default, asserts truthful /compute-profile, opens smoke workflow. Verified end-to-end against workspace/converted real package. |
| 15j: Burn WGPU validation propagation | WGPU panic hook guard makes async validation errors fail the owning operation. The 8-byte/16-byte binding mismatch remains a CubeCL upstream limitation; documented, not fixed here. |
| 15i: Burn capability and compute-profile truth alignment | Capability set now LoadBundle/TextEncode/CreateEmptyLatent/DiffusionSample/LatentDecode/ImageSave/ImagePreview. ImageImport dropped. Burn variant added. No Candle fallback. |
| 15g: Burn real SDXL smoke to image artifact | Backend-direct proof. UNet 1676/1676, 256×256 PNG. |
| 15h: Module snapshot keys → diffusers/Candle | All Module fields aligned. |
| 15f, 15e, 15e0 | Full-profile VAE/UNet + architecture doc. |
| 15d8, 15d7 | Performance envelope + sampler audit. |
| 15d6–15d5 / Candle issues | Archived in `.scratch/` filesystem — not active. |
