### Now

| Issue | Status | Notes |
| --- | --- | --- |
| [15i: Burn capability and compute-profile truth alignment](inference-backends/burn/issues/15i-burn-capability-and-compute-profile-truth.md) | ready-for-agent | Align Burn runtime/profile capabilities, remove false image.import advertisement, prove explicit Burn selection has no Candle fallback. |
| [15j: Burn WGPU validation error elimination and propagation](inference-backends/burn/issues/15j-burn-wgpu-validation-error-elimination-and-propagation.md) | ready-for-agent | Fix the 8-byte/16-byte WGPU binding mismatch; make async validation fail the operation/test. |

### Next (blocked)

| Issue | Status | Blocked by |
| --- | --- | --- |
| [05a: Burn package workspace bootstrap](e2e-workflow/issues/05a-burn-package-workspace-and-smoke-workflow.md) | blocked | 15i |
| [05b: Burn Axum HTTP workflow-to-PNG E2E](e2e-workflow/issues/05b-burn-axum-http-workflow-to-png-e2e.md) | blocked | 05a + 15j |

### After Burn Axum E2E

| Issue | Status | Notes |
| --- | --- | --- |
| [15d9: Burn full-topology sampler parity](inference-backends/burn/issues/15d9-burn-full-topology-sampler-parity.md) | in_progress | Resume after e2e-workflow/05. Deterministic evidence done, WGPU/Flex tolerances pending. |
| [16a: Burn LoRA/training design gate](inference-backends/burn/issues/16a-burn-lora-training-readiness-design-gate.md) | blocked | Wait for Axum E2E + 15d9. |

### Recently Landed

| Issue | Notes |
| --- | --- |
| 15g: Burn real SDXL smoke to image artifact | Backend-direct proof. UNet 1676/1676, 256×256 PNG. WGPU noise → 15j. |
| 15h: Module snapshot keys → diffusers/Candle | All Module fields aligned. |
| 15f, 15e, 15e0 | Full-profile VAE/UNet + architecture doc. |
| 15d8, 15d7 | Performance envelope + sampler audit. |
| 15d6–15d5 / Candle issues | Archived in `.scratch/` filesystem — not active. |
