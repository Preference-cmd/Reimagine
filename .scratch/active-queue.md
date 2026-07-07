# Active Queues

> Current navigation surface for active `.scratch/**/issues`.
> Use this file as the first stop for implementation agents. The full
> historical ledger remains in [`issue-status.md`](issue-status.md).

## How To Use

- Pick a lane first. Lanes can move in parallel when they use separate
  worktrees and do not edit the same files.
- Inside a lane, pick from `Now` before `Next`.
- `Design Gates` are not AFK implementation work; resolve the decision before
  dispatching dependent issues.
- Keep issue files as the durable briefs; this file is only the current
  navigation surface.
- When an item lands, update this file and `issue-status.md` together.

## Lane: Burn Backend

Purpose: continue the Burn-native backend runtime line and close the remaining
Module fidelity gaps without regressing the WGPU/Flex execution target.

Current policy:

- V1 runtime target is WGPU by default with Flex as the CPU fallback.
- `burn-ndarray` is no longer a production fallback target.
- New implementation should use Burn-native `Module<B: Backend>`, active
  backend tensors, and burn-store loading seams.
- Do not expand old handwritten tensor helpers; treat them as historical
  context only.
- Keep full SDXL graph execution scoped to landed Module topology and loader
  tranches; real-package image smoke remains the 15g proof point.

### Now

| Issue | Status | Notes |
| --- | --- | --- |
| [15g: Burn real SDXL package smoke to image artifact](inference-backends/burn/issues/15g-burn-real-sdxl-package-smoke-to-image.md) | ready-for-agent | Prove the public Burn chain can produce an image artifact from a converted real SDXL package now that full-profile UNet and VAE topology tranches are in place. |

### Next

| Issue | Status | Notes |
| --- | --- | --- |
| [15d9: Burn full-topology sampler parity](inference-backends/burn/issues/15d9-burn-full-topology-sampler-parity.md) | blocked | Wait for 15e and 15g before claiming numeric sampler parity against a trusted reference. |

### Design Gates

| Issue | Status | Notes |
| --- | --- | --- |
| [16a: Burn LoRA and training readiness design gate](inference-backends/burn/issues/16a-burn-lora-training-readiness-design-gate.md) | blocked | Wait for executable full UNet/VAE topology and real-package image smoke so adapter attachment points are stable. |

### Recently Landed

| Issue | Status | Notes |
| --- | --- | --- |
| [15f: Burn full SDXL VAE decoder fidelity](inference-backends/burn/issues/15f-burn-full-sdxl-vae-decoder-fidelity.md) | done | Full-profile VAE decoder Module now uses latent projection, residual blocks, x8 Burn interpolate upsampling, and output projection; package/source mapping/contract/loading require the 15f decoder key-space while the tiny e2e fixture remains explicitly profile-gated. `latent.decode` stays decode-only; real-package artifact proof remains 15g. |
| [15e: Burn full SDXL UNet executable topology](inference-backends/burn/issues/15e-burn-full-sdxl-unet-executable-topology.md) | done | Full-profile UNet Module graph guard is open with skip-stack traversal, downsample/upsample stages, added-conditioning path, burn-store load policy without deferred topology families, and WGPU/Flex-safe Burn crate verification; sampler parity remains 15d9 and real-package image smoke remains 15g. |
| [15e0: Burn full SDXL execution architecture](inference-backends/burn/issues/15e0-burn-full-sdxl-execution-architecture.md) | done | Accepted design in `docs/architecture/modules/burn-full-sdxl-execution.md`; defines full UNet/VAE Module boundaries, burn-store key-space policy, guard-removal conditions, real-package smoke boundary, sampler parity boundary, and LoRA/training attachment vocabulary. |
| [15d8: Burn WGPU/Flex performance envelope](inference-backends/burn/issues/15d8-burn-wgpu-flex-performance-envelope.md) | done | Landed on main (`654d4bc`); adds deterministic Burn performance envelope probes for WGPU/Flex, scenario catalog, store/cache byte/count observations, and repeated model.load_bundle cache reuse coverage without threshold-based timing. |
| [15d7: Burn sampler fidelity audit](inference-backends/burn/issues/15d7-burn-sampler-fidelity-audit.md) | done | Added sampler-to-UNet forward evidence for scheduler timestep, CFG branch order, and branch-specific conditioning shapes; full-topology numeric parity split to 15d9. |
| [15d6: Burn VAE decoder topology and key-space](inference-backends/burn/issues/15d6-burn-vae-decoder-topology-and-keyspace.md) | done | First VAE key-space tranche landed: converted VAE packages now use `conv_out.*` Burn Module snapshot names, stale package reuse is blocked by converter-version bump, dead VAE raw weight buffers are removed, and full decoder residual/attention/upsample fidelity remains deferred. |
| [15d5b: Burn UNet attention and stage fidelity tranche](inference-backends/burn/issues/15d5b-burn-unet-attention-stage-fidelity.md) | done | First down-block attention projection tranche landed: maps `attn1/attn2` source projection weights into Burn MHA/context snapshots, loader applies them through burn-store, and remaining attention/stage/topology families stay deferred. |
| [15d3a: Burn SDXL added-conditioning UNet injection](inference-backends/burn/issues/15d3a-burn-sdxl-added-conditioning-unet-injection.md) | done | Landed on main (`200b412`); adds Burn-private pooled/time-id added-conditioning, a Burn-native projection Module in `SdxlUnet<B>`, positive/negative CFG threading, and missing-pooled diagnostics before sampled latent mutation. |
| [15d5a: Burn UNet source mapping and loader alignment](inference-backends/burn/issues/15d5a-burn-unet-source-mapping-alignment.md) | done | Landed on main (`dba094b`); maps the first full-profile UNet tranche from `model.diffusion.*` source keys into Burn Module snapshot names. |
| [15d5: Burn UNet block fidelity tranches](inference-backends/burn/issues/15d5-burn-unet-block-fidelity-tranches.md) | done | First full-profile UNet tranche landed: time embedding plus first down resblock/time-projection source keys load through burn-store; full graph execution remains disabled. |

## Lane: Support / Acquisition

Purpose: model acquisition work that may unblock model-manager, app-host, and
Burn package import, but should not be folded into backend runtime work.

### Now

| Issue | Status | Notes |
| --- | --- | --- |
| [MA-04: Axum IPC commands](model-acquisition/issues/MA-04-axum-ipc-commands.md) | ready-for-agent | Add Axum HTTP endpoint for model download (mirror of MA-03 Tauri command). |

### Next

(empty)

### Design Gates

(empty)

### Recently Landed

| Issue | Status | Notes |
| --- | --- | --- |
| [MA-03: Tauri IPC commands](model-acquisition/issues/MA-03-tauri-ipc-commands.md) | done | Landed on main. `download_huggingface_model` Tauri command with progress streaming via `TauriDownloadEventHub`, 67+30+12+21 tests passing. |
| [MA-02: app-host integration](model-acquisition/issues/MA-02-app-host-integration.md) | done | Landed on main (`3932b0b`). ModelAcquisitionService with config load/save, acquire() via spawn_blocking, model.download Agent tool, IPC DTO, 67+30 tests passing. |
| [MA-01: model-acquisition crate](model-acquisition/issues/MA-01-model-acquisition-crate.md) | done | Landed on main (`659c762`). Backend-neutral model-acquisition library crate; wraps hf-hub, ConfigDocument (model_acquisition.json), staging/promote with atomic rename + backup rollback, path safety validation, AcquisitionReport JSON, progress sink trait. 30 unit tests passing, 4 #[ignore] integration tests. |

## Lane: Tauri Host

Purpose: desktop IPC and host integration.

### Now

(empty)

### Next

(empty)

### Recently Landed

| Issue | Status | Notes |
| --- | --- | --- |
| [05: Artifact and desktop affordances](tauri-host/issues/05-artifact-and-desktop-affordances.md) | done | Implemented on main (`357a816`) and pushed to origin. |
| [06: Agent panel IPC](tauri-host/issues/06-agent-panel-ipc.md) | dispatched | Implemented on main; left as dispatched in the historical ledger until separately reviewed/closed. |

## Parking Lot

| Issue | Status | Notes |
| --- | --- | --- |
| [SDXL added-conditioning forward beyond Candle example compatibility](real-inference/issues/10c-sdxl-added-conditioning-forward.md) | needs-triage | Historical Candle-side note; Burn-side 15d3a is done. Re-triage only if Candle real-inference work resumes. |

## Historical Ledger

Most earlier issues are `done` or `split` and remain in place so old links,
handoffs, and implementation notes keep working. Use
[`issue-status.md`](issue-status.md) when you need the full history.
