# Burn capability and compute-profile truth alignment

Status: done

Depends on: burn/15g

## Parent

burn/15g: Burn real SDXL package smoke to image artifact

## Architecture source

- [Burn Backend Adapter Architecture](../../../../docs/architecture/modules/burn-integration.md)
- [Axum E2E Workflow Guide](../../../../docs/architecture/e2e-workflow-axum.md)

## What to build

Make Burn capability discovery describe the operations the selected Burn
runtime can actually execute. The app-host compute profile and backend runtime
contract must agree before the Axum E2E uses capability discovery as readiness
evidence.

The current mismatch has two directions: the Burn profile omits sampling,
decode, save, and preview capabilities that are implemented, while the backend
advertises `image.import` even though that operation returns
`BackendNotImplemented`.

An explicit `selected_instance=burn:wgpu:default` must resolve to the Burn
runtime without installing or routing through Candle fallback hooks. Candle may
remain visible as another candidate in the aggregate profile.

## Acceptance criteria

- [x] Burn runtime capabilities include load bundle, text encode, empty latent,
      diffusion sample, latent decode, image save, and image preview.
- [x] Every available Burn WGPU/Flex instance profile advertises that same
      implemented text-to-image capability set.
- [x] The diffusion profile exposes the supported Euler/normal pair used by the
      smoke workflow.
- [x] `image.import` is no longer advertised while it still returns a precise
      unsupported-operation error.
- [x] `GET /compute-profile` projects the corrected Burn capabilities without
      backend-private types.
- [x] Explicit Burn WGPU selection resolves to Burn and registers Burn runtime
      hooks only; it does not add a Candle fallback execution route.
- [x] Focused Burn, app-host, config, and Axum compute-profile tests cover the
      aligned contract.

## Non-goals

- Implementing Burn image import or latent encode.
- Changing the default backend when no explicit instance is configured.
- Hardware adapter enumeration or performance tuning.

## Blocked by

- burn/15g (done)
