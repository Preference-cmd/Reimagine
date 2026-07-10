# Burn full-topology sampler parity

Status: in_progress

Depends on: burn/15e, burn/15g

Dependency status: 15e (full UNet topology), 15g (real-package smoke to image
artifact), and 15e0 (execution architecture) are done.
Both `TinySdxlE2e` and `SdxlBase` profiles return `is_module_graph_supported()`
as `true`. The architecture doc defines the parity boundary at
`docs/architecture/modules/burn-full-sdxl-execution.md`. 15d9 adds numeric
sampler parity evidence against the full-profile Euler/normal path.

## Parent

burn/15d: Burn Module fidelity gap breakdown

## What to build

After the full SDXL UNet Module graph is executable on WGPU/Flex and the public
real-package smoke can produce an image artifact, compare the Burn sampler path
against a trusted SDXL Euler/normal reference for a tiny deterministic fixture
and a minimal full-profile smoke case.

This issue should validate sampler semantics, not broaden model loading or
performance work.

The parity claim should be limited to sampler/scheduler semantics. If the
reference comparison exposes a mismatch caused by conditioning projection,
loader key-space, or UNet block fidelity, publish a follow-up in that owner
area instead of broadening this issue.

## Acceptance criteria

- [x] Run sampler parity against an executable full-profile UNet Module path.
- [x] Verify timestep scaling, sigma update, CFG ordering, and positive/negative
      conditioning provenance against the reference.
- [ ] Define acceptable numeric tolerances separately for WGPU and Flex.
- [ ] Publish follow-up issues for any mismatch with clear ownership: scheduler,
      sampler, conditioning projection, or UNet block fidelity.
- [x] Do not enable full SDXL graph execution unless the required topology and
      loader tranches have landed.

## Blocked by

- ~~burn/15e~~ done
- ~~burn/15g~~ done
