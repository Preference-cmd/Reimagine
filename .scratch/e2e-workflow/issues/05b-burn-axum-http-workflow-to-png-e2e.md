# Burn Axum HTTP workflow-to-PNG E2E

Status: ready-for-agent

Depends on: e2e-workflow/05a (done), burn/15j (done)

## Parent

[Burn Axum real-image E2E](./05-burn-axum-real-image-e2e.md)

## Architecture source

- [Axum E2E Workflow Guide](../../../docs/architecture/e2e-workflow-axum.md)
- [Axum Host Architecture](../../../docs/architecture/modules/axum-host.md)

## What to build

Add an ignored, opt-in real-package test that uses the bootstrapped Burn
workspace and the repository smoke workflow to complete the entire lifecycle
through Axum HTTP routes. The execution path must never call `BurnBackend` or
poll `RuntimeService` directly.

Open the workflow, run the explicit `node_save_image` target, poll the run
route to a terminal summary, inspect run events, and download the artifact.
The test is a structural HTTP-to-PNG milestone, not an image-quality test.

## Acceptance criteria

- [ ] The ignored test skips clearly only when a required workspace/package
      environment variable is absent; present but invalid configuration fails.
- [ ] Workflow open and run use Axum HTTP routes and return HTTP 200.
- [ ] Run polling uses `GET /runs/:id`, reaches `Completed` within 300 seconds,
      and reports stage, run id, state, diagnostics, last event, and resolved
      backend instance on failure.
- [ ] `GET /runs/:id/events` contains run completion and artifact creation
      evidence for `node_save_image`.
- [ ] `GET /artifacts/:artifact_id` returns HTTP 200,
      `content-type: image/png`, a valid PNG signature, and an image that
      decodes to exactly 256x256.
- [ ] The run and test fail on any WGPU validation error or background WGPU
      panic; false-green completion is not accepted.
- [ ] Cleanup is limited to files created by this run and never deletes package
      data or unrelated workspace state.
- [ ] The first successful real-package run records the package identity,
      selected backend instance, smoke parameters, duration, run id, and
      artifact id/path as completion evidence.

## Non-goals

- Direct Burn backend testing.
- Visual quality or Diffusers numeric parity assertions.
- Performance targets or multi-step production defaults.

## Blocked by

- ~~e2e-workflow/05a~~ done
- ~~burn/15j~~ done (WSL guard lands async errors as deterministic failures)
