# Axum E2E Workflow Guide

> Status: V1 developer workflow guide

This guide describes the repeatable Axum path for preparing a local workspace,
starting the HTTP host, running the canonical SDXL example workflow, and
locating the generated PNG artifact.

The current Candle backend can complete this path and write a PNG, but the
SDXL math is still placeholder implementation. It validates the workspace
shape, workflow execution path, model manifest resolution, runtime events, and
artifact writing. It does not yet prove that real SDXL CLIP, diffusion, or VAE
weights are being used for final image quality.

## Workspace Layout

Choose a local workspace `base_path`. Do not put this directory under git.

```text
base_path/
  models/
    manifest.json
    checkpoints/
      sdxl_base_1.0.safetensors
  input/
  output/
  workflows/
  config/
    inference_backend.json
    model_series.json
```

The Axum dev server creates the standard directories if they are missing:

```text
models/
input/
output/
workflows/
config/
```

Real model weights are local developer state. They must not be committed.

## Prepare A Workspace

Set a workspace path:

```bash
export REIMAGINE_WORKSPACE="$HOME/ReimagineWorkspace"
mkdir -p \
  "$REIMAGINE_WORKSPACE/models/checkpoints" \
  "$REIMAGINE_WORKSPACE/input" \
  "$REIMAGINE_WORKSPACE/output" \
  "$REIMAGINE_WORKSPACE/workflows" \
  "$REIMAGINE_WORKSPACE/config"
```

Place the SDXL base checkpoint here:

```text
$REIMAGINE_WORKSPACE/models/checkpoints/sdxl_base_1.0.safetensors
```

For the current placeholder Candle path, the file only needs to exist and be a
readable `.safetensors` file. For future real SDXL inference, this should be
the real SDXL base checkpoint.

## Backend Config

Write `<base_path>/config/inference_backend.json`:

```bash
cat > "$REIMAGINE_WORKSPACE/config/inference_backend.json" <<'JSON'
{
  "schema_version": "1",
  "backend": "candle",
  "candle_device": "cpu"
}
JSON
```

`candle_device` is the device/profile label passed through app-host into the
Candle backend configuration. V1 commonly uses `cpu`; local builds may also
use labels such as `metal` when the backend supports them.

## Model Manifest

Workflow JSON stores a stable `ModelRef`, not a local file path. The SDXL
example references:

```json
{
  "id": "sdxl-base-1.0",
  "model_series": "stable_diffusion",
  "variant": "sdxl",
  "role": "CheckpointBundle"
}
```

That id resolves through `<base_path>/models/manifest.json`. A minimal manual
manifest for this guide is:

```bash
cat > "$REIMAGINE_WORKSPACE/models/manifest.json" <<'JSON'
{
  "schema_version": "reimagine.model_manifest.v1",
  "model_roots": [
    {
      "id": "base",
      "path": ".",
      "kind": "base_path_models"
    }
  ],
  "models": [
    {
      "id": "sdxl-base-1.0",
      "model_series": "stable_diffusion",
      "variant": "sdxl",
      "roles": [
        "CheckpointBundle",
        "DiffusionModel",
        "TextEncoder",
        "Vae"
      ],
      "source": {
        "type": "local_file_relative",
        "root_id": "base",
        "path": "checkpoints/sdxl_base_1.0.safetensors"
      },
      "source_status": "Unverified",
      "format": "safetensors"
    }
  ]
}
JSON
```

The `base` model root points at `<base_path>/models`, so the relative source
above resolves to:

```text
<base_path>/models/checkpoints/sdxl_base_1.0.safetensors
```

`Unverified` is intentional for a hand-written manifest: the source exists,
but no fingerprint has been computed yet. Readiness may warn, but it can still
allow the run. A verified manifest can later set `source_status` to
`Available` and include observed size, modified time, and fingerprint fields.

## Model Series Config

`model_series.json` is used by model-manager classification and scanning. The
manual manifest above is enough for direct resolution, but keeping a minimal
series config in the workspace makes later scans classify SDXL checkpoint-like
files consistently:

```bash
cat > "$REIMAGINE_WORKSPACE/config/model_series.json" <<'JSON'
{
  "schema_version": "reimagine.model_series.v1",
  "rules": [
    {
      "model_series": "stable_diffusion",
      "variant": "sdxl",
      "root_id": "base",
      "filename_pattern": "*sdxl*",
      "extension": "safetensors",
      "roles": [
        "CheckpointBundle",
        "DiffusionModel",
        "TextEncoder",
        "Vae"
      ],
      "format": "safetensors"
    }
  ]
}
JSON
```

## Start Axum

From the repository root:

```bash
cargo run -p reimagine-axum-host -- \
  --base-path "$REIMAGINE_WORKSPACE" \
  --addr 127.0.0.1:7878 \
  --log-filter 'info,tower_http=debug'
```

The server logs the listening address, `base_path`, workspace directories,
selected backend, and selected device.

If `--base-path` is omitted, the dev server chooses and prints a temporary
development workspace path. For repeatable E2E runs, pass `--base-path`
explicitly.

## Health Check

```bash
curl -fsS http://127.0.0.1:7878/health | jq .
```

Expected shape:

```json
{
  "status": "ok",
  "workspace": "reimagine-axum-host"
}
```

## Open The Example Workflow

Open the canonical SDXL workflow inline:

```bash
jq -n \
  --slurpfile workflow docs/architecture/examples/sdxl-base-workflow.json \
  '{ workflow: $workflow[0] }' \
  | curl -fsS \
      -H 'content-type: application/json' \
      --data-binary @- \
      http://127.0.0.1:7878/workflows/open \
  | tee /tmp/reimagine-open-workflow.json \
  | jq .
```

Expected shape:

```json
{
  "workflow_id": "workflow_sdxl_base_demo",
  "source": "inline"
}
```

The workflow's checkpoint node uses `ModelRef.id = "sdxl-base-1.0"`, which
maps to the manifest entry written above.

## Run The Save Image Target

Run the explicit `node_save_image` target. This target forces the graph up
through prompt strings, CLIP encode, latent creation, sampling, VAE decode, and
image save.

```bash
curl -fsS \
  -H 'content-type: application/json' \
  --data-binary '{
    "target_selection": {
      "kind": "explicit",
      "targets": [
        { "kind": "node", "node_id": "node_save_image" }
      ]
    },
    "correlation_id": "manual-sdxl-e2e"
  }' \
  http://127.0.0.1:7878/workflows/workflow_sdxl_base_demo/run \
  | tee /tmp/reimagine-run.json \
  | jq .
```

Expected started shape:

```json
{
  "outcome": "started",
  "run_id": "...",
  "workflow_id": "workflow_sdxl_base_demo",
  "workflow_version": 1,
  "initial_snapshot": {
    "state": "Queued"
  },
  "diagnostics": []
}
```

If the response is `"outcome": "blocked"`, inspect the returned diagnostics.
Common causes are a missing manifest entry, mismatched `model_series` /
`variant`, missing checkpoint file, or a missing node executor.

## Poll The Run

Save the run id and poll until the response becomes a terminal summary:

```bash
export REIMAGINE_RUN_ID="$(jq -r '.run_id' /tmp/reimagine-run.json)"

curl -fsS "http://127.0.0.1:7878/runs/$REIMAGINE_RUN_ID" | jq .
```

Completed response shape:

```json
{
  "kind": "summary",
  "run_id": "...",
  "workflow_id": "workflow_sdxl_base_demo",
  "state": "Completed",
  "artifacts": [
    {
      "id": "...",
      "node_id": "node_save_image",
      "reference": "output/..."
    }
  ]
}
```

V1 does not yet include an artifact download route. Use the `reference` field
to identify the corresponding file under the workspace `output/` directory.

## Inspect Events

```bash
curl -fsS "http://127.0.0.1:7878/runs/$REIMAGINE_RUN_ID/events" | jq .
```

Useful events include `RunQueued`, `RunStarted`, `NodeStarted`,
`NodeCompleted`, `ArtifactCreated`, `RunCompleted`, and `RunFailed`. The
`correlation_id` supplied in the run request should appear in route logs and
runtime events when available.

## Locate The PNG

List the generated PNG files:

```bash
find "$REIMAGINE_WORKSPACE/output" -maxdepth 1 -type f -name '*.png' -print
```

The `save_image` node uses the workflow's `filename_prefix` parameter:

```json
{
  "filename_prefix": {
    "type": "string",
    "value": "sdxl_demo"
  }
}
```

The exact filename may include runtime-generated uniqueness data, but it
should live under:

```text
<base_path>/output/
```

## Current Placeholder Boundary

The current end-to-end path proves:

- Axum can bootstrap a real workspace `base_path`.
- Config and manifest files are read through app-host/model-manager paths.
- `ModelRef("sdxl-base-1.0")` resolves through `models/manifest.json`.
- Runtime can execute the SDXL-shaped node graph to `node_save_image`.
- Run events and snapshots are visible through HTTP.
- A PNG artifact is written under `<base_path>/output`.

The current path does not yet prove:

- real SDXL CLIP text encoding;
- real diffusion sampling against UNet weights;
- real VAE decode against VAE weights;
- production image quality or performance.

Future real SDXL work should replace the placeholder Candle internals behind
the same inference backend capabilities and keep this workspace shape stable:

```text
workflow ModelRef -> model-manager descriptor -> app-host inference runtime
  -> backend load/execute capabilities -> runtime events/artifacts
```

The guide should remain valid as the backend implementation becomes real; only
the note about placeholder math should shrink or disappear.
