# Model Manager Module Architecture

> Status: working draft
> Crate: `crates/model-manager`

## Role

`model-manager` owns the local model manifest, model scanning, model identity, model descriptors, and model reference resolution. It lets workflow JSON store stable `ModelRef` values instead of local file paths.

It may depend on `core` and `config`. It must not depend on Tauri, Axum, Candle, runtime scheduling, or UI state.

## Responsibilities

- Maintain `ModelManifest` under the app `base_path`.
- Scan configured model directories.
- Add, update, and remove manifest entries.
- Assign and validate stable `ModelId` values.
- Resolve `core::model::ModelRef` into `ModelDescriptor`.
- Emit diagnostics for missing, stale, mismatched, or unsupported model entries.

## Non-Responsibilities

- Loading model weights.
- Holding loaded model handles.
- Device placement or dtype conversion.
- Downloading models from remote providers in V1.
- Workflow mutation. Model manager actions are not `WorkflowCommand`.
- UI rendering or file picker behavior.

## Base Path and Manifest Location

V1 stores app-managed local folders under a selected `base_path`:

```text
base_path/
  models/
    manifest.json
  input/
  output/
  workflows/
  config/
```

The app creates a default `base_path` under the app data directory. Users may choose a different `base_path` in settings.

`model-manager` uses `config::AppPaths` for workspace path layout, but owns the model manifest semantics itself.

The model manifest lives at:

```text
<base_path>/models/manifest.json
```

Workflow JSON never stores absolute model paths. It stores `ModelRef` values:

```json
{
  "id": "sdxl-base-1.0",
  "model_series": "stable_diffusion",
  "variant": "sdxl",
  "role": "CheckpointBundle"
}
```

The manifest resolves `ModelId` to model sources.

## Manifest Shape

```text
ModelManifest
  schema_version
  model_roots: Vec<ModelRoot>
  models: Vec<ModelDescriptor>
```

```text
ModelDescriptor
  id: ModelId
  model_series: ModelSeries
  variant: ModelVariant
  roles: Vec<ModelRole>
  source: ModelSource
  source_status: ModelSourceStatus
  format: ModelFormat
  size_bytes
  observed_size_bytes
  observed_modified_at
  fingerprint
  verified_at
  discovered_at
  updated_at
  metadata optional
```

V1 does not include `display_name`. The UI displays `ModelId` as the primary label and may show model series, variant, roles, and source path as secondary metadata.

Example:

```json
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
      "source_status": "Available",
      "format": "safetensors",
      "size_bytes": 6938078336,
      "observed_size_bytes": 6938078336,
      "observed_modified_at": "2026-06-07T00:00:00Z",
      "fingerprint": {
        "kind": "sha256",
        "value": "..."
      },
      "verified_at": "2026-06-07T00:00:00Z",
      "discovered_at": "2026-06-07T00:00:00Z",
      "updated_at": "2026-06-07T00:00:00Z"
    }
  ]
}
```

`model_roots` is manifest-owned configuration, not workflow data. V1 supports a default `base_path` root plus optional user-selected roots.

The manifest is not a normal `ConfigHandle<T>` document because it lives under `<base_path>/models/manifest.json`, not under `<base_path>/config/`.

## Model Sources

V1 source types:

```text
ModelSource
  LocalFileRelative { root_id, path }
  LocalFileAbsolute { path }
```

Relative paths are relative to a manifest model root. The default root resolves under `<base_path>/models/`. Workflow JSON still stores only `ModelId`.

Absolute sources are allowed for local convenience but make the manifest less portable. They should produce portability warnings when exporting or packaging.

## Source Status

`ModelSourceStatus` is persisted so the UI can immediately show the last known state before a background scan refreshes it.

```text
ModelSourceStatus
  Available
  Missing
  Stale
  Unverified
```

Readiness behavior:

```text
Available
  allow run

Missing
  block run

Stale
  block run until manual refresh/verify updates fingerprint and observed metadata

Unverified
  warn and allow run
```

`Unverified` is used when a source exists but no fingerprint is available. `Stale` is used when size or modified time changed after the last verification.

## ModelId

`ModelId` is the stable user-visible model name in V1.

User-created manifest entries may choose readable ids:

```text
sdxl-base-1.0
```

Auto-generated ids should be deterministic and collision-resistant enough for local use:

```text
sdxl-checkpoint-sdxl_base_1_0-a1b2c3d4
```

V1 does not need aliases, tags, notes, or display names.

Unknown values are allowed in the manifest:

```text
model_series = "unknown"
variant = "unknown"
roles = []
```

Unknown model entries can be listed and edited, but they are not considered runnable until the user or series config supplies roles and a concrete series/variant.

## Suggested Module Layout

```text
src/
  lib.rs
  manager.rs
  error.rs

  manifest.rs
  manifest/
    descriptor.rs
    source.rs
    status.rs
    root.rs
    format.rs
    fingerprint.rs
    validation.rs

  store.rs
  store/
    path.rs
    atomic_write.rs

  scan.rs
  scan/
    config.rs
    candidate.rs
    filesystem.rs
    update.rs

  classify.rs
  classify/
    series_config.rs
    matcher.rs

  identity.rs
  identity/
    id_policy.rs
    conflict.rs

  verify.rs
  verify/
    sha256.rs
    refresh.rs

  resolve.rs
  resolve/
    readiness.rs
    descriptor.rs

  event.rs
```

Use modern Rust module layout. Do not introduce `mod.rs`, and prefer ordinary `mod foo;` declarations over `#[path = "..."]` attributes.

`lib.rs` is the public facade. It should re-export stable API types such as `ModelManager`, `ModelManifest`, `ModelDescriptor`, `ModelManifestStore`, `ScanConfig`, `ModelSeriesConfig`, resolver traits, and result/report types. It should not expose every internal helper module.

The top-level module files are facades for cohesive subdomains. For example, `manifest.rs` wires private files under `manifest/`, and `scan.rs` wires private files under `scan/`. This keeps related code together without using `mod.rs`.

## Internal Architecture

```text
ModelManager
  coordinates store, scan, classification, identity, verification, and resolution

manifest
  serializable schema and validation data

store
  base_path path resolution and atomic persistence

scan
  filesystem observation and scan report generation

classify
  model_series.json loading and candidate classification

identity
  model id generation and conflict handling

verify
  explicit fingerprint computation and refresh

resolve
  readiness and full descriptor resolution

event
  model-manager event constructors using core DomainEvent and Diagnostic
```

Scanner does not directly write the manifest. It returns observations. Manifest update logic lives in `scan/update.rs` or a closely related updater component, so scan preview and tests can reuse the same observation layer.

## Manifest Store

`ModelManifestStore` owns manifest persistence:

```text
ModelManifestStore
  load()
  save(manifest)
  update(mutator)
```

Persistence rules:

- load missing manifest as an empty V1 manifest;
- preserve unknown future fields only if the chosen serde strategy supports it without complexity;
- save atomically by writing a temporary file and renaming it into place;
- never partially write a manifest;
- report parse errors as diagnostics, not panics.

The store does not scan directories by itself. Scanning is a separate service that proposes manifest updates.

## Model Manager Service

`ModelManager` coordinates store, scanner, and resolver:

```text
ModelManager
  manifest_store
  scanner
  classifier
  id_policy
  verifier
  resolver
```

V1 operations:

```text
load_manifest()
save_manifest()
list_models()
get_model(ModelId)
resolve(ModelRef)
scan_roots()
add_or_update_model(ModelDescriptor)
remove_model(ModelId)
validate_manifest()
```

These are model-management actions, not `WorkflowCommand`.

Removing a model in V1 removes only the manifest entry. It does not delete the underlying file. Destructive file deletion remains outside model-manager.

## Series and Variant Configuration

Model series and variant inference should come from a local configuration file, not hard-coded filename guesses alone.

V1 stores this under:

```text
<base_path>/config/model_series.json
```

V1 shape:

```text
ModelSeriesConfig
  schema_version
  rules: Vec<ModelSeriesRule>
```

```text
ModelSeriesRule
  root_id optional
  path_pattern optional
  filename_pattern optional
  extension optional
  model_series
  variant
  roles
  format optional
```

Rules may match by path pattern, filename pattern, extension, or manifest root. A rule matches only when every field present on the rule matches the candidate. Rules are evaluated in order; the first matching rule wins.

V1 matching semantics:

```text
extension
  exact match after trimming leading "." and ASCII-lowercasing both sides

path_pattern
  glob match against the candidate path

filename_pattern
  glob match against the candidate filename

root_id
  exact match against the candidate manifest root id
```

The classifier does not walk directories or stat files. It classifies a `ClassificationCandidate` supplied by the scanner:

```text
ClassificationCandidate
  root_id optional
  path
  filename
  extension
  observed_format optional
```

If no rule matches, the classifier returns `model_series = "unknown"`, `variant = "unknown"`, no roles, and either the observed format or `Unknown`.

`ModelSeriesConfig` is defined by model-manager and implements `config::ConfigDocument`. `ModelManager` owns a `ConfigHandle<ModelSeriesConfig>` created from `AppConfig::config::<ModelSeriesConfig>()`.

Users may edit and save the inferred model series and variant. Saving user edits updates the manifest and should emit diagnostics/events if the change affects model readiness or referenced workflows.

## Scanner

`ModelScanner` walks configured roots and returns scan observations:

```text
ScanObservation
  root_id
  source
  format
  size_bytes
  modified_at optional
  relative_path
  filename
  extension
```

Scan behavior is configurable per root:

```text
ScanConfig
  recursive
  ignore_hidden
  include_patterns
  exclude_patterns
  supported_extensions
```

Default V1 scan config:

```text
recursive = true
ignore_hidden = true
supported_extensions = [".safetensors"]
exclude_patterns include .git, target, node_modules, and common cache/build dirs
```

The scanner should support checkpoint-like files under configured roots and ignore hidden files/directories by default. It does not write the manifest and does not compute fingerprints during normal scans.

Manifest update logic consumes observations plus `ModelSeriesConfig` classification and id policy results:

```text
ManifestUpdatePolicy
  apply_observations(manifest, observations)
  apply_root_observations(manifest, root_id, observations)
```

`apply_observations` is for a full scan snapshot. `apply_root_observations` is for a single-root scan and must only mark entries from that root as missing. This prevents scanning one available root from marking unrelated roots as missing.

V1 does not need remote download, Hugging Face snapshots, or automatic metadata extraction beyond simple format/model_series/variant/role inference.

## Fingerprint Strategy

Fingerprints support stale-file diagnostics and model identity conflict resolution.

V1 can use:

```text
Fingerprint
  kind: Sha256
  value
```

For large models, hashing may be expensive. V1 can compute SHA-256 when a model is added or refreshed, and avoid rehashing on every startup by comparing size and modified time first.

Manifest stores:

```text
size_bytes
modified_at optional
fingerprint optional
```

Readiness can treat missing fingerprints as warning-level weaker validation, not as failure.

Fingerprint calculation frequency:

```text
first add
  compute fingerprint

manual refresh / verify
  recompute fingerprint and update verified_at

normal startup
  do not compute fingerprint

normal scan
  stat files only
  if size/modified_at unchanged, keep verified state
  if size/modified_at changed, mark stale

runtime load
  do not compute fingerprint
```

Readiness behavior:

```text
source missing
  block run

size/modified_at changed and fingerprint has not been refreshed
  block run

fingerprint mismatch
  block run

fingerprint missing but source exists
  warn and allow run
```

The purpose of the fingerprint is to detect silent model replacement without hashing large checkpoint files on every startup or run. Fingerprints also participate in auto-id collision handling: when an auto-generated id collides, matching fingerprint plus matching full `ModelSource` means the candidate is treated as the same model.

## ID Policy

Manual IDs are allowed and are the cleanest V1 path.

Auto-generated IDs should be deterministic from descriptor data:

```text
model_series-variant-role-normalized_filename-short_hash
```

Collision handling:

```text
if id is unused
  use it
else if same fingerprint and same full ModelSource
  treat as same model
else
  append a longer deterministic hash suffix
  if the suffixed id is also taken, append a deterministic counter
  emit notification/diagnostic describing the conflict resolution
```

Manual id conflicts are rejected and reported to the user. Auto-generated id conflicts are resolved by suffixing, but still produce a notification/diagnostic so the user can see what happened. The id policy is pure model-manager logic over candidate descriptor data and existing manifest descriptors; it does not save manifests.

Model IDs are user-visible. Avoid opaque UUID-only IDs for V1.

## Scan and Update Policy

Scanning should not silently delete manifest entries.

Recommended V1 behavior:

```text
new file found
  add manifest entry automatically
  emit notification/diagnostic describing the added model

known file changed
  mark stale
  emit notification/diagnostic
  do not silently trust the old fingerprint

known file missing
  keep manifest entry, mark source unavailable
  emit notification/diagnostic

file removed from scan root
  do not delete manifest entry automatically
```

This avoids breaking workflows when a drive is temporarily unavailable.

Updates that change user-visible model identity, availability, or run readiness must produce a notification or diagnostic. Silent manifest mutation is allowed only for harmless metadata refreshes that do not affect resolution or readiness.

If a model root is removed from the manifest or becomes unavailable, model entries that depend on that root are not deleted automatically. They are marked missing/unavailable and can be manually pruned.

## Events and Diagnostics

Model manager should not introduce a Tauri-specific `ModelManagerEvent` concept. Instead, model manager emits domain events/diagnostics into the shared core event/diagnostic stream.

The host decides how to bridge those events:

```text
Tauri host
  shared event -> Tauri event

Axum host
  shared event -> SSE/WebSocket event
```

Conceptual event categories:

```text
ModelAdded
ModelMarkedMissing
ModelMarkedStale
ModelVerified
ModelIdConflictResolved
ModelRootChanged
```

Each event may carry diagnostics. Diagnostics remain the source of actionable user/agent information; events provide timeline and notification semantics.

## Manifest Validation

Manifest validation checks:

- unique model ids;
- valid model series, variant, and roles;
- at least one role per descriptor;
- source path shape is valid;
- relative source root exists;
- format is supported;
- source file existence;
- source status consistency;
- size/fingerprint consistency when available.

Manifest validation diagnostics are separate from workflow diagnostics, but workflow readiness may surface relevant model diagnostics at the referencing node param.

## Schema Versioning

V1 only accepts `reimagine.model_manifest.v1`.

Unsupported schema versions produce diagnostics. V1 does not perform automatic manifest migration. Migration support can be added when the manifest has a second real schema version.

## Resolution

```text
ModelResolver
  resolve(ModelRef) -> ModelDescriptor
```

Resolution checks:

- id exists in manifest;
- manifest model_series and variant match `ModelRef.model_series` and `ModelRef.variant`;
- requested role is provided by the descriptor;
- source file exists;
- stale status blocks until explicit verify refresh updates fingerprint and observed metadata;
- missing fingerprint with an existing source emits a warning but still allows readiness;
- fingerprint-backed descriptors block when observed file metadata no longer matches the verified snapshot.

The resolver returns descriptors only. Loaded backend handles belong to backend stores assembled outside model-manager.

There are two resolver surfaces:

```text
ModelReadinessResolver
  lightweight view passed to core readiness

ModelDescriptorResolver
  full descriptor lookup used by app-host/backend loading capabilities
```

The readiness resolver can project a descriptor into:

```text
ResolvedModelInfo
  id
  model_series
  variant
  roles
  format
  source_available
```

The full descriptor remains owned by model-manager and includes source paths, size, fingerprint, and metadata.

Resolution should not load model weights. It only proves that a manifest entry exists and points to an available source that can later be loaded by a backend.

Explicit verify/refresh remains the only path that computes SHA-256, updates `fingerprint`, `verified_at`, `observed_size_bytes`, `observed_modified_at`, `source_status`, and `updated_at`, and clears stale state for an existing descriptor or a first-add descriptor.

## Backend Handoff

`app-host` assembles backend loading capabilities that receive a full `ModelDescriptor` and requested role:

```text
BackendModelStore
  get_or_load(descriptor, role, device, precision) -> LoadedModelHandle
```

`model-manager` never returns loaded handles. It also does not decide device, precision, cache policy, or eviction.

## Diagnostics

Model manager emits diagnostics such as:

```text
ModelRefNotFound
ModelSourceMissing
ModelFingerprintMismatch
ModelSeriesMismatch
ModelVariantMismatch
ModelRoleMissing
UnsupportedModelFormat
```

These diagnostics can be surfaced by workflow readiness validation when a `ModelRef` appears in a node param.

Diagnostic targets:

```text
manifest
manifest.model
workflow.node params.<slot>
```

Examples:

```text
ModelSourceMissing
  manifest target: manifest.model id=sdxl-base-1.0 path=source
  workflow target: workflow.node id=node_checkpoint path=params.checkpoint

ModelRoleMissing
  workflow target: workflow.node id=node_checkpoint path=params.checkpoint.role
```

## Concurrency

V1 can keep concurrency simple:

- one in-memory manifest per app state;
- async scans run one at a time;
- manifest updates are serialized by the model manager;
- backend loading can read descriptors while scans are not mutating them;
- long hash computations should not block UI.

If a workflow run starts while a scan is in progress, `app-host` builds readiness and backend loading capabilities from the last committed manifest snapshot.

## V1 Limits

V1 intentionally excludes:

- remote model download;
- alias/display name/tag editing;
- semantic version constraints;
- automatic LoRA composition;
- model card metadata;
- background eviction of loaded models;
- manifest sync across machines.

## Dependency Direction

```text
model-manager -> core
model-manager -> config
model-manager must not -> candle-integration
model-manager must not -> runtime
model-manager must not -> tauri
```

`app-host` and Candle integration consume resolved descriptors:

```text
Workflow ModelRef
  -> app-host/model service resolves ModelDescriptor
  -> candle-integration-backed capability loads backend model payload
  -> runtime receives Model / Clip / Vae handles from node executors
```
