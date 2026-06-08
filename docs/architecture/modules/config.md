# Config Module Architecture

> Status: working draft
> Crate: `crates/config`

## Role

`config` provides workspace-scoped path layout and typed JSON configuration infrastructure. It lets module services reuse the same base path, config key validation, atomic persistence, and config IO diagnostics without making `AppConfig` understand every module's business configuration.

It must not depend on Tauri, Axum, Candle, runtime scheduling, model-manager, agent, UI state, or provider SDKs.

It uses Tokio directly for async filesystem IO. `core` remains Tokio-free.

## Responsibilities

- Represent the active workspace `base_path`.
- Derive standard workspace directories.
- Create V1 workspace directories.
- Validate config keys under `<base_path>/config/`.
- Load, save, and update typed JSON config documents.
- Provide atomic JSON write helpers.
- Convert config IO failures into core diagnostics.

## Non-Responsibilities

- Owning `WorkspaceState`.
- Owning service lifecycle or dependency injection.
- Defining module-specific config semantics.
- Emitting semantic domain events.
- Implementing an event bus or host bridge.
- Treating `models/manifest.json` as a normal config document.

## Workspace Ownership

The top-level composition model is:

```text
AppState
  owns Arc<WorkspaceState>

WorkspaceState
  owns Arc<AppConfig>
  owns Arc<ModelManager>
  owns Arc<AgentRuntime>
  owns Arc<RuntimeStore>
  ...

AppConfig
  owns AppPaths
  owns ConfigStore

Module services
  own their own ConfigHandle<T>
```

Changing `base_path` rebuilds `WorkspaceState`.

`AppConfig` can create typed config handles, but it does not import or own concrete module config types such as `ModelSeriesConfig`, `AgentConfig`, `ProviderConfig`, or `RuntimeConfig`.

## App Paths

V1 workspace layout:

```text
base_path/
  models/
  input/
  output/
  workflows/
  config/
```

```text
AppPaths
  base_path
  models_dir
  input_dir
  output_dir
  workflows_dir
  config_dir
  async ensure_all()
```

The host chooses the default `base_path` and may let the user replace it through settings. `config` only models and prepares the selected path.

## Config Store and Handles

```text
AppConfig
  paths: AppPaths
  store: Arc<ConfigStore>
  config<T: ConfigDocument>() -> ConfigHandle<T>
```

```text
ConfigStore
  async load_json<T>(key)
  async save_json<T>(key, value)
  async update_json<T>(key, mutator)
```

```text
ConfigHandle<T>
  store: Arc<ConfigStore>
  key: ConfigKey
  async load()
  async save(value)
  async update(mutator)
  path()
```

`ConfigHandle<T>` is a typed accessor. It is not required to cache the current config value. Module services may cache loaded values themselves when useful.

Config APIs use one infrastructure report type:

```text
ConfigReport
  key: ConfigKey
  path: PathBuf
  diagnostics: Vec<Diagnostic>
```

Suggested return shape:

```text
async load() -> ConfigResult<(T, ConfigReport)>
async save(value) -> ConfigResult<ConfigReport>
async update(mutator) -> ConfigResult<(T, ConfigReport)>
```

`ConfigReport` is not a business operation envelope. Module services lift config diagnostics into `OperationReport` when a business action needs semantic diagnostics and events.

## Config Document

Each module defines its own config struct and implements:

```text
ConfigDocument
  KEY: &'static str
  SCHEMA_VERSION: &'static str
  validate(&self, context: &ConfigValidationContext) -> Vec<Diagnostic>
```

The trait constant `SCHEMA_VERSION` is used for defaults, diagnostics, and validation. Config files should still store their own `schema_version` field when a module's config schema needs compatibility checks.

`ConfigDocument::validate(context)` checks only the configuration document itself:

```text
schema version support
field shape
enum/newtype values
path or pattern syntax
duplicate rules inside the document
```

It does not perform readiness or external state checks:

```text
whether a configured model root currently exists
whether a provider API key works
whether a model is stale or missing
whether an output directory is currently writable for a run
```

Those checks belong to the owning module service.

```text
ConfigValidationContext
  key: ConfigKey
  path: PathBuf
  correlation_id: Option<CorrelationId>
```

Validation diagnostics should use deterministic ids. Do not add UUID or clock dependencies:

```text
config:<key>:<code>
config:<key>:validation:<index>
```

## Config Keys

`ConfigKey` is always relative to `<base_path>/config/`.

Allowed:

```text
model_series.json
agent/providers.json
runtime/defaults.json
```

Rejected:

```text
absolute paths
../ escapes
empty paths
```

## Persistence Rules

- Missing config files load as `T::default()`.
- Invalid JSON returns a config error and user-facing diagnostic.
- Saves use a temp-write-and-rename strategy.
- `ConfigStore` does not understand business schema versions beyond calling `ConfigDocument::validate(context)`.
- Business validation remains in the owning crate.
- Readiness and external state validation remain in the owning module service.

`models/manifest.json` is not a regular config document. Model manager owns its manifest store, but may reuse `AppPaths`, atomic write helpers, and diagnostic style.

## Atomic Write Helper

The config crate exposes a small reusable atomic file write helper for JSON/text-like persistence:

```text
async atomic_write(path, bytes)
```

Rules:

- write to a temporary file in the same directory;
- rename the temporary file into place;
- avoid leaving a partially written target file on failure;
- do not add business validation or semantic events.
- use Tokio filesystem APIs.

## Tokio Dependency

`config` uses Tokio because both the Tauri host and future Axum host are async/Tokio-oriented, and config IO belongs outside the pure `core` crate.

```text
[workspace.dependencies]
tokio = "1"
```

```text
[dependencies]
tokio = { workspace = true, features = ["fs", "io-util"] }

[dev-dependencies]
tokio = { workspace = true, features = ["macros", "rt"] }
```

If implementation or tests need a multi-threaded runtime, add `rt-multi-thread` explicitly. Do not add Tokio to `reimagine-core`.

## Events and Diagnostics

`config` depends on `core` for diagnostic/event payload types.

Config IO failures can become diagnostics:

```text
ConfigPathInvalid
ConfigJsonInvalid
ConfigReadFailed
ConfigWriteFailed
```

The config crate should not emit semantic domain events. Module services emit semantic events such as `ModelSeriesConfigSaved` or `ProviderConfigChanged`.
