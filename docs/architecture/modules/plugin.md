# Plugin Module Architecture

> Status: working draft
> Crate: `crates/plugin` planned

## Role

`plugin` defines the shared metadata contract for host extensions. It gives
the application a stable way to describe built-in and future external plugin
packages without coupling the base plugin contract to inference backends,
nodes, Agent providers, Tauri, or dynamic loading.

V1 uses plugin-shaped static registration. Built-in pieces such as the Candle
backend can be described as plugins even though they are linked at compile
time. Runtime loading of third-party plugins is explicitly deferred.

## Core Terms

```text
PluginPackage
  an object that exposes metadata for a plugin package

PluginDescriptor
  stable metadata for the package itself, including its Plugin identity

Plugin
  stable identity of a plugin package

PluginExtension
  one capability the plugin contributes to a host surface

Extension
  stable identity of a plugin extension

HostSurface
  a host-owned surface that plugins may extend
```

Use `extends` as the field that connects a `PluginExtension` to a
`HostSurface`:

```rust
pub trait PluginPackage: Send + Sync + 'static {
    fn descriptor(&self) -> PluginDescriptor;
    fn extensions(&self) -> Vec<PluginExtension>;
}

pub struct PluginDescriptor {
    pub plugin: Plugin,
    pub name: String,
    pub version: PluginVersion,
    pub api_version: PluginApiVersion,
    pub origin: PluginOrigin,
}

pub struct PluginExtension {
    pub extension: Extension,
    pub extends: HostSurface,
    pub name: String,
}

pub enum HostSurface {
    InferenceBackend,
    NodeCatalog,
    NodeExecutor,
    WorkflowAdapter,
    AgentTool,
    AgentProvider,
}
```

Avoid the name `extension_points` for plugin-provided entries. An extension
point is usually a host-provided slot. In this architecture, the host surface
is the slot and the plugin declares extensions for that surface.

Avoid suffixes such as `Kind` when the enum name already carries the domain
meaning. Prefer `HostSurface` over `PluginExtensionKind`.

Avoid `Id` suffixes for concepts that are primarily domain identities. Prefer
`Plugin`, `Extension`, `Backend`, and `BackendInstance` over `PluginId`,
`ExtensionId`, `BackendId`, and `BackendInstanceId`. They should still be typed
newtypes internally; the point is to keep public names in the domain language.

## Responsibilities

- Define typed plugin metadata identities such as `Plugin`, `PluginVersion`,
  `PluginApiVersion`, and `Extension`.
- Define `PluginDescriptor`, `PluginOrigin`, `PluginExtension`,
  `HostSurface`, and the base `PluginPackage` trait.
- Keep metadata serializable and testable without any concrete host.
- Provide a vocabulary that app-host can use when collecting static built-in
  extensions and later dynamic extensions.

## Non-Responsibilities

- Dynamic plugin loading ABI.
- Backend factories.
- Node executor factories.
- Agent provider factories.
- Tauri plugin integration.
- Config persistence.
- Runtime scheduling or resource policy.

Extension-specific construction traits belong to the owning domain crate. For
example, inference owns backend/router contracts, agent owns provider/tool
contracts, and core/nodes own node schema and built-in catalog data.

## Dependency Direction

```text
plugin -> no domain crate

inference may depend on plugin metadata for backend provenance.
agent may depend on plugin metadata for provider/tool provenance.
app-host depends on plugin and wires concrete extensions into domain registries.
concrete backend crates may expose plugin descriptors, but must not depend on
app-host.
```

The base plugin crate must not depend on `runtime`, `inference-backends/*`,
`app-host`, `tauri`, or `axum`.

## V1 Static Registration

The V1 loading model is static:

```text
app-host
  -> BuiltinPluginLoader
  -> CandlePluginPackage
  -> PluginDescriptor
  -> [PluginExtension { extends: HostSurface::InferenceBackend }]
  -> app-host constructs Candle backend instance
  -> app-host registers backend instance with inference router
```

This preserves the eventual plugin shape without deciding external binary
loading, process isolation, WASM, or native dynamic library protocols too
early.

Example Candle metadata:

```text
PluginDescriptor
  plugin: "builtin.candle"
  name: "Candle"
  origin: Builtin

PluginExtension
  extension: "backend.candle"
  extends: HostSurface::InferenceBackend
  name: "Candle inference backend"
```

## Relationship To Backend Identity

Plugin identity and backend instance identity are related but not equivalent:

```text
Plugin         -> which package provided the extension
Extension      -> which plugin extension was registered
Backend        -> open inference-owned backend implementation label
BackendInstance -> concrete configured backend instance, possibly including
                   device/config profile
```

A single plugin can provide multiple extensions. A single backend extension can
produce multiple backend instances if configuration asks for that later.

Backend instance runtime hooks are not a separate V1 host surface. They are
wired as part of an inference backend instance:

```text
PluginExtension { extends: HostSurface::InferenceBackend }
  -> BackendInstanceDescriptor { plugin, extension, backend, instance }
  -> typed inference backend adapter
  -> backend-instance lifecycle/observation hooks
```

This keeps plugin identity, backend routing, and backend-instance observations
attached to the same configured `BackendInstance`. If future plugin packages
need to contribute standalone resource monitors or schedulers, that should be
a new architecture decision rather than an implicit extension of the inference
backend surface.
