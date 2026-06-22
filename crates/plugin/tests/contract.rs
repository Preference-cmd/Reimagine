//! Integration tests for the plugin metadata contract.

use reimagine_plugin::{
    Extension, HostSurface, Plugin, PluginApiVersion, PluginDescriptor, PluginExtension,
    PluginOrigin, PluginPackage, PluginVersion,
};

struct TestPlugin {
    descriptor: PluginDescriptor,
    extensions: Vec<PluginExtension>,
}

impl TestPlugin {
    fn new(descriptor: PluginDescriptor, extensions: Vec<PluginExtension>) -> Self {
        Self {
            descriptor,
            extensions,
        }
    }
}

impl PluginPackage for TestPlugin {
    fn descriptor(&self) -> PluginDescriptor {
        self.descriptor.clone()
    }

    fn extensions(&self) -> Vec<PluginExtension> {
        self.extensions.clone()
    }
}

fn candle_descriptor() -> PluginDescriptor {
    PluginDescriptor {
        plugin: Plugin::try_from("builtin.candle").unwrap(),
        name: "Candle".to_string(),
        version: PluginVersion::try_from("0.1.0").unwrap(),
        api_version: PluginApiVersion::try_from("plugin/v1").unwrap(),
        origin: PluginOrigin::Builtin,
    }
}

fn candle_extension() -> PluginExtension {
    PluginExtension {
        extension: Extension::try_from("backend.candle").unwrap(),
        extends: HostSurface::InferenceBackend,
        name: "Candle inference backend".to_string(),
    }
}

#[test]
fn descriptor_round_trips_through_json() {
    let descriptor = candle_descriptor();

    let json = serde_json::to_string(&descriptor).expect("serialize");
    let restored: PluginDescriptor = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(restored, descriptor);
    assert_eq!(restored.plugin.as_str(), "builtin.candle");
    assert_eq!(restored.version.as_str(), "0.1.0");
    assert_eq!(restored.api_version.as_str(), "plugin/v1");
    assert_eq!(restored.origin.as_str(), "builtin");
}

#[test]
fn plugin_can_declare_multiple_extensions() {
    let descriptor = candle_descriptor();
    let backend_ext = candle_extension();
    let executor_ext = PluginExtension {
        extension: Extension::try_from("executor.candle.ksampler").unwrap(),
        extends: HostSurface::NodeExecutor,
        name: "Candle KSampler executor".to_string(),
    };

    let plugin = TestPlugin::new(descriptor, vec![backend_ext.clone(), executor_ext.clone()]);
    let extensions = plugin.extensions();

    assert_eq!(extensions.len(), 2);
    assert_eq!(extensions[0], backend_ext);
    assert_eq!(extensions[1], executor_ext);
    assert_eq!(extensions[0].extends, HostSurface::InferenceBackend);
    assert_eq!(extensions[1].extends, HostSurface::NodeExecutor);
}

#[test]
fn empty_identity_is_rejected_on_construction() {
    let err = Plugin::try_from("").unwrap_err();
    assert_eq!(
        err.to_string(),
        "plugin identity must not be empty".to_string()
    );

    assert!(Extension::try_from("").is_err());
    assert!(PluginVersion::try_from("").is_err());
    assert!(PluginApiVersion::try_from("").is_err());
}

#[test]
fn empty_identity_is_rejected_on_deserialize() {
    let json = r#""""#;
    let result: Result<Plugin, _> = serde_json::from_str(json);
    assert!(result.is_err());

    let result: Result<Extension, _> = serde_json::from_str(json);
    assert!(result.is_err());

    let result: Result<PluginVersion, _> = serde_json::from_str(json);
    assert!(result.is_err());

    let result: Result<PluginApiVersion, _> = serde_json::from_str(json);
    assert!(result.is_err());
}

#[test]
fn host_surface_label_is_stable() {
    assert_eq!(HostSurface::InferenceBackend.as_str(), "inference_backend");
    assert_eq!(HostSurface::NodeCatalog.as_str(), "node_catalog");
    assert_eq!(HostSurface::NodeExecutor.as_str(), "node_executor");
    assert_eq!(HostSurface::WorkflowAdapter.as_str(), "workflow_adapter");
    assert_eq!(HostSurface::AgentTool.as_str(), "agent_tool");
    assert_eq!(HostSurface::AgentProvider.as_str(), "agent_provider");
}

#[test]
fn plugin_extension_round_trips_through_json() {
    let extension = candle_extension();

    let json = serde_json::to_string(&extension).expect("serialize");
    let restored: PluginExtension = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(restored, extension);
    assert_eq!(restored.extension.as_str(), "backend.candle");
    assert_eq!(restored.extends, HostSurface::InferenceBackend);
    assert_eq!(restored.name, "Candle inference backend");
}

#[test]
fn host_surface_round_trips_through_json() {
    for surface in [
        HostSurface::InferenceBackend,
        HostSurface::NodeCatalog,
        HostSurface::NodeExecutor,
        HostSurface::WorkflowAdapter,
        HostSurface::AgentTool,
        HostSurface::AgentProvider,
    ] {
        let json = serde_json::to_string(&surface).expect("serialize");
        let restored: HostSurface = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored, surface);
    }
}

#[test]
fn external_origin_records_source_metadata() {
    let descriptor = PluginDescriptor {
        plugin: Plugin::try_from("external.foo").unwrap(),
        name: "Foo".to_string(),
        version: PluginVersion::try_from("1.2.3").unwrap(),
        api_version: PluginApiVersion::try_from("plugin/v1").unwrap(),
        origin: PluginOrigin::External {
            source: "reimagine-plugin-foo".to_string(),
        },
    };

    assert_eq!(descriptor.origin.as_str(), "external");

    let json = serde_json::to_string(&descriptor).expect("serialize");
    let restored: PluginDescriptor = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored, descriptor);
}
