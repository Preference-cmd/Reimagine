//! The host adopts a started external `ModRegistry` and stops it once on
//! shutdown.

use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use mod_sdk::{
    Invocation, Mod, ModContext, ModManifest, ModRegistry, OperationAnnotations, OperationSpec,
};
use reimagine_app_host::{OperationProvider, WorkspaceHost, WorkspaceScope};
use reimagine_core::diagnostic::Diagnostic;
use reimagine_core::event::EventBus;
use reimagine_core::model::ProjectId;
use serde_json::{Value, json};

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-app-host-registry-{prefix}-{nonce}"))
}

struct StopProbeMod {
    manifest: ModManifest,
    stop_calls: AtomicUsize,
}

impl StopProbeMod {
    fn new() -> Self {
        Self {
            manifest: ModManifest {
                id: "probe".to_owned(),
                name: "Stop probe".to_owned(),
                version: "0.1.0".to_owned(),
                api: "0.1".to_owned(),
                operations: vec![OperationSpec {
                    name: "probe.ping".to_owned(),
                    description: "Prove the external registry reached the host".to_owned(),
                    input_schema: json!({ "type": "object", "properties": {} }),
                    output_schema: json!({ "type": "object", "properties": {} }),
                    annotations: OperationAnnotations {
                        read_only: true,
                        destructive: false,
                        idempotent: true,
                        deterministic: true,
                        open_world: false,
                    },
                }],
                events: vec![],
            },
            stop_calls: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl Mod for StopProbeMod {
    fn manifest(&self) -> &ModManifest {
        &self.manifest
    }

    async fn start(&self, _ctx: ModContext) -> Result<(), Diagnostic> {
        Ok(())
    }

    async fn invoke(&self, _invocation: Invocation) -> Result<Value, Diagnostic> {
        Ok(json!({}))
    }

    async fn stop(&self) -> Result<(), Diagnostic> {
        self.stop_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[tokio::test]
async fn external_registry_is_listed_and_stopped_once_on_shutdown() {
    let base = temp_dir("external");
    let app_data_root = base.join("app-data");
    fs::create_dir_all(&app_data_root).expect("create app data root");

    let event_bus = EventBus::new();
    let probe = Arc::new(StopProbeMod::new());
    let mod_registry =
        Arc::new(ModRegistry::start(vec![probe.clone() as Arc<dyn Mod>], &event_bus).await);

    let host = WorkspaceHost::try_with_app_data_root_and_event_bus_and_mod_registry(
        WorkspaceScope::new("test-external-registry"),
        &base,
        &app_data_root,
        event_bus,
        Arc::clone(&mod_registry),
    )
    .await
    .expect("bootstrap with an external mod registry");

    let operations = host
        .operation_service()
        .list(&ProjectId::new("default"))
        .await;
    let ping = operations
        .iter()
        .find(|info| info.spec.name == "probe.ping")
        .expect("op list must include the registered mod's operation");
    assert_eq!(
        ping.provider,
        OperationProvider::Mod {
            id: "probe".to_owned()
        }
    );
    assert!(ping.health.is_available());

    host.shutdown().await;
    assert_eq!(
        probe.stop_calls.load(Ordering::SeqCst),
        1,
        "shutdown must stop the external mod registry"
    );

    host.shutdown().await;
    assert_eq!(
        probe.stop_calls.load(Ordering::SeqCst),
        1,
        "shutdown must not stop the mods twice"
    );

    let _ = fs::remove_dir_all(&base);
}
