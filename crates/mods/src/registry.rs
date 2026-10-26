use std::collections::HashSet;
use std::sync::Arc;

use reimagine_core::diagnostic::Diagnostic;
use reimagine_core::event::EventBus;
use reimagine_core::health::Health;
use reimagine_core::task::CancelToken;
use semver::VersionReq;
use serde_json::Value;

use crate::context::ModContext;
use crate::diagnostic::caller_error;
use crate::interface::Mod;
use crate::manifest::{API_VERSION, ModManifest};

/// Namespaces the kernel keeps for itself; no mod may take one as its id.
pub const RESERVED_NAMESPACES: &[&str] =
    &["project", "board", "workflow", "run", "job", "asset", "mod"];

/// A registered mod as the kernel sees it.
#[derive(Debug, Clone)]
pub struct ModEntry {
    pub manifest: Arc<ModManifest>,
    /// Unavailable when the manifest was rejected or `start` failed.
    pub health: Health,
}

struct Registered {
    entry: ModEntry,
    instance: Arc<dyn Mod>,
}

/// The started mods of one application. A rejected mod stays listed as
/// unavailable and never affects the others.
pub struct ModRegistry {
    mods: Vec<Registered>,
}

impl ModRegistry {
    /// Validate every manifest, then start each accepted mod in order.
    ///
    /// A manifest is rejected when its id is reserved, not snake_case or
    /// already taken, when its api requirement excludes [`API_VERSION`], or
    /// when an operation or event leaves the mod's namespace or repeats.
    pub async fn start(mods: Vec<Arc<dyn Mod>>, event_bus: &EventBus) -> Self {
        let mut registered: Vec<Registered> = Vec::with_capacity(mods.len());
        for instance in mods {
            let manifest = Arc::new(instance.manifest().clone());
            let taken = registered
                .iter()
                .any(|other| other.entry.manifest.id == manifest.id);
            let health = match validate(&manifest, taken) {
                Err(diagnostic) => Health::Unavailable(diagnostic),
                Ok(()) => {
                    let ctx = ModContext::new(Arc::clone(&manifest), event_bus.clone());
                    match instance.start(ctx).await {
                        Ok(()) => Health::Available,
                        Err(diagnostic) => Health::Unavailable(diagnostic),
                    }
                }
            };
            if let Health::Unavailable(diagnostic) = &health {
                tracing::warn!(
                    mod_id = %manifest.id,
                    message = diagnostic.message(),
                    "mod unavailable"
                );
            }
            registered.push(Registered {
                entry: ModEntry { manifest, health },
                instance,
            });
        }
        Self { mods: registered }
    }

    /// Stop every mod that started.
    pub async fn stop(&self) {
        for registered in self.available() {
            if let Err(diagnostic) = registered.instance.stop().await {
                tracing::warn!(
                    mod_id = %registered.entry.manifest.id,
                    message = diagnostic.message(),
                    "mod failed to stop"
                );
            }
        }
    }

    /// Call `operation` on mod `mod_id`. Fails before reaching the mod when
    /// the mod is unknown or unavailable or does not declare the operation.
    pub async fn invoke(
        &self,
        mod_id: &str,
        operation: &str,
        payload: Value,
    ) -> Result<Value, Diagnostic> {
        let registered = self
            .mods
            .iter()
            .find(|registered| registered.entry.manifest.id == mod_id)
            .ok_or_else(|| {
                caller_error(
                    mod_id,
                    "UNKNOWN_MOD",
                    format!("no mod `{mod_id}` is registered"),
                )
            })?;
        if let Health::Unavailable(diagnostic) = &registered.entry.health {
            return Err(diagnostic.clone());
        }
        let manifest = &registered.entry.manifest;
        if !manifest.operations.iter().any(|op| op.name == operation) {
            return Err(caller_error(
                mod_id,
                "UNKNOWN_OPERATION",
                format!("mod `{mod_id}` declares no operation `{operation}`"),
            ));
        }
        registered
            .instance
            .invoke(operation, payload, CancelToken::new())
            .await
    }

    /// Every registered mod, in registration order.
    pub fn list(&self) -> Vec<ModEntry> {
        self.mods
            .iter()
            .map(|registered| registered.entry.clone())
            .collect()
    }

    fn available(&self) -> impl Iterator<Item = &Registered> {
        self.mods
            .iter()
            .filter(|registered| registered.entry.health.is_available())
    }
}

fn validate(manifest: &ModManifest, id_taken: bool) -> Result<(), Diagnostic> {
    let id = manifest.id.as_str();
    let reject = |code: &str, message: String| Err(caller_error(id, code, message));

    if RESERVED_NAMESPACES.contains(&id) {
        return reject(
            "RESERVED_ID",
            format!("mod id `{id}` is reserved by the kernel"),
        );
    }
    if !is_snake_case(id) {
        return reject(
            "INVALID_ID",
            format!("mod id `{id}` is not lowercase snake_case"),
        );
    }
    if id_taken {
        return reject(
            "DUPLICATE_ID",
            format!("mod id `{id}` is already registered"),
        );
    }
    match VersionReq::parse(&manifest.api) {
        Ok(requirement) if requirement.matches(&API_VERSION) => {}
        Ok(_) => {
            return reject(
                "INCOMPATIBLE_API",
                format!(
                    "mod `{id}` requires api `{}`; this kernel implements {API_VERSION}",
                    manifest.api
                ),
            );
        }
        Err(error) => {
            return reject(
                "INVALID_API",
                format!(
                    "mod `{id}` api `{}` is not a semver requirement: {error}",
                    manifest.api
                ),
            );
        }
    }

    let in_namespace = |name: &str| {
        name.strip_prefix(id)
            .and_then(|rest| rest.strip_prefix('.'))
            .is_some_and(|rest| !rest.is_empty())
    };
    let mut names = HashSet::new();
    for operation in &manifest.operations {
        if !in_namespace(&operation.name) {
            return reject(
                "OPERATION_OUTSIDE_NAMESPACE",
                format!("operation `{}` is not under `{id}.`", operation.name),
            );
        }
        if !names.insert(operation.name.as_str()) {
            return reject(
                "DUPLICATE_OPERATION",
                format!("operation `{}` is declared twice", operation.name),
            );
        }
    }
    let mut kinds = HashSet::new();
    for event in &manifest.events {
        if !in_namespace(&event.kind) {
            return reject(
                "EVENT_OUTSIDE_NAMESPACE",
                format!("event kind `{}` is not under `{id}.`", event.kind),
            );
        }
        if !kinds.insert(event.kind.as_str()) {
            return reject(
                "DUPLICATE_EVENT",
                format!("event kind `{}` is declared twice", event.kind),
            );
        }
    }
    Ok(())
}

fn is_snake_case(id: &str) -> bool {
    id.starts_with(|c: char| c.is_ascii_lowercase())
        && !id.ends_with('_')
        && !id.contains("__")
        && id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use async_trait::async_trait;
    use serde_json::json;

    use super::*;
    use crate::diagnostic::internal_error;
    use crate::manifest::{ModEvent, ModOperation, OperationAnnotations};

    struct Fake {
        manifest: ModManifest,
        fail_start: bool,
        calls: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl Mod for Fake {
        fn manifest(&self) -> &ModManifest {
            &self.manifest
        }

        async fn start(&self, _ctx: ModContext) -> Result<(), Diagnostic> {
            if self.fail_start {
                return Err(internal_error(&self.manifest.id, "START", "no backend"));
            }
            Ok(())
        }

        async fn invoke(
            &self,
            operation: &str,
            payload: Value,
            _cancel: CancelToken,
        ) -> Result<Value, Diagnostic> {
            self.calls.lock().unwrap().push(operation.to_owned());
            Ok(json!({ "echo": payload }))
        }

        async fn stop(&self) -> Result<(), Diagnostic> {
            Ok(())
        }
    }

    fn manifest(id: &str) -> ModManifest {
        ModManifest {
            id: id.to_owned(),
            name: id.to_owned(),
            version: "0.1.0".to_owned(),
            api: "0.1".to_owned(),
            operations: vec![operation(&format!("{id}.ping"))],
            events: vec![ModEvent {
                kind: format!("{id}.pinged"),
                schema: json!({ "type": "object" }),
            }],
        }
    }

    fn operation(name: &str) -> ModOperation {
        ModOperation {
            name: name.to_owned(),
            description: "ping".to_owned(),
            input_schema: json!({ "type": "object" }),
            output_schema: json!({ "type": "object" }),
            annotations: OperationAnnotations::default(),
        }
    }

    fn fake(manifest: ModManifest) -> Arc<Fake> {
        Arc::new(Fake {
            manifest,
            fail_start: false,
            calls: Mutex::new(Vec::new()),
        })
    }

    async fn registry(mods: Vec<Arc<Fake>>) -> ModRegistry {
        let mods = mods.into_iter().map(|m| m as Arc<dyn Mod>).collect();
        ModRegistry::start(mods, &EventBus::new()).await
    }

    fn rejection(registry: &ModRegistry, index: usize) -> String {
        match &registry.list()[index].health {
            Health::Unavailable(diagnostic) => diagnostic.code().as_str().to_owned(),
            Health::Available => panic!("mod {index} was accepted"),
        }
    }

    #[tokio::test]
    async fn invalid_manifests_are_unavailable_with_a_reason() {
        let mut outside_op = manifest("outside_op");
        outside_op.operations.push(operation("other.ping"));
        let mut prefix_only = manifest("prefix_only");
        prefix_only.operations.push(operation("prefix_onlyx.ping"));
        let mut twice_op = manifest("twice_op");
        twice_op.operations.push(operation("twice_op.ping"));
        let mut outside_event = manifest("outside_event");
        outside_event.events[0].kind = "board.changed".to_owned();
        let mut twice_event = manifest("twice_event");
        twice_event.events.push(twice_event.events[0].clone());
        let mut newer_api = manifest("newer_api");
        newer_api.api = "0.2".to_owned();
        let mut bad_api = manifest("bad_api");
        bad_api.api = "latest".to_owned();

        let cases = [
            (manifest("board"), "MOD/CALLER_RESERVED_ID"),
            (manifest("Bad-Id"), "MOD/CALLER_INVALID_ID"),
            (outside_op, "MOD/CALLER_OPERATION_OUTSIDE_NAMESPACE"),
            (prefix_only, "MOD/CALLER_OPERATION_OUTSIDE_NAMESPACE"),
            (twice_op, "MOD/CALLER_DUPLICATE_OPERATION"),
            (outside_event, "MOD/CALLER_EVENT_OUTSIDE_NAMESPACE"),
            (twice_event, "MOD/CALLER_DUPLICATE_EVENT"),
            (newer_api, "MOD/CALLER_INCOMPATIBLE_API"),
            (bad_api, "MOD/CALLER_INVALID_API"),
        ];
        for (manifest, code) in cases {
            let id = manifest.id.clone();
            let registry = registry(vec![fake(manifest), fake(self::manifest("good"))]).await;
            assert_eq!(rejection(&registry, 0), code, "{id}");
            assert!(registry.list()[1].health.is_available(), "{id} spared good");
            registry
                .invoke("good", "good.ping", json!({}))
                .await
                .expect("the other mod still works");
        }
    }

    #[tokio::test]
    async fn a_second_mod_with_the_same_id_is_rejected() {
        let first = fake(manifest("demo"));
        let registry = registry(vec![Arc::clone(&first), fake(manifest("demo"))]).await;

        assert!(registry.list()[0].health.is_available());
        assert_eq!(rejection(&registry, 1), "MOD/CALLER_DUPLICATE_ID");
        registry
            .invoke("demo", "demo.ping", json!({}))
            .await
            .expect("first registration serves");
        assert_eq!(first.calls.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn invoke_reaches_only_available_mods_and_declared_operations() {
        let broken = Arc::new(Fake {
            manifest: manifest("broken"),
            fail_start: true,
            calls: Mutex::new(Vec::new()),
        });
        let demo = fake(manifest("demo"));
        let registry = registry(vec![Arc::clone(&broken), Arc::clone(&demo)]).await;

        let output = registry
            .invoke("demo", "demo.ping", json!({ "n": 1 }))
            .await
            .expect("declared operation");
        assert_eq!(output, json!({ "echo": { "n": 1 } }));

        let code = |result: Result<Value, Diagnostic>| {
            result.expect_err("rejected").code().as_str().to_owned()
        };
        assert_eq!(
            code(registry.invoke("broken", "broken.ping", json!({})).await),
            "MOD/INTERNAL_START"
        );
        assert_eq!(
            code(registry.invoke("demo", "demo.pong", json!({})).await),
            "MOD/CALLER_UNKNOWN_OPERATION"
        );
        assert_eq!(
            code(registry.invoke("missing", "missing.ping", json!({})).await),
            "MOD/CALLER_UNKNOWN_MOD"
        );
        assert!(broken.calls.lock().unwrap().is_empty());
        assert_eq!(*demo.calls.lock().unwrap(), ["demo.ping"]);
    }
}
