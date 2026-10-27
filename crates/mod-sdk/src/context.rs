use std::sync::Arc;

use reimagine_core::diagnostic::Diagnostic;
use reimagine_core::event::{Event, EventBus};
use serde_json::Value;

use crate::diagnostic::caller_error;
use crate::manifest::ModManifest;

/// What the kernel hands a started mod.
#[derive(Debug, Clone)]
pub struct ModContext {
    manifest: Arc<ModManifest>,
    event_bus: EventBus,
}

impl ModContext {
    pub fn new(manifest: Arc<ModManifest>, event_bus: EventBus) -> Self {
        Self {
            manifest,
            event_bus,
        }
    }

    pub fn manifest(&self) -> &ModManifest {
        &self.manifest
    }

    /// Publish a mod event. `kind` must be declared in the manifest, which
    /// also keeps it inside the mod's namespace.
    pub fn publish(&self, kind: impl Into<String>, payload: Value) -> Result<(), Diagnostic> {
        let kind = kind.into();
        if !self.manifest.events.iter().any(|event| event.covers(&kind)) {
            return Err(caller_error(
                &self.manifest.id,
                "UNDECLARED_EVENT",
                format!(
                    "mod `{}` did not declare event kind `{kind}`",
                    self.manifest.id
                ),
            ));
        }
        self.event_bus.publish(Event::mod_event(kind, payload));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::manifest::ModEvent;

    #[test]
    fn publish_accepts_declared_kind() {
        let manifest = Arc::new(ModManifest {
            id: "sample".to_owned(),
            name: "Sample Mod".to_owned(),
            version: "0.1.0".to_owned(),
            api: "0.1".to_owned(),
            events: vec![ModEvent {
                kind: "sample.tick".to_owned(),
                schema: json!({}),
            }],
            operations: Vec::new(),
        });
        let bus = EventBus::new();
        let mut rx = bus.subscribe();
        let ctx = ModContext::new(manifest, bus);

        assert!(ctx.publish("sample.tick", json!({"n": 1})).is_ok());
        match rx.try_recv() {
            Ok(Event::Mod { kind, payload }) => {
                assert_eq!(kind, "sample.tick");
                assert_eq!(payload, json!({"n": 1}));
            }
            other => panic!("expected mod event, got {other:?}"),
        }
    }

    #[test]
    fn publish_rejects_undeclared_kind() {
        let manifest = Arc::new(ModManifest {
            id: "sample".to_owned(),
            name: "Sample Mod".to_owned(),
            version: "0.1.0".to_owned(),
            api: "0.1".to_owned(),
            events: Vec::new(),
            operations: Vec::new(),
        });
        let bus = EventBus::new();
        let ctx = ModContext::new(manifest, bus);

        let err = ctx
            .publish("sample.tick", json!({}))
            .expect_err("should reject undeclared");
        assert_eq!(err.code().as_str(), "MOD/CALLER_UNDECLARED_EVENT");
    }
}
