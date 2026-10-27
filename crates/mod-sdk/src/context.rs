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

    fn context(bus: &EventBus) -> ModContext {
        let event = |kind: &str| ModEvent {
            kind: kind.to_owned(),
            schema: json!({ "type": "object" }),
        };
        let manifest = ModManifest {
            id: "demo".to_owned(),
            name: "Demo".to_owned(),
            version: "0.1.0".to_owned(),
            api: "0.1".to_owned(),
            operations: Vec::new(),
            events: vec![event("demo.changed"), event("demo.download.*")],
        };
        ModContext::new(Arc::new(manifest), bus.clone())
    }

    #[test]
    fn declared_kinds_reach_the_bus() {
        let bus = EventBus::new();
        let mut rx = bus.subscribe();
        let ctx = context(&bus);

        ctx.publish("demo.changed", json!({})).expect("exact kind");
        ctx.publish("demo.download.progress", json!({ "n": 1 }))
            .expect("family member");

        let kinds: Vec<String> = std::iter::from_fn(|| rx.try_recv().ok())
            .map(|event| match event {
                Event::Mod { kind, .. } => kind,
                other => panic!("unexpected event {other:?}"),
            })
            .collect();
        assert_eq!(kinds, ["demo.changed", "demo.download.progress"]);
    }

    #[test]
    fn undeclared_and_foreign_kinds_are_rejected() {
        let bus = EventBus::new();
        let mut rx = bus.subscribe();
        let ctx = context(&bus);

        for kind in [
            "demo.other",
            "other.changed",
            "demo.download",
            "demo.downloads.x",
        ] {
            let error = ctx.publish(kind, json!({})).expect_err(kind);
            assert_eq!(error.code().as_str(), "MOD/CALLER_UNDECLARED_EVENT");
        }
        assert!(rx.try_recv().is_err(), "nothing was published");
    }
}
