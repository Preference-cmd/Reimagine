//! Interface conformance, run against the sample mod. Every behavior test
//! runs once per way of hosting the mod and must hold for each.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use mod_sdk::testing::{SampleMod, check_manifest, validate};
use mod_sdk::{
    Mod, ModContext, ModManifest, ModOperation, ModRegistry, OperationAnnotations, ProcessMod,
    ProcessModSpec,
};
use reimagine_core::diagnostic::Diagnostic;
use reimagine_core::event::{Event, EventBus, EventReceiver};
use reimagine_core::health::Health;
use reimagine_core::task::CancelToken;
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy)]
enum Host {
    InProcess,
    Process,
}

const HOSTS: &[Host] = &[Host::InProcess, Host::Process];

/// The sample mod hosted as `host`; `failing` makes its `start` fail.
async fn sample(host: Host, failing: bool) -> Arc<dyn Mod> {
    match host {
        Host::InProcess if failing => Arc::new(SampleMod::failing_start()),
        Host::InProcess => Arc::new(SampleMod::new()),
        Host::Process => {
            let bin = env!("CARGO_BIN_EXE_sample-mod");
            let mut spec = ProcessModSpec::new(bin);
            if failing {
                spec.args.push("failing".into());
            }
            let proc = ProcessMod::spawn(spec).await.expect("spawn sample-mod");
            Arc::new(proc)
        }
    }
}

async fn start(mods: Vec<Arc<dyn Mod>>) -> (ModRegistry, EventReceiver) {
    let bus = EventBus::new();
    let rx = bus.subscribe();
    (ModRegistry::start(mods, &bus).await, rx)
}

async fn call(
    registry: &ModRegistry,
    operation: &str,
    payload: Value,
) -> Result<Value, Diagnostic> {
    registry
        .invoke("sample", operation, payload, CancelToken::new())
        .await
}

fn code(result: Result<Value, Diagnostic>) -> String {
    result.expect_err("call fails").code().as_str().to_owned()
}

/// The payload of the next mod event of `kind`, skipping any other event.
async fn next_event(rx: &mut EventReceiver, kind: &str) -> Value {
    let wait = async {
        loop {
            match rx.recv().await.expect("bus open") {
                Event::Mod { kind: k, payload } if k == kind => return payload,
                _ => {}
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(5), wait)
        .await
        .unwrap_or_else(|_| panic!("no `{kind}` event"))
}

fn event_schema<'m>(manifest: &'m ModManifest, kind: &str) -> &'m Value {
    &manifest
        .events
        .iter()
        .find(|event| event.covers(kind))
        .expect("declared event")
        .schema
}

/// A second, always-working mod to show a broken one does not affect it.
struct Pong(ModManifest);

impl Pong {
    fn new() -> Self {
        Self(ModManifest {
            id: "pong".to_owned(),
            name: "Pong".to_owned(),
            version: "0.1.0".to_owned(),
            api: "0.1".to_owned(),
            operations: vec![ModOperation {
                name: "pong.ping".to_owned(),
                description: "Answer.".to_owned(),
                input_schema: json!({ "type": "object" }),
                output_schema: json!({ "type": "object" }),
                annotations: OperationAnnotations::default(),
            }],
            events: Vec::new(),
        })
    }
}

#[async_trait]
impl Mod for Pong {
    fn manifest(&self) -> &ModManifest {
        &self.0
    }
    async fn start(&self, _ctx: ModContext) -> Result<(), Diagnostic> {
        Ok(())
    }
    async fn invoke(&self, _: &str, _: Value, _: CancelToken) -> Result<Value, Diagnostic> {
        Ok(json!({ "pong": true }))
    }
    async fn stop(&self) -> Result<(), Diagnostic> {
        Ok(())
    }
}

#[tokio::test]
async fn the_sample_manifest_conforms() {
    for &host in HOSTS {
        let sample = sample(host, false).await;
        assert_eq!(check_manifest(sample.manifest()), Ok(()), "{host:?}");
    }
}

#[test]
fn check_manifest_reports_every_problem() {
    let mut manifest = SampleMod::new().manifest().clone();
    manifest.operations[0].output_schema = json!({ "type": "string" });
    manifest.operations[1].input_schema["examples"] = json!([{ "kind": "other" }]);
    manifest.operations[2]
        .input_schema
        .as_object_mut()
        .unwrap()
        .remove("examples");
    manifest.operations[2].description = " ".to_owned();
    manifest.events[0].schema = json!({ "type": 7 });

    let problems = check_manifest(&manifest).expect_err("problems");
    let expected = [
        "event `sample.echoed` schema is not valid JSON Schema",
        "`sample.echo` output_schema root must be",
        "`sample.fail` example 0 fails its input_schema",
        "`sample.wait` input_schema has no `examples`",
        "`sample.wait` has no description",
    ];
    for needle in expected {
        assert!(
            problems.iter().any(|p| p.contains(needle)),
            "missing `{needle}` in {problems:?}"
        );
    }
    assert_eq!(problems.len(), expected.len(), "{problems:?}");

    let mut bad_name = SampleMod::new().manifest().clone();
    bad_name.operations[0].name = format!("sample.{}", "x".repeat(130));
    let problems = check_manifest(&bad_name).expect_err("too long");
    assert!(
        problems
            .iter()
            .any(|p| p.contains("not a valid MCP tool name"))
    );
}

#[tokio::test]
async fn echo_returns_its_input_and_publishes_a_declared_event() {
    for &host in HOSTS {
        let sample = sample(host, false).await;
        let manifest = sample.manifest().clone();
        let (registry, mut rx) = start(vec![sample]).await;

        let output = call(&registry, "sample.echo", json!({ "text": "hi" }))
            .await
            .expect("echo");
        assert_eq!(output, json!({ "text": "hi" }), "{host:?}");
        validate(&manifest.operations[0].output_schema, &output).expect("output schema");

        let payload = next_event(&mut rx, "sample.echoed").await;
        assert_eq!(payload, json!({ "text": "hi" }), "{host:?}");
        validate(event_schema(&manifest, "sample.echoed"), &payload).expect("event schema");
    }
}

#[tokio::test]
async fn input_that_fails_the_schema_never_reaches_the_mod() {
    for &host in HOSTS {
        let (registry, mut rx) = start(vec![sample(host, false).await]).await;

        for bad in [
            json!({}),
            json!({ "text": 1 }),
            json!({ "text": "a", "x": 1 }),
        ] {
            let error = call(&registry, "sample.echo", bad.clone())
                .await
                .expect_err("rejected");
            assert_eq!(
                error.code().as_str(),
                "MOD/CALLER_INVALID_INPUT",
                "{host:?} {bad}"
            );
        }
        // The first echo event comes from the first valid call.
        call(&registry, "sample.echo", json!({ "text": "ok" }))
            .await
            .expect("valid call");
        assert_eq!(
            next_event(&mut rx, "sample.echoed").await,
            json!({ "text": "ok" }),
            "{host:?}"
        );
    }
}

#[tokio::test]
async fn caller_and_internal_errors_keep_their_class() {
    for &host in HOSTS {
        let (registry, _rx) = start(vec![sample(host, false).await]).await;

        let caller = call(&registry, "sample.fail", json!({ "kind": "caller" })).await;
        assert_eq!(code(caller), "MOD/CALLER_SAMPLE", "{host:?}");
        let internal = call(&registry, "sample.fail", json!({ "kind": "internal" })).await;
        assert_eq!(code(internal), "MOD/INTERNAL_SAMPLE", "{host:?}");
    }
}

#[tokio::test]
async fn cancelling_a_call_returns_cancelled() {
    for &host in HOSTS {
        let (registry, mut rx) = start(vec![sample(host, false).await]).await;
        let registry = Arc::new(registry);
        let cancel = CancelToken::new();

        let call = tokio::spawn({
            let registry = Arc::clone(&registry);
            let cancel = cancel.clone();
            async move {
                registry
                    .invoke("sample", "sample.wait", json!({}), cancel)
                    .await
            }
        });
        assert_eq!(
            next_event(&mut rx, "sample.progress.started").await,
            json!({ "stage": "started" })
        );
        cancel.cancel();

        let result = tokio::time::timeout(Duration::from_secs(5), call)
            .await
            .expect("call ends after cancel")
            .expect("no panic");
        assert_eq!(code(result), "MOD/CALLER_CANCELLED", "{host:?}");
    }
}

#[tokio::test]
async fn a_mod_that_fails_to_start_is_unavailable_alone() {
    for &host in HOSTS {
        let (registry, _rx) = start(vec![sample(host, true).await, Arc::new(Pong::new())]).await;

        let entries = registry.list();
        match &entries[0].health {
            Health::Unavailable(diagnostic) => {
                assert_eq!(
                    diagnostic.code().as_str(),
                    "MOD/INTERNAL_START_FAILED",
                    "{host:?}"
                );
            }
            Health::Available => panic!("{host:?}: failed start is listed available"),
        }
        let echo = call(&registry, "sample.echo", json!({ "text": "hi" })).await;
        assert_eq!(code(echo), "MOD/INTERNAL_START_FAILED", "{host:?}");

        assert!(entries[1].health.is_available());
        registry
            .invoke("pong", "pong.ping", json!({}), CancelToken::new())
            .await
            .expect("the other mod works");
    }
}

#[tokio::test]
async fn stop_reaches_the_started_mod() {
    for &host in HOSTS {
        let (registry, mut rx) = start(vec![sample(host, false).await]).await;

        registry.stop().await;

        assert_eq!(
            next_event(&mut rx, "sample.progress.stopped").await,
            json!({ "stage": "stopped" }),
            "{host:?}"
        );
    }
}
