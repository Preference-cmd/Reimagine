//! End-to-end host test for the text mod.
//!
//! The real `text-mod` binary is spawned as a child process, registered in a
//! [`ModRegistry`] and driven through a [`WorkspaceHost`] built on a temp base
//! path. The host performs the asset resolution, schema validation, step
//! recording and output ingestion, so this covers the contract the desktop app
//! uses rather than the mod in isolation.

#![allow(clippy::result_large_err)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use mod_sdk::serde_json::{Value, json};
use mod_sdk::{EventBus, Mod, ModRegistry, ProcessMod, ProcessModSpec};
use reimagine_app_host::{
    AssetStore, InvokeRequest, OperationProvider, WorkspaceHost, WorkspaceHostBuilder,
    WorkspaceScope,
};
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandProvenance, OperationContext,
};
use reimagine_core::model::{AssetId, ProjectId};
use reimagine_core::project::ProjectMetadata;
use reimagine_core::step::StepOutcome;
use reimagine_core::task::CancelToken;
use tempfile::TempDir;

const PROJECT: &str = "text-e2e";

struct Fixture {
    host: WorkspaceHost,
    project: ProjectId,
    project_dir: PathBuf,
    base: TempDir,
}

impl Fixture {
    async fn new() -> Self {
        let base = tempfile::tempdir().expect("temp dir");
        let event_bus = EventBus::new();
        let process_mod = ProcessMod::spawn(ProcessModSpec::new(env!("CARGO_BIN_EXE_text-mod")))
            .await
            .expect("spawn the real text-mod");
        let module: Arc<dyn Mod> = Arc::new(process_mod);
        let registry = Arc::new(ModRegistry::start(vec![module], &event_bus).await);
        assert!(
            registry
                .list()
                .first()
                .is_some_and(|entry| entry.health.is_available()),
            "the text mod must start"
        );

        let host = WorkspaceHostBuilder::new(WorkspaceScope::new("text-mod-host-e2e"), base.path())
            .event_bus(event_bus)
            .mod_registry(Arc::clone(&registry))
            .build()
            .await
            .expect("build the workspace host");

        let project = ProjectId::new(PROJECT);
        host.project_service()
            .create_project(
                project.clone(),
                ProjectMetadata::new(
                    "Text e2e",
                    "Host integration fixture",
                    "2026-01-01T00:00:00Z".into(),
                    "2026-01-01T00:00:00Z".into(),
                ),
            )
            .await
            .expect("create the project");
        let project_dir = host.config().paths().project_dir(project.as_str());

        Self {
            host,
            project,
            project_dir,
            base,
        }
    }

    fn request(&self, name: &str, input: Value) -> InvokeRequest {
        InvokeRequest {
            name: name.to_owned(),
            input,
            context: OperationContext::new(
                CommandActor::new(CommandActorKind::Human).with_id("test-user"),
                CommandProvenance::Direct,
            ),
            cancel: CancelToken::new(),
            run: None,
        }
    }

    async fn invoke(&self, name: &str, input: Value) -> Value {
        self.host
            .operation_service()
            .invoke(&self.project, self.request(name, input))
            .await
            .unwrap_or_else(|error| panic!("{name} failed: {error:?}"))
            .output
    }

    async fn preview(&self, name: &str, input: Value, key: &str) -> Value {
        self.host
            .operation_service()
            .preview(&self.project, name, input, key, CancelToken::new())
            .await
            .unwrap_or_else(|error| panic!("preview of {name} failed: {error:?}"))
    }

    async fn import(&self, path: &Path) -> String {
        let output = self
            .invoke("asset.import", json!({ "paths": [path.to_string_lossy()] }))
            .await;
        output["assets"]
            .as_array()
            .and_then(|assets| assets.first())
            .and_then(Value::as_str)
            .expect("imported asset id")
            .to_owned()
    }

    fn store(&self) -> AssetStore {
        AssetStore::new(&self.project_dir)
    }

    async fn asset_text(&self, asset_id: &str) -> String {
        let id = AssetId::try_new(asset_id).expect("valid asset id");
        let record = self
            .store()
            .get_asset(&id)
            .await
            .expect("read the asset record")
            .expect("the asset exists");
        std::fs::read_to_string(self.store().content_path(&record.hash)).expect("read asset bytes")
    }

    fn write_fixture(&self, name: &str, text: &str) -> PathBuf {
        let path = self.base.path().join(name);
        std::fs::write(&path, text).expect("write fixture");
        path
    }

    fn step_lines(&self) -> usize {
        std::fs::read_to_string(self.project_dir.join("steps.jsonl"))
            .map(|content| {
                content
                    .lines()
                    .filter(|line| !line.trim().is_empty())
                    .count()
            })
            .unwrap_or(0)
    }
}

#[tokio::test]
async fn op_list_reports_the_six_text_operations() {
    let fixture = Fixture::new().await;
    let operations = fixture
        .host
        .operation_service()
        .list(&fixture.project)
        .await;

    let text: Vec<&str> = operations
        .iter()
        .filter(|info| {
            info.provider
                == OperationProvider::Mod {
                    id: "text".to_owned(),
                }
        })
        .map(|info| info.spec.name.as_str())
        .collect();
    assert_eq!(
        text,
        [
            "text.normalize",
            "text.replace",
            "text.inspect",
            "text.template",
            "text.split",
            "text.join",
        ]
    );
    assert!(
        operations
            .iter()
            .filter(|info| info.provider
                == OperationProvider::Mod {
                    id: "text".to_owned()
                })
            .all(|info| info.health.is_available()),
        "every text operation must be available"
    );
    fixture.host.shutdown().await;
}

#[tokio::test]
async fn replace_imports_an_asset_and_records_the_step_origin() {
    let fixture = Fixture::new().await;
    let source = fixture.write_fixture(
        "replace.md",
        "The quick brown fox jumps over the lazy dog.\n",
    );
    let input_asset = fixture.import(&source).await;

    let request = fixture.request(
        "text.replace",
        json!({
            "doc": input_asset,
            "pairs": [
                { "find": "fox", "replace": "cat", "mode": "word" },
                { "find": "dog", "replace": "wolf", "mode": "word" }
            ]
        }),
    );
    let result = fixture
        .host
        .operation_service()
        .invoke(&fixture.project, request)
        .await
        .expect("text.replace");

    assert_eq!(result.step.op, "text.replace");
    assert_eq!(result.step.op_version, "0.1.0");
    assert!(matches!(result.step.outcome, StepOutcome::Succeeded { .. }));
    assert_eq!(result.output["counts"], json!([1, 1]));

    let output_asset = result.output["doc"]
        .as_str()
        .expect("replaced output must be an asset")
        .to_owned();
    assert_ne!(output_asset, input_asset);

    let record = fixture
        .store()
        .get_asset(&AssetId::try_new(&output_asset).expect("asset id"))
        .await
        .expect("read output record")
        .expect("the output asset is registered");
    assert_eq!(
        record.origin, result.step.id,
        "the output asset must trace back to the replace step"
    );
    assert_eq!(record.media_type, "text/markdown");
    assert_eq!(
        fixture.asset_text(&output_asset).await,
        "The quick brown cat jumps over the lazy wolf.\n"
    );

    fixture.host.shutdown().await;
}

#[tokio::test]
async fn preview_returns_inline_text_without_writing_a_step() {
    let fixture = Fixture::new().await;
    let before = fixture.step_lines();

    let preview = fixture
        .preview(
            "text.replace",
            json!({
                "text": "AI is useful",
                "pairs": [{ "find": "AI", "replace": "A.I.", "mode": "literal" }]
            }),
            "replace-inline",
        )
        .await;

    assert_eq!(preview["text"], "A.I. is useful");
    assert_eq!(preview["counts"], json!([1]));
    assert_eq!(
        fixture.step_lines(),
        before,
        "a preview must not write a step record"
    );

    fixture.host.shutdown().await;
}

#[tokio::test]
async fn split_creates_a_collection_and_join_keeps_slot_order() {
    let fixture = Fixture::new().await;
    let source = fixture.write_fixture(
        "split.md",
        "# One\n\nfirst\n\n# Two\n\nsecond\n\n# Three\n\nthird\n",
    );
    let input_asset = fixture.import(&source).await;

    let split = fixture
        .invoke(
            "text.split",
            json!({
                "doc": input_asset,
                "by": { "kind": "heading", "level": 1 }
            }),
        )
        .await;
    let pieces = split["docs"].as_array().expect("split docs").clone();
    assert_eq!(pieces.len(), 3);
    assert_eq!(
        pieces
            .iter()
            .map(|piece| piece["title"].as_str().expect("title"))
            .collect::<Vec<_>>(),
        ["One", "Two", "Three"]
    );
    let piece_ids: Vec<String> = pieces
        .iter()
        .map(|piece| piece["doc"].as_str().expect("piece asset").to_owned())
        .collect();

    let created = fixture
        .invoke(
            "collection.create",
            json!({ "name": "Split pieces", "assets": piece_ids }),
        )
        .await;
    let collection_id = created["collection_id"].as_str().expect("collection id");

    let collection = fixture
        .invoke("collection.get", json!({ "collection_id": collection_id }))
        .await;
    let slots = collection["slots"].as_array().expect("collection slots");
    let slot_assets: Vec<String> = slots
        .iter()
        .map(|slot| slot["asset_id"].as_str().expect("slot asset").to_owned())
        .collect();
    assert_eq!(
        slot_assets, piece_ids,
        "the collection slots must keep the split order"
    );

    let join_docs: Vec<Value> = slot_assets
        .iter()
        .map(|asset| json!({ "doc": asset }))
        .collect();
    let joined = fixture
        .invoke(
            "text.join",
            json!({ "docs": join_docs, "separator": "\n\n" }),
        )
        .await;

    let mut expected = Vec::new();
    for asset in &slot_assets {
        expected.push(fixture.asset_text(asset).await);
    }
    let expected = expected.join("\n\n");
    let joined_asset = joined["doc"].as_str().expect("joined asset");
    assert_eq!(fixture.asset_text(joined_asset).await, expected);
    assert!(expected.starts_with("# One"));
    assert!(expected.contains("# Two"));
    assert!(expected.ends_with("# Three\n\nthird\n"));

    fixture.host.shutdown().await;
}

#[tokio::test]
async fn an_ambiguous_document_is_diagnosed() {
    let fixture = Fixture::new().await;
    let source = fixture.write_fixture("ambiguous.md", "text\n");
    let input_asset = fixture.import(&source).await;

    let error = fixture
        .host
        .operation_service()
        .invoke(
            &fixture.project,
            fixture.request(
                "text.replace",
                json!({
                    "doc": input_asset,
                    "text": "also inline",
                    "pairs": [{ "find": "text", "replace": "x", "mode": "literal" }]
                }),
            ),
        )
        .await
        .expect_err("both doc and text must be rejected");

    assert_eq!(error.code().as_str(), "MOD/CALLER_AMBIGUOUS_DOCUMENT");

    fixture.host.shutdown().await;
}
