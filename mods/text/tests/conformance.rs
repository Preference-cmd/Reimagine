//! Conformance of the real `text-mod` binary.
//!
//! The binary is spawned as a child process, started through `ModRegistry`,
//! and every operation is driven over the JSON-RPC stdio protocol. Both the
//! asset (`doc`) and inline (`text`) forms are covered, as are the array
//! shapes of `split` and `join`.

#![allow(clippy::result_large_err)]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use mod_sdk::diagnostic::Diagnostic;
use mod_sdk::serde_json::{Value, json};
use mod_sdk::{CancelToken, EventBus, Invocation, Mod, ModRegistry, ProcessMod, ProcessModSpec};
use tempfile::TempDir;

const MOD_ID: &str = "text";

/// Spawn the real binary, start it through a registry and keep a temp dir for
/// inputs and output directories.
async fn start() -> (ModRegistry, TempDir) {
    let binary = env!("CARGO_BIN_EXE_text-mod");
    let module = ProcessMod::spawn(ProcessModSpec::new(binary))
        .await
        .expect("spawn text-mod");
    let module: Arc<dyn Mod> = Arc::new(module);
    let registry = ModRegistry::start(vec![module], &EventBus::new()).await;
    assert!(
        registry.list()[0].health.is_available(),
        "the text mod did not start"
    );
    let temp = tempfile::tempdir().expect("temp dir");
    (registry, temp)
}

async fn call(
    registry: &ModRegistry,
    operation: &str,
    input: Value,
    output_dir: &Path,
) -> Result<Value, Diagnostic> {
    registry
        .invoke(
            MOD_ID,
            Invocation::new(operation, input, output_dir, CancelToken::new()),
        )
        .await
}

fn code(result: Result<Value, Diagnostic>) -> String {
    result.expect_err("call fails").code().as_str().to_owned()
}

fn out_dir(temp: &TempDir, name: &str) -> std::path::PathBuf {
    let dir = temp.path().join(name);
    std::fs::create_dir_all(&dir).expect("output dir");
    dir
}

/// Check `output` against the operation's declared output schema.
fn validate_output(registry: &ModRegistry, operation: &str, output: &Value) {
    let manifest = &registry.list()[0].manifest;
    let spec = manifest
        .operations
        .iter()
        .find(|spec| spec.name == operation)
        .expect("declared operation");
    let validator = mod_sdk::schema::compile(&spec.output_schema).expect("compiled schema");
    assert_eq!(
        mod_sdk::schema::violations(&validator, output),
        None,
        "output of {operation} does not match its schema: {output}"
    );
}

#[tokio::test]
async fn manifest_declares_six_deterministic_project_local_operations() {
    let (registry, _temp) = start().await;
    let entry = registry.list().remove(0);
    assert_eq!(entry.manifest.id, "text");
    assert_eq!(entry.manifest.version, "0.1.0");
    assert_eq!(entry.manifest.api, "0.1");
    assert!(entry.manifest.events.is_empty());

    let names: Vec<&str> = entry
        .manifest
        .operations
        .iter()
        .map(|operation| operation.name.as_str())
        .collect();
    assert_eq!(
        names,
        [
            "text.normalize",
            "text.replace",
            "text.inspect",
            "text.template",
            "text.split",
            "text.join"
        ]
    );
    for operation in &entry.manifest.operations {
        assert!(operation.annotations.deterministic, "{}", operation.name);
        assert!(!operation.annotations.open_world, "{}", operation.name);
        assert!(!operation.annotations.destructive, "{}", operation.name);
        if operation.name != "text.inspect" {
            let properties = &operation.input_schema["properties"];
            if operation.name != "text.join" {
                assert_eq!(properties["doc"]["format"], "asset", "{}", operation.name);
                assert_eq!(
                    properties["doc"]["contentMediaType"], "text/*",
                    "{}",
                    operation.name
                );
                assert_eq!(
                    properties["text"]["format"], "inline-text",
                    "{}",
                    operation.name
                );
            }
        }
    }
    let annotation = |name: &str| {
        entry
            .manifest
            .operations
            .iter()
            .find(|operation| operation.name == name)
            .expect("declared operation")
            .annotations
    };
    assert!(annotation("text.normalize").idempotent);
    assert!(annotation("text.inspect").read_only);
    assert!(!annotation("text.inspect").idempotent);
    assert!(!annotation("text.replace").idempotent);
    registry.stop().await;
}

#[tokio::test]
async fn normalize_agrees_between_asset_and_inline_and_mirrors_the_form() {
    let (registry, temp) = start().await;
    let source = "Hello,  world!\r\n\r\nwide，punctuation！\r\n";
    let input_path = temp.path().join("input.md");
    std::fs::write(&input_path, source).expect("write input");

    let asset_dir = out_dir(&temp, "normalize-asset");
    let asset = call(
        &registry,
        "text.normalize",
        json!({
            "doc": input_path.to_str().expect("utf8 path"),
            "punctuation": "halfwidth"
        }),
        &asset_dir,
    )
    .await
    .expect("asset normalize");
    validate_output(&registry, "text.normalize", &asset);
    let name = asset["doc"].as_str().expect("doc output");
    assert!(name.ends_with(".md"), "{name}");
    assert!(asset.get("text").is_none());
    let from_asset = std::fs::read_to_string(asset_dir.join(name)).expect("read output");

    let inline_dir = out_dir(&temp, "normalize-inline");
    let inline = call(
        &registry,
        "text.normalize",
        json!({ "text": source, "punctuation": "halfwidth" }),
        &inline_dir,
    )
    .await
    .expect("inline normalize");
    validate_output(&registry, "text.normalize", &inline);
    let from_inline = inline["text"].as_str().expect("text output");
    assert_eq!(from_asset, from_inline);
    assert_eq!(from_inline, "Hello, world!\n\nwide,punctuation!\n");
    assert!(
        !inline_dir.join("output.md").exists(),
        "inline input must not produce an asset"
    );
    registry.stop().await;
}

#[tokio::test]
async fn replace_reports_counts_captures_words_and_literal_dollars() {
    let (registry, temp) = start().await;
    let input = json!({
        "text": "cat catalog\ncat.\nAlice@example.com",
        "pairs": [
            { "find": "cat", "replace": "dog", "mode": "word" },
            { "find": "@example.com", "replace": "$1", "mode": "literal" },
            { "find": r"(\w+)\$1", "replace": "$1 (at)", "mode": "regex" }
        ]
    });
    let dir = out_dir(&temp, "replace");
    let output = call(&registry, "text.replace", input.clone(), &dir)
        .await
        .expect("replace");
    validate_output(&registry, "text.replace", &output);
    assert_eq!(output["counts"], json!([2, 1, 1]));
    assert_eq!(output["text"], "dog catalog\ndog.\nAlice (at)");

    // The asset form writes the same text and still reports counts.
    let input_path = temp.path().join("replace.md");
    std::fs::write(&input_path, input["text"].as_str().expect("text")).expect("write");
    let mut asset_input = input.clone();
    asset_input.as_object_mut().expect("object").remove("text");
    asset_input["doc"] = json!(input_path.to_str().expect("utf8 path"));
    let asset_dir = out_dir(&temp, "replace-asset");
    let asset = call(&registry, "text.replace", asset_input, &asset_dir)
        .await
        .expect("asset replace");
    validate_output(&registry, "text.replace", &asset);
    assert_eq!(asset["counts"], json!([2, 1, 1]));
    assert_eq!(
        std::fs::read_to_string(asset_dir.join(asset["doc"].as_str().expect("doc"))).expect("read"),
        "dog catalog\ndog.\nAlice (at)"
    );
    registry.stop().await;
}

#[tokio::test]
async fn inspect_reports_title_headings_front_matter_and_stats() {
    let (registry, temp) = start().await;
    let body = "# Heading One\n\nIntro paragraph.\n\n## Section\n\n```rust\n# not a heading\n```\n\n### Deep\n";
    let text = format!("---\ntitle: Interview\nauthor: A\n---\n{body}");
    let output = call(
        &registry,
        "text.inspect",
        json!({ "text": text }),
        &out_dir(&temp, "inspect"),
    )
    .await
    .expect("inspect");
    validate_output(&registry, "text.inspect", &output);
    assert_eq!(output["title"], "Interview");
    assert_eq!(
        output["headings"],
        json!([
            { "level": 1, "text": "Heading One" },
            { "level": 2, "text": "Section" },
            { "level": 3, "text": "Deep" }
        ])
    );
    assert_eq!(
        output["front_matter"],
        json!({ "title": "Interview", "author": "A" })
    );
    assert_eq!(
        output["stats"],
        json!({
            "chars": body.chars().count(),
            "lines": body.lines().count(),
            "paragraphs": 5
        })
    );
    assert!(output.get("doc").is_none());
    assert!(output.get("text").is_none());
    registry.stop().await;
}

#[tokio::test]
async fn template_resolves_values_then_front_matter_then_body() {
    let (registry, temp) = start().await;
    let output = call(
        &registry,
        "text.template",
        json!({
            "text": "---\ntitle: From front matter\nauthor: A\n---\nBody text",
            "template": "{{title}}|{{author}}|{{count}}|{{body}}",
            "values": { "title": "From values", "count": 3 }
        }),
        &out_dir(&temp, "template"),
    )
    .await
    .expect("template");
    validate_output(&registry, "text.template", &output);
    assert_eq!(output["text"], "From values|A|3|Body text");
    registry.stop().await;
}

#[tokio::test]
async fn split_yields_docs_for_assets_and_pieces_for_inline() {
    let (registry, temp) = start().await;
    let source = "Intro\n\n## One\n\nfirst\n\n### Sub\n\nsub text\n\n## Two\n\nsecond\n";
    let input_path = temp.path().join("split.md");
    std::fs::write(&input_path, source).expect("write");
    let asset_dir = out_dir(&temp, "split-asset");
    let asset = call(
        &registry,
        "text.split",
        json!({
            "doc": input_path.to_str().expect("utf8 path"),
            "by": { "kind": "heading", "level": 2 }
        }),
        &asset_dir,
    )
    .await
    .expect("asset split");
    validate_output(&registry, "text.split", &asset);
    let docs = asset["docs"].as_array().expect("docs");
    assert_eq!(docs.len(), 2);
    assert_eq!(docs[0]["title"], "One");
    assert_eq!(docs[1]["title"], "Two");
    assert_eq!(
        std::fs::read_to_string(asset_dir.join(docs[0]["doc"].as_str().expect("doc")))
            .expect("read first piece"),
        "## One\n\nfirst\n\n### Sub\n\nsub text\n\n"
    );
    assert!(asset.get("pieces").is_none());

    let inline = call(
        &registry,
        "text.split",
        json!({
            "text": "one\n\n---\n\ntwo\n\n---\n\n",
            "by": { "kind": "separator", "separator": "\n---\n" }
        }),
        &out_dir(&temp, "split-inline"),
    )
    .await
    .expect("inline split");
    validate_output(&registry, "text.split", &inline);
    assert_eq!(
        inline["pieces"],
        json!([
            { "text": "one", "title": "one", "chars": 3 },
            { "text": "two", "title": "two", "chars": 3 }
        ])
    );

    let error = call(
        &registry,
        "text.split",
        json!({
            "text": "no headings here",
            "by": { "kind": "heading", "level": 1 }
        }),
        &out_dir(&temp, "split-none"),
    )
    .await;
    assert_eq!(code(error), "MOD/CALLER_NO_HEADINGS");
    registry.stop().await;
}

#[tokio::test]
async fn join_accepts_the_docs_from_split_and_renders_texts_with_a_template() {
    let (registry, temp) = start().await;
    let source = "## One\n\nfirst\n\n## Two\n\nsecond\n";
    let input_path = temp.path().join("join.md");
    std::fs::write(&input_path, source).expect("write");
    let split_dir = out_dir(&temp, "join-split");
    let split = call(
        &registry,
        "text.split",
        json!({
            "doc": input_path.to_str().expect("utf8 path"),
            "by": { "kind": "heading", "level": 2 }
        }),
        &split_dir,
    )
    .await
    .expect("split");
    let docs: Vec<Value> = split["docs"]
        .as_array()
        .expect("docs")
        .iter()
        .map(|item| {
            let path = split_dir.join(item["doc"].as_str().expect("doc"));
            json!({ "doc": path.to_str().expect("utf8 path"), "title": item["title"] })
        })
        .collect();

    let join_dir = out_dir(&temp, "join-asset");
    let joined = call(
        &registry,
        "text.join",
        json!({ "docs": docs, "separator": "\n\n" }),
        &join_dir,
    )
    .await
    .expect("asset join");
    validate_output(&registry, "text.join", &joined);
    assert_eq!(joined["doc"], "joined.md");
    assert_eq!(
        std::fs::read_to_string(join_dir.join("joined.md")).expect("read joined"),
        "## One\n\nfirst\n\n\n\n## Two\n\nsecond\n"
    );

    let texts = call(
        &registry,
        "text.join",
        json!({ "texts": ["one", "two"], "template": "<p>{{body}}</p>" }),
        &out_dir(&temp, "join-text"),
    )
    .await
    .expect("template join");
    validate_output(&registry, "text.join", &texts);
    assert_eq!(texts["text"], "<p>one</p><p>two</p>");
    registry.stop().await;
}

#[tokio::test]
async fn doc_and_text_are_exclusive() {
    let (registry, temp) = start().await;
    let input_path = temp.path().join("input.md");
    std::fs::write(&input_path, "text").expect("write");
    let dir = out_dir(&temp, "exclusive");

    let both = call(
        &registry,
        "text.normalize",
        json!({
            "doc": input_path.to_str().expect("utf8 path"),
            "text": "inline"
        }),
        &dir,
    )
    .await;
    assert_eq!(code(both), "MOD/CALLER_AMBIGUOUS_DOCUMENT");

    let neither = call(&registry, "text.normalize", json!({}), &dir).await;
    assert_eq!(code(neither), "MOD/CALLER_MISSING_DOCUMENT");
    registry.stop().await;
}

#[tokio::test]
async fn invalid_utf8_and_the_inline_cap_are_caller_diagnostics() {
    let (registry, temp) = start().await;
    let broken = temp.path().join("broken.md");
    std::fs::write(&broken, [0xff, 0xfe, 0x00]).expect("write");
    let error = call(
        &registry,
        "text.normalize",
        json!({ "doc": broken.to_str().expect("utf8 path") }),
        &out_dir(&temp, "utf8"),
    )
    .await;
    assert_eq!(code(error), "MOD/CALLER_INVALID_UTF8");

    let oversized = "x".repeat(256 * 1024 + 1);
    let error = call(
        &registry,
        "text.normalize",
        json!({ "text": oversized }),
        &out_dir(&temp, "cap"),
    )
    .await;
    assert_eq!(code(error), "MOD/CALLER_INLINE_TOO_LARGE");

    // Exactly at the cap is accepted.
    let at_cap = "x".repeat(256 * 1024);
    let accepted = call(
        &registry,
        "text.normalize",
        json!({ "text": at_cap }),
        &out_dir(&temp, "cap-ok"),
    )
    .await
    .expect("at the cap");
    assert_eq!(accepted["text"].as_str().expect("text").len(), 256 * 1024);
    registry.stop().await;
}

#[tokio::test]
async fn a_pathological_regex_returns_promptly() {
    let (registry, temp) = start().await;
    let input = format!("{}b", "a".repeat(100_000));
    let dir = out_dir(&temp, "regex");
    let call_future = call(
        &registry,
        "text.replace",
        json!({
            "text": input,
            "pairs": [{ "find": "(a+)+$", "replace": "x", "mode": "regex" }]
        }),
        &dir,
    );
    let output = tokio::time::timeout(Duration::from_secs(5), call_future)
        .await
        .expect("catastrophic backtracking regex must return promptly")
        .expect("replace");
    assert_eq!(output["counts"], json!([0]));
    assert_eq!(output["text"].as_str().expect("text"), input);
    registry.stop().await;
}
