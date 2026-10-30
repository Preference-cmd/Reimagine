//! Read-only list operations behind the workbench library table:
//! `collection.list`, `dynamic_workflow.list` and `asset.list`.

use std::fs;
use std::path::Path;

use reimagine_app_host::{InvokeRequest, WorkspaceHost, WorkspaceHostBuilder, WorkspaceScope};
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandProvenance, OperationContext,
};
use reimagine_core::model::ProjectId;
use reimagine_core::project::ProjectMetadata;
use reimagine_core::task::CancelToken;
use serde_json::{Value, json};

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-library-lists-{prefix}-{nonce}"))
}

fn test_context() -> OperationContext {
    OperationContext::new(
        CommandActor::new(CommandActorKind::Human).with_id("test-user"),
        CommandProvenance::Direct,
    )
}

async fn build_host(base: &Path) -> WorkspaceHost {
    WorkspaceHostBuilder::new(WorkspaceScope::new("test-library-lists"), base)
        .build()
        .await
        .expect("build host")
}

fn project_dir(host: &WorkspaceHost, project: &ProjectId) -> std::path::PathBuf {
    host.config().paths().project_dir(project.as_str())
}

async fn create_project(host: &WorkspaceHost, id: &str) -> ProjectId {
    let project_id = ProjectId::new(id);
    host.project_service()
        .create_project(
            project_id.clone(),
            ProjectMetadata::new(
                "Library test",
                "test project",
                "2026-01-01T00:00:00Z".into(),
                "2026-01-01T00:00:00Z".into(),
            ),
        )
        .await
        .expect("create project");
    project_id
}

async fn invoke(host: &WorkspaceHost, project: &ProjectId, name: &str, input: Value) -> Value {
    host.operation_service()
        .invoke(
            project,
            InvokeRequest {
                name: name.to_owned(),
                input,
                context: test_context(),
                cancel: CancelToken::new(),
                run: None,
            },
        )
        .await
        .unwrap_or_else(|error| panic!("{name} failed: {}", error.message()))
        .output
}

async fn import_files(
    host: &WorkspaceHost,
    project: &ProjectId,
    base: &Path,
    files: &[(&str, &str)],
) -> Vec<String> {
    let mut paths = Vec::new();
    for (name, content) in files {
        let path = base.join(format!("import-{name}"));
        fs::write(&path, content).expect("write import source");
        paths.push(path.to_string_lossy().to_string());
    }
    let output = invoke(host, project, "asset.import", json!({ "paths": paths })).await;
    output["assets"]
        .as_array()
        .expect("assets array")
        .iter()
        .map(|value| value.as_str().expect("asset id").to_owned())
        .collect()
}

async fn create_collection(
    host: &WorkspaceHost,
    project: &ProjectId,
    name: &str,
    assets: &[&str],
) -> String {
    let output = invoke(
        host,
        project,
        "collection.create",
        json!({ "name": name, "assets": assets }),
    )
    .await;
    output["collection_id"]
        .as_str()
        .expect("collection id")
        .to_owned()
}

async fn create_workflow(host: &WorkspaceHost, project: &ProjectId, name: &str) -> String {
    let output = invoke(
        host,
        project,
        "dynamic_workflow.create",
        json!({
            "name": name,
            "workflow": {
                "inputs": {},
                "params": { "type": "object" },
                "steps": [
                    { "id": "first", "op": "test.step", "input": {} },
                    { "id": "second", "op": "test.step", "input": {} }
                ],
                "output": { "$from": "steps.second.out" }
            }
        }),
    )
    .await;
    output["dynamic_workflow_id"]
        .as_str()
        .expect("dynamic workflow id")
        .to_owned()
}

#[tokio::test]
async fn list_operations_return_the_project_library() {
    let base = temp_dir("library");
    let host = build_host(&base).await;
    let project = create_project(&host, "proj-library").await;

    let assets = import_files(
        &host,
        &project,
        &base,
        &[("a.txt", "alpha"), ("b.txt", "beta")],
    )
    .await;
    assert_eq!(assets.len(), 2);

    let alpha_id = create_collection(&host, &project, "Alpha", &[assets[0].as_str()]).await;
    let beta_id = create_collection(
        &host,
        &project,
        "Beta",
        &[assets[0].as_str(), assets[1].as_str()],
    )
    .await;
    let workflow_id = create_workflow(&host, &project, "Flow One").await;

    let listed = invoke(&host, &project, "collection.list", json!({})).await;
    let collections = listed["collections"].as_array().expect("collections array");
    assert_eq!(collections.len(), 2);
    let alpha = collections
        .iter()
        .find(|collection| collection["name"] == json!("Alpha"))
        .expect("Alpha listed");
    assert_eq!(alpha["id"].as_str(), Some(alpha_id.as_str()));
    assert_eq!(alpha["version"].as_u64(), Some(0));
    assert_eq!(alpha["slot_count"].as_u64(), Some(1));
    let beta = collections
        .iter()
        .find(|collection| collection["name"] == json!("Beta"))
        .expect("Beta listed");
    assert_eq!(beta["id"].as_str(), Some(beta_id.as_str()));
    assert_eq!(beta["version"].as_u64(), Some(0));
    assert_eq!(beta["slot_count"].as_u64(), Some(2));

    let listed = invoke(&host, &project, "dynamic_workflow.list", json!({})).await;
    let workflows = listed["workflows"].as_array().expect("workflows array");
    assert_eq!(workflows.len(), 1);
    assert_eq!(workflows[0]["id"].as_str(), Some(workflow_id.as_str()));
    assert_eq!(workflows[0]["name"].as_str(), Some("Flow One"));
    assert_eq!(workflows[0]["version"].as_u64(), Some(1));
    assert_eq!(workflows[0]["step_count"].as_u64(), Some(2));

    let listed = invoke(&host, &project, "asset.list", json!({})).await;
    let records = listed["assets"].as_array().expect("assets array");
    assert_eq!(records.len(), 2);
    assert_eq!(
        records[0]["id"].as_str(),
        Some(assets[1].as_str()),
        "assets are listed newest first"
    );
    assert_eq!(records[1]["id"].as_str(), Some(assets[0].as_str()));
    assert_eq!(records[0]["media_type"].as_str(), Some("text/plain"));
    assert_eq!(records[0]["size"].as_u64(), Some("beta".len() as u64));
    assert_eq!(records[1]["size"].as_u64(), Some("alpha".len() as u64));
    assert!(
        records[0]["origin"]
            .as_str()
            .is_some_and(|origin| !origin.is_empty())
    );

    // The three list operations are registered read-only, so invoking them
    // records no step and mutates no document.
    let infos = host.operation_service().list(&project).await;
    for name in ["collection.list", "dynamic_workflow.list", "asset.list"] {
        let info = infos
            .iter()
            .find(|info| info.spec.name == name)
            .unwrap_or_else(|| panic!("{name} is registered"));
        assert!(info.spec.annotations.read_only, "{name} must be read-only");
    }

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn list_operations_are_empty_for_unknown_and_empty_projects() {
    let base = temp_dir("empty");
    let host = build_host(&base).await;
    let empty = create_project(&host, "proj-empty").await;
    let unknown = ProjectId::new("proj-missing");

    for project in [&empty, &unknown] {
        for (name, field) in [
            ("collection.list", "collections"),
            ("dynamic_workflow.list", "workflows"),
            ("asset.list", "assets"),
        ] {
            let listed = invoke(&host, project, name, json!({})).await;
            assert_eq!(
                listed[field].as_array().map(Vec::len),
                Some(0),
                "{name} must list nothing for `{}`",
                project.as_str()
            );
        }
    }

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn list_operations_record_no_step() {
    let base = temp_dir("no-steps");
    let host = build_host(&base).await;
    let project = create_project(&host, "proj-no-steps").await;

    for name in ["collection.list", "dynamic_workflow.list", "asset.list"] {
        let _ = invoke(&host, &project, name, json!({})).await;
    }

    let steps_path = project_dir(&host, &project).join("steps.jsonl");
    let empty = match fs::metadata(&steps_path) {
        Ok(metadata) => metadata.len() == 0,
        Err(error) => error.kind() == std::io::ErrorKind::NotFound,
    };
    assert!(
        empty,
        "read-only list operations must not append to `{}`",
        steps_path.display()
    );

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn list_operations_return_records_after_a_host_reopen() {
    let base = temp_dir("reopen");
    let host = build_host(&base).await;
    let project = create_project(&host, "proj-reopen").await;

    let assets = import_files(&host, &project, &base, &[("a.txt", "alpha")]).await;
    let collection_id = create_collection(&host, &project, "Alpha", &[assets[0].as_str()]).await;
    let workflow_id = create_workflow(&host, &project, "Flow One").await;

    host.shutdown().await;
    drop(host);

    // A new host over the same root sees the persisted documents.
    let host = build_host(&base).await;

    let listed = invoke(&host, &project, "collection.list", json!({})).await;
    let collections = listed["collections"].as_array().expect("collections array");
    assert_eq!(collections.len(), 1);
    assert_eq!(collections[0]["id"].as_str(), Some(collection_id.as_str()));
    assert_eq!(collections[0]["name"].as_str(), Some("Alpha"));
    assert_eq!(collections[0]["slot_count"].as_u64(), Some(1));

    let listed = invoke(&host, &project, "dynamic_workflow.list", json!({})).await;
    let workflows = listed["workflows"].as_array().expect("workflows array");
    assert_eq!(workflows.len(), 1);
    assert_eq!(workflows[0]["id"].as_str(), Some(workflow_id.as_str()));
    assert_eq!(workflows[0]["name"].as_str(), Some("Flow One"));
    assert_eq!(workflows[0]["step_count"].as_u64(), Some(2));

    let listed = invoke(&host, &project, "asset.list", json!({})).await;
    let records = listed["assets"].as_array().expect("assets array");
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["id"].as_str(), Some(assets[0].as_str()));

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn asset_list_ignores_a_torn_last_line() {
    let base = temp_dir("torn");
    let host = build_host(&base).await;
    let project = create_project(&host, "proj-torn").await;

    let assets = import_files(&host, &project, &base, &[("a.txt", "alpha")]).await;
    assert_eq!(assets.len(), 1);

    let jsonl_path = host
        .config()
        .paths()
        .project_dir(project.as_str())
        .join("assets.jsonl");
    let mut contents = fs::read_to_string(&jsonl_path).expect("read assets.jsonl");
    contents.push_str(r#"{"id":"asset-torn","hash":"#);
    fs::write(&jsonl_path, contents).expect("append torn line");

    let listed = invoke(&host, &project, "asset.list", json!({})).await;
    let records = listed["assets"].as_array().expect("assets array");
    assert_eq!(
        records.len(),
        1,
        "a torn trailing line is not a completed record"
    );
    assert_eq!(records[0]["id"].as_str(), Some(assets[0].as_str()));

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn asset_import_after_a_torn_tail_keeps_both_records() {
    let base = temp_dir("torn-append");
    let host = build_host(&base).await;
    let project = create_project(&host, "proj-torn-append").await;

    let first = import_files(&host, &project, &base, &[("a.txt", "alpha")]).await;
    let jsonl_path = project_dir(&host, &project).join("assets.jsonl");
    let mut contents = fs::read_to_string(&jsonl_path).expect("read assets.jsonl");
    contents.push_str(r#"{"id":"asset-torn","hash":"#);
    fs::write(&jsonl_path, contents).expect("append torn line");

    // The next import must terminate the torn line before appending, so the
    // new record parses instead of merging with the partial one.
    let second = import_files(&host, &project, &base, &[("b.txt", "beta")]).await;
    assert_eq!(second.len(), 1);

    let listed = invoke(&host, &project, "asset.list", json!({})).await;
    let records = listed["assets"].as_array().expect("assets array");
    assert_eq!(
        records.len(),
        2,
        "both complete records are listed and the torn line is ignored"
    );
    assert_eq!(records[0]["id"].as_str(), Some(second[0].as_str()));
    assert_eq!(records[1]["id"].as_str(), Some(first[0].as_str()));

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn asset_list_skips_a_complete_but_invalid_line() {
    let base = temp_dir("invalid-line");
    let host = build_host(&base).await;
    let project = create_project(&host, "proj-invalid-line").await;

    let assets = import_files(&host, &project, &base, &[("a.txt", "alpha")]).await;
    let jsonl_path = project_dir(&host, &project).join("assets.jsonl");
    let mut contents = fs::read_to_string(&jsonl_path).expect("read assets.jsonl");
    // Syntactically valid JSON that is not an asset record, terminated by a
    // newline so it counts as a completed line.
    contents.push_str("{\"id\":\"asset-invalid\",\"media_type\":\"text/plain\"}\n");
    fs::write(&jsonl_path, contents).expect("append invalid record");

    let listed = invoke(&host, &project, "asset.list", json!({})).await;
    let records = listed["assets"].as_array().expect("assets array");
    assert_eq!(records.len(), 1, "an invalid completed line is ignored");
    assert_eq!(records[0]["id"].as_str(), Some(assets[0].as_str()));

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn list_operations_are_isolated_per_project() {
    let base = temp_dir("isolation");
    let host = build_host(&base).await;
    let alpha = create_project(&host, "proj-alpha").await;
    let beta = create_project(&host, "proj-beta").await;

    let alpha_assets = import_files(&host, &alpha, &base, &[("alpha.txt", "alpha")]).await;
    let beta_assets = import_files(&host, &beta, &base, &[("beta.txt", "beta")]).await;
    assert_eq!(alpha_assets.len(), 1);
    assert_eq!(beta_assets.len(), 1);
    assert_ne!(alpha_assets[0], beta_assets[0]);

    let alpha_collection =
        create_collection(&host, &alpha, "Alpha only", &[alpha_assets[0].as_str()]).await;
    let beta_collection =
        create_collection(&host, &beta, "Beta only", &[beta_assets[0].as_str()]).await;
    let alpha_workflow = create_workflow(&host, &alpha, "Alpha flow").await;
    let beta_workflow = create_workflow(&host, &beta, "Beta flow").await;

    let cases = [
        (
            &alpha,
            alpha_collection.as_str(),
            alpha_workflow.as_str(),
            alpha_assets[0].as_str(),
        ),
        (
            &beta,
            beta_collection.as_str(),
            beta_workflow.as_str(),
            beta_assets[0].as_str(),
        ),
    ];
    for (project, collection_id, workflow_id, asset_id) in cases {
        let listed = invoke(&host, project, "collection.list", json!({})).await;
        let collections = listed["collections"].as_array().expect("collections array");
        assert_eq!(
            collections.len(),
            1,
            "only `{}`'s collection is listed",
            project.as_str()
        );
        assert_eq!(collections[0]["id"].as_str(), Some(collection_id));

        let listed = invoke(&host, project, "dynamic_workflow.list", json!({})).await;
        let workflows = listed["workflows"].as_array().expect("workflows array");
        assert_eq!(
            workflows.len(),
            1,
            "only `{}`'s workflow is listed",
            project.as_str()
        );
        assert_eq!(workflows[0]["id"].as_str(), Some(workflow_id));

        let listed = invoke(&host, project, "asset.list", json!({})).await;
        let records = listed["assets"].as_array().expect("assets array");
        assert_eq!(
            records.len(),
            1,
            "only `{}`'s assets are listed",
            project.as_str()
        );
        assert_eq!(records[0]["id"].as_str(), Some(asset_id));
    }

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn list_operations_skip_malformed_documents() {
    let base = temp_dir("malformed");
    let host = build_host(&base).await;
    let project = create_project(&host, "proj-malformed").await;

    let assets = import_files(&host, &project, &base, &[("a.txt", "alpha")]).await;
    let collection_id = create_collection(&host, &project, "Alpha", &[assets[0].as_str()]).await;
    let workflow_id = create_workflow(&host, &project, "Flow One").await;

    // A malformed document is logged as a warning and skipped; one bad file
    // must not hide the rest of the library or fail the list operation.
    let project_root = project_dir(&host, &project);
    let malformed_collection = project_root.join("collections").join("broken.json");
    fs::create_dir_all(malformed_collection.parent().expect("collection dir"))
        .expect("create collections dir");
    fs::write(&malformed_collection, "{ not a collection document")
        .expect("write malformed collection");
    let malformed_workflow = project_root.join("dynamic_workflows").join("broken.json");
    fs::create_dir_all(malformed_workflow.parent().expect("workflow dir"))
        .expect("create dynamic workflows dir");
    fs::write(&malformed_workflow, "{ not a dynamic workflow document")
        .expect("write malformed workflow");

    let listed = invoke(&host, &project, "collection.list", json!({})).await;
    let collections = listed["collections"].as_array().expect("collections array");
    assert_eq!(collections.len(), 1, "the valid collection is still listed");
    assert_eq!(collections[0]["id"].as_str(), Some(collection_id.as_str()));

    let listed = invoke(&host, &project, "dynamic_workflow.list", json!({})).await;
    let workflows = listed["workflows"].as_array().expect("workflows array");
    assert_eq!(workflows.len(), 1, "the valid workflow is still listed");
    assert_eq!(workflows[0]["id"].as_str(), Some(workflow_id.as_str()));

    let _ = fs::remove_dir_all(&base);
}
