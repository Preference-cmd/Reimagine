//! Wiremock-driven tests for `LlmModelCatalog`: load/fallback semantics,
//! override replacement, refresh persistence, and corrupt-file
//! tolerance.

use reimagine_agent_harness::{LlmCatalogError, LlmModelCatalog};
use reimagine_config::AppPaths;
use serde_json::{Value, json};
use tempfile::TempDir;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

const ENDPOINT: &str = "/api.json";

fn temp_paths() -> (TempDir, AppPaths) {
    let dir = TempDir::new().expect("temp dir");
    let paths = AppPaths::new(dir.path());
    (dir, paths)
}

fn api_payload() -> Value {
    json!({
        "openai": {
            "name": "OpenAI",
            "models": [
                {
                    "id": "gpt-4o-mini",
                    "name": "GPT-4o mini",
                    "reasoning": false,
                    "input": ["text", "image"],
                    "contextWindow": 128000,
                    "maxTokens": 16384,
                    "cost": { "input": 0.15, "output": 0.6, "cacheRead": 0.075, "cacheWrite": 0.15 }
                },
                {
                    "id": "o3",
                    "name": "o3",
                    "reasoning": true,
                    "input": ["text", "image"],
                    "contextWindow": 200000,
                    "maxTokens": 100000,
                    "cost": { "input": 2.0, "output": 8.0, "cacheRead": 1.0, "cacheWrite": 2.5 },
                    "futureField": "ignored"
                }
            ]
        },
        "anthropic": {
            "name": "Anthropic",
            "models": [
                {
                    "id": "claude-3-5-sonnet-latest",
                    "name": "Claude 3.5 Sonnet",
                    "reasoning": false,
                    "input": ["text", "image"],
                    "contextWindow": 200000,
                    "maxTokens": 8192,
                    "cost": { "input": 3.0, "output": 15.0, "cacheRead": 0.3, "cacheWrite": 3.75 }
                }
            ]
        },
        "openrouter": {
            "name": "OpenRouter",
            "models": [
                {
                    "id": "meta-llama/llama-3.3-70b-instruct",
                    "name": "Llama 3.3 70B",
                    "reasoning": false,
                    "input": ["text"],
                    "contextWindow": 131072,
                    "maxTokens": 4096,
                    "cost": { "input": 0.12, "output": 0.3, "cacheRead": 0.12, "cacheWrite": 0.12 }
                }
            ]
        }
    })
}

async fn serve_payload(server: &MockServer, payload: Value) {
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_json(payload))
        .mount(server)
        .await;
}

async fn refresh_all(
    catalog: &mut LlmModelCatalog,
    paths: &AppPaths,
) -> Result<(), LlmCatalogError> {
    let server = MockServer::start().await;
    serve_payload(&server, api_payload()).await;
    let client = reqwest::Client::new();
    catalog
        .refresh_with_http_client(
            paths,
            None,
            &client,
            &format!("{}{}", server.uri(), ENDPOINT),
        )
        .await
}

#[tokio::test]
async fn load_with_missing_cache_returns_empty_catalog() {
    let (_dir, paths) = temp_paths();
    let catalog = LlmModelCatalog::load(&paths).await.expect("load ok");
    assert!(catalog.is_empty());
    assert!(catalog.providers().is_empty());
    assert!(catalog.model("openai", "gpt-4o-mini").is_none());
}

#[tokio::test]
async fn load_with_corrupt_cache_returns_error() {
    let (_dir, paths) = temp_paths();
    std::fs::create_dir_all(paths.config_dir()).expect("create config dir");
    std::fs::write(paths.config_dir().join("model-catalog.json"), "{not json").expect("write");
    let error = LlmModelCatalog::load(&paths)
        .await
        .expect_err("corrupt cache errors");
    assert!(matches!(error, LlmCatalogError::Parse(_)));
}

#[tokio::test]
async fn load_with_corrupt_override_returns_error() {
    let (_dir, paths) = temp_paths();
    std::fs::create_dir_all(paths.config_dir()).expect("create config dir");
    std::fs::write(
        paths.config_dir().join("model-catalog.override.json"),
        "not json",
    )
    .expect("write");
    let error = LlmModelCatalog::load(&paths)
        .await
        .expect_err("corrupt override errors");
    assert!(matches!(error, LlmCatalogError::Parse(_)));
}

#[tokio::test]
async fn refresh_fetches_persists_and_replaces_catalog() {
    let (_dir, paths) = temp_paths();
    let mut catalog = LlmModelCatalog::default();
    refresh_all(&mut catalog, &paths).await.expect("refresh ok");

    assert_eq!(catalog.providers(), ["anthropic", "openai", "openrouter"]);

    let gpt = catalog
        .model("openai", "gpt-4o-mini")
        .expect("gpt-4o-mini present");
    assert_eq!(gpt.provider().unwrap().as_str(), "openai");
    assert!(!gpt.reasoning());
    assert_eq!(gpt.input_modalities(), ["text", "image"]);
    assert_eq!(gpt.context_window(), Some(128000));
    assert_eq!(gpt.max_tokens(), Some(16384));
    let cost = gpt.cost().expect("cost present");
    assert_eq!(cost.input(), 0.15);
    assert_eq!(cost.output(), 0.6);
    assert_eq!(cost.cache_read(), 0.075);
    assert_eq!(cost.cache_write(), 0.15);

    let o3 = catalog.model("openai", "o3").expect("o3 present");
    assert!(o3.reasoning());

    let persisted = LlmModelCatalog::load(&paths)
        .await
        .expect("load after refresh");
    assert_eq!(persisted, catalog);
}

#[tokio::test]
async fn refresh_with_provider_filter_keeps_only_matching_providers() {
    let server = MockServer::start().await;
    serve_payload(&server, api_payload()).await;
    let client = reqwest::Client::new();
    let mut catalog = LlmModelCatalog::default();
    catalog
        .refresh_with_http_client(
            &AppPaths::new(temp_paths().0.path()),
            Some(&["openai".to_string()]),
            &client,
            &format!("{}{}", server.uri(), ENDPOINT),
        )
        .await
        .expect("refresh ok");

    assert_eq!(catalog.providers(), ["openai"]);
    assert!(catalog.model("openai", "gpt-4o-mini").is_some());
}

#[tokio::test]
async fn refresh_failure_without_cache_falls_back_to_snapshot() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    let client = reqwest::Client::new();
    let (_dir, paths) = temp_paths();
    let mut catalog = LlmModelCatalog::default();
    let error = catalog
        .refresh_with_http_client(
            &paths,
            None,
            &client,
            &format!("{}{}", server.uri(), ENDPOINT),
        )
        .await
        .expect_err("refresh fails");

    assert!(matches!(error, LlmCatalogError::Fetch(_)));
    assert_eq!(
        catalog.providers(),
        ["anthropic", "openai", "openrouter"],
        "snapshot supplies representative providers"
    );
    assert!(catalog.model("openai", "gpt-4o-mini").is_some());
    assert!(
        catalog
            .model("anthropic", "claude-3-5-sonnet-latest")
            .is_some()
    );
    assert!(catalog.model("openrouter", "openai/gpt-4o-mini").is_some());
    assert!(!paths.config_dir().join("model-catalog.json").exists());
}

#[tokio::test]
async fn refresh_failure_with_cache_keeps_existing_data() {
    let (_dir, paths) = temp_paths();
    let mut catalog = LlmModelCatalog::default();
    refresh_all(&mut catalog, &paths).await.expect("refresh ok");
    let before = catalog.clone();

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    let client = reqwest::Client::new();
    let error = catalog
        .refresh_with_http_client(
            &paths,
            None,
            &client,
            &format!("{}{}", server.uri(), ENDPOINT),
        )
        .await
        .expect_err("refresh fails");

    assert!(matches!(error, LlmCatalogError::Fetch(_)));
    assert_eq!(catalog, before, "catalog unchanged on failed refresh");
    assert!(paths.config_dir().join("model-catalog.json").exists());
}

#[tokio::test]
async fn override_replaces_provider_entry_wholesale() {
    let (_dir, paths) = temp_paths();
    let mut catalog = LlmModelCatalog::default();
    refresh_all(&mut catalog, &paths).await.expect("refresh ok");

    let override_dir = paths.config_dir().to_path_buf();
    std::fs::create_dir_all(&override_dir).expect("create config dir");
    let override_payload = json!({
        "openai": {
            "name": "Custom OpenAI",
            "models": [
                {
                    "name": "custom-model",
                    "provider": "openai",
                    "capabilities": [],
                    "reasoning": true,
                    "input_modalities": ["text"],
                    "context_window": 1000,
                    "max_tokens": 500,
                    "cost": { "input": 1.0, "output": 2.0, "cache_read": 0.5, "cache_write": 1.0 }
                }
            ]
        }
    });
    std::fs::write(
        override_dir.join("model-catalog.override.json"),
        override_payload.to_string(),
    )
    .expect("write override");

    let loaded = LlmModelCatalog::load(&paths)
        .await
        .expect("load with override");
    let openai = loaded.provider("openai").expect("openai present");
    assert_eq!(openai.name, "Custom OpenAI");
    assert_eq!(openai.models.len(), 1);
    assert_eq!(openai.models[0].name().as_str(), "custom-model");
    assert!(
        loaded.model("openai", "gpt-4o-mini").is_none(),
        "no deep merge"
    );
    assert!(
        loaded
            .model("anthropic", "claude-3-5-sonnet-latest")
            .is_some(),
        "other providers untouched"
    );
}

#[tokio::test]
async fn override_missing_file_is_noop() {
    let (_dir, paths) = temp_paths();
    let mut catalog = LlmModelCatalog::default();
    refresh_all(&mut catalog, &paths).await.expect("refresh ok");

    let loaded = LlmModelCatalog::load(&paths).await.expect("load ok");
    assert_eq!(loaded, catalog);
}

#[tokio::test]
async fn refresh_reapplies_override_to_in_memory_catalog() {
    let (_dir, paths) = temp_paths();
    let mut catalog = LlmModelCatalog::default();
    refresh_all(&mut catalog, &paths).await.expect("refresh ok");

    let override_dir = paths.config_dir().to_path_buf();
    std::fs::create_dir_all(&override_dir).expect("create config dir");
    std::fs::write(
        override_dir.join("model-catalog.override.json"),
        json!({
            "openai": {
                "name": "Custom OpenAI",
                "models": [
                    {
                        "name": "custom-model",
                        "provider": "openai",
                        "capabilities": [],
                        "reasoning": true,
                        "input_modalities": ["text"],
                        "context_window": 1000,
                        "max_tokens": 500,
                        "cost": null
                    }
                ]
            }
        })
        .to_string(),
    )
    .expect("write override");

    refresh_all(&mut catalog, &paths).await.expect("refresh ok");
    let openai = catalog.provider("openai").expect("openai present");
    assert_eq!(openai.name, "Custom OpenAI");
    assert_eq!(openai.models.len(), 1);
    assert_eq!(openai.models[0].name().as_str(), "custom-model");
    assert!(catalog.model("openai", "gpt-4o-mini").is_none());
    assert_eq!(
        catalog.model("openai", "custom-model").unwrap().cost(),
        None
    );
}
