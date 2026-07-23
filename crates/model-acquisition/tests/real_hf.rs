//! Integration tests for `reimagine-model-acquisition`.
//!
//! These tests require network access to HuggingFace Hub and are marked `#[ignore]`
//! so they are skipped by default. Run with:
//!
//! ```text
//! cargo test -p reimagine-model-acquisition -- --ignored
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use reimagine_model_acquisition::hf::provider::{AcquisitionProgressSink, HuggingFaceProvider};
use reimagine_model_acquisition::{
    AcquireProvider, AcquisitionReport, AllowPatterns, ModelAcquisitionConfig,
    ModelAcquisitionRequest, OverwritePolicy, RepoId, Revision, TargetRelativeDir,
};

struct TestSink;

impl AcquisitionProgressSink for TestSink {
    fn file_done(&self, relative_path: &str, bytes: u64, outcome: &str) {
        eprintln!("  [{outcome}] {relative_path} ({bytes} bytes)");
    }

    fn done(&self, report: &AcquisitionReport) {
        eprintln!(
            "Acquisition finished: {} files, {} total bytes",
            report.files.len(),
            report.total_bytes,
        );
    }
}

/// Build a request for a well-known small model repo.
fn tiny_model_request(overwrite: OverwritePolicy) -> ModelAcquisitionRequest {
    let repo_id = RepoId::new("google-bert/bert-base-uncased").unwrap();
    let target_dir = TargetRelativeDir::new(PathBuf::from("test-integ/bert-base-uncased")).unwrap();

    ModelAcquisitionRequest {
        provider: AcquireProvider::HuggingFace,
        repo_id,
        revision: Revision::default(),
        allow_patterns: AllowPatterns::new(vec!["config.json".to_owned()]),
        target_relative_dir: target_dir,
        overwrite_policy: overwrite,
    }
}

/// Build a request that downloads just a single small file.
fn tokenizer_request() -> ModelAcquisitionRequest {
    let repo_id = RepoId::new("google-bert/bert-base-uncased").unwrap();
    let target_dir = TargetRelativeDir::new(PathBuf::from("test-integ/tokenizer")).unwrap();

    ModelAcquisitionRequest {
        provider: AcquireProvider::HuggingFace,
        repo_id,
        revision: Revision::default(),
        allow_patterns: AllowPatterns::new(vec!["tokenizer.json".to_owned()]),
        target_relative_dir: target_dir,
        overwrite_policy: OverwritePolicy::Overwrite,
    }
}

#[ignore = "requires network access to HuggingFace Hub"]
#[tokio::test]
async fn test_snapshot_download_small_model() {
    // Set up a provider using a public repo (no token needed).
    let config = ModelAcquisitionConfig::default();
    let provider = HuggingFaceProvider::new(&config);

    let tmp = tempfile::tempdir().unwrap();
    let request = tiny_model_request(OverwritePolicy::Overwrite);
    let sink: Option<Arc<dyn AcquisitionProgressSink>> = Some(Arc::new(TestSink));

    let report = provider
        .download(tmp.path().to_path_buf(), request, sink)
        .await
        .expect("snapshot download should succeed");

    assert!(!report.files.is_empty(), "expected at least one file");
    assert!(
        report.total_bytes > 0,
        "expected some bytes downloaded, got 0"
    );

    // Verify the file was placed at the expected target path.
    let target = tmp.path().join("test-integ/bert-base-uncased/config.json");
    assert!(
        target.exists(),
        "expected config.json to exist at {}",
        target.display()
    );

    // Verify report was written.
    let report_path = tmp.path().join("test-integ/acquisition-report.json");
    assert!(
        report_path.exists(),
        "expected acquisition-report.json at {}",
        report_path.display()
    );
}

#[ignore = "requires network access to HuggingFace Hub"]
#[tokio::test]
async fn test_overwrite_skips_existing() {
    let tmp = tempfile::tempdir().unwrap();

    // First download.
    let config = ModelAcquisitionConfig::default();
    let request = tiny_model_request(OverwritePolicy::Skip);
    HuggingFaceProvider::new(&config)
        .download(tmp.path().to_path_buf(), request, None)
        .await
        .expect("first download should succeed");

    // Second download — with Skip policy, if target exists it should fail.
    let request2 = tiny_model_request(OverwritePolicy::Skip);
    let result = HuggingFaceProvider::new(&config)
        .download(tmp.path().to_path_buf(), request2, None)
        .await;

    // The TargetExists error is expected.
    assert!(
        result.is_err(),
        "second download should fail when target exists and policy is Skip"
    );
}

#[ignore = "requires network access to HuggingFace Hub"]
#[tokio::test]
async fn test_overwrite_replaces() {
    let tmp = tempfile::tempdir().unwrap();

    // First download with Overwrite policy.
    let config = ModelAcquisitionConfig::default();
    let request1 = tiny_model_request(OverwritePolicy::Overwrite);
    HuggingFaceProvider::new(&config)
        .download(tmp.path().to_path_buf(), request1, None)
        .await
        .expect("first download should succeed");

    // Second download — overwrite should succeed.
    let request2 = tokenizer_request();
    HuggingFaceProvider::new(&config)
        .download(tmp.path().to_path_buf(), request2, None)
        .await
        .expect("overwrite download should succeed");

    // Should find tokenizer.json.
    let tokenizer_path = tmp.path().join("test-integ/tokenizer/tokenizer.json");
    assert!(
        tokenizer_path.exists(),
        "expected tokenizer.json at {}",
        tokenizer_path.display()
    );
}

#[ignore = "requires network access to HuggingFace Hub"]
#[tokio::test]
async fn test_download_with_revision() {
    let config = ModelAcquisitionConfig::default();
    let provider = HuggingFaceProvider::new(&config);

    let tmp = tempfile::tempdir().unwrap();

    let repo_id = RepoId::new("hf-internal-testing/tiny-random-bert").unwrap();
    let target_dir = TargetRelativeDir::new(PathBuf::from("test-integ/tiny-bert")).unwrap();

    let request = ModelAcquisitionRequest {
        provider: AcquireProvider::HuggingFace,
        repo_id,
        revision: Revision::new("main"),
        allow_patterns: AllowPatterns::new(vec!["config.json".to_owned()]),
        target_relative_dir: target_dir,
        overwrite_policy: OverwritePolicy::Overwrite,
    };

    let report = provider
        .download(tmp.path().to_path_buf(), request, None)
        .await
        .expect("download with explicit revision should succeed");

    assert!(!report.files.is_empty(), "expected at least one file");
}
