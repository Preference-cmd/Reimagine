//! Workspace bootstrap helpers used by the Axum host binary.
//!
//! Keeping this small module in the library makes the binary thin and
//! lets us unit-test startup behavior such as invalid backend config
//! handling.

use std::path::PathBuf;
use std::sync::Arc;

use reimagine_agent::WorkspaceScope;
use reimagine_app_host::{AppHostError, WorkspaceHost};
use reimagine_config::AppPaths;
use reimagine_runtime::RunEventSink;

use crate::recorder::RunEventRecorder;

/// Default development workspace path used when `--base-path` is omitted.
///
/// The path lives under the system temp directory so the server can start
/// without requiring an explicit workspace, while still being easy to
/// discover in logs.
pub fn default_workspace_path() -> PathBuf {
    std::env::temp_dir().join("reimagine-axum-host-workspace")
}

/// Ensure the workspace directory layout exists on disk.
pub async fn ensure_workspace_dirs(base_path: impl Into<PathBuf>) -> std::io::Result<()> {
    AppPaths::new(base_path)
        .ensure_all()
        .await
        .map_err(|error| std::io::Error::other(format!("failed to create workspace dirs: {error}")))
}

/// Bootstrap a [`WorkspaceHost`] for the Axum server, wiring the shared
/// [`RunEventRecorder`] as the runtime event sink.
///
/// This is the async path that loads backend config from disk and returns
/// a clear error when the config JSON is invalid.
pub async fn bootstrap_workspace(
    base_path: impl Into<PathBuf>,
    recorder: Arc<RunEventRecorder>,
) -> Result<Arc<WorkspaceHost>, AppHostError> {
    let base_path = base_path.into();
    let host = WorkspaceHost::try_with_defaults_and_event_sink(
        WorkspaceScope::new("reimagine-axum-host"),
        &base_path,
        recorder as Arc<dyn RunEventSink>,
    )
    .await?;
    Ok(Arc::new(host))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(prefix: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("reimagine-axum-host-bootstrap-{prefix}-{nonce}"))
    }

    #[tokio::test]
    async fn ensure_workspace_dirs_creates_expected_layout() {
        let base = temp_dir("ensure-dirs");
        ensure_workspace_dirs(&base)
            .await
            .expect("ensure dirs should succeed");

        let paths = AppPaths::new(&base);
        assert!(paths.base_path().exists());
        assert!(paths.models_dir().exists());
        assert!(paths.output_dir().exists());
        assert!(paths.workflows_dir().exists());
        assert!(paths.config_dir().exists());

        let _ = tokio::fs::remove_dir_all(&base).await;
    }

    #[tokio::test]
    async fn bootstrap_workspace_returns_error_for_invalid_config() {
        let base = temp_dir("invalid-config");
        let config_dir = base.join("config");
        tokio::fs::create_dir_all(&config_dir)
            .await
            .expect("create config dir");
        tokio::fs::write(
            config_dir.join("inference_backend.json"),
            r#"{"backend": "not_a_backend"}"#,
        )
        .await
        .expect("write invalid config");

        let recorder = Arc::new(RunEventRecorder::new());
        let err = bootstrap_workspace(&base, recorder)
            .await
            .expect_err("invalid config should fail bootstrap");
        let msg = err.to_string();
        assert!(
            msg.contains("inference_backend.json") || msg.contains("config"),
            "error should mention the config file, got: {msg}"
        );

        let _ = tokio::fs::remove_dir_all(&base).await;
    }

    #[tokio::test]
    async fn bootstrap_workspace_succeeds_with_missing_config() {
        let base = temp_dir("missing-config");
        let recorder = Arc::new(RunEventRecorder::new());
        let workspace = bootstrap_workspace(&base, recorder)
            .await
            .expect("missing config should bootstrap with defaults");
        assert_eq!(workspace.base_path(), base);

        let _ = tokio::fs::remove_dir_all(&base).await;
    }
}
