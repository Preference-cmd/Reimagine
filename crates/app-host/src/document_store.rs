use std::path::Path;

use reimagine_config::atomic_write;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::error::{AppHostError, AppHostResult};

/// Reads one JSON project document; `Ok(None)` when the file does not exist.
/// `what` names the document in error messages.
pub(crate) async fn read_document<T: DeserializeOwned>(
    path: &Path,
    what: &str,
) -> AppHostResult<Option<T>> {
    match tokio::fs::read(path).await {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|error| AppHostError::Io {
                path: path.to_path_buf(),
                message: format!("invalid {what} document: {error}"),
            }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(AppHostError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        }),
    }
}

/// Writes one JSON project document atomically, creating the parent directory.
pub(crate) async fn write_document<T: Serialize>(
    path: &Path,
    value: &T,
    what: &str,
) -> AppHostResult<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|error| AppHostError::Io {
                path: parent.to_path_buf(),
                message: error.to_string(),
            })?;
    }
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| AppHostError::Io {
        path: path.to_path_buf(),
        message: format!("failed to serialize {what}: {error}"),
    })?;
    atomic_write(path, bytes)
        .await
        .map_err(|error| AppHostError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn read_missing_document_returns_none() {
        let temp = std::env::temp_dir().join(format!(
            "doc-store-missing-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let missing = read_document::<serde_json::Value>(&temp.join("board.json"), "board")
            .await
            .expect("missing file is not an error");
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn read_invalid_document_returns_io_error() {
        let temp = std::env::temp_dir().join(format!(
            "doc-store-invalid-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = temp.join("proj").join("board.json");
        tokio::fs::create_dir_all(path.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&path, b"not json").await.unwrap();

        let err = read_document::<serde_json::Value>(&path, "board")
            .await
            .unwrap_err();
        match err {
            AppHostError::Io { path: p, message } => {
                assert_eq!(p, path);
                assert!(message.starts_with("invalid board document:"));
            }
            other => panic!("expected Io error, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&temp);
    }
}
