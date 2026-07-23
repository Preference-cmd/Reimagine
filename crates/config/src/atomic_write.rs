use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::io::AsyncWriteExt;

use crate::{ConfigError, ConfigResult};

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Atomically write bytes to a file by using a sibling temporary file.
pub async fn atomic_write(path: impl AsRef<Path>, bytes: impl AsRef<[u8]>) -> ConfigResult<()> {
    let path = path.as_ref();
    let parent = path.parent().ok_or_else(|| ConfigError::WriteFailed {
        path: path.to_path_buf(),
        message: "target path has no parent directory".to_owned(),
    })?;
    tokio::fs::create_dir_all(parent)
        .await
        .map_err(|error| ConfigError::WriteFailed {
            path: parent.to_path_buf(),
            message: error.to_string(),
        })?;

    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("config");
    let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temp_path = parent.join(format!(
        ".{file_name}.{}.{}.tmp",
        std::process::id(),
        sequence
    ));

    let result = async {
        let mut file = tokio::fs::File::create(&temp_path).await.map_err(|error| {
            ConfigError::WriteFailed {
                path: temp_path.clone(),
                message: error.to_string(),
            }
        })?;
        file.write_all(bytes.as_ref())
            .await
            .map_err(|error| ConfigError::WriteFailed {
                path: temp_path.clone(),
                message: error.to_string(),
            })?;
        file.sync_all()
            .await
            .map_err(|error| ConfigError::WriteFailed {
                path: temp_path.clone(),
                message: error.to_string(),
            })?;
        drop(file);
        tokio::fs::rename(&temp_path, path)
            .await
            .map_err(|error| ConfigError::WriteFailed {
                path: path.to_path_buf(),
                message: error.to_string(),
            })
    }
    .await;

    if result.is_err() {
        let _ = tokio::fs::remove_file(&temp_path).await;
    }

    result
}
