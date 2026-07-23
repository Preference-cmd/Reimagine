use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Outcome of a single-file download attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcquisitionOutcome {
    /// File was downloaded successfully.
    Downloaded,
    /// File was already present and skipped (OverwritePolicy::Skip).
    Skipped,
    /// File was overwritten.
    Overwritten,
    /// File failed to download.
    Failed { message: String },
}

/// Record for a single file within the acquisition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcquisitionFileEntry {
    /// Relative path within the target directory.
    pub relative_path: String,
    /// Size in bytes.
    pub bytes: u64,
    /// Outcome for this file.
    pub outcome: AcquisitionOutcome,
}

/// High-level summary of a completed acquisition.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AcquisitionReport {
    /// The provider that performed the acquisition.
    pub provider: String,
    /// The repository identifier.
    pub repo_id: String,
    /// The revision that was fetched.
    pub revision: String,
    /// The target directory relative to the workspace base path.
    pub target_dir: String,
    /// Per-file records.
    pub files: Vec<AcquisitionFileEntry>,
    /// Total bytes downloaded (sum of `Downloaded` + `Overwritten` outcomes).
    pub total_bytes: u64,
    /// ISO 8601 timestamp of completion.
    pub finished_at: String,
}

impl AcquisitionReport {
    /// Create a new report skeleton.
    pub fn new(
        provider: impl Into<String>,
        repo_id: impl Into<String>,
        revision: impl Into<String>,
        target_dir: impl Into<String>,
    ) -> Self {
        Self {
            provider: provider.into(),
            repo_id: repo_id.into(),
            revision: revision.into(),
            target_dir: target_dir.into(),
            files: Vec::new(),
            total_bytes: 0,
            finished_at: crate::timestamp::now_utc(),
        }
    }

    /// Add a file entry and accumulate bytes for successful outcomes.
    pub fn add_file(&mut self, entry: AcquisitionFileEntry) {
        match &entry.outcome {
            AcquisitionOutcome::Downloaded | AcquisitionOutcome::Overwritten => {
                self.total_bytes += entry.bytes;
            }
            _ => {}
        }
        self.files.push(entry);
    }

    /// Write the report as `acquisition-report.json` alongside the target directory.
    pub async fn write(
        &self,
        base_path: &std::path::Path,
    ) -> crate::ModelAcquisitionResult<PathBuf> {
        use tokio::io::AsyncWriteExt;

        let path = base_path.join("acquisition-report.json");
        let json =
            serde_json::to_vec_pretty(self).map_err(|e| crate::ModelAcquisitionError::Json {
                path: Some(path.clone()),
                message: e.to_string(),
            })?;

        let mut file =
            tokio::fs::File::create(&path)
                .await
                .map_err(|e| crate::ModelAcquisitionError::Io {
                    path: path.clone(),
                    message: e.to_string(),
                })?;

        file.write_all(&json)
            .await
            .map_err(|e| crate::ModelAcquisitionError::Io {
                path: path.clone(),
                message: e.to_string(),
            })?;

        file.flush()
            .await
            .map_err(|e| crate::ModelAcquisitionError::Io {
                path: path.clone(),
                message: e.to_string(),
            })?;

        Ok(path)
    }
}
