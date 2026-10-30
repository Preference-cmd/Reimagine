#![allow(clippy::result_large_err)]
//! Content-addressed asset storage for projects.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use reimagine_core::asset::AssetRecord;
use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::model::{AssetId, DiagnosticId, StepId};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

fn asset_caller_error(code: &str, message: impl Into<String>) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new(format!("asset:{code}")),
        DiagnosticCode::new(format!("ASSET/CALLER_{code}")),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("assets"),
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new("asset")),
    )
}

fn asset_internal_error(code: &str, message: impl Into<String>) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new(format!("asset:{code}")),
        DiagnosticCode::new(format!("ASSET/INTERNAL_{code}")),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("assets"),
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new("asset")),
    )
}

/// Resolves the media type for an asset given its path and an optional schema hint.
pub fn resolve_media_type(path: &Path, hint: Option<&str>) -> String {
    if let Some(h) = hint {
        let trimmed = h.trim();
        if !trimmed.is_empty() && !trimmed.contains('*') {
            return trimmed.to_owned();
        }
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    match ext.as_deref() {
        Some("txt") => "text/plain",
        Some("md" | "markdown") => "text/markdown",
        Some("json") => "application/json",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("svg") => "image/svg+xml",
        Some("pdf") => "application/pdf",
        Some("html" | "htm") => "text/html",
        Some("csv") => "text/csv",
        _ => "application/octet-stream",
    }
    .to_owned()
}

/// Checks if a concrete media type matches an expected pattern (e.g. `text/*` matches `text/markdown`).
pub(crate) fn matches_media_type_pattern(pattern: &str, concrete: &str) -> bool {
    let pattern = pattern.trim();
    let concrete = concrete.trim();
    if pattern == concrete || pattern == "*/*" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix("/*") {
        let concrete_prefix = concrete.split('/').next().unwrap_or("");
        return prefix.eq_ignore_ascii_case(concrete_prefix);
    }
    pattern.eq_ignore_ascii_case(concrete)
}

/// Per-project content-addressed asset store.
#[derive(Debug)]
pub struct AssetStore {
    project_dir: PathBuf,
    append_lock: tokio::sync::Mutex<()>,
}

impl AssetStore {
    pub fn new(project_dir: impl Into<PathBuf>) -> Self {
        Self {
            project_dir: project_dir.into(),
            append_lock: tokio::sync::Mutex::new(()),
        }
    }

    fn assets_dir(&self) -> PathBuf {
        self.project_dir.join("assets")
    }

    fn assets_jsonl_path(&self) -> PathBuf {
        self.project_dir.join("assets.jsonl")
    }

    pub fn content_path(&self, hash: &str) -> PathBuf {
        let prefix = if hash.len() >= 2 { &hash[0..2] } else { "xx" };
        self.assets_dir().join(prefix).join(hash)
    }

    /// Ingest a file into the project asset store.
    pub async fn ingest_file(
        &self,
        origin: StepId,
        source_path: &Path,
        media_type_hint: Option<&str>,
    ) -> Result<AssetRecord, Diagnostic> {
        let meta = match tokio::fs::symlink_metadata(source_path).await {
            Ok(m) => m,
            Err(e) => {
                return Err(asset_caller_error(
                    "FILE_NOT_FOUND",
                    format!(
                        "failed to read source asset `{}`: {e}",
                        source_path.display()
                    ),
                ));
            }
        };

        if meta.file_type().is_symlink() {
            return Err(asset_caller_error(
                "SYMLINK_REJECTED",
                format!("symlink `{}` is not allowed", source_path.display()),
            ));
        }

        if !meta.is_file() {
            return Err(asset_caller_error(
                "NOT_A_REGULAR_FILE",
                format!("source `{}` is not a regular file", source_path.display()),
            ));
        }

        let bytes = tokio::fs::read(source_path).await.map_err(|e| {
            asset_internal_error(
                "IO",
                format!("failed to read `{}`: {e}", source_path.display()),
            )
        })?;

        let size = bytes.len() as u64;
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let hash = format!("{:x}", hasher.finalize());

        let media_type = resolve_media_type(source_path, media_type_hint);

        let dest = self.content_path(&hash);
        if !dest.exists() {
            let parent = dest.parent().ok_or_else(|| {
                asset_internal_error("IO", format!("dest has no parent: `{}`", dest.display()))
            })?;
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                asset_internal_error(
                    "IO",
                    format!("failed to create dir `{}`: {e}", parent.display()),
                )
            })?;

            static TMP_COUNTER: AtomicU64 = AtomicU64::new(1);
            let tmp_count = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let tmp_nanos = chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default();
            let tmp_file = parent.join(format!(".tmp-{tmp_nanos:x}-{tmp_count:x}"));

            if let Err(e) = tokio::fs::write(&tmp_file, &bytes).await {
                let _ = tokio::fs::remove_file(&tmp_file).await;
                return Err(asset_internal_error(
                    "IO",
                    format!(
                        "failed to write temp asset file `{}`: {e}",
                        tmp_file.display()
                    ),
                ));
            }

            // Order: write temp -> rename to dest; if rename fails and dest exists -> dedupe,
            // remove temp file (still writable) and continue; on success set dest read-only.
            // Also remove temp file on every other error path.
            match tokio::fs::rename(&tmp_file, &dest).await {
                Ok(()) => {
                    if let Ok(perms_meta) = tokio::fs::metadata(&dest).await {
                        let mut perms = perms_meta.permissions();
                        perms.set_readonly(true);
                        let _ = tokio::fs::set_permissions(&dest, perms).await;
                    }
                }
                Err(e) => {
                    let _ = tokio::fs::remove_file(&tmp_file).await;
                    if !dest.exists() {
                        return Err(asset_internal_error(
                            "IO",
                            format!("failed to rename asset to `{}`: {e}", dest.display()),
                        ));
                    }
                }
            }
        }

        static COUNTER: AtomicU64 = AtomicU64::new(1);
        let count = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default();
        let id = AssetId::new(format!("asset-{nanos:x}-{count:x}"));

        let record = AssetRecord::new(id, hash, media_type, size, origin);

        let jsonl_path = self.assets_jsonl_path();
        let line = serde_json::to_string(&record).map_err(|e| {
            asset_internal_error(
                "SERIALIZATION",
                format!("failed to serialize asset record: {e}"),
            )
        })? + "\n";

        if let Some(parent) = jsonl_path.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }

        {
            let _guard = self.append_lock.lock().await;
            let mut file = tokio::fs::OpenOptions::new()
                .create(true)
                .read(true)
                .append(true)
                .open(&jsonl_path)
                .await
                .map_err(|e| {
                    asset_internal_error(
                        "IO",
                        format!("failed to open `{}`: {e}", jsonl_path.display()),
                    )
                })?;
            Self::terminate_partial_line(&mut file, &jsonl_path).await?;
            file.write_all(line.as_bytes()).await.map_err(|e| {
                asset_internal_error(
                    "IO",
                    format!("failed to write `{}`: {e}", jsonl_path.display()),
                )
            })?;
            file.flush().await.map_err(|e| {
                asset_internal_error(
                    "IO",
                    format!("failed to flush `{}`: {e}", jsonl_path.display()),
                )
            })?;
        }

        Ok(record)
    }

    /// Terminate a partial trailing line before appending a record.
    ///
    /// A process abort can interrupt an append, leaving a line without its
    /// newline. Appending directly after it would merge the torn line and the
    /// new record into one unparseable line, so `list` would drop both. Adding
    /// the missing newline keeps the torn line a single ignored record and
    /// starts the new record on its own line.
    async fn terminate_partial_line(
        file: &mut tokio::fs::File,
        jsonl_path: &Path,
    ) -> Result<(), Diagnostic> {
        let len = file
            .metadata()
            .await
            .map_err(|e| {
                asset_internal_error(
                    "IO",
                    format!("failed to stat `{}`: {e}", jsonl_path.display()),
                )
            })?
            .len();
        if len == 0 {
            return Ok(());
        }

        file.seek(std::io::SeekFrom::End(-1)).await.map_err(|e| {
            asset_internal_error(
                "IO",
                format!("failed to seek `{}`: {e}", jsonl_path.display()),
            )
        })?;
        let mut last = [0u8; 1];
        file.read_exact(&mut last).await.map_err(|e| {
            asset_internal_error(
                "IO",
                format!("failed to read `{}`: {e}", jsonl_path.display()),
            )
        })?;
        if last[0] != b'\n' {
            file.write_all(b"\n").await.map_err(|e| {
                asset_internal_error(
                    "IO",
                    format!("failed to repair `{}`: {e}", jsonl_path.display()),
                )
            })?;
        }
        Ok(())
    }

    /// Every asset record in file order (oldest first).
    ///
    /// A record is appended one line at a time, so a trailing line without a
    /// newline is an interrupted append and is ignored; unparseable lines are
    /// skipped rather than failing the listing.
    pub async fn list(&self) -> Result<Vec<AssetRecord>, Diagnostic> {
        let jsonl_path = self.assets_jsonl_path();
        if !jsonl_path.exists() {
            return Ok(Vec::new());
        }
        let content = tokio::fs::read_to_string(&jsonl_path).await.map_err(|e| {
            asset_internal_error(
                "IO",
                format!("failed to read `{}`: {e}", jsonl_path.display()),
            )
        })?;
        let completed = match content.rfind('\n') {
            Some(last_nl) => &content[..=last_nl],
            None => "",
        };
        let mut records = Vec::new();
        for line in completed.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(record) = serde_json::from_str::<AssetRecord>(line) {
                records.push(record);
            }
        }
        Ok(records)
    }

    /// Read an asset record by AssetId synchronously.
    pub fn get_asset_sync(&self, id: &AssetId) -> Result<Option<AssetRecord>, Diagnostic> {
        let jsonl_path = self.assets_jsonl_path();
        if !jsonl_path.exists() {
            return Ok(None);
        }
        let content = std::fs::read_to_string(&jsonl_path).map_err(|e| {
            asset_internal_error(
                "IO",
                format!("failed to read `{}`: {e}", jsonl_path.display()),
            )
        })?;
        let valid_slice = match content.rfind('\n') {
            Some(last_nl) => &content[..=last_nl],
            None => "",
        };
        for line in valid_slice.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(rec) = serde_json::from_str::<AssetRecord>(line)
                && rec.id == *id
            {
                return Ok(Some(rec));
            }
        }
        Ok(None)
    }

    /// Read an asset record by AssetId.
    pub async fn get_asset(&self, id: &AssetId) -> Result<Option<AssetRecord>, Diagnostic> {
        let jsonl_path = self.assets_jsonl_path();
        if !jsonl_path.exists() {
            return Ok(None);
        }
        let content = tokio::fs::read_to_string(&jsonl_path).await.map_err(|e| {
            asset_internal_error(
                "IO",
                format!("failed to read `{}`: {e}", jsonl_path.display()),
            )
        })?;
        let valid_slice = match content.rfind('\n') {
            Some(last_nl) => &content[..=last_nl],
            None => "",
        };
        for line in valid_slice.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(rec) = serde_json::from_str::<AssetRecord>(line)
                && rec.id == *id
            {
                return Ok(Some(rec));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn media_type_resolution() {
        assert_eq!(resolve_media_type(Path::new("foo.txt"), None), "text/plain");
        assert_eq!(
            resolve_media_type(Path::new("doc.md"), None),
            "text/markdown"
        );
        assert_eq!(resolve_media_type(Path::new("img.png"), None), "image/png");
        assert_eq!(
            resolve_media_type(Path::new("file.xyz"), None),
            "application/octet-stream"
        );
        // Explicit hint overrides extension
        assert_eq!(
            resolve_media_type(Path::new("foo.txt"), Some("text/custom")),
            "text/custom"
        );
        // Wildcard hint uses extension
        assert_eq!(
            resolve_media_type(Path::new("doc.md"), Some("text/*")),
            "text/markdown"
        );
    }

    #[test]
    fn media_type_matching() {
        assert!(matches_media_type_pattern("text/*", "text/plain"));
        assert!(matches_media_type_pattern("text/*", "text/markdown"));
        assert!(!matches_media_type_pattern("text/*", "image/png"));
        assert!(matches_media_type_pattern("image/png", "image/png"));
        assert!(!matches_media_type_pattern("image/png", "image/jpeg"));
        assert!(matches_media_type_pattern("*/*", "application/json"));
    }

    #[tokio::test]
    async fn concurrent_ingest_of_identical_content() {
        let temp = tempfile::tempdir().unwrap();
        let store = Arc::new(AssetStore::new(temp.path()));

        let source = temp.path().join("source.txt");
        tokio::fs::write(&source, b"identical content")
            .await
            .unwrap();

        let store1 = Arc::clone(&store);
        let store2 = Arc::clone(&store);
        let src1 = source.clone();
        let src2 = source.clone();

        let (res1, res2) = tokio::join!(
            tokio::spawn(async move {
                store1
                    .ingest_file(StepId::new("step-1"), &src1, Some("text/plain"))
                    .await
            }),
            tokio::spawn(async move {
                store2
                    .ingest_file(StepId::new("step-2"), &src2, Some("text/plain"))
                    .await
            }),
        );

        let rec1 = res1.unwrap().expect("task 1 succeeds");
        let rec2 = res2.unwrap().expect("task 2 succeeds");

        assert_eq!(rec1.hash, rec2.hash);
        assert_ne!(rec1.id, rec2.id);

        // Two records in assets.jsonl
        let jsonl_content = tokio::fs::read_to_string(temp.path().join("assets.jsonl"))
            .await
            .unwrap();
        let lines: Vec<&str> = jsonl_content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .collect();
        assert_eq!(lines.len(), 2);

        // Exactly one content file on disk
        let content_path = store.content_path(&rec1.hash);
        assert!(content_path.is_file());
    }

    #[tokio::test]
    async fn ingest_after_a_torn_tail_keeps_the_new_record() {
        let temp = tempfile::tempdir().unwrap();
        let store = AssetStore::new(temp.path());
        let jsonl_path = temp.path().join("assets.jsonl");

        // Simulate a crash mid-append: a partial record without its newline.
        tokio::fs::write(&jsonl_path, b"{\"id\":\"asset-torn\",\"hash\":")
            .await
            .unwrap();

        let source = temp.path().join("source.txt");
        tokio::fs::write(&source, b"after the crash").await.unwrap();

        let record = store
            .ingest_file(StepId::new("step-1"), &source, Some("text/plain"))
            .await
            .expect("ingest after a torn tail");

        let listed = store.list().await.expect("list assets");
        assert_eq!(listed.len(), 1, "the torn line stays one ignored record");
        assert_eq!(listed[0].id, record.id);

        let contents = tokio::fs::read_to_string(&jsonl_path).await.unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2, "the repaired newline splits the two lines");
        assert_eq!(lines[0], "{\"id\":\"asset-torn\",\"hash\":");
        assert!(serde_json::from_str::<AssetRecord>(lines[1]).is_ok());
    }
}
