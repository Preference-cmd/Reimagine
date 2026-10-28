//! Append-only step execution log for projects.

use std::path::PathBuf;

use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::model::DiagnosticId;
use reimagine_core::step::StepRecord;
use tokio::io::AsyncWriteExt;

fn step_internal_error(code: &str, message: impl Into<String>) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new(format!("step_log:{code}")),
        DiagnosticCode::new(format!("STEPLOG/INTERNAL_{code}")),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("step_log"),
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new("step_log")),
    )
}

/// Append-only step execution log for a project.
#[derive(Debug)]
pub struct StepLog {
    project_dir: PathBuf,
    append_lock: tokio::sync::Mutex<()>,
}

impl StepLog {
    /// Construct a new `StepLog` for the given project directory.
    pub fn new(project_dir: impl Into<PathBuf>) -> Self {
        Self {
            project_dir: project_dir.into(),
            append_lock: tokio::sync::Mutex::new(()),
        }
    }

    fn steps_jsonl_path(&self) -> PathBuf {
        self.project_dir.join("steps.jsonl")
    }

    async fn append_record(&self, record: &StepRecord) -> Result<(), Diagnostic> {
        let _guard = self.append_lock.lock().await;

        let path = self.steps_jsonl_path();
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| step_internal_error("IO", format!("create step log dir: {e}")))?;
        }

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await
            .map_err(|e| step_internal_error("IO", format!("open steps.jsonl: {e}")))?;

        let line = serde_json::to_string(record).map_err(|e| {
            step_internal_error("SERIALIZATION", format!("serialize step record: {e}"))
        })?;

        file.write_all(format!("{line}\n").as_bytes())
            .await
            .map_err(|e| step_internal_error("IO", format!("write step line: {e}")))?;

        file.flush()
            .await
            .map_err(|e| step_internal_error("IO", format!("flush steps.jsonl: {e}")))?;

        Ok(())
    }

    /// Append a start line for a step record to `steps.jsonl`.
    pub async fn record_start(&self, record: &StepRecord) -> Result<(), Diagnostic> {
        self.append_record(record).await
    }

    /// Append a finish line for a step record to `steps.jsonl`.
    pub async fn record_finish(&self, record: &StepRecord) -> Result<(), Diagnostic> {
        self.append_record(record).await
    }

    /// All step records, with each step's start and finish lines merged into
    /// one record (the last line wins). Order follows first appearance.
    ///
    /// Malformed lines are skipped: a process abort can leave a torn tail, and
    /// one bad line must not make a run's history unreadable.
    pub async fn records(&self) -> Result<Vec<StepRecord>, Diagnostic> {
        let path = self.steps_jsonl_path();
        let content = match tokio::fs::read_to_string(&path).await {
            Ok(content) => content,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(step_internal_error(
                    "IO",
                    format!("read steps.jsonl: {error}"),
                ));
            }
        };

        let mut order: Vec<String> = Vec::new();
        let mut merged: std::collections::HashMap<String, StepRecord> =
            std::collections::HashMap::new();
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let Ok(record) = serde_json::from_str::<StepRecord>(line) else {
                continue;
            };
            let step_id = record.id.to_string();
            if !merged.contains_key(&step_id) {
                order.push(step_id.clone());
            }
            merged.insert(step_id, record);
        }

        Ok(order
            .into_iter()
            .filter_map(|step_id| merged.remove(&step_id))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_core::command::{
        CommandActor, CommandActorKind, CommandProvenance, OperationContext,
    };
    use reimagine_core::event::Timestamp;
    use reimagine_core::model::StepId;
    use reimagine_core::step::{StepOutcome, StepRecord};
    use serde_json::json;
    use std::sync::Arc;

    #[tokio::test]
    async fn records_merges_start_and_finish_lines() {
        let temp = tempfile::tempdir().unwrap();
        let log = StepLog::new(temp.path());
        let context = OperationContext::new(
            CommandActor::new(CommandActorKind::Human).with_id("test-user"),
            CommandProvenance::Direct,
        );

        for (index, finish) in [(0usize, true), (1, false)] {
            let step_id = StepId::new(format!("step-merge-{index}"));
            let mut record = StepRecord::new_running(
                step_id,
                "test.op",
                "0.1.0",
                json!({ "index": index }),
                context.clone(),
                Timestamp::new("2026-10-27T00:00:00Z"),
            );
            log.record_start(&record).await.unwrap();
            if finish {
                record.finish(
                    StepOutcome::Succeeded {
                        output: json!({ "ok": true }),
                    },
                    Timestamp::new("2026-10-27T00:00:01Z"),
                );
                log.record_finish(&record).await.unwrap();
            }
        }

        let records = log.records().await.unwrap();
        assert_eq!(records.len(), 2);
        assert!(matches!(records[0].outcome, StepOutcome::Succeeded { .. }));
        assert!(matches!(records[1].outcome, StepOutcome::Running));
        assert!(records[1].finished_at.is_none());
    }

    #[tokio::test]
    async fn step_log_concurrent_appends() {
        let temp = tempfile::tempdir().unwrap();
        let log = Arc::new(StepLog::new(temp.path()));

        let task_count = 50;
        let mut handles = Vec::new();

        for i in 0..task_count {
            let log_clone = Arc::clone(&log);
            handles.push(tokio::spawn(async move {
                let step_id = StepId::new(format!("step-concurrent-{i}"));
                let start_time = Timestamp::new("2026-10-27T00:00:00Z");
                let context = OperationContext::new(
                    CommandActor::new(CommandActorKind::Human).with_id("test-user"),
                    CommandProvenance::Direct,
                );
                let mut record = StepRecord::new_running(
                    step_id.clone(),
                    format!("op-{i}"),
                    "0.1.0".to_string(),
                    json!({ "input": i }),
                    context,
                    start_time,
                );
                log_clone.record_start(&record).await.unwrap();

                let finish_time = Timestamp::new("2026-10-27T00:00:05Z");
                record.finish(
                    StepOutcome::Succeeded {
                        output: json!({ "res": i }),
                    },
                    finish_time,
                );
                log_clone.record_finish(&record).await.unwrap();
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        let content = tokio::fs::read_to_string(log.steps_jsonl_path())
            .await
            .unwrap();
        let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
        assert_eq!(lines.len(), task_count * 2);

        let mut starts = std::collections::HashMap::new();
        let mut finishes = std::collections::HashMap::new();

        for line in lines {
            let record: StepRecord = serde_json::from_str(line).expect("valid StepRecord JSON");
            if matches!(record.outcome, StepOutcome::Running) {
                *starts.entry(record.id.to_string()).or_insert(0) += 1;
            } else {
                *finishes.entry(record.id.to_string()).or_insert(0) += 1;
            }
        }

        assert_eq!(starts.len(), task_count);
        assert_eq!(finishes.len(), task_count);
        for i in 0..task_count {
            let key = format!("step-concurrent-{i}");
            assert_eq!(starts.get(&key), Some(&1));
            assert_eq!(finishes.get(&key), Some(&1));
        }
    }
}
