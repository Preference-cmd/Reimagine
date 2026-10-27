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
