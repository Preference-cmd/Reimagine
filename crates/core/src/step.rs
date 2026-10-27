use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::command::OperationContext;
use crate::diagnostic::Diagnostic;
use crate::event::Timestamp;
use crate::model::StepId;

/// Outcome of an operation step execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum StepOutcome {
    Running,
    Succeeded { output: Value },
    Failed { diagnostic: Box<Diagnostic> },
    Cancelled,
}

impl StepOutcome {
    pub fn failed(diagnostic: Diagnostic) -> Self {
        Self::Failed {
            diagnostic: Box::new(diagnostic),
        }
    }
}

/// A durable record of one operation invocation in a project.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepRecord {
    pub id: StepId,
    pub op: String,
    pub op_version: String,
    pub input: Value,
    pub context: OperationContext,
    pub outcome: StepOutcome,
    pub started_at: Timestamp,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<Timestamp>,
}

impl StepRecord {
    pub fn new_running(
        id: StepId,
        op: impl Into<String>,
        op_version: impl Into<String>,
        input: Value,
        context: OperationContext,
        started_at: Timestamp,
    ) -> Self {
        Self {
            id,
            op: op.into(),
            op_version: op_version.into(),
            input,
            context,
            outcome: StepOutcome::Running,
            started_at,
            finished_at: None,
        }
    }

    pub fn finish(&mut self, outcome: StepOutcome, finished_at: Timestamp) {
        self.outcome = outcome;
        self.finished_at = Some(finished_at);
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::command::{CommandActor, CommandActorKind, CommandProvenance};

    #[test]
    fn step_record_serde_roundtrip() {
        let context = OperationContext::new(
            CommandActor::new(CommandActorKind::Human),
            CommandProvenance::Direct,
        );
        let mut record = StepRecord::new_running(
            StepId::new("step-1"),
            "test.op",
            "1.0.0",
            json!({ "foo": "bar" }),
            context,
            Timestamp::new("2026-10-27T10:00:00Z"),
        );
        assert!(matches!(record.outcome, StepOutcome::Running));
        assert!(record.finished_at.is_none());

        let serialized = serde_json::to_string(&record).expect("serialize running");
        let deserialized: StepRecord =
            serde_json::from_str(&serialized).expect("deserialize running");
        assert_eq!(record, deserialized);

        record.finish(
            StepOutcome::Succeeded {
                output: json!({ "baz": 42 }),
            },
            Timestamp::new("2026-10-27T10:00:05Z"),
        );
        assert!(matches!(record.outcome, StepOutcome::Succeeded { .. }));
        assert!(record.finished_at.is_some());

        let serialized = serde_json::to_string(&record).expect("serialize finished");
        let deserialized: StepRecord =
            serde_json::from_str(&serialized).expect("deserialize finished");
        assert_eq!(record, deserialized);
    }
}
