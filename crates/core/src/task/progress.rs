//! Task progress reporting and sinks.

/// Structured progress report for an in-flight task or node.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Progress {
    pub sequence: u64,
    pub completed: u64,
    pub total: Option<u64>,
    pub message: Option<String>,
}

impl Progress {
    pub fn new(sequence: u64, completed: u64, total: Option<u64>, message: Option<String>) -> Self {
        Self {
            sequence,
            completed,
            total,
            message,
        }
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn completed(&self) -> u64 {
        self.completed
    }

    pub fn total(&self) -> Option<u64> {
        self.total
    }

    pub fn message(&self) -> Option<&str> {
        self.message.as_deref()
    }
}

/// Sink that receives in-flight task progress reports.
pub trait ProgressSink: Send + Sync + 'static {
    fn report(&self, progress: Progress);
}

/// No-op progress sink that discards all reported progress events.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopProgressSink;

impl ProgressSink for NoopProgressSink {
    fn report(&self, _progress: Progress) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_accessors_and_serialization() {
        let progress = Progress::new(1, 5, Some(10), Some("processing".to_string()));
        assert_eq!(progress.sequence(), 1);
        assert_eq!(progress.completed(), 5);
        assert_eq!(progress.total(), Some(10));
        assert_eq!(progress.message(), Some("processing"));

        let serialized = serde_json::to_string(&progress).expect("serialize progress");
        let deserialized: Progress =
            serde_json::from_str(&serialized).expect("deserialize progress");
        assert_eq!(progress, deserialized);
    }

    #[test]
    fn noop_sink_discards_without_error() {
        let sink = NoopProgressSink;
        sink.report(Progress::new(1, 1, None, None));
    }
}
