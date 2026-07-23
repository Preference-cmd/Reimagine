//! Runtime error types.

use std::fmt;

/// Errors returned from the public runtime surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    /// The runtime was asked to operate on a run id that is not known.
    UnknownRun { run_id: String },
    /// Attempted to start a run with an empty plan.
    EmptyPlan { run_id: String },
    /// The plan has zero stages or zero nodes.
    EmptyExecutionGraph { run_id: String },
    /// A plan node references a `NodeTypeId` that has no registered executor.
    MissingExecutor { run_id: String, type_id: String },
    /// A host-provided event sink failed to emit an event.
    EventSink { message: String },
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownRun { run_id } => write!(f, "unknown run id: {run_id}"),
            Self::EmptyPlan { run_id } => write!(f, "execution plan is empty for run {run_id}"),
            Self::EmptyExecutionGraph { run_id } => {
                write!(f, "execution plan has no nodes for run {run_id}")
            }
            Self::MissingExecutor { run_id, type_id } => write!(
                f,
                "no executor registered for node type {type_id} in run {run_id}"
            ),
            Self::EventSink { message } => write!(f, "run event sink failed: {message}"),
        }
    }
}

impl std::error::Error for RuntimeError {}
