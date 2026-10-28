//! Dynamic workflows: declarative step graphs and their batch runs.
//!
//! A [`WorkflowSpec`] is a small JSON-defined graph of operation calls. Steps
//! reference declared inputs, run parameters and earlier step outputs through
//! [`Binding`] paths. Binding resolution and static validation are pure; the
//! host schedules and persists runs.

mod binding;
mod run;
mod workflow;

pub use binding::{Binding, BindingError, BindingScope, collect_bindings, resolve_value};
pub use run::{RunDocument, RunItem, RunState, RunTarget, SlotState};
pub use workflow::{DynamicWorkflowDocument, WorkflowSpec, WorkflowStep};
