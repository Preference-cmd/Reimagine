//! Domain event and operation report payloads.

#[path = "event/domain_event.rs"]
mod domain_event;
#[path = "event/kind.rs"]
mod kind;
#[path = "event/report.rs"]
mod report;
#[path = "event/source.rs"]
mod source;

pub use domain_event::{DomainEvent, DomainEventId, Timestamp};
pub use kind::DomainEventKind;
pub use report::OperationReport;
pub use source::DomainEventSource;
