//! The worker host contract: identity, profile, and payload types shared
//! by the host and the workers it talks to.
//!
//! The wire contract itself is the `reimagine.worker.v2` protobuf service
//! (defined in `backend-worker-transport-grpc`); this crate carries the
//! Rust types the protobuf messages map to, plus the protocol version the
//! compatibility checks are written against.

mod error;
mod handshake;
mod identity;
mod progress;
mod request;
mod response;
pub mod transport;
mod worker_profile;

pub use error::BackendExecutionError;
pub use handshake::{ProtocolRange, ProtocolVersion, WORKER_PROTOCOL_VERSION, WorkerHello};
pub use identity::{BackendInstanceId, WorkerIdentity, WorkerIncarnationId, WorkerInstallationId};
pub use progress::ProgressFrame;
pub use request::request_operation;
pub use response::TerminalOutcome;
pub use transport::{TransportDescription, TransportError, TransportKind};
pub use worker_profile::{WorkerInstanceProfile, WorkerProfile};
