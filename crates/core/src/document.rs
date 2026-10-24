//! Generic document editing session, history, and document trait.

mod history;
mod session;
mod traits;

pub use history::{History, HistoryEntry};
pub use session::DocumentSession;
pub use traits::{CommandApplication, Document};
