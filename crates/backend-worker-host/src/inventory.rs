pub mod compatibility;
pub mod error;
pub mod records;
pub mod store;

pub use compatibility::{ProtocolIncompatibility, check_installed_protocol, check_record_protocol};
pub use error::{InventoryError, InventoryResult};
pub use records::{InstallationRecord, InventoryIndex, InventorySnapshot, VersionPolicy};
pub use store::InventoryStore;
