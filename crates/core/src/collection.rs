//! Collection domain: ordered slots of assets and slot modification commands.

mod command;
mod document;
mod slot;

pub use command::{CollectionChange, CollectionCommand};
pub use document::CollectionDocument;
pub use slot::Slot;
