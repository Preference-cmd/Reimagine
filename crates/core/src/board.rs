//! Board domain: project canvas document with items and commands.
//!
//! A Board is the persisted project canvas document that holds visual items
//! (workflow references, asset references, notes, run references) arranged
//! spatially on a canvas.

mod command;
mod document;
mod item;
mod kind;
mod session;

pub use command::{
    BoardChange, BoardCommand, BoardCommandBatch, BoardCommandResult, BoardCommandResultStatus,
};
pub use document::{BoardDocument, BoardSchemaVersion};
pub use item::{BoardItem, BoardItemPosition, BoardItemSize};
pub use kind::BoardItemKind;
pub use session::BoardSession;
