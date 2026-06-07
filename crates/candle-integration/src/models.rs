//! Typed model implementations, dispatched by family.
//!
//! Each submodule corresponds to a model family. The submodule's `pub fn`
//! or struct constructor is the way core obtains a `Model` impl for that
//! family; `loader::load` is the dispatcher.

#[path = "models/clip.rs"]
pub mod clip;
