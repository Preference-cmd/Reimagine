//! The `text` mod: six deterministic operations over Markdown or plain text.
//!
//! Every operation that takes one document accepts it either as an asset
//! (`doc`, a path the host resolved) or as inline text (`text`); exactly one
//! must be present and the result mirrors the form. `manifest` declares the
//! operation schemas.
//!
//! `text.split` and `text.join` operate on arrays of documents; their input
//! shape decides whether the result is an asset or inline text.

#![deny(unsafe_code)]
// A diagnostic is the error of every operation. It is large but only built on
// the failure path, so boxing it would only add noise to every signature.
#![allow(clippy::result_large_err)]

mod document;
mod front_matter;
mod manifest;
mod markdown;
mod ops;
mod template;

use std::path::Path;

use mod_sdk::async_trait;
use mod_sdk::diagnostic::{Diagnostic, caller_error};
use mod_sdk::serde_json::Value;
use mod_sdk::{Invocation, Mod, ModContext, ModManifest};

/// The mod id, also the operation namespace.
pub const ID: &str = "text";

/// The `text` mod. Every operation is stateless; the manifest is the only
/// state, and it never changes after registration.
pub struct TextMod {
    manifest: ModManifest,
}

impl TextMod {
    pub fn new() -> Self {
        Self {
            manifest: manifest::manifest(),
        }
    }
}

impl Default for TextMod {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Mod for TextMod {
    fn manifest(&self) -> &ModManifest {
        &self.manifest
    }

    async fn start(&self, _ctx: ModContext) -> Result<(), Diagnostic> {
        Ok(())
    }

    async fn invoke(&self, invocation: Invocation) -> Result<Value, Diagnostic> {
        let input = &invocation.input;
        let output_dir: &Path = &invocation.output_dir;
        match invocation.operation.as_str() {
            "text.normalize" => ops::normalize::run(input, output_dir),
            "text.replace" => ops::replace::run(input, output_dir),
            "text.inspect" => ops::inspect::run(input, output_dir),
            "text.template" => ops::template::run(input, output_dir),
            "text.split" => ops::split::run(input, output_dir),
            "text.join" => ops::join::run(input, output_dir),
            other => Err(caller_error(
                ID,
                "UNKNOWN_OPERATION",
                format!("the text mod has no operation `{other}`"),
            )),
        }
    }

    async fn stop(&self) -> Result<(), Diagnostic> {
        Ok(())
    }
}
