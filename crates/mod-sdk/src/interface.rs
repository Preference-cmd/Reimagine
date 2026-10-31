use std::path::PathBuf;

use async_trait::async_trait;
use reimagine_core::diagnostic::Diagnostic;
use reimagine_core::task::CancelToken;
use serde_json::Value;

use crate::context::ModContext;
use crate::manifest::ModManifest;

/// An operation invocation passed to [`Mod::invoke`].
#[derive(Debug, Clone)]
pub struct Invocation {
    pub operation: String,
    pub input: Value,
    pub output_dir: PathBuf,
    pub cancel: CancelToken,
}

impl Invocation {
    pub fn new(
        operation: impl Into<String>,
        input: Value,
        output_dir: impl Into<PathBuf>,
        cancel: CancelToken,
    ) -> Self {
        Self {
            operation: operation.into(),
            input,
            output_dir: output_dir.into(),
            cancel,
        }
    }
}

/// A domain mod. The kernel calls it only through this trait, whether the mod
/// runs in process or behind a process boundary.
#[allow(clippy::double_must_use)]
#[async_trait]
pub trait Mod: Send + Sync {
    /// The mod's contributions. Must not change after registration.
    fn manifest(&self) -> &ModManifest;

    /// Called once before the first `invoke`.
    async fn start(&self, ctx: ModContext) -> Result<(), Diagnostic>;

    /// Run one operation declared in the manifest.
    async fn invoke(&self, invocation: Invocation) -> Result<Value, Diagnostic>;

    /// Stop the mod; no `invoke` follows.
    ///
    /// The kernel may call this again when an earlier call was cancelled before
    /// it completed, so `stop` must be idempotent.
    async fn stop(&self) -> Result<(), Diagnostic>;
}
