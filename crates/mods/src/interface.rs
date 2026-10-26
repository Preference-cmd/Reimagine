use async_trait::async_trait;
use reimagine_core::diagnostic::Diagnostic;
use reimagine_core::task::CancelToken;
use serde_json::Value;

use crate::context::ModContext;
use crate::manifest::ModManifest;

/// A domain mod. The kernel calls it only through this trait, whether the mod
/// runs in process or behind a process boundary.
#[async_trait]
pub trait Mod: Send + Sync {
    /// The mod's contributions. Must not change after registration.
    fn manifest(&self) -> &ModManifest;

    /// Called once before the first `invoke`.
    async fn start(&self, ctx: ModContext) -> Result<(), Diagnostic>;

    /// Run one operation declared in the manifest. `operation` is the full
    /// `<mod id>.<name>`; `payload` is the operation's input.
    async fn invoke(
        &self,
        operation: &str,
        payload: Value,
        cancel: CancelToken,
    ) -> Result<Value, Diagnostic>;

    /// Called once at shutdown; no `invoke` follows.
    async fn stop(&self) -> Result<(), Diagnostic>;
}
