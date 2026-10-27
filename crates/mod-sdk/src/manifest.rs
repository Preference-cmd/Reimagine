use semver::Version;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The version of the mod interface this kernel implements.
pub const API_VERSION: Version = Version::new(0, 1, 0);

/// What a mod contributes. Plain data: the kernel can list it without
/// running the mod.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModManifest {
    /// Mod id and namespace: lowercase snake_case.
    pub id: String,
    pub name: String,
    pub version: String,
    /// Semver requirement on [`API_VERSION`], e.g. `"0.1"`.
    pub api: String,
    pub operations: Vec<OperationSpec>,
    pub events: Vec<ModEvent>,
}

/// An operation callable through [`crate::Mod::invoke`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationSpec {
    /// `<namespace>.<name>`.
    pub name: String,
    pub description: String,
    /// JSON Schema 2020-12 of the input; always an object schema.
    pub input_schema: Value,
    /// JSON Schema 2020-12 of the output.
    pub output_schema: Value,
    pub annotations: OperationAnnotations,
}

/// Behavior hints for an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct OperationAnnotations {
    /// Changes no state.
    pub read_only: bool,
    /// May destroy data or interrupt work in progress.
    pub destructive: bool,
    /// Repeating the call with the same input has no further effect.
    pub idempotent: bool,
    /// Produces identical results given identical inputs.
    pub deterministic: bool,
    /// May affect or depend on state outside the project (network, exports).
    pub open_world: bool,
}

/// An event kind the mod publishes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModEvent {
    /// `<mod id>.<name>`, or `<mod id>.<prefix>.*` for a family of kinds.
    pub kind: String,
    /// JSON Schema 2020-12 of the payload.
    pub schema: Value,
}

impl ModEvent {
    /// Whether `kind` is this declaration or a member of its family.
    pub fn covers(&self, kind: &str) -> bool {
        match self.kind.strip_suffix('*') {
            Some(family) => kind.len() > family.len() && kind.starts_with(family),
            None => kind == self.kind,
        }
    }
}
