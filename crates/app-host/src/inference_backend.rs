//! App-host inference backend selection.
//!
//! Backend selection is owned at the app-host/config boundary. V1 supports
//! Candle and Burn. Burn selection resolves to `burn:wgpu:default` and does
//! not install a Candle fallback execution route.

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum BackendSelection {
    #[default]
    Candle,
    Burn,
}

impl std::fmt::Display for BackendSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Candle => f.write_str("candle"),
            Self::Burn => f.write_str("burn"),
        }
    }
}

impl From<reimagine_config::InferenceBackendKind> for BackendSelection {
    fn from(kind: reimagine_config::InferenceBackendKind) -> Self {
        match kind {
            reimagine_config::InferenceBackendKind::Candle => Self::Candle,
            reimagine_config::InferenceBackendKind::Burn => Self::Burn,
        }
    }
}
