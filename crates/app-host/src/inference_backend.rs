//! App-host inference backend selection.
//!
//! Backend selection is owned at the app-host/config boundary. V1 only
//! supports Candle, but the enum leaves room for future backends without
//! changing the runtime or inference executor APIs.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendSelection {
    Candle,
}

impl Default for BackendSelection {
    fn default() -> Self {
        Self::Candle
    }
}

impl std::fmt::Display for BackendSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Candle => f.write_str("candle"),
        }
    }
}

impl From<reimagine_config::InferenceBackendKind> for BackendSelection {
    fn from(kind: reimagine_config::InferenceBackendKind) -> Self {
        match kind {
            reimagine_config::InferenceBackendKind::Candle => Self::Candle,
        }
    }
}
