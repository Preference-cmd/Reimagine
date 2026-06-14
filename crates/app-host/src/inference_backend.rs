//! App-host inference backend selection.
//!
//! Backend selection is owned at the app-host/config boundary. V1 only
//! supports Candle, but the enum leaves room for future backends without
//! changing the runtime or inference executor APIs.

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendSelection {
    Candle,
}

impl Default for BackendSelection {
    fn default() -> Self {
        Self::Candle
    }
}
