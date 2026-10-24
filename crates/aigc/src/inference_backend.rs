//! App-host inference backend selection.
//!
//! Backend selection is owned at the app-host/config boundary. Inference
//! runs via the out-of-process Burn backend.

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum BackendSelection {
    #[default]
    Burn,
}

impl std::fmt::Display for BackendSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Burn => f.write_str("burn"),
        }
    }
}

impl From<reimagine_config::InferenceBackendKind> for BackendSelection {
    fn from(kind: reimagine_config::InferenceBackendKind) -> Self {
        match kind {
            reimagine_config::InferenceBackendKind::Burn => Self::Burn,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebootstrap_backend_selection_rejects_candle() {
        let err = serde_json::from_str::<BackendSelection>("\"candle\"").unwrap_err();
        assert!(
            err.to_string().contains("unknown variant `candle`"),
            "{err}"
        );
    }

    #[test]
    fn deserialize_burn_succeeds() {
        let sel: BackendSelection = serde_json::from_str("\"burn\"").unwrap();
        assert_eq!(sel, BackendSelection::Burn);
    }
}
