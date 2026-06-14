use reimagine_inference::operation::InferenceOperationId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendNotImplementedError {
    backend_kind: String,
    operation_id: InferenceOperationId,
    model_series: Option<String>,
    model_variant: Option<String>,
    message: String,
}

impl BackendNotImplementedError {
    pub fn new(
        backend_kind: impl Into<String>,
        operation_id: InferenceOperationId,
        message: impl Into<String>,
    ) -> Self {
        Self {
            backend_kind: backend_kind.into(),
            operation_id,
            model_series: None,
            model_variant: None,
            message: message.into(),
        }
    }

    pub fn with_model(mut self, series: Option<String>, variant: Option<String>) -> Self {
        self.model_series = series;
        self.model_variant = variant;
        self
    }

    pub fn backend_kind(&self) -> &str {
        &self.backend_kind
    }
    pub fn operation_id(&self) -> &InferenceOperationId {
        &self.operation_id
    }
    pub fn model_series(&self) -> Option<&str> {
        self.model_series.as_deref()
    }
    pub fn model_variant(&self) -> Option<&str> {
        self.model_variant.as_deref()
    }
    pub fn message(&self) -> &str {
        &self.message
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandleBackendError {
    BackendNotImplemented(BackendNotImplementedError),
    InvalidRequest(String),
}

impl std::fmt::Display for CandleBackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BackendNotImplemented(err) => write!(
                f,
                "{} not implemented for {}",
                err.operation_id(),
                err.backend_kind()
            ),
            Self::InvalidRequest(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for CandleBackendError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_not_implemented_error_stores_fields() {
        let err = BackendNotImplementedError::new(
            "candle",
            "text.encode".into(),
            "text encode not implemented",
        );
        assert_eq!(err.backend_kind(), "candle");
        assert_eq!(err.operation_id().as_str(), "text.encode");
        assert_eq!(err.message(), "text encode not implemented");
        assert!(err.model_series().is_none());
        assert!(err.model_variant().is_none());
    }

    #[test]
    fn backend_not_implemented_error_with_model() {
        let err =
            BackendNotImplementedError::new("candle", "diffusion.sample".into(), "not implemented")
                .with_model(
                    Some("stable_diffusion".to_string()),
                    Some("sdxl".to_string()),
                );
        assert_eq!(err.model_series(), Some("stable_diffusion"));
        assert_eq!(err.model_variant(), Some("sdxl"));
    }

    #[test]
    fn backend_not_implemented_error_display() {
        let err = CandleBackendError::BackendNotImplemented(BackendNotImplementedError::new(
            "candle",
            "diffusion.sample".into(),
            "not implemented",
        ));
        let msg = err.to_string();
        assert!(msg.contains("diffusion.sample"), "{msg}");
        assert!(msg.contains("candle"), "{msg}");
    }

    #[test]
    fn invalid_request_display() {
        let err = CandleBackendError::InvalidRequest("bad param".to_string());
        assert_eq!(err.to_string(), "bad param");
    }
}
