use reimagine_inference::InferenceCapability;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendNotImplementedError {
    backend_kind: String,
    capability: InferenceCapability,
    model_series: Option<String>,
    model_variant: Option<String>,
    message: String,
}

impl BackendNotImplementedError {
    pub fn new(
        backend_kind: impl Into<String>,
        capability: InferenceCapability,
        message: impl Into<String>,
    ) -> Self {
        Self {
            backend_kind: backend_kind.into(),
            capability,
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
    pub fn capability(&self) -> InferenceCapability {
        self.capability
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
    DeviceUnavailable {
        requested: String,
        reason: String,
    },
    UnsupportedModelFamily {
        model_id: String,
        series: String,
        variant: String,
    },
}

impl std::fmt::Display for CandleBackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BackendNotImplemented(err) => write!(
                f,
                "{} not implemented for {}",
                err.capability(),
                err.backend_kind()
            ),
            Self::InvalidRequest(msg) => f.write_str(msg),
            Self::DeviceUnavailable { requested, reason } => {
                write!(f, "candle device `{requested}` unavailable: {reason}")
            }
            Self::UnsupportedModelFamily {
                model_id,
                series,
                variant,
            } => write!(
                f,
                "candle backend has no loader for model `{model_id}` (series `{series}`, variant `{variant}`)"
            ),
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
            InferenceCapability::TextEncode,
            "text encode not implemented",
        );
        assert_eq!(err.backend_kind(), "candle");
        assert_eq!(err.capability(), InferenceCapability::TextEncode);
        assert_eq!(err.message(), "text encode not implemented");
        assert!(err.model_series().is_none());
        assert!(err.model_variant().is_none());
    }

    #[test]
    fn backend_not_implemented_error_with_model() {
        let err = BackendNotImplementedError::new(
            "candle",
            InferenceCapability::DiffusionSample,
            "not implemented",
        )
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
            InferenceCapability::DiffusionSample,
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

    #[test]
    fn device_unavailable_display() {
        let err = CandleBackendError::DeviceUnavailable {
            requested: "tpu".to_string(),
            reason: "unsupported".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("tpu"), "{msg}");
        assert!(msg.contains("unsupported"), "{msg}");
    }

    #[test]
    fn unsupported_model_family_display() {
        let err = CandleBackendError::UnsupportedModelFamily {
            model_id: "flux-dev".to_string(),
            series: "flux".to_string(),
            variant: "dev".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("flux-dev"), "{msg}");
        assert!(msg.contains("flux"), "{msg}");
        assert!(msg.contains("dev"), "{msg}");
    }
}
