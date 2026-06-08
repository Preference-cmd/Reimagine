//! Model series classification configuration and candidate classification.

mod candidate;
mod classifier;
mod series_config;

pub use candidate::ClassificationCandidate;
pub use classifier::{ClassificationResult, Classifier};
pub use series_config::{MODEL_SERIES_SCHEMA_VERSION, ModelSeriesConfig, ModelSeriesRule};
