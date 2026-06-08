use crate::manifest::{ModelFormat, ModelRootId};

/// A file observed by a scanner, provided to the classifier without filesystem access.
#[derive(Debug, Clone)]
pub struct ClassificationCandidate {
    root_id: Option<ModelRootId>,
    path: String,
    filename: String,
    extension: String,
    observed_format: Option<ModelFormat>,
}

impl ClassificationCandidate {
    pub fn new(
        root_id: Option<ModelRootId>,
        path: impl Into<String>,
        filename: impl Into<String>,
        extension: impl Into<String>,
    ) -> Self {
        Self {
            root_id,
            path: path.into(),
            filename: filename.into(),
            extension: extension.into(),
            observed_format: None,
        }
    }

    pub fn with_observed_format(mut self, format: ModelFormat) -> Self {
        self.observed_format = Some(format);
        self
    }

    pub fn root_id(&self) -> Option<&ModelRootId> {
        self.root_id.as_ref()
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn extension(&self) -> &str {
        &self.extension
    }

    pub fn observed_format(&self) -> Option<ModelFormat> {
        self.observed_format
    }
}
