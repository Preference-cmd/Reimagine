use crate::{ModelFormat, ModelRootId, ModelSource};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanObservation {
    root_id: ModelRootId,
    relative_path: String,
    filename: String,
    extension: String,
    format: ModelFormat,
    size_bytes: u64,
    modified_at: Option<String>,
    source: ModelSource,
}

impl ScanObservation {
    pub(crate) fn new(
        root_id: ModelRootId,
        relative_path: String,
        filename: String,
        extension: String,
        format: ModelFormat,
        size_bytes: u64,
        modified_at: Option<String>,
    ) -> Self {
        let source = ModelSource::relative(root_id.clone(), relative_path.clone());
        Self {
            root_id,
            relative_path,
            filename,
            extension,
            format,
            size_bytes,
            modified_at,
            source,
        }
    }

    pub fn root_id(&self) -> &ModelRootId {
        &self.root_id
    }

    pub fn relative_path(&self) -> &str {
        &self.relative_path
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn extension(&self) -> &str {
        &self.extension
    }

    pub fn format(&self) -> ModelFormat {
        self.format
    }

    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    pub fn modified_at(&self) -> Option<&str> {
        self.modified_at.as_deref()
    }

    pub fn source(&self) -> &ModelSource {
        &self.source
    }
}
