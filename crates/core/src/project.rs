//! Project domain aggregate root.
//!
//! A Project is the top-level container for all creative work in Reimagine.
//! It owns boards, workflows, assets, runs, and agent threads.

use crate::event::Timestamp;
use crate::model::ProjectId;

/// Project domain aggregate root.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Project {
    id: ProjectId,
    metadata: ProjectMetadata,
}

impl Project {
    /// Create a new project with the given id and metadata.
    pub fn new(id: ProjectId, metadata: ProjectMetadata) -> Self {
        Self { id, metadata }
    }

    /// Get the project id.
    pub fn id(&self) -> &ProjectId {
        &self.id
    }

    /// Get the project metadata.
    pub fn metadata(&self) -> &ProjectMetadata {
        &self.metadata
    }

    /// Get a mutable reference to the project metadata.
    pub fn metadata_mut(&mut self) -> &mut ProjectMetadata {
        &mut self.metadata
    }
}

/// Metadata for a project.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProjectMetadata {
    name: String,
    description: String,
    created_at: Timestamp,
    updated_at: Timestamp,
}

impl ProjectMetadata {
    /// Create new project metadata.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        created_at: Timestamp,
        updated_at: Timestamp,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            created_at,
            updated_at,
        }
    }

    /// Get the project name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the project description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Get the creation timestamp.
    pub fn created_at(&self) -> &Timestamp {
        &self.created_at
    }

    /// Get the last update timestamp.
    pub fn updated_at(&self) -> &Timestamp {
        &self.updated_at
    }

    /// Set the project name.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    /// Set the project description.
    pub fn set_description(&mut self, description: impl Into<String>) {
        self.description = description.into();
    }

    /// Set the update timestamp.
    pub fn set_updated_at(&mut self, updated_at: Timestamp) {
        self.updated_at = updated_at;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ProjectId;

    #[test]
    fn project_creation() {
        let id = ProjectId::new("my-project");
        let metadata = ProjectMetadata::new(
            "My Project",
            "A test project",
            Timestamp::new("2026-01-01T00:00:00Z"),
            Timestamp::new("2026-01-01T00:00:00Z"),
        );

        let project = Project::new(id.clone(), metadata.clone());
        assert_eq!(project.id(), &id);
        assert_eq!(project.metadata(), &metadata);
    }

    #[test]
    fn project_metadata_accessors() {
        let metadata = ProjectMetadata::new(
            "Test",
            "Description",
            Timestamp::new("2026-01-01T00:00:00Z"),
            Timestamp::new("2026-01-02T00:00:00Z"),
        );

        assert_eq!(metadata.name(), "Test");
        assert_eq!(metadata.description(), "Description");
        assert_eq!(metadata.created_at().as_str(), "2026-01-01T00:00:00Z");
        assert_eq!(metadata.updated_at().as_str(), "2026-01-02T00:00:00Z");
    }

    #[test]
    fn project_metadata_mutation() {
        let mut metadata = ProjectMetadata::new(
            "Old Name",
            "Old Description",
            Timestamp::new("2026-01-01T00:00:00Z"),
            Timestamp::new("2026-01-01T00:00:00Z"),
        );

        metadata.set_name("New Name");
        metadata.set_description("New Description");
        metadata.set_updated_at(Timestamp::new("2026-01-03T00:00:00Z"));

        assert_eq!(metadata.name(), "New Name");
        assert_eq!(metadata.description(), "New Description");
        assert_eq!(metadata.updated_at().as_str(), "2026-01-03T00:00:00Z");
    }

    #[test]
    fn project_serde_roundtrip() {
        let id = ProjectId::new("test-project");
        let metadata = ProjectMetadata::new(
            "Test Project",
            "A test project",
            Timestamp::new("2026-01-01T00:00:00Z"),
            Timestamp::new("2026-01-02T00:00:00Z"),
        );

        let project = Project::new(id, metadata);
        let json = serde_json::to_string(&project).expect("serialize");
        let back: Project = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(project, back);
    }

    #[test]
    fn project_metadata_serde_roundtrip() {
        let metadata = ProjectMetadata::new(
            "Test",
            "Description",
            Timestamp::new("2026-01-01T00:00:00Z"),
            Timestamp::new("2026-01-02T00:00:00Z"),
        );

        let json = serde_json::to_string(&metadata).expect("serialize");
        let back: ProjectMetadata = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(metadata, back);
    }
}
