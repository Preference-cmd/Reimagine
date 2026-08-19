use reimagine_core::event::Timestamp;
use reimagine_core::project::{Project, ProjectMetadata};
use serde::{Deserialize, Serialize};

/// Stable project projection shared by Tauri and other host adapters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectDto {
    pub id: String,
    pub name: String,
    pub description: String,
    pub created_at: String,
    pub updated_at: String,
}

impl From<Project> for ProjectDto {
    fn from(project: Project) -> Self {
        Self {
            id: project.id().to_string(),
            name: project.metadata().name().to_owned(),
            description: project.metadata().description().to_owned(),
            created_at: project.metadata().created_at().to_string(),
            updated_at: project.metadata().updated_at().to_string(),
        }
    }
}

/// Metadata accepted by project create/update commands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectMetadataInputDto {
    pub name: String,
    pub description: String,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
}

impl ProjectMetadataInputDto {
    pub fn into_domain(self, now: &str) -> ProjectMetadata {
        ProjectMetadata::new(
            self.name,
            self.description,
            Timestamp::new(self.created_at.unwrap_or_else(|| now.to_owned())),
            Timestamp::new(self.updated_at.unwrap_or_else(|| now.to_owned())),
        )
    }

    pub fn into_updated_domain(self, created_at: Timestamp, now: &str) -> ProjectMetadata {
        ProjectMetadata::new(
            self.name,
            self.description,
            created_at,
            Timestamp::new(self.updated_at.unwrap_or_else(|| now.to_owned())),
        )
    }
}
