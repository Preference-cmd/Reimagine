use std::collections::VecDeque;
use std::path::Path;
use std::time::UNIX_EPOCH;

use crate::{ModelFormat, ModelManagerError, ModelManagerResult, ModelRoot, ModelRootKind};

use super::{ScanConfig, ScanObservation};

pub struct ModelScanner {
    config: ScanConfig,
}

impl ModelScanner {
    pub fn new(config: ScanConfig) -> Self {
        Self { config }
    }

    pub async fn scan_root(
        &self,
        root: &ModelRoot,
        root_path: impl AsRef<Path>,
    ) -> ModelManagerResult<Vec<ScanObservation>> {
        let root_path = root_path.as_ref();
        if !tokio::fs::try_exists(root_path).await.map_err(|error| {
            ModelManagerError::ReadFailed {
                path: root_path.display().to_string(),
                message: error.to_string(),
            }
        })? {
            return Ok(Vec::new());
        }

        let mut observations = Vec::new();
        let mut queue = VecDeque::from([root_path.to_path_buf()]);

        while let Some(dir) = queue.pop_front() {
            let mut entries =
                tokio::fs::read_dir(&dir)
                    .await
                    .map_err(|error| ModelManagerError::ReadFailed {
                        path: dir.display().to_string(),
                        message: error.to_string(),
                    })?;

            while let Some(entry) =
                entries
                    .next_entry()
                    .await
                    .map_err(|error| ModelManagerError::ReadFailed {
                        path: dir.display().to_string(),
                        message: error.to_string(),
                    })?
            {
                let path = entry.path();
                let relative = relative_path(root_path, &path);

                if self.config.ignore_hidden() && has_hidden_component(&relative) {
                    continue;
                }

                if matches_any(self.config.exclude_patterns(), &relative)
                    || is_generated_package_path(root.kind(), &relative)
                {
                    continue;
                }

                let file_type =
                    entry
                        .file_type()
                        .await
                        .map_err(|error| ModelManagerError::ReadFailed {
                            path: path.display().to_string(),
                            message: error.to_string(),
                        })?;

                if file_type.is_dir() {
                    if self.config.recursive() {
                        queue.push_back(path);
                    }
                    continue;
                }

                if !file_type.is_file() {
                    continue;
                }

                if !matches_include(self.config.include_patterns(), &relative) {
                    continue;
                }

                let extension = extension(&path);
                if !self
                    .config
                    .supported_extensions()
                    .iter()
                    .any(|supported| supported == &extension)
                {
                    continue;
                }

                let metadata =
                    entry
                        .metadata()
                        .await
                        .map_err(|error| ModelManagerError::ReadFailed {
                            path: path.display().to_string(),
                            message: error.to_string(),
                        })?;
                let filename = path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or_default()
                    .to_owned();
                observations.push(ScanObservation::new(
                    root.id().clone(),
                    relative,
                    filename,
                    extension.clone(),
                    format_for_extension(&extension),
                    metadata.len(),
                    modified_at_string(&metadata),
                ));
            }
        }

        observations.sort_by(|left, right| left.relative_path().cmp(right.relative_path()));
        Ok(observations)
    }
}

fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .components()
        .filter_map(|component| match component {
            std::path::Component::Normal(part) => part.to_str(),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn has_hidden_component(relative: &str) -> bool {
    relative.split('/').any(|part| part.starts_with('.'))
}

fn matches_any(patterns: &[String], relative: &str) -> bool {
    patterns
        .iter()
        .any(|pattern| glob_match::glob_match(pattern, relative))
}

fn matches_include(patterns: &[String], relative: &str) -> bool {
    patterns.is_empty()
        || patterns
            .iter()
            .any(|pattern| glob_match::glob_match(pattern, relative))
}

fn is_generated_package_path(root_kind: ModelRootKind, relative: &str) -> bool {
    matches!(root_kind, ModelRootKind::BasePathModels)
        && (relative == "converted" || relative.starts_with("converted/"))
}

fn extension(path: &Path) -> String {
    path.extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
}

fn format_for_extension(extension: &str) -> ModelFormat {
    match extension {
        "safetensors" => ModelFormat::Safetensors,
        "gguf" => ModelFormat::Gguf,
        "ckpt" => ModelFormat::Ckpt,
        _ => ModelFormat::Unknown,
    }
}

fn modified_at_string(metadata: &std::fs::Metadata) -> Option<String> {
    let duration = metadata.modified().ok()?.duration_since(UNIX_EPOCH).ok()?;
    Some(duration.as_secs().to_string())
}
