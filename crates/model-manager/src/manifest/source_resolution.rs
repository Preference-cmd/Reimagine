use std::path::{Path, PathBuf};

use super::{ModelManifest, ModelRoot, ModelSource};

pub(crate) fn resolve_source_path(
    manifest: &ModelManifest,
    source: &ModelSource,
    models_dir: &Path,
) -> Option<PathBuf> {
    match source {
        ModelSource::LocalFileRelative { root_id, path } => {
            if root_id.as_str() == "base" {
                Some(models_dir.join(path))
            } else {
                let root = manifest
                    .model_roots()
                    .iter()
                    .find(|root| root.id() == root_id)?;
                Some(resolve_root_path(root, models_dir).join(path))
            }
        }
        ModelSource::LocalFileAbsolute { path } => Some(PathBuf::from(path)),
    }
}

pub(crate) fn resolve_root_path(root: &ModelRoot, models_dir: &Path) -> PathBuf {
    let root_path = Path::new(root.path());
    if root_path.is_absolute() {
        root_path.to_path_buf()
    } else {
        models_dir.join(root_path)
    }
}
