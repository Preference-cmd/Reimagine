use std::collections::BTreeMap;
use tokio::sync::RwLock;
use reimagine_config::{AppPaths, atomic_write};
use reimagine_core::model::ProjectId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectMemory {
    pub entries: BTreeMap<String, MemoryEntry>,
    pub version: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub value: serde_json::Value,
    pub updated_by: String,
    pub updated_at: String,
}

#[derive(Debug)]
pub struct ProjectMemoryService {
    paths: AppPaths,
    memories: RwLock<BTreeMap<ProjectId, ProjectMemory>>,
}

impl ProjectMemoryService {
    pub fn new(paths: AppPaths) -> Self {
        Self { paths, memories: RwLock::new(BTreeMap::new()) }
    }
    fn memory_path(&self, project_id: &ProjectId) -> std::path::PathBuf {
        self.paths.project_dir(project_id.as_str()).join("memory.json")
    }
    pub async fn get(&self, project_id: &ProjectId, key: &str) -> Option<MemoryEntry> {
        let mems = self.memories.read().await;
        mems.get(project_id).and_then(|m| m.entries.get(key).cloned())
    }
    pub async fn set(&self, project_id: ProjectId, key: String, value: serde_json::Value, updated_by: String) -> Result<(), crate::AppHostError> {
        let mut mems = self.memories.write().await;
        let mem = mems.entry(project_id.clone()).or_default();
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs().to_string();
        let entry = MemoryEntry { value, updated_by, updated_at: now };
        mem.entries.insert(key, entry);
        mem.version += 1;
        let path = self.memory_path(&project_id);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| crate::AppHostError::Io { path: parent.to_path_buf(), message: e.to_string() })?;
        }
        let data = serde_json::to_vec_pretty(&*mem).map_err(|e| crate::AppHostError::Io { path: path.clone(), message: format!("serialize memory: {e}") })?;
        atomic_write(&path, data).await.map_err(|e| crate::AppHostError::Io { path: path.clone(), message: e.to_string() })?;
        Ok(())
    }
    pub async fn all(&self, project_id: &ProjectId) -> Option<ProjectMemory> {
        self.memories.read().await.get(project_id).cloned()
    }
}
