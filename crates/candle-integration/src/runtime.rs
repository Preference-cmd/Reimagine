//! Long-lived state: workdir + device + dtype + cache of loaded models.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use reimagine_core::inference::{DeviceSpec, DType, Model, ModelSpec};

use crate::load;

/// A running inference session. Holds the workdir, compute device, dtype, and
/// a cache of loaded models keyed by `family::variant`.
pub struct Session {
    workdir: PathBuf,
    device: DeviceSpec,
    dtype: DType,
    cache: HashMap<String, Arc<dyn Model>>,
}

impl Session {
    pub fn new(workdir: PathBuf, device: DeviceSpec, dtype: DType) -> Self {
        Self {
            workdir,
            device,
            dtype,
            cache: HashMap::new(),
        }
    }

    pub fn workdir(&self) -> &PathBuf {
        &self.workdir
    }
    pub fn device(&self) -> &DeviceSpec {
        &self.device
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Load (or fetch from cache) a model.
    pub fn load_model(&mut self, spec: ModelSpec) -> Result<Arc<dyn Model>, reimagine_core::inference::Error> {
        let key = format!("{}::{}", spec.family, spec.variant);
        if let Some(m) = self.cache.get(&key) {
            return Ok(Arc::clone(m));
        }
        let m: Arc<dyn Model> = Arc::from(load(&spec)?);
        self.cache.insert(key, Arc::clone(&m));
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_core::inference::{Error, NodeInput};

    #[test]
    fn session_loads_clip_stub_and_infer_returns_not_implemented() {
        let mut session = Session::new(
            PathBuf::from("/tmp/reimagine-test-workdir"),
            DeviceSpec::Cpu,
            DType::F32,
        );

        // Write a sentinel file so the loader's existence check passes.
        let weights = std::env::temp_dir().join("reimagine-clip-stub-weights.safetensors");
        std::fs::write(&weights, b"stub").unwrap();

        let spec = ModelSpec {
            family: "clip".into(),
            variant: "vit-b-32".into(),
            dtype: DType::F32,
            device: DeviceSpec::Cpu,
            weights: weights.clone(),
        };

        let m = session.load_model(spec).expect("clip stub should load");
        let result = m.infer(NodeInput::Text("hello".into()));
        assert!(matches!(result, Err(Error::NotImplemented(_))));

        // Cache hit on second load_model call returns the same Arc.
        let spec2 = ModelSpec {
            family: "clip".into(),
            variant: "vit-b-32".into(),
            dtype: DType::F32,
            device: DeviceSpec::Cpu,
            weights,
        };
        let m2 = session.load_model(spec2).expect("second load should hit cache");
        assert!(Arc::ptr_eq(&m, &m2));

        let _ = std::fs::remove_file(session.workdir()); // not the weights; harmless
    }
}