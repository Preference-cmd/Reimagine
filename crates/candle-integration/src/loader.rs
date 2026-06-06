//! Model loader. Takes a `ModelSpec` (with the weights path already resolved
//! by core) and dispatches by `spec.family` to a concrete model impl.

use reimagine_core::inference::{Error, Model, ModelSpec};

use crate::models::clip::ClipTextEncoder;

pub fn load(spec: &ModelSpec) -> Result<Box<dyn Model>, Error> {
    if !spec.weights.exists() {
        return Err(Error::Loader(format!(
            "weights not found: {}",
            spec.weights.display()
        )));
    }
    match spec.family.as_str() {
        "clip" => Ok(Box::new(ClipTextEncoder::new(spec.clone()))),
        other => Err(Error::Loader(format!("unknown model family: {other}"))),
    }
}