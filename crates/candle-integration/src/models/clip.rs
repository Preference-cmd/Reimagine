//! CLIP text encoder. M0 stub: `infer` returns `NotImplemented`.
//! Real Candle wiring lands at M1.

use reimagine_core::inference::{Error, Model, ModelSpec, NodeInput, NodeOutput};

pub struct ClipTextEncoder {
    spec: ModelSpec,
}

impl ClipTextEncoder {
    pub fn new(spec: ModelSpec) -> Self {
        Self { spec }
    }

    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }
}

impl Model for ClipTextEncoder {
    fn infer(&self, input: NodeInput) -> Result<NodeOutput, Error> {
        match input {
            NodeInput::Text(_) => Err(Error::NotImplemented("ClipTextEncoder::infer (M1)")),
            other => Err(Error::Model(format!(
                "ClipTextEncoder expects Text input, got {other:?}"
            ))),
        }
    }
}