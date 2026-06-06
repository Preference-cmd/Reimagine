//! Backend-agnostic inference types and trait.
//!
//! These types form the public boundary between `reimagine-core` and any
//! inference backend (e.g. `reimagine-candle-integration`). A backend
//! implements the [`Model`] trait; core talks to it only through these types.

use std::path::PathBuf;

use crate::model::TensorData;

/// Self-contained description of a model to load.
///
/// `weights` is the path to the safetensors / onnx / etc. file. Core is
/// responsible for resolving the path; the loader does not concatenate.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub family: String,
    pub variant: String,
    pub dtype: DType,
    pub device: DeviceSpec,
    pub weights: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
}

impl Default for DType {
    fn default() -> Self {
        Self::F32
    }
}

#[derive(Debug, Clone)]
pub enum DeviceSpec {
    Cpu,
    Cuda(usize),
    Metal,
}

impl Default for DeviceSpec {
    fn default() -> Self {
        Self::Cpu
    }
}

/// Backend-agnostic input to a model's `infer` call. The backend dispatches
/// based on the loaded model's family.
#[derive(Debug, Clone)]
pub enum NodeInput {
    Text(String),
    Image(TensorData),
    Latent(TensorData),
    Conditioning(TensorData),
}

/// Backend-agnostic output from a model's `infer` call.
#[derive(Debug, Clone)]
pub enum NodeOutput {
    Embedding(TensorData),
    Image(TensorData),
    Latent(TensorData),
}

/// A loaded, ready-to-call model. Backends implement this trait for each
/// concrete model family (e.g. CLIP, UNet, VAE).
pub trait Model: Send + Sync {
    fn infer(&self, input: NodeInput) -> Result<NodeOutput, Error>;
}

/// Inference-related errors. Concrete backends may return richer error
/// chains; the trait only requires this top-level shape.
#[derive(Debug)]
pub enum Error {
    Io(String),
    Device(String),
    Loader(String),
    Model(String),
    Shape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    NotImplemented(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(s) => write!(f, "io error: {s}"),
            Self::Device(s) => write!(f, "device error: {s}"),
            Self::Loader(s) => write!(f, "loader error: {s}"),
            Self::Model(s) => write!(f, "model error: {s}"),
            Self::Shape { expected, actual } => {
                write!(f, "shape mismatch: expected {expected:?}, got {actual:?}")
            }
            Self::NotImplemented(s) => write!(f, "not implemented: {s}"),
        }
    }
}

impl std::error::Error for Error {}
