use std::sync::Arc;

use super::models::ModelRef;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum ParamValue {
    String(String),
    Text(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
    Seed(u64),
    Select(String),
    Path(String),
    ModelRef(ModelRef),
    Null,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeValue {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Tensor(TensorData),
    Empty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TensorDType {
    F32,
    F16,
    BF16,
    U8,
    I64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TensorShape {
    dims: Vec<usize>,
}

impl TensorShape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }
}

#[derive(Debug, Clone)]
pub struct TensorData {
    data: Arc<[f32]>,
    shape: TensorShape,
}

impl TensorData {
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data: data.into(),
            shape: TensorShape::new(shape),
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.data.to_vec()
    }

    pub fn shape(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

impl PartialEq for TensorData {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data[..] == other.data[..]
    }
}
