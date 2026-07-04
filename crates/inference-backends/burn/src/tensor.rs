//! Backend-generic tensor wrapper for the Burn inference backend.
//!
//! `BurnTensor<D>` wraps a concrete `Tensor<B, D>` behind a feature-gated
//! enum so that the rest of the crate can remain backend-agnostic.  The
//! active variant is chosen at compile time by the `wgpu` / `flex` feature
//! flags, or falls back to `burn-ndarray`.
//!
//! This is the V1 single-variant (ndarray-only) version — only
//! `BurnTensor::Ndarray` is compiled under the default `wgpu` feature
//! (which still uses `burn-ndarray` as a compile-time base).  Future
//! issues will add the `Wgpu` and `Flex` variants.

use burn_ndarray::NdArray;
use burn_tensor::{Tensor, TensorData};

/// A backend-generic `D`-dimensional tensor.
///
/// # Feature behaviour
///
/// | Active feature  | Variant        | Concrete type       |
/// |-----------------|----------------|---------------------|
/// | `wgpu` (default)| `Ndarray`      | `Tensor<NdArray, D>`|
/// | `flex`          | `Flex`         | `Tensor<Flex, D>`   |
/// | _(none of the above)_ | `Ndarray` | `Tensor<NdArray, D>`|
///
/// V1 uses `Ndarray` under all feature configurations; the `Wgpu` / `Flex`
/// variants will be wired in a follow-up (burn/14f).
#[derive(Debug, Clone)]
pub enum BurnTensor<const D: usize> {
    /// Tensor backed by `burn-ndarray` (legacy CPU path, also the
    /// compile-time base under the `wgpu` feature).
    Ndarray(Tensor<NdArray, D>),
}

use burn_ndarray::NdArrayDevice;

// ---------------------------------------------------------------------------
// Construction helpers
// ---------------------------------------------------------------------------
impl<const D: usize> BurnTensor<D> {
    /// Create a zero-filled tensor on the ndarray CPU device.
    pub fn zeros(shape: [usize; D]) -> Self {
        Self::Ndarray(Tensor::<NdArray, D>::zeros(shape, &NdArrayDevice::Cpu))
    }
}

// ---------------------------------------------------------------------------
// Accessors (match-based, backend-agnostic)
// ---------------------------------------------------------------------------
impl<const D: usize> BurnTensor<D> {
    /// Return the shape dimensions.
    pub fn dims(&self) -> [usize; D] {
        match self {
            Self::Ndarray(t) => t.shape().dims(),
        }
    }

    /// Number of elements.
    pub fn num_elements(&self) -> usize {
        match self {
            Self::Ndarray(t) => t.shape().num_elements(),
        }
    }

    /// Approximate byte size (f32 × element count).
    pub fn byte_size(&self) -> usize {
        self.num_elements() * 4
    }

    /// Extract raw tensor data (useful for testing / serialisation).
    pub fn to_data(&self) -> TensorData {
        match self {
            Self::Ndarray(t) => t.to_data(),
        }
    }
}

// ---------------------------------------------------------------------------
// Map / transform helpers
// ---------------------------------------------------------------------------
impl<const D: usize> BurnTensor<D> {
    /// Apply a closure to the inner ndarray tensor, returning a new
    /// `BurnTensor`.
    pub fn map(self, f: impl FnOnce(Tensor<NdArray, D>) -> Tensor<NdArray, D>) -> Self {
        match self {
            Self::Ndarray(t) => Self::Ndarray(f(t)),
        }
    }
}
