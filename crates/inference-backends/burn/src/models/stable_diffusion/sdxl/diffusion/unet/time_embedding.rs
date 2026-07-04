//! SDXL sinusoidal time embedding and time embedding MLP.
//!
//! These functions compute the time-step conditioning that is broadcast
//! into each ResBlock.  The diffusion timestep (a scalar per batch item)
//! is first encoded as a sinusoidal position embedding, then projected
//! through a two-layer MLP.

use burn_ndarray::NdArray;
use burn_ndarray::NdArrayDevice;
use burn_tensor::Shape;
use burn_tensor::{Tensor, TensorData, activation};

use crate::error::BurnBackendError;
use crate::tensor::BurnTensor;

// ---------------------------------------------------------------------------
// Sinusoidal time embedding
// ---------------------------------------------------------------------------

/// Transformer-style sinusoidal position embedding for diffusion timesteps.
///
/// Each element `t` in the batch is mapped to a sinusoid:
///
/// ```text
/// for i in 0..dim/2:
///     freq = 1.0 / (max_period ^ (2*i/dim))
///     embedding[b, 2*i]   = sin(t_b * freq)
///     embedding[b, 2*i+1] = cos(t_b * freq)
/// ```
///
/// # Arguments
///
/// * `timesteps` — int-like values (u64) representing the diffusion timestep
///   index, shaped `[batch]`.
/// * `dim` — embedding dimension (must be even; 256 for SDXL).
/// * `max_period` — maximum wavelength (10000 for SDXL).
pub fn sinusoidal_time_embedding(
    timesteps: &BurnTensor<1>,
    dim: usize,
    max_period: u32,
) -> Result<BurnTensor<2>, BurnBackendError> {
    let timesteps_nd = match timesteps {
        BurnTensor::Ndarray(t) => t.clone(),
    };

    if dim % 2 != 0 {
        return Err(BurnBackendError::InvalidRequest(
            "sinusoidal_time_embedding: dim must be even".into(),
        ));
    }

    let batch = timesteps_nd.dims()[0];
    let half_dim = dim / 2;

    // Build frequency vector using exp(-(2i/dim) * ln(max_period))
    let log_max = (max_period as f64).ln();
    let mut freqs_data = vec![0.0f64; half_dim];
    for i in 0..half_dim {
        let exponent = -(2.0 * i as f64) / dim as f64 * log_max;
        freqs_data[i] = exponent.exp();
    }

    // Read timestep values as f32
    let ts_data = timesteps_nd.to_data();
    let ts_slice = ts_data.as_slice::<f32>().unwrap();

    // Build embedding matrix [batch, dim]
    let mut embedding_data = vec![0.0f32; batch * dim];
    for b in 0..batch {
        let t = ts_slice[b] as f64;
        for i in 0..half_dim {
            let angle = t * freqs_data[i];
            embedding_data[b * dim + 2 * i] = angle.sin() as f32;
            embedding_data[b * dim + 2 * i + 1] = angle.cos() as f32;
        }
    }

    let tensor = Tensor::<NdArray, 2>::from_data(
        TensorData::new(embedding_data, Shape::new([batch, dim])),
        &NdArrayDevice::Cpu,
    );

    Ok(BurnTensor::Ndarray(tensor))
}

// ---------------------------------------------------------------------------
// Time embedding MLP
// ---------------------------------------------------------------------------

/// Project the sinusoidal time embedding through a two-layer MLP.
///
/// ```text
/// fc1: Linear(time_dim_in, time_dim_out)
/// act: SiLU
/// fc2: Linear(time_dim_out, time_dim_out)
/// ```
///
/// # Arguments
///
/// * `x` — sinusoidal embedding `[batch, time_dim_in]` (e.g. `[batch, 256]`).
/// * `fc1_weight` — `[time_dim_out, time_dim_in]`
/// * `fc1_bias` — `[time_dim_out]`
/// * `fc2_weight` — `[time_dim_out, time_dim_out]`
/// * `fc2_bias` — `[time_dim_out]`
pub fn time_embedding_mlp(
    x: &BurnTensor<2>,
    fc1_weight: &[f32],
    fc1_bias: &[f32],
    fc2_weight: &[f32],
    fc2_bias: &[f32],
    time_dim_in: usize,
    time_dim_out: usize,
) -> Result<BurnTensor<2>, BurnBackendError> {
    let x_nd = match x {
        BurnTensor::Ndarray(t) => t.clone(),
    };
    let dims = x_nd.dims();
    let batch = dims[0];

    // fc1: [batch, time_dim_in] @ [time_dim_in, time_dim_out] (with transpose)
    let w1 = Tensor::<NdArray, 2>::from_data(
        TensorData::new(fc1_weight.to_vec(), Shape::new([time_dim_out, time_dim_in])),
        &NdArrayDevice::Cpu,
    );
    let b1 = Tensor::<NdArray, 1>::from_data(
        TensorData::new(fc1_bias.to_vec(), Shape::new([time_dim_out])),
        &NdArrayDevice::Cpu,
    );

    let h = x_nd.matmul(w1.transpose()) + b1.reshape([1, time_dim_out]);
    let h = activation::silu(h);

    // fc2: [batch, time_dim_out] @ [time_dim_out, time_dim_out]
    let w2 = Tensor::<NdArray, 2>::from_data(
        TensorData::new(
            fc2_weight.to_vec(),
            Shape::new([time_dim_out, time_dim_out]),
        ),
        &NdArrayDevice::Cpu,
    );
    let b2 = Tensor::<NdArray, 1>::from_data(
        TensorData::new(fc2_bias.to_vec(), Shape::new([time_dim_out])),
        &NdArrayDevice::Cpu,
    );

    let out = h.matmul(w2.transpose()) + b2.reshape([1, time_dim_out]);

    Ok(BurnTensor::Ndarray(out.reshape([batch, time_dim_out])))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinusoidal_time_embedding_dim_4_hand_computed() {
        // dim=4 -> half_dim=2
        // freq[0] = 1 / (10000^(0/4)) = 1.0
        // freq[1] = 1 / (10000^(2/4)) = 1 / 100 = 0.01
        let timesteps = BurnTensor::Ndarray(Tensor::<NdArray, 1>::from_data(
            TensorData::new(vec![0.0f32, 1.0f32], Shape::new([2])),
            &NdArrayDevice::Cpu,
        ));

        let result = sinusoidal_time_embedding(&timesteps, 4, 10000).unwrap();
        let data = match result {
            BurnTensor::Ndarray(t) => t.to_data(),
        };
        assert_eq!(data.shape.dims(), [2, 4]);

        let slice = data.as_slice::<f32>().unwrap();

        // Row 0: timestep 0 -> sin(0)=0, cos(0)=1
        assert!((slice[0] - 0.0).abs() < 1e-5, "row0[0]: got {}", slice[0]);
        assert!((slice[1] - 1.0).abs() < 1e-5, "row0[1]: got {}", slice[1]);
        assert!((slice[2] - 0.0).abs() < 1e-5, "row0[2]: got {}", slice[2]);
        assert!((slice[3] - 1.0).abs() < 1e-5, "row0[3]: got {}", slice[3]);

        // Row 1: timestep 1
        assert!(
            (slice[4] - 1.0f32.sin()).abs() < 1e-5,
            "row1[0]: got {}",
            slice[4]
        );
        assert!(
            (slice[5] - 1.0f32.cos()).abs() < 1e-5,
            "row1[1]: got {}",
            slice[5]
        );
        assert!(
            (slice[6] - 0.01f32.sin()).abs() < 1e-5,
            "row1[2]: got {}",
            slice[6]
        );
        assert!(
            (slice[7] - 0.01f32.cos()).abs() < 1e-5,
            "row1[3]: got {}",
            slice[7]
        );
    }

    #[test]
    fn time_embedding_mlp_shapes_match() {
        let batch = 2usize;
        let time_dim_in = 256usize;
        let time_dim_out = 1280usize;

        let x = BurnTensor::Ndarray(Tensor::<NdArray, 2>::from_data(
            TensorData::new(
                vec![0.5f32; batch * time_dim_in],
                Shape::new([batch, time_dim_in]),
            ),
            &NdArrayDevice::Cpu,
        ));

        let w1 = vec![0.001f32; time_dim_out * time_dim_in];
        let b1 = vec![0.0f32; time_dim_out];
        let w2 = vec![0.001f32; time_dim_out * time_dim_out];
        let b2 = vec![0.0f32; time_dim_out];

        let result = time_embedding_mlp(&x, &w1, &b1, &w2, &b2, time_dim_in, time_dim_out).unwrap();

        let data = match result {
            BurnTensor::Ndarray(t) => t.to_data(),
        };
        assert_eq!(data.shape.dims(), [batch, time_dim_out]);
        let slice = data.as_slice::<f32>().unwrap();
        assert!(slice.iter().all(|v| v.is_finite()));
    }
}
