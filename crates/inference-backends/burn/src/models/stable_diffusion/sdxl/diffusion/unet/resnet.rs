//! SDXL UNet building blocks — ResBlock, Conv2d, GroupNorm, SiLU.

use burn_ndarray::NdArray;
use burn_ndarray::NdArrayDevice;
use burn_tensor::Shape;
use burn_tensor::{
    Tensor, TensorData, activation, module::conv2d as burn_conv2d, ops::ConvOptions,
};

use crate::error::BurnBackendError;
use crate::tensor::BurnTensor;

// ---------------------------------------------------------------------------
// SiLU
// ---------------------------------------------------------------------------

/// SiLU activation: `silu(x) = x * sigmoid(x)`.
pub fn silu(x: &BurnTensor<4>) -> Result<BurnTensor<4>, BurnBackendError> {
    Ok(BurnTensor::Ndarray(activation::silu(match x {
        BurnTensor::Ndarray(t) => t.clone(),
    })))
}

// ---------------------------------------------------------------------------
// Conv2d
// ---------------------------------------------------------------------------

/// 2D convolution over NCHW tensors.
///
/// * `input` — `[batch, in_ch, h, w]`
/// * `weight` — raw f32 slice shaped `[out_ch, in_ch, k_h, k_w]`
/// * `bias` — raw f32 slice shaped `[out_ch]`, or `None`
pub fn conv2d(
    input: &BurnTensor<4>,
    weight: &[f32],
    bias: Option<&[f32]>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
) -> Result<BurnTensor<4>, BurnBackendError> {
    let input_nd = match input {
        BurnTensor::Ndarray(t) => t.clone(),
    };

    // Build weight tensor: [out_ch, in_ch, k_h, k_w]
    let w = Tensor::<NdArray, 4>::from_data(
        TensorData::new(
            weight.to_vec(),
            Shape::new([out_channels, in_channels, kernel_size, kernel_size]),
        ),
        &NdArrayDevice::Cpu,
    );

    // Build bias tensor if present
    let b_opt = bias.map(|b| {
        Tensor::<NdArray, 1>::from_data(
            TensorData::new(b.to_vec(), Shape::new([out_channels])),
            &NdArrayDevice::Cpu,
        )
    });

    let options = ConvOptions::new([stride, stride], [padding, padding], [1, 1], 1);

    let out = burn_conv2d(input_nd, w, b_opt, options);
    Ok(BurnTensor::Ndarray(out))
}

// ---------------------------------------------------------------------------
// GroupNorm
// ---------------------------------------------------------------------------

/// Group normalisation for NCHW tensors.
///
/// * `num_groups` — number of groups (32 for SDXL)
/// * `num_channels` — total number of channels (must be divisible by `num_groups`)
pub fn group_norm(
    input: &BurnTensor<4>,
    weight: &[f32],
    bias: &[f32],
    num_groups: usize,
    num_channels: usize,
    eps: f32,
) -> Result<BurnTensor<4>, BurnBackendError> {
    let input_nd = match input {
        BurnTensor::Ndarray(t) => t.clone(),
    };

    let dims = input_nd.dims();
    let batch = dims[0];
    let h = dims[2];
    let w = dims[3];

    let c_per_group = num_channels / num_groups;

    // Reshape to [batch, groups, c_per_group, h, w]
    let x = input_nd.reshape([batch, num_groups, c_per_group, h, w]);

    // Compute mean over (c_per_group, h, w) — dims 2, 3, 4
    let mean = x.clone().sum_dims(&[2, 3, 4]);
    let per_group_elements = c_per_group * h * w;
    let divisor = per_group_elements as f32;
    let mean = mean / divisor;

    let centered = x.clone() - mean.reshape([batch, num_groups, 1, 1, 1]);
    let var = (centered.clone() * centered.clone()).sum_dims(&[2, 3, 4]) / divisor;
    let std = (var + eps).reshape([batch, num_groups, 1, 1, 1]).sqrt();

    let normalized = centered / std;

    // Load per-channel weight / bias
    let w_t = Tensor::<NdArray, 1>::from_data(
        TensorData::new(weight.to_vec(), Shape::new([num_channels])),
        &NdArrayDevice::Cpu,
    )
    .reshape([1, num_channels, 1, 1]);

    let b_t = Tensor::<NdArray, 1>::from_data(
        TensorData::new(bias.to_vec(), Shape::new([num_channels])),
        &NdArrayDevice::Cpu,
    )
    .reshape([1, num_channels, 1, 1]);

    let result = (normalized.reshape([batch, num_channels, h, w]) * w_t) + b_t;
    Ok(BurnTensor::Ndarray(result))
}

// ---------------------------------------------------------------------------
// ResBlock
// ---------------------------------------------------------------------------

/// A single SDXL ResBlock with time-conditioning:
///
/// ```text
/// x ──GroupNorm──SiLU──Conv2(3x3)──+──SiLU──GroupNorm──SiLU──Conv2(3x3)──+──> out
///             ^                    |                                    ^
///         time_emb               (broadcast)                           skip
/// ```
///
/// * If `in_channels != out_channels`, the skip connection uses a 1x1 Conv2d
///   with `skip_weight` / `skip_bias`.  If those are `None`, the skip is an
///   identity (caller is responsible for providing them when channels differ).
/// * `time_emb` is expected to already be SiLU-activated and linearly projected
///   to `[batch, out_channels]` by `time_embedding_mlp`.
pub fn res_block(
    x: &BurnTensor<4>,
    time_emb: &BurnTensor<2>,
    in_channels: usize,
    out_channels: usize,
    num_groups: usize,
    in_norm_weight: &[f32],
    in_norm_bias: &[f32],
    in_conv_weight: &[f32],
    in_conv_bias: &[f32],
    emb_linear_1_weight: &[f32],
    emb_linear_1_bias: &[f32],
    emb_linear_2_weight: &[f32],
    emb_linear_2_bias: &[f32],
    out_norm_weight: &[f32],
    out_norm_bias: &[f32],
    out_conv_weight: &[f32],
    out_conv_bias: &[f32],
    skip_weight: Option<&[f32]>,
    skip_bias: Option<&[f32]>,
) -> Result<BurnTensor<4>, BurnBackendError> {
    let x_nd = match x {
        BurnTensor::Ndarray(t) => t.clone(),
    };
    let batch = x_nd.dims()[0];

    // --- GroupNorm + SiLU + Conv2d (in) ---
    let h = group_norm(
        x,
        in_norm_weight,
        in_norm_bias,
        num_groups,
        in_channels,
        1e-5,
    )?;
    let h = silu(&h)?;
    let h = conv2d(
        &h,
        in_conv_weight,
        Some(in_conv_bias),
        in_channels,
        out_channels,
        3,
        1,
        1,
    )?;

    // --- Time embedding branch ---
    let te_nd = match time_emb {
        BurnTensor::Ndarray(t) => t.clone(),
    };

    // Linear 1: [batch, time_dim] @ [out_channels, time_dim]^T -> [batch, out_channels]
    let te_emb_1 = linear_2d(
        &BurnTensor::Ndarray(te_nd),
        emb_linear_1_weight,
        Some(emb_linear_1_bias),
        time_emb.dims()[1],
        out_channels,
    )?;
    let te_emb_1_nd = match &te_emb_1 {
        BurnTensor::Ndarray(t) => t.clone(),
    };
    let te_act = BurnTensor::Ndarray(activation::silu(te_emb_1_nd));

    // Linear 2: [batch, out_channels] @ [out_channels, out_channels]^T -> [batch, out_channels]
    let emb_proj = linear_2d(
        &te_act,
        emb_linear_2_weight,
        Some(emb_linear_2_bias),
        out_channels,
        out_channels,
    )?;

    // Add time embedding: h [batch, out_ch, h, w] + emb_proj[:, :, None, None]
    let emb_nd = match emb_proj {
        BurnTensor::Ndarray(t) => t,
    };
    let emb_bc = emb_nd.reshape([batch, out_channels, 1, 1]);
    let h_cur = match &h {
        BurnTensor::Ndarray(t) => t.clone(),
    };
    let h = BurnTensor::Ndarray(h_cur + emb_bc);

    // --- GroupNorm + SiLU + Conv2d (out) ---
    let h = group_norm(
        &h,
        out_norm_weight,
        out_norm_bias,
        num_groups,
        out_channels,
        1e-5,
    )?;
    let h = silu(&h)?;
    let h = conv2d(
        &h,
        out_conv_weight,
        Some(out_conv_bias),
        out_channels,
        out_channels,
        3,
        1,
        1,
    )?;

    // --- Skip connection ---
    let skip = if in_channels == out_channels {
        x.clone()
    } else {
        conv2d(
            x,
            skip_weight.ok_or_else(|| {
                BurnBackendError::InvalidRequest(
                    "skip_weight required when in_channels != out_channels".into(),
                )
            })?,
            skip_bias,
            in_channels,
            out_channels,
            1,
            0,
            1,
        )?
    };

    // out_conv + skip
    let h_nd = match &h {
        BurnTensor::Ndarray(t) => t.clone(),
    };
    let skip_nd = match &skip {
        BurnTensor::Ndarray(t) => t.clone(),
    };
    Ok(BurnTensor::Ndarray(h_nd + skip_nd))
}

// ---------------------------------------------------------------------------
// Linear helper for 2-D tensors [batch, in] -> [batch, out]
// ---------------------------------------------------------------------------

fn linear_2d(
    input: &BurnTensor<2>,
    weight: &[f32],
    bias: Option<&[f32]>,
    in_features: usize,
    out_features: usize,
) -> Result<BurnTensor<2>, BurnBackendError> {
    let input_nd = match input {
        BurnTensor::Ndarray(t) => t.clone(),
    };
    let dims = input_nd.dims();
    let batch = dims[0];

    let w = Tensor::<NdArray, 2>::from_data(
        TensorData::new(weight.to_vec(), Shape::new([out_features, in_features])),
        &NdArrayDevice::Cpu,
    );

    let input_2d = input_nd.reshape([batch, in_features]);
    let y = input_2d.matmul(w.transpose());

    if let Some(bias) = bias {
        let b = Tensor::<NdArray, 1>::from_data(
            TensorData::new(bias.to_vec(), Shape::new([out_features])),
            &NdArrayDevice::Cpu,
        );
        Ok(BurnTensor::Ndarray(y + b.reshape([1, out_features])))
    } else {
        Ok(BurnTensor::Ndarray(y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::stable_diffusion::sdxl::diffusion::unet::time_embedding::sinusoidal_time_embedding;
    use crate::models::stable_diffusion::sdxl::diffusion::unet::time_embedding::time_embedding_mlp;

    // ---------- silu tests ----------

    #[test]
    fn silu_zero_is_zero() {
        let x = BurnTensor::Ndarray(Tensor::<NdArray, 4>::from_data(
            TensorData::new(vec![0.0f32; 4], Shape::new([1, 1, 2, 2])),
            &NdArrayDevice::Cpu,
        ));
        let result = silu(&x).unwrap();
        let data = match result {
            BurnTensor::Ndarray(t) => t.to_data(),
        };
        let slice = data.as_slice::<f32>().unwrap();
        for v in slice {
            assert!((*v - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn res_block_zero_weights_constant_output() {
        let in_ch = 4usize;
        let out_ch = 4usize;
        let h = 4usize;
        let w = 4usize;
        let batch = 1usize;

        let x = BurnTensor::Ndarray(Tensor::<NdArray, 4>::from_data(
            TensorData::new(
                vec![1.0f32; batch * in_ch * h * w],
                Shape::new([batch, in_ch, h, w]),
            ),
            &NdArrayDevice::Cpu,
        ));
        let time_emb = BurnTensor::Ndarray(Tensor::<NdArray, 2>::from_data(
            TensorData::new(vec![0.5f32; 256], Shape::new([1, 256])),
            &NdArrayDevice::Cpu,
        ));

        // Zero weight slices with correct sizes for (in_ch=4, out_ch=4, groups=2)
        let in_norm_w = vec![0.0f32; in_ch];
        let in_norm_b = vec![0.0f32; in_ch];
        let in_conv_w = vec![0.0f32; out_ch * in_ch * 3 * 3];
        let in_conv_b = vec![0.0f32; out_ch];
        let emb_lin1_w = vec![0.0f32; out_ch * 256];
        let emb_lin1_b = vec![0.0f32; out_ch];
        let emb_lin2_w = vec![0.0f32; out_ch * out_ch];
        let emb_lin2_b = vec![0.0f32; out_ch];
        let out_norm_w = vec![0.0f32; out_ch];
        let out_norm_b = vec![0.0f32; out_ch];
        let out_conv_w = vec![0.0f32; out_ch * out_ch * 3 * 3];
        let out_conv_b = vec![0.0f32; out_ch];

        let out = res_block(
            &x,
            &time_emb,
            in_ch,
            out_ch,
            2,
            &in_norm_w,
            &in_norm_b,
            &in_conv_w,
            &in_conv_b,
            &emb_lin1_w,
            &emb_lin1_b,
            &emb_lin2_w,
            &emb_lin2_b,
            &out_norm_w,
            &out_norm_b,
            &out_conv_w,
            &out_conv_b,
            None,
            None,
        )
        .unwrap();

        let data = match &out {
            BurnTensor::Ndarray(t) => t.to_data(),
        };
        assert_eq!(data.shape.dims(), [batch, out_ch, h, w]);
        let slice = data.as_slice::<f32>().unwrap();
        assert!(slice.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn res_block_skip_conv_shapes() {
        let in_ch = 4usize;
        let out_ch = 8usize;
        let h = 2usize;
        let w = 2usize;
        let batch = 1usize;

        let x = BurnTensor::Ndarray(Tensor::<NdArray, 4>::from_data(
            TensorData::new(
                vec![1.0f32; batch * in_ch * h * w],
                Shape::new([batch, in_ch, h, w]),
            ),
            &NdArrayDevice::Cpu,
        ));
        let time_emb = BurnTensor::Ndarray(Tensor::<NdArray, 2>::from_data(
            TensorData::new(vec![0.0f32; 256], Shape::new([1, 256])),
            &NdArrayDevice::Cpu,
        ));

        // Zero weight slices with correct sizes for (in_ch=4, out_ch=8, groups=2)
        let in_norm_w = vec![0.0f32; in_ch];
        let in_norm_b = vec![0.0f32; in_ch];
        let in_conv_w = vec![0.0f32; out_ch * in_ch * 3 * 3];
        let in_conv_b = vec![0.0f32; out_ch];
        let emb_lin1_w = vec![0.0f32; out_ch * 256];
        let emb_lin1_b = vec![0.0f32; out_ch];
        let emb_lin2_w = vec![0.0f32; out_ch * out_ch];
        let emb_lin2_b = vec![0.0f32; out_ch];
        let out_norm_w = vec![0.0f32; out_ch];
        let out_norm_b = vec![0.0f32; out_ch];
        let out_conv_w = vec![0.0f32; out_ch * out_ch * 3 * 3];
        let out_conv_b = vec![0.0f32; out_ch];
        let skip_w = vec![0.0f32; out_ch * in_ch * 1 * 1];
        let skip_b = vec![0.0f32; out_ch];

        let out = res_block(
            &x,
            &time_emb,
            in_ch,
            out_ch,
            2,
            &in_norm_w,
            &in_norm_b,
            &in_conv_w,
            &in_conv_b,
            &emb_lin1_w,
            &emb_lin1_b,
            &emb_lin2_w,
            &emb_lin2_b,
            &out_norm_w,
            &out_norm_b,
            &out_conv_w,
            &out_conv_b,
            Some(&skip_w),
            Some(&skip_b),
        )
        .unwrap();

        let data = match &out {
            BurnTensor::Ndarray(t) => t.to_data(),
        };
        assert_eq!(data.shape.dims(), [batch, out_ch, h, w]);
        let slice = data.as_slice::<f32>().unwrap();
        assert!(slice.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn silu_positive() {
        let x = BurnTensor::Ndarray(Tensor::<NdArray, 4>::from_data(
            TensorData::new(vec![1.0f32], Shape::new([1, 1, 1, 1])),
            &NdArrayDevice::Cpu,
        ));
        let result = silu(&x).unwrap();
        let data = match result {
            BurnTensor::Ndarray(t) => t.to_data(),
        };
        let slice = data.as_slice::<f32>().unwrap();
        let v = slice[0];
        let expected = 1.0f32 * (1.0f32 / (1.0f32 + (-1.0f32).exp()));
        assert!((v - expected).abs() < 1e-5, "expected {expected}, got {v}");
    }

    #[test]
    fn conv2d_single_pixel_identity_weights() {
        // 1x1 kernel with weight=1, no bias should pass the value through
        let input = BurnTensor::Ndarray(Tensor::<NdArray, 4>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], Shape::new([1, 1, 2, 2])),
            &NdArrayDevice::Cpu,
        ));
        let result = conv2d(&input, &[1.0f32], None, 1, 1, 1, 0, 1).unwrap();
        let data = match result {
            BurnTensor::Ndarray(t) => t.to_data(),
        };
        assert_eq!(data.shape.dims(), [1, 1, 2, 2]);
        let slice = data.as_slice::<f32>().unwrap();
        let expected = [1.0, 2.0, 3.0, 4.0];
        for (i, &v) in slice.iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-5,
                "at {i}: got {v}, expected {}",
                expected[i]
            );
        }
    }

    #[test]
    fn group_norm_basic() {
        let batch = 1usize;
        let channels = 4usize;
        let h = 2usize;
        let w = 2usize;
        // All-ones input -> norm=0 -> scaled = bias
        let input = BurnTensor::Ndarray(Tensor::<NdArray, 4>::from_data(
            TensorData::new(
                vec![1.0f32; batch * channels * h * w],
                Shape::new([batch, channels, h, w]),
            ),
            &NdArrayDevice::Cpu,
        ));
        let weight = vec![1.0f32; channels];
        let bias = vec![0.0f32; channels];
        let result = group_norm(&input, &weight, &bias, 2, channels, 1e-5).unwrap();
        let data = match result {
            BurnTensor::Ndarray(t) => t.to_data(),
        };
        // All-ones should give ~0.0 after centering
        let slice = data.as_slice::<f32>().unwrap();
        for &v in slice {
            assert!(v.abs() < 1e-4, "expected ~0.0 for all-ones input, got {v}");
        }
    }

    #[test]
    fn sinusoidal_time_embedding_dim_4() {
        // dim=4 -> half_dim=2
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
