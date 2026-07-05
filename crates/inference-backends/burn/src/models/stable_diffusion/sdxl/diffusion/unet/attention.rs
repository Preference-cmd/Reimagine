//! SDXL UNet self-attention and cross-attention blocks.
//!
//! Implements the two attention variants used in the SDXL UNet:
//!
//! 1. **Self-attention block** — Q=K=V from the same spatial feature map
//!    `[batch, channels, h, w]`. Used in the middle block and inner
//!    downsample/upsample stages.
//! 2. **Cross-attention block** — Q from hidden state, K=V from text embedding
//!    `[batch, 77, context_dim]`. Injects text conditioning into the UNet.
//!
//! Both blocks reshape `[batch, channels, h, w]` → `[batch, seq, channels]`
//! (`seq = h * w`) for attention, then reshape back.

use burn_ndarray::{NdArray, NdArrayDevice};
use burn_tensor::{Shape, Tensor, TensorData, activation};

use crate::error::BurnBackendError;
use crate::models::stable_diffusion::sdxl::text_conditioning::module::ClipWeightData;
use crate::tensor::BurnTensor;

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------

/// Weights for a self-attention block (Q, K, V all from the same input).
pub struct SelfAttnBlockWeights {
    pub norm_weight: ClipWeightData,
    pub norm_bias: ClipWeightData,
    pub q_weight: ClipWeightData, // [channels, channels]
    pub q_bias: ClipWeightData,
    pub k_weight: ClipWeightData, // [channels, channels]
    pub k_bias: ClipWeightData,
    pub v_weight: ClipWeightData, // [channels, channels]
    pub v_bias: ClipWeightData,
    pub out_weight: ClipWeightData, // [channels, channels]
    pub out_bias: ClipWeightData,
}

/// Weights for a cross-attention block (Q from hidden, K/V from text).
pub struct CrossAttnBlockWeights {
    pub norm_weight: ClipWeightData,
    pub norm_bias: ClipWeightData,
    pub q_weight: ClipWeightData, // [channels, channels]
    pub q_bias: ClipWeightData,
    pub k_weight: ClipWeightData, // [channels, context_dim]
    pub k_bias: ClipWeightData,
    pub v_weight: ClipWeightData, // [channels, context_dim]
    pub v_bias: ClipWeightData,
    pub out_weight: ClipWeightData, // [channels, channels]
    pub out_bias: ClipWeightData,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build a 2-D weight tensor: `[out, in]`.
fn weight2d(w: &ClipWeightData, out: usize, inn: usize) -> Tensor<NdArray, 2> {
    Tensor::from_data(
        TensorData::new(w.data.clone(), Shape::new([out, inn])),
        &NdArrayDevice::Cpu,
    )
}

/// Build a 1-D bias tensor: `[len]`.
fn bias1d(b: &ClipWeightData, len: usize) -> Tensor<NdArray, 1> {
    Tensor::from_data(
        TensorData::new(b.data.clone(), Shape::new([len])),
        &NdArrayDevice::Cpu,
    )
}

/// Reshape `[batch, channels, h, w]` → `[batch, seq, channels]` where `seq = h*w`.
fn to_seq(t: Tensor<NdArray, 4>) -> Tensor<NdArray, 3> {
    let d = t.dims();
    t.reshape([d[0], d[2] * d[3], d[1]])
}

/// Reshape `[batch, seq, channels]` → `[batch, channels, h, w]`.
fn to_spatial(t: Tensor<NdArray, 3>, h: usize, w: usize) -> Tensor<NdArray, 4> {
    let d = t.dims();
    t.reshape([d[0], d[2], h, w])
}

// ---------------------------------------------------------------------------
// Multi-head attention on [batch, seq, dim]
// ---------------------------------------------------------------------------

fn mha_3d(
    q: Tensor<NdArray, 3>,
    k: Tensor<NdArray, 3>,
    v: Tensor<NdArray, 3>,
    num_heads: usize,
) -> Result<Tensor<NdArray, 3>, BurnBackendError> {
    let d = q.dims();
    let [batch, q_seq, dim] = d;

    if dim % num_heads != 0 {
        return Err(BurnBackendError::InvalidRequest(format!(
            "mha: dim ({dim}) must be divisible by num_heads ({num_heads})"
        )));
    }

    let head_dim = dim / num_heads;

    let k_seq = k.dims()[1];
    let v_seq = v.dims()[1];

    // [batch, heads, seq, head_dim]
    let q = q
        .reshape([batch, q_seq, num_heads, head_dim])
        .swap_dims(1, 2);
    let k = k
        .reshape([batch, k_seq, num_heads, head_dim])
        .swap_dims(1, 2);
    let v = v
        .reshape([batch, v_seq, num_heads, head_dim])
        .swap_dims(1, 2);

    let scale = (head_dim as f32).sqrt();
    let scores = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);
    let attn_w = activation::softmax(scores, 3);
    let out = attn_w.matmul(v);
    // [batch, heads, q_seq, head_dim] -> [batch, q_seq, heads, head_dim] -> [batch, q_seq, dim]
    let out = out.swap_dims(1, 2).reshape([batch, q_seq, dim]);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Linear: y = x @ W^T + b  (3D input: [batch, seq, in])
// ---------------------------------------------------------------------------

fn linear3d(
    x: Tensor<NdArray, 3>,
    w: Tensor<NdArray, 2>,
    b: Option<Tensor<NdArray, 1>>,
) -> Tensor<NdArray, 3> {
    let [batch, seq, _] = x.dims();
    let w_in = w.dims()[1];
    let w_out = w.dims()[0];
    let out = x.reshape([batch * seq, w_in]).matmul(w.clone().transpose());
    let reshaped = match b {
        Some(bias) => out + bias.reshape([1, w_out]),
        None => out,
    };
    reshaped.reshape([batch, seq, w_out])
}

// ---------------------------------------------------------------------------
// Self-attention block
// ---------------------------------------------------------------------------

/// Self-attention block on `[batch, channels, h, w]`.
///
/// Q, K, V all come from the same spatial feature map via GroupNorm + linear.
///
/// `num_groups` is passed through to `group_norm` (32 for SDXL).
pub fn self_attention_block(
    x: &BurnTensor<4>,
    weights: &SelfAttnBlockWeights,
    num_heads: usize,
    num_groups: usize,
) -> Result<BurnTensor<4>, BurnBackendError> {
    let x_nd = match x {
        BurnTensor::Ndarray(t) => t.clone(),
    };
    let [_batch, channels, h, w] = x_nd.dims();

    // GroupNorm over channels
    let normed = super::resnet::group_norm(
        x,
        &weights.norm_weight.data,
        &weights.norm_bias.data,
        num_groups,
        channels,
        1e-5,
    )?;
    let normed_nd = match &normed {
        BurnTensor::Ndarray(t) => t.clone(),
    };

    // [batch, seq, channels] where seq = h*w
    let seq = to_seq(normed_nd);

    // Q, K, V linear projections
    let q = linear3d(
        seq.clone(),
        weight2d(&weights.q_weight, channels, channels),
        Some(bias1d(&weights.q_bias, channels)),
    );
    let k = linear3d(
        seq.clone(),
        weight2d(&weights.k_weight, channels, channels),
        Some(bias1d(&weights.k_bias, channels)),
    );
    let v = linear3d(
        seq,
        weight2d(&weights.v_weight, channels, channels),
        Some(bias1d(&weights.v_bias, channels)),
    );

    // Multi-head attention
    let attn_out = mha_3d(q, k, v, num_heads)?;

    // Output projection
    let out_proj = linear3d(
        attn_out,
        weight2d(&weights.out_weight, channels, channels),
        Some(bias1d(&weights.out_bias, channels)),
    );

    // [batch, channels, h, w]
    Ok(BurnTensor::Ndarray(to_spatial(out_proj, h, w)))
}

// ---------------------------------------------------------------------------
// Cross-attention block
// ---------------------------------------------------------------------------

/// Cross-attention block: Q from hidden state, K/V from text embedding.
///
/// * `x` — hidden state `[batch, channels, h, w]`
/// * `text_emb` — CLIP text embeddings `[batch, 77, context_dim]`
///
/// `num_groups` is passed through to `group_norm` (32 for SDXL).
pub fn cross_attention_block(
    x: &BurnTensor<4>,
    text_emb: &BurnTensor<3>,
    weights: &CrossAttnBlockWeights,
    num_heads: usize,
    num_groups: usize,
) -> Result<BurnTensor<4>, BurnBackendError> {
    let x_nd = match x {
        BurnTensor::Ndarray(t) => t.clone(),
    };
    let [_batch, channels, h, w] = x_nd.dims();

    let text_nd = match text_emb {
        BurnTensor::Ndarray(t) => t.clone(),
    };
    let context_dim = text_nd.dims()[2];

    // GroupNorm over channels of hidden state
    let normed = super::resnet::group_norm(
        x,
        &weights.norm_weight.data,
        &weights.norm_bias.data,
        num_groups,
        channels,
        1e-5,
    )?;
    let normed_nd = match &normed {
        BurnTensor::Ndarray(t) => t.clone(),
    };

    // [batch, seq, channels] where seq = h*w
    let seq = to_seq(normed_nd);

    // Q from hidden state
    let q = linear3d(
        seq.clone(),
        weight2d(&weights.q_weight, channels, channels),
        Some(bias1d(&weights.q_bias, channels)),
    );

    // K, V from text embedding: [batch, 77, context_dim]
    let k = linear3d(
        text_nd.clone(),
        weight2d(&weights.k_weight, channels, context_dim),
        Some(bias1d(&weights.k_bias, channels)),
    );
    let v = linear3d(
        text_nd,
        weight2d(&weights.v_weight, channels, context_dim),
        Some(bias1d(&weights.v_bias, channels)),
    );

    // Multi-head attention
    let attn_out = mha_3d(q, k, v, num_heads)?;

    // Output projection
    let out_proj = linear3d(
        attn_out,
        weight2d(&weights.out_weight, channels, channels),
        Some(bias1d(&weights.out_bias, channels)),
    );

    // [batch, channels, h, w]
    Ok(BurnTensor::Ndarray(to_spatial(out_proj, h, w)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;

    fn zeros_self_attn(channels: usize) -> SelfAttnBlockWeights {
        SelfAttnBlockWeights {
            norm_weight: ClipWeightData {
                data: vec![0.0f32; channels],
            },
            norm_bias: ClipWeightData {
                data: vec![0.0f32; channels],
            },
            q_weight: ClipWeightData {
                data: vec![0.0f32; channels * channels],
            },
            q_bias: ClipWeightData {
                data: vec![0.0f32; channels],
            },
            k_weight: ClipWeightData {
                data: vec![0.0f32; channels * channels],
            },
            k_bias: ClipWeightData {
                data: vec![0.0f32; channels],
            },
            v_weight: ClipWeightData {
                data: vec![0.0f32; channels * channels],
            },
            v_bias: ClipWeightData {
                data: vec![0.0f32; channels],
            },
            out_weight: ClipWeightData {
                data: vec![0.0f32; channels * channels],
            },
            out_bias: ClipWeightData {
                data: vec![0.0f32; channels],
            },
        }
    }

    fn zeros_cross_attn(channels: usize, context_dim: usize) -> CrossAttnBlockWeights {
        CrossAttnBlockWeights {
            norm_weight: ClipWeightData {
                data: vec![0.0f32; channels],
            },
            norm_bias: ClipWeightData {
                data: vec![0.0f32; channels],
            },
            q_weight: ClipWeightData {
                data: vec![0.0f32; channels * channels],
            },
            q_bias: ClipWeightData {
                data: vec![0.0f32; channels],
            },
            k_weight: ClipWeightData {
                data: vec![0.0f32; channels * context_dim],
            },
            k_bias: ClipWeightData {
                data: vec![0.0f32; channels],
            },
            v_weight: ClipWeightData {
                data: vec![0.0f32; channels * context_dim],
            },
            v_bias: ClipWeightData {
                data: vec![0.0f32; channels],
            },
            out_weight: ClipWeightData {
                data: vec![0.0f32; channels * channels],
            },
            out_bias: ClipWeightData {
                data: vec![0.0f32; channels],
            },
        }
    }

    fn make_x(batch: usize, channels: usize, h: usize, w: usize) -> BurnTensor<4> {
        BurnTensor::Ndarray(Tensor::<NdArray, 4>::from_data(
            TensorData::new(
                vec![0.5f32; batch * channels * h * w],
                Shape::new([batch, channels, h, w]),
            ),
            &NdArrayDevice::Cpu,
        ))
    }

    #[test]
    fn self_attention_block_shape() {
        let channels = 4usize;
        let x = make_x(1, channels, 2, 2);
        let result = self_attention_block(&x, &zeros_self_attn(channels), 2, 1).unwrap();
        let data = match &result {
            BurnTensor::Ndarray(t) => t.to_data(),
        };
        assert_eq!(data.shape.dims(), [1, channels, 2, 2]);
    }

    #[test]
    fn self_attention_block_zero_weights() {
        let channels = 4usize;
        let x = make_x(1, channels, 2, 2);
        let result = self_attention_block(&x, &zeros_self_attn(channels), 2, 1).unwrap();
        let data = match &result {
            BurnTensor::Ndarray(t) => t.to_data(),
        };
        // All-zero weights: GroupNorm(num_groups=1) with zero w/b → zero output
        let slice = data.as_slice::<f32>().unwrap();
        assert!(slice.iter().all(|v| v.abs() < 1e-5));
    }

    #[test]
    fn cross_attention_block_shape() {
        let channels = 4usize;
        let x = make_x(1, channels, 2, 2);
        let text_emb = BurnTensor::Ndarray(Tensor::<NdArray, 3>::from_data(
            TensorData::new(vec![1.0f32; 1 * 77 * 128], Shape::new([1, 77, 128])),
            &NdArrayDevice::Cpu,
        ));
        let result =
            cross_attention_block(&x, &text_emb, &zeros_cross_attn(channels, 128), 2, 1).unwrap();
        let data = match &result {
            BurnTensor::Ndarray(t) => t.to_data(),
        };
        assert_eq!(data.shape.dims(), [1, channels, 2, 2]);
    }

    #[test]
    fn cross_attention_block_zero_weights() {
        let channels = 4usize;
        let x = make_x(1, channels, 2, 2);
        let text_emb = BurnTensor::Ndarray(Tensor::<NdArray, 3>::from_data(
            TensorData::new(vec![1.0f32; 1 * 77 * 128], Shape::new([1, 77, 128])),
            &NdArrayDevice::Cpu,
        ));
        let result =
            cross_attention_block(&x, &text_emb, &zeros_cross_attn(channels, 128), 2, 1).unwrap();
        let data = match &result {
            BurnTensor::Ndarray(t) => t.to_data(),
        };
        let slice = data.as_slice::<f32>().unwrap();
        assert!(slice.iter().all(|v| v.abs() < 1e-5));
    }

    #[test]
    fn mha_forward_shapes_match() {
        // batch=1, seq=2, dim=4, heads=2, head_dim=2
        let q = Tensor::<NdArray, 3>::zeros([1, 2, 4], &NdArrayDevice::Cpu);
        let k = Tensor::<NdArray, 3>::zeros([1, 2, 4], &NdArrayDevice::Cpu);
        let v = Tensor::<NdArray, 3>::zeros([1, 2, 4], &NdArrayDevice::Cpu);
        let out = mha_3d(q, k, v, 2).unwrap();
        assert_eq!(out.dims(), [1, 2, 4]);
        assert!(
            out.to_data()
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn mha_bad_head_count() {
        let q = Tensor::<NdArray, 3>::zeros([1, 2, 4], &NdArrayDevice::Cpu);
        let k = q.clone();
        let v = q.clone();
        let err = mha_3d(q, k, v, 3).unwrap_err();
        match err {
            BurnBackendError::InvalidRequest(msg) => {
                assert!(msg.contains("divisible by num_heads"));
            }
            _ => panic!("expected InvalidRequest"),
        }
    }

    #[test]
    fn self_attn_diff_hw_dims() {
        // h=2, w=3 -> seq=6
        let channels = 8usize;
        let x = make_x(1, channels, 2, 3);
        let result = self_attention_block(&x, &zeros_self_attn(channels), 4, 2).unwrap();
        assert_eq!(result.dims(), [1, channels, 2, 3]);
    }

    #[test]
    fn cross_attn_different_text_seq() {
        let channels = 8usize;
        let x = make_x(1, channels, 2, 2);
        let text_emb = BurnTensor::Ndarray(Tensor::<NdArray, 3>::from_data(
            TensorData::new(vec![0.7f32; 1 * 50 * 128], Shape::new([1, 50, 128])),
            &NdArrayDevice::Cpu,
        ));
        let result =
            cross_attention_block(&x, &text_emb, &zeros_cross_attn(channels, 128), 2, 2).unwrap();
        assert_eq!(result.dims(), [1, channels, 2, 2]);
    }
}
