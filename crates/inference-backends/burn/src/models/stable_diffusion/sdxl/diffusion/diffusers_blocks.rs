//! Diffusers-shaped UNet attention / transformer blocks.
//!
//! Snapshot paths match HuggingFace `UNet2DConditionModel` package keys:
//! `attentions.N.transformer_blocks.M.attn{1,2}.to_{q,k,v,out.0}.*`.

use burn::module::Module;
use burn_core as burn;
use burn_nn::{
    Linear, LinearConfig,
    norm::{GroupNorm, GroupNormConfig, LayerNorm, LayerNormConfig},
};
use burn_tensor::{Tensor, activation, backend::Backend};

/// Package: `*.attn1` / `*.attn2` leaf projections.
#[derive(Module, Debug)]
pub struct SdxlDiffusersAttention<B: Backend> {
    pub to_q: Linear<B>,
    pub to_k: Linear<B>,
    pub to_v: Linear<B>,
    /// Length-1 so snapshots land on `to_out.0.{weight,bias}`.
    pub to_out: Vec<Linear<B>>,
    num_heads: usize,
}

impl<B: Backend> SdxlDiffusersAttention<B> {
    pub fn init(
        query_dim: usize,
        context_dim: usize,
        num_heads: usize,
        device: &B::Device,
    ) -> Self {
        assert!(
            query_dim.is_multiple_of(num_heads),
            "query_dim {query_dim} must be divisible by num_heads {num_heads}"
        );
        Self {
            to_q: LinearConfig::new(query_dim, query_dim)
                .with_bias(false)
                .init(device),
            to_k: LinearConfig::new(context_dim, query_dim)
                .with_bias(false)
                .init(device),
            to_v: LinearConfig::new(context_dim, query_dim)
                .with_bias(false)
                .init(device),
            to_out: vec![LinearConfig::new(query_dim, query_dim).init(device)],
            num_heads,
        }
    }

    /// `hidden` / `context`: [B, S, C]
    pub fn forward(&self, hidden: Tensor<B, 3>, context: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, channels] = hidden.dims();
        let head_dim = channels / self.num_heads;
        let q = self.to_q.forward(hidden);
        let k = self.to_k.forward(context.clone());
        let v = self.to_v.forward(context);

        let q = q
            .reshape([batch, seq, self.num_heads, head_dim])
            .swap_dims(1, 2);
        let k_seq = k.dims()[1];
        let k = k
            .reshape([batch, k_seq, self.num_heads, head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, k_seq, self.num_heads, head_dim])
            .swap_dims(1, 2);

        let scale = (head_dim as f64).sqrt().recip();
        let scores = Tensor::matmul(q, k.swap_dims(2, 3)) * scale;
        let weights = activation::softmax(scores, 3);
        let attn = Tensor::matmul(weights, v)
            .swap_dims(1, 2)
            .reshape([batch, seq, channels]);

        self.to_out
            .first()
            .expect("diffusers attention requires to_out.0")
            .forward(attn)
    }
}

/// Package: `ff` with nested `net.0.proj` + `net.2` via length-3 vec of wrappers.
#[derive(Module, Debug)]
pub struct SdxlDiffusersFeedForward<B: Backend> {
    pub net: Vec<SdxlDiffusersFfNetEntry<B>>,
}

/// Homogeneous slot so `net.{i}` indices match ModuleList.
#[derive(Module, Debug)]
pub struct SdxlDiffusersFfNetEntry<B: Backend> {
    /// Present on GEGLU slot (index 0).
    pub proj: Option<Linear<B>>,
    /// Present on final Linear slot (index 2) — stored as nested linear module
    /// fields by flattening: when only `linear` is set, keys are `net.2.weight`.
    /// Use a dedicated Linear module field named to match... Linear's own weight
    /// is at entry path + weight if we use Linear as the entry type only.
    ///
    /// For index 2 we need `net.2.weight`. That means net[2] itself is Linear.
    /// Heterogeneous lists force a remapper for FF; keep GEGLU keys perfect and
    /// map `ff.net.2` via a sibling Linear field `net2` with remapper, OR accept
    /// `net.2.linear.weight` temporarily.
    pub linear: Option<Linear<B>>,
}

impl<B: Backend> SdxlDiffusersFeedForward<B> {
    pub fn init(channels: usize, device: &B::Device) -> Self {
        let inner = channels * 4;
        Self {
            net: vec![
                SdxlDiffusersFfNetEntry {
                    proj: Some(LinearConfig::new(channels, inner * 2).init(device)),
                    linear: None,
                },
                // Dropout placeholder (no tensors).
                SdxlDiffusersFfNetEntry {
                    proj: None,
                    linear: None,
                },
                SdxlDiffusersFfNetEntry {
                    proj: None,
                    linear: Some(LinearConfig::new(inner, channels).init(device)),
                },
            ],
        }
    }

    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let proj = self.net[0]
            .proj
            .as_ref()
            .expect("ff.net.0.proj required");
        let projected = proj.forward(hidden);
        let [batch, seq, dual] = projected.dims();
        let half = dual / 2;
        let value = projected.clone().slice([0..batch, 0..seq, 0..half]);
        let gate = projected.slice([0..batch, 0..seq, half..dual]);
        let hidden = value * activation::gelu(gate);
        self.net[2]
            .linear
            .as_ref()
            .expect("ff.net.2 linear required")
            .forward(hidden)
    }
}

/// Package: `transformer_blocks.M.*`
#[derive(Module, Debug)]
pub struct SdxlBasicTransformerBlock<B: Backend> {
    pub norm1: LayerNorm<B>,
    pub attn1: SdxlDiffusersAttention<B>,
    pub norm2: LayerNorm<B>,
    pub attn2: SdxlDiffusersAttention<B>,
    pub norm3: LayerNorm<B>,
    pub ff: SdxlDiffusersFeedForward<B>,
}

impl<B: Backend> SdxlBasicTransformerBlock<B> {
    pub fn init(
        channels: usize,
        context_dim: usize,
        num_heads: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            norm1: LayerNormConfig::new(channels).init(device),
            attn1: SdxlDiffusersAttention::init(channels, channels, num_heads, device),
            norm2: LayerNormConfig::new(channels).init(device),
            attn2: SdxlDiffusersAttention::init(channels, context_dim, num_heads, device),
            norm3: LayerNormConfig::new(channels).init(device),
            ff: SdxlDiffusersFeedForward::init(channels, device),
        }
    }

    pub fn forward(&self, hidden: Tensor<B, 3>, context: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = hidden.clone();
        let hidden = residual.clone()
            + self
                .attn1
                .forward(self.norm1.forward(hidden), residual.clone());

        let residual = hidden.clone();
        let hidden = residual.clone()
            + self
                .attn2
                .forward(self.norm2.forward(hidden), context);

        let residual = hidden.clone();
        residual + self.ff.forward(self.norm3.forward(hidden))
    }
}

/// Package: `attentions.N.*` (spatial transformer).
#[derive(Module, Debug)]
pub struct SdxlSpatialTransformer<B: Backend> {
    pub norm: GroupNorm<B>,
    pub proj_in: Linear<B>,
    pub transformer_blocks: Vec<SdxlBasicTransformerBlock<B>>,
    pub proj_out: Linear<B>,
}

impl<B: Backend> SdxlSpatialTransformer<B> {
    pub fn init(
        channels: usize,
        context_dim: usize,
        num_heads: usize,
        num_layers: usize,
        num_groups: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            norm: GroupNormConfig::new(num_groups, channels).init(device),
            proj_in: LinearConfig::new(channels, channels).init(device),
            transformer_blocks: (0..num_layers)
                .map(|_| SdxlBasicTransformerBlock::init(channels, context_dim, num_heads, device))
                .collect(),
            proj_out: LinearConfig::new(channels, channels).init(device),
        }
    }

    /// `hidden`: [B, C, H, W], `context`: [B, S, context_dim]
    pub fn forward(&self, hidden: Tensor<B, 4>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let residual = hidden.clone();
        let [batch, channels, height, width] = hidden.dims();
        let seq = height * width;
        let hidden = self
            .norm
            .forward(hidden)
            .swap_dims(1, 2)
            .swap_dims(2, 3)
            .reshape([batch, seq, channels]);
        let mut hidden = self.proj_in.forward(hidden);
        for block in &self.transformer_blocks {
            hidden = block.forward(hidden, context.clone());
        }
        let hidden = self
            .proj_out
            .forward(hidden)
            .reshape([batch, height, width, channels])
            .swap_dims(2, 3)
            .swap_dims(1, 2);
        hidden + residual
    }
}

#[cfg(test)]
mod tests {
    use burn_tensor::Tensor;

    use crate::active_backend::{ActiveBurnBackend, active_device};
    use crate::config::BurnBackendConfig;

    #[test]
    fn spatial_transformer_preserves_nchw_shape() {
        let config = BurnBackendConfig::new("/models", "/output");
        let device = active_device(config.device());
        let block = super::SdxlSpatialTransformer::<ActiveBurnBackend>::init(
            8, 16, 2, 1, 2, &device,
        );
        let hidden = Tensor::<ActiveBurnBackend, 4>::zeros([1, 8, 4, 4], &device);
        let context = Tensor::<ActiveBurnBackend, 3>::zeros([1, 3, 16], &device);
        let out = block.forward(hidden, context);
        assert_eq!(out.dims(), [1, 8, 4, 4]);
    }
}
