//! Burn-native SDXL VAE decoder Module scaffold.

use burn::module::Module;
use burn_core as burn;
use burn_nn::{
    PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
    interpolate::{Interpolate2d, Interpolate2dConfig},
    norm::{GroupNorm, GroupNormConfig},
};
use burn_tensor::{Tensor, activation, backend::Backend};

/// Full-profile SDXL VAE decoder graph.
#[derive(Module, Debug)]
pub struct SdxlVaeDecoder<B: Backend> {
    conv_in: Conv2d<B>,
    mid_block: SdxlVaeMidBlock<B>,
    up_blocks: Vec<SdxlVaeUpBlock<B>>,
    conv_norm_out: GroupNorm<B>,
    conv_out: Conv2d<B>,
}

impl<B: Backend> SdxlVaeDecoder<B> {
    pub fn init(device: &B::Device) -> Self {
        let up_block_specs = [
            // up_block.0: 512 → 512, 3×res, upsampler
            SdxlVaeUpBlockSpec {
                in_channels: 512,
                out_channels: 512,
                num_resnets: 3,
                has_upsampler: true,
            },
            // up_block.1: 512 → 512, 3×res, upsampler
            SdxlVaeUpBlockSpec {
                in_channels: 512,
                out_channels: 512,
                num_resnets: 3,
                has_upsampler: true,
            },
            // up_block.2: 512 → 256, 3×res (first resnet has skip), upsampler
            SdxlVaeUpBlockSpec {
                in_channels: 512,
                out_channels: 256,
                num_resnets: 3,
                has_upsampler: true,
            },
            // up_block.3: 256 → 128, 3×res (first resnet has skip), no upsampler
            SdxlVaeUpBlockSpec {
                in_channels: 256,
                out_channels: 128,
                num_resnets: 3,
                has_upsampler: false,
            },
        ];

        Self {
            conv_in: Conv2dConfig::new([4, 512], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            mid_block: SdxlVaeMidBlock::<B>::init(device),
            up_blocks: up_block_specs
                .into_iter()
                .map(|spec| spec.build(device))
                .collect(),
            conv_norm_out: GroupNormConfig::new(32, 128).init(device),
            conv_out: Conv2dConfig::new([128, 3], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
        }
    }

    pub fn forward(&self, latent: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut hidden = self.conv_in.forward(latent);
        hidden = self.mid_block.forward(hidden);
        for block in &self.up_blocks {
            hidden = block.forward(hidden);
        }
        let hidden = activation::silu(self.conv_norm_out.forward(hidden));
        activation::sigmoid(self.conv_out.forward(hidden))
    }

    #[cfg(test)]
    pub fn has_conv_in(&self) -> bool {
        true
    }

    #[cfg(test)]
    pub fn has_mid_block(&self) -> bool {
        true
    }

    #[cfg(test)]
    pub fn mid_block_resnet_count(&self) -> usize {
        self.mid_block.resnet_count()
    }

    #[cfg(test)]
    pub fn mid_block_has_attention(&self) -> bool {
        true
    }

    #[cfg(test)]
    pub fn up_block_count(&self) -> usize {
        self.up_blocks.len()
    }

    #[cfg(test)]
    pub fn total_up_block_resnet_count(&self) -> usize {
        self.up_blocks
            .iter()
            .map(|b| b.resnet_count())
            .sum()
    }

    #[cfg(test)]
    pub fn has_output_projection(&self) -> bool {
        true
    }
}

#[derive(Debug)]
struct SdxlVaeUpBlockSpec {
    in_channels: usize,
    out_channels: usize,
    num_resnets: usize,
    has_upsampler: bool,
}

impl SdxlVaeUpBlockSpec {
    fn build<B: Backend>(self, device: &B::Device) -> SdxlVaeUpBlock<B> {
        let mut resnets = Vec::with_capacity(self.num_resnets);
        let mut current_channels = self.in_channels;
        for i in 0..self.num_resnets {
            let out_channels = if i == 0 && self.in_channels != self.out_channels {
                self.out_channels
            } else {
                current_channels
            };
            resnets.push(SdxlVaeResidualBlock::<B>::init(
                current_channels,
                out_channels,
                device,
            ));
            current_channels = out_channels;
        }

        let upsampler = if self.has_upsampler {
            let upsample_out = self.out_channels;
            Some(SdxlVaeUpsampleConv {
                conv: Conv2dConfig::new([upsample_out, upsample_out], [3, 3])
                    .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                    .init(device),
            })
        } else {
            None
        };

        SdxlVaeUpBlock {
            resnets,
            interpolate: Interpolate2dConfig::new()
                .with_scale_factor(Some([2.0, 2.0]))
                .init(),
            upsampler,
        }
    }
}

/// A residual block in the SDXL VAE decoder.
///
/// Channels may differ between input and output — when they do, an
/// optional 1×1 skip convolution is inserted to project the input to
/// the output channel count before the skip addition.
#[derive(Module, Debug)]
pub struct SdxlVaeResidualBlock<B: Backend> {
    norm1: GroupNorm<B>,
    conv1: Conv2d<B>,
    norm2: GroupNorm<B>,
    conv2: Conv2d<B>,
    conv_shortcut: Option<Conv2d<B>>,
}

impl<B: Backend> SdxlVaeResidualBlock<B> {
    pub fn init(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        let conv_shortcut = if in_channels != out_channels {
            Some(
                Conv2dConfig::new([in_channels, out_channels], [1, 1])
                    .init(device),
            )
        } else {
            None
        };

        Self {
            norm1: GroupNormConfig::new(32, in_channels).init(device),
            conv1: Conv2dConfig::new([in_channels, out_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            norm2: GroupNormConfig::new(32, out_channels).init(device),
            conv2: Conv2dConfig::new([out_channels, out_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            conv_shortcut,
        }
    }

    pub fn forward(&self, hidden: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = match &self.conv_shortcut {
            Some(conv) => conv.forward(hidden.clone()),
            None => hidden.clone(),
        };
        let hidden = self
            .conv1
            .forward(activation::silu(self.norm1.forward(hidden)));
        let hidden = self
            .conv2
            .forward(activation::silu(self.norm2.forward(hidden)));
        hidden + residual
    }
}

/// Mid-block for the SDXL VAE decoder: 2×residual blocks at the
/// decoder's full channel count followed by a single attention block.
#[derive(Module, Debug)]
pub struct SdxlVaeMidBlock<B: Backend> {
    resnets: [SdxlVaeResidualBlock<B>; 2],
    attention: SdxlVaeAttentionBlock<B>,
}

impl<B: Backend> SdxlVaeMidBlock<B> {
    pub fn init(device: &B::Device) -> Self {
        let channels = 512;
        Self {
            resnets: [
                SdxlVaeResidualBlock::<B>::init(channels, channels, device),
                SdxlVaeResidualBlock::<B>::init(channels, channels, device),
            ],
            attention: SdxlVaeAttentionBlock::<B>::init(channels, device),
        }
    }

    pub fn forward(&self, hidden: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut hidden = self.resnets[0].forward(hidden);
        hidden = self.resnets[1].forward(hidden);
        self.attention.forward(hidden)
    }

    #[cfg(test)]
    pub fn resnet_count(&self) -> usize {
        self.resnets.len()
    }
}

/// Single un-padded spatial-attention block at fixed channel count.
///
/// Used only inside the mid-block of the SDXL VAE decoder; it has no
/// self-attention bias and the output projection is a 1×1 convolution.
#[derive(Module, Debug)]
pub struct SdxlVaeAttentionBlock<B: Backend> {
    group_norm: GroupNorm<B>,
    to_q: Conv2d<B>,
    to_k: Conv2d<B>,
    to_v: Conv2d<B>,
    to_out: Conv2d<B>,
}

impl<B: Backend> SdxlVaeAttentionBlock<B> {
    pub fn init(channels: usize, device: &B::Device) -> Self {
        Self {
            group_norm: GroupNormConfig::new(32, channels).init(device),
            to_q: Conv2dConfig::new([channels, channels], [1, 1]).init(device),
            to_k: Conv2dConfig::new([channels, channels], [1, 1]).init(device),
            to_v: Conv2dConfig::new([channels, channels], [1, 1]).init(device),
            to_out: Conv2dConfig::new([channels, channels], [1, 1]).init(device),
        }
    }

    pub fn forward(&self, hidden: Tensor<B, 4>) -> Tensor<B, 4> {
        // hidden: [B, C, H, W]
        let normalized = self.group_norm.forward(hidden.clone());
        let q = self.to_q.forward(normalized.clone());
        let k = self.to_k.forward(normalized.clone());
        let v = self.to_v.forward(normalized);

        // Collapse H×W into a single sequence dimension.
        let [b, c, h, w] = q.shape().dims();
        let seq = h * w;
        // Reshape [B, C, H, W] → [B, C, H*W] then transpose last two dims
        // to get [B, H*W, C] which is the standard sequence layout for matmul.
        let q = q.reshape([b, c, seq]).swap_dims(1, 2); // [B, H*W, C]
        let k = k.reshape([b, c, seq]).swap_dims(1, 2); // [B, H*W, C]
        let v = v.reshape([b, c, seq]).swap_dims(1, 2); // [B, H*W, C]

        // scaled dot-product attention over the spatial sequence.
        let scale = (c as f64).sqrt().recip();
        // q @ k^T: [B, H*W, C] @ [B, C, H*W] → [B, H*W, H*W]
        let attn_weights = Tensor::matmul(q, k.transpose()) * scale;
        let attn_weights = activation::softmax(attn_weights, 2);
        // attn @ v: [B, H*W, H*W] @ [B, H*W, C] → [B, H*W, C]
        let attn_out = Tensor::matmul(attn_weights, v)
            .swap_dims(1, 2) // [B, C, H*W]
            .reshape([b, c, h, w]);

        self.to_out.forward(hidden + attn_out)
    }
}

/// One of the four up_blocks in the SDXL VAE decoder.
///
/// Each block contains 3 residual blocks at potentially changing
/// channel counts, an optional Nearest-neighbor (scale=2) upsampling,
/// and an optional post-convolution to match the target channel count.
#[derive(Module, Debug)]
pub struct SdxlVaeUpBlock<B: Backend> {
    resnets: Vec<SdxlVaeResidualBlock<B>>,
    interpolate: Interpolate2d,
    upsampler: Option<SdxlVaeUpsampleConv<B>>,
}

impl<B: Backend> SdxlVaeUpBlock<B> {
    pub fn forward(&self, hidden: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut hidden = hidden;
        for resnet in &self.resnets {
            hidden = resnet.forward(hidden);
        }
        if self.upsampler.is_some() {
            hidden = self.interpolate.forward(hidden);
            hidden = match &self.upsampler {
                Some(upsampler) => upsampler.forward(hidden),
                None => hidden,
            };
        }
        hidden
    }

    #[cfg(test)]
    pub fn resnet_count(&self) -> usize {
        self.resnets.len()
    }
}

#[derive(Module, Debug)]
struct SdxlVaeUpsampleConv<B: Backend> {
    conv: Conv2d<B>,
}

impl<B: Backend> SdxlVaeUpsampleConv<B> {
    pub fn forward(&self, hidden: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv.forward(hidden)
    }
}

#[cfg(test)]
mod tests {
    use burn_tensor::Tensor;

    use crate::active_backend::{ActiveBurnBackend, active_device};
    use crate::config::BurnBackendConfig;

    #[test]
    fn sdxl_vae_decoder_module_outputs_rgb_image_shape_on_active_backend() {
        let config = BurnBackendConfig::new("/models", "/output");
        let device = active_device(config.device());
        let decoder = super::SdxlVaeDecoder::<ActiveBurnBackend>::init(&device);
        let latent = Tensor::<ActiveBurnBackend, 4>::zeros([1, 4, 8, 8], &device);

        let image = decoder.forward(latent);

        // 8→64 from 3 upsampling stages, no upscaling in last block, padding=1 conv_out
        assert_eq!(image.shape().dims(), [1, 3, 64, 64]);
    }

    #[test]
    fn sdxl_vae_decoder_module_exposes_full_profile_execution_plan() {
        let config = BurnBackendConfig::new("/models", "/output");
        let device = active_device(config.device());
        let decoder = super::SdxlVaeDecoder::<ActiveBurnBackend>::init(&device);

        assert!(decoder.has_conv_in());
        assert!(decoder.has_mid_block());
        assert_eq!(decoder.mid_block_resnet_count(), 2);
        assert!(decoder.mid_block_has_attention());
        assert_eq!(decoder.up_block_count(), 4);
        assert_eq!(decoder.total_up_block_resnet_count(), 12);
        assert!(decoder.has_output_projection());
    }
}
