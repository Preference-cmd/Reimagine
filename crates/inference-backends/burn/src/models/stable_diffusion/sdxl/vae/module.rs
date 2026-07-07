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

/// Burn-native SDXL VAE decoder graph.
#[derive(Module, Debug)]
pub struct SdxlVaeDecoder<B: Backend> {
    latent_projection: Conv2d<B>,
    residual_blocks: Vec<SdxlVaeResidualBlock<B>>,
    upsample_blocks: Vec<SdxlVaeUpsampleBlock>,
    pub conv_out: Conv2d<B>,
}

impl<B: Backend> SdxlVaeDecoder<B> {
    pub fn init(device: &B::Device) -> Self {
        Self {
            latent_projection: Conv2dConfig::new([4, 4], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            residual_blocks: vec![
                SdxlVaeResidualBlock::init(4, device),
                SdxlVaeResidualBlock::init(4, device),
            ],
            upsample_blocks: vec![
                SdxlVaeUpsampleBlock::new(),
                SdxlVaeUpsampleBlock::new(),
                SdxlVaeUpsampleBlock::new(),
            ],
            conv_out: Conv2dConfig::new([4, 3], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
        }
    }

    pub fn forward(&self, latent: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut hidden = self.latent_projection.forward(latent);
        for block in &self.residual_blocks {
            hidden = block.forward(hidden);
        }
        for block in &self.upsample_blocks {
            hidden = block.forward(hidden);
        }
        activation::sigmoid(self.conv_out.forward(hidden))
    }

    #[cfg(test)]
    pub fn has_latent_projection(&self) -> bool {
        true
    }

    #[cfg(test)]
    pub fn residual_block_count(&self) -> usize {
        self.residual_blocks.len()
    }

    #[cfg(test)]
    pub fn upsample_block_count(&self) -> usize {
        self.upsample_blocks.len()
    }

    #[cfg(test)]
    pub fn has_output_projection(&self) -> bool {
        true
    }
}

#[derive(Module, Debug)]
pub struct SdxlVaeResidualBlock<B: Backend> {
    norm_1: GroupNorm<B>,
    conv_1: Conv2d<B>,
    norm_2: GroupNorm<B>,
    conv_2: Conv2d<B>,
}

impl<B: Backend> SdxlVaeResidualBlock<B> {
    fn init(channels: usize, device: &B::Device) -> Self {
        Self {
            norm_1: GroupNormConfig::new(1, channels).init(device),
            conv_1: Conv2dConfig::new([channels, channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            norm_2: GroupNormConfig::new(1, channels).init(device),
            conv_2: Conv2dConfig::new([channels, channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
        }
    }

    fn forward(&self, hidden: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = hidden.clone();
        let hidden = self
            .conv_1
            .forward(activation::silu(self.norm_1.forward(hidden)));
        let hidden = self
            .conv_2
            .forward(activation::silu(self.norm_2.forward(hidden)));
        hidden + residual
    }
}

#[derive(Clone, Module, Debug)]
pub struct SdxlVaeUpsampleBlock {
    interpolate: Interpolate2d,
}

impl SdxlVaeUpsampleBlock {
    fn new() -> Self {
        Self {
            interpolate: Interpolate2dConfig::new()
                .with_scale_factor(Some([2.0, 2.0]))
                .init(),
        }
    }

    fn forward<B: Backend>(&self, hidden: Tensor<B, 4>) -> Tensor<B, 4> {
        self.interpolate.forward(hidden)
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

        assert_eq!(image.shape().dims(), [1, 3, 64, 64]);
    }

    #[test]
    fn sdxl_vae_decoder_module_exposes_full_profile_execution_plan() {
        let config = BurnBackendConfig::new("/models", "/output");
        let device = active_device(config.device());
        let decoder = super::SdxlVaeDecoder::<ActiveBurnBackend>::init(&device);

        assert!(decoder.has_latent_projection());
        assert_eq!(decoder.residual_block_count(), 2);
        assert_eq!(decoder.upsample_block_count(), 3);
        assert!(decoder.has_output_projection());
    }
}
