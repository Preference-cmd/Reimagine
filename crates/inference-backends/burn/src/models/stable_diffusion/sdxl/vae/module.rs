//! Burn-native SDXL VAE decoder Module scaffold.

use burn::module::Module;
use burn_core as burn;
use burn_nn::{
    PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
};
use burn_tensor::{Tensor, activation, backend::Backend};

/// Minimal Burn-native SDXL VAE decoder graph.
///
/// The first 14l cutover keeps the production path on the active WGPU/Flex
/// backend. It projects latent channels to RGB and performs deterministic VAE
/// scale-factor upsampling as Module/tensor graph glue; fuller residual,
/// attention, and upsample blocks can expand this scaffold without reviving the
/// old ndarray placeholder.
#[derive(Module, Debug)]
pub struct SdxlVaeDecoder<B: Backend> {
    pub conv_out: Conv2d<B>,
}

impl<B: Backend> SdxlVaeDecoder<B> {
    pub fn init(device: &B::Device) -> Self {
        Self {
            conv_out: Conv2dConfig::new([4, 3], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
        }
    }

    pub fn forward(&self, latent: Tensor<B, 4>) -> Tensor<B, 4> {
        let image = self.conv_out.forward(latent);
        let image = image.repeat_dim(2, 8).repeat_dim(3, 8);
        activation::sigmoid(image)
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
}
