//! Burn-native SDXL CLIP module structs.
//!
//! These backend-generic Burn modules are composed from Burn's reusable `nn`
//! layers and loaded through the typed runtime store seam.

use burn::module::Module;
use burn_core as burn;
use burn_nn::{
    Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    modules::attention::{
        MhaInput, MultiHeadAttention, MultiHeadAttentionConfig, generate_autoregressive_mask,
    },
};
use burn_tensor::{Int, Tensor, activation, backend::Backend};

use crate::text_encoder::clip::ClipTextEncoderProfile;

/// Burn-native SDXL text encoder pair.
#[derive(Module, Debug)]
pub struct SdxlTextEncoders<B: Backend> {
    pub clip_l: ClipTextEncoderModule<B>,
    pub open_clip_g: ClipTextEncoderModule<B>,
}

impl<B: Backend> SdxlTextEncoders<B> {
    /// Initialize both SDXL text encoders with Burn-native layer modules.
    pub fn init(device: &B::Device) -> Self {
        Self::init_from_profiles(
            &ClipTextEncoderProfile::sdxl_clip_l(),
            &ClipTextEncoderProfile::sdxl_open_clip_g(),
            device,
        )
    }

    pub(crate) fn init_from_profiles(
        clip_l: &ClipTextEncoderProfile,
        open_clip_g: &ClipTextEncoderProfile,
        device: &B::Device,
    ) -> Self {
        Self {
            clip_l: ClipTextEncoderModule::init(clip_l, device),
            open_clip_g: ClipTextEncoderModule::init(open_clip_g, device),
        }
    }
}

/// Burn-native CLIP/OpenCLIP text encoder graph.
#[derive(Module, Debug)]
pub struct ClipTextEncoderModule<B: Backend> {
    pub token_embedding: Embedding<B>,
    pub position_embedding: Embedding<B>,
    pub final_layer_norm: LayerNorm<B>,
    pub text_projection: Option<Linear<B>>,
    blocks: Vec<ClipTransformerBlockModule<B>>,
}

/// Burn-native CLIP/OpenCLIP encoder output.
#[derive(Debug, Clone)]
pub struct ClipTextEncoderModuleOutput<B: Backend> {
    pub hidden: Tensor<B, 3>,
    pub pooled: Option<Tensor<B, 2>>,
}

impl<B: Backend> ClipTextEncoderModule<B> {
    /// Initialize the encoder graph from a CLIP profile.
    pub fn init(profile: &ClipTextEncoderProfile, device: &B::Device) -> Self {
        let width = profile.width as usize;
        let blocks = (0..profile.num_layers)
            .map(|_| ClipTransformerBlockModule::init(profile, device))
            .collect();
        let text_projection = profile.produces_pooled_output.then(|| {
            LinearConfig::new(width, width)
                .with_bias(false)
                .init(device)
        });

        Self {
            token_embedding: EmbeddingConfig::new(profile.vocab_size as usize, width).init(device),
            position_embedding: EmbeddingConfig::new(profile.sequence_length as usize, width)
                .init(device),
            final_layer_norm: LayerNormConfig::new(width).init(device),
            text_projection,
            blocks,
        }
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn blocks(&self) -> &[ClipTransformerBlockModule<B>] {
        &self.blocks
    }

    pub fn uses_pooled_projection(&self) -> bool {
        self.text_projection.is_some()
    }

    pub fn forward(&self, token_ids: Tensor<B, 2, Int>) -> ClipTextEncoderModuleOutput<B> {
        let [batch, seq_len] = token_ids.dims();
        let token_embeddings = self.token_embedding.forward(token_ids);
        let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &token_embeddings.device())
            .reshape([1, seq_len])
            .repeat_dim(0, batch);
        let position_embeddings = self.position_embedding.forward(positions);

        let mut hidden = token_embeddings + position_embeddings;
        for block in &self.blocks {
            hidden = block.forward(hidden);
        }

        let hidden = self.final_layer_norm.forward(hidden);
        let pooled = self.text_projection.as_ref().map(|projection| {
            let [batch, _seq_len, width] = hidden.dims();
            let first_token = hidden.clone().slice([0..batch, 0..1, 0..width]);
            projection.forward(first_token).reshape([batch, width])
        });

        ClipTextEncoderModuleOutput { hidden, pooled }
    }
}

/// Burn-native CLIP transformer block.
#[derive(Module, Debug)]
pub struct ClipTransformerBlockModule<B: Backend> {
    pub ln_1: LayerNorm<B>,
    pub attention: MultiHeadAttention<B>,
    pub ln_2: LayerNorm<B>,
    pub mlp_fc1: Linear<B>,
    pub mlp_fc2: Linear<B>,
}

impl<B: Backend> ClipTransformerBlockModule<B> {
    fn init(profile: &ClipTextEncoderProfile, device: &B::Device) -> Self {
        let width = profile.width as usize;
        let inner_width = profile.inner_width as usize;

        Self {
            ln_1: LayerNormConfig::new(width).init(device),
            attention: MultiHeadAttentionConfig::new(width, profile.heads as usize)
                .with_dropout(0.0)
                .init(device),
            ln_2: LayerNormConfig::new(width).init(device),
            mlp_fc1: LinearConfig::new(width, inner_width).init(device),
            mlp_fc2: LinearConfig::new(inner_width, width).init(device),
        }
    }

    fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _width] = hidden.dims();
        let attn_input = self.ln_1.forward(hidden.clone());
        let attn_mask = generate_autoregressive_mask(batch, seq_len, &attn_input.device());
        let attn_output = self
            .attention
            .forward(MhaInput::self_attn(attn_input).mask_attn(attn_mask))
            .context;
        let hidden = hidden + attn_output;

        let mlp_input = self.ln_2.forward(hidden.clone());
        let mlp_hidden = quick_gelu(self.mlp_fc1.forward(mlp_input));
        let mlp_output = self.mlp_fc2.forward(mlp_hidden);

        hidden + mlp_output
    }
}

fn quick_gelu<B: Backend>(tensor: Tensor<B, 3>) -> Tensor<B, 3> {
    let scaled = tensor.clone().mul_scalar(1.702);
    tensor * activation::sigmoid(scaled)
}

#[cfg(test)]
mod tests {
    use crate::active_backend::{ActiveBurnBackend, active_device};
    use crate::config::BurnBackendConfig;
    use burn_core::module::Module;
    use burn_store::{ModuleSnapshot, SafetensorsStore};
    use burn_tensor::{Int, Tensor};

    use crate::runtime::BurnRuntime;
    use crate::text_encoder::clip::{ClipTextEncoderProfile, ClipTextEncoderVariant};

    use super::SdxlTextEncoders;

    #[test]
    fn sdxl_text_encoders_are_burn_modules_loaded_through_runtime_store() {
        type B = ActiveBurnBackend;

        let clip_l_profile = tiny_profile(ClipTextEncoderVariant::ClipL, false);
        let open_clip_g_profile = tiny_profile(ClipTextEncoderVariant::OpenClipG, true);
        let runtime = BurnRuntime::<B>::new(active_test_device());
        let module = SdxlTextEncoders::<B>::init_from_profiles(
            &clip_l_profile,
            &open_clip_g_profile,
            runtime.device(),
        );

        assert_eq!(
            module.clip_l.block_count(),
            clip_l_profile.num_layers as usize
        );
        assert_eq!(
            module.open_clip_g.block_count(),
            open_clip_g_profile.num_layers as usize
        );
        assert!(!module.clip_l.uses_pooled_projection());
        assert!(module.open_clip_g.uses_pooled_projection());

        let first_clip_l_block = module
            .clip_l
            .blocks()
            .first()
            .expect("CLIP-L should initialize transformer blocks");
        assert_eq!(
            first_clip_l_block.attention.d_model,
            clip_l_profile.width as usize
        );
        assert_eq!(
            first_clip_l_block.attention.n_heads,
            clip_l_profile.heads as usize
        );

        let mut save_store = SafetensorsStore::from_bytes(None);
        module
            .save_into(&mut save_store)
            .expect("text encoder skeleton should save into burn-store");
        let bytes = save_store
            .get_bytes()
            .expect("saved text encoder bytes should be readable");
        let mut load_store = SafetensorsStore::from_bytes(Some(bytes));
        let mut loaded = SdxlTextEncoders::<B>::init_from_profiles(
            &clip_l_profile,
            &open_clip_g_profile,
            runtime.device(),
        );

        let result = runtime
            .load_module_store(&mut loaded, &mut load_store)
            .expect("text encoder skeleton should load from burn-store");

        assert!(result.errors.is_empty(), "unexpected load errors: {result}");
        assert_eq!(module.num_params(), loaded.num_params());
    }

    #[test]
    fn clip_text_encoder_module_forward_returns_hidden_and_optional_pooled_shapes() {
        type B = ActiveBurnBackend;

        let device = active_test_device();
        let clip_l_profile = tiny_attention_profile();
        let clip_g_profile = ClipTextEncoderProfile {
            produces_pooled_output: true,
            ..tiny_attention_profile()
        };
        let clip_l = super::ClipTextEncoderModule::<B>::init(&clip_l_profile, &device);
        let clip_g = super::ClipTextEncoderModule::<B>::init(&clip_g_profile, &device);
        let token_ids = Tensor::<B, 2, Int>::from_ints([[1, 2, 3, 4, 5]], &device);

        let clip_l_out = clip_l.forward(token_ids.clone());
        let clip_g_out = clip_g.forward(token_ids);

        assert_eq!(clip_l_out.hidden.dims(), [1, 5, 2]);
        assert!(clip_l_out.pooled.is_none());
        assert_eq!(clip_g_out.hidden.dims(), [1, 5, 2]);
        assert_eq!(
            clip_g_out.pooled.expect("OpenCLIP-G pooled output").dims(),
            [1, 2]
        );
    }

    fn tiny_profile(
        variant: ClipTextEncoderVariant,
        produces_pooled_output: bool,
    ) -> ClipTextEncoderProfile {
        ClipTextEncoderProfile {
            variant,
            target_prefix: "test.text_encoder".to_string(),
            num_layers: 2,
            width: 4,
            heads: 2,
            inner_width: 8,
            vocab_size: 16,
            sequence_length: 5,
            produces_pooled_output,
        }
    }

    fn tiny_attention_profile() -> ClipTextEncoderProfile {
        ClipTextEncoderProfile {
            variant: ClipTextEncoderVariant::ClipL,
            target_prefix: "test.text_encoder".to_string(),
            num_layers: 1,
            width: 2,
            heads: 1,
            inner_width: 8,
            vocab_size: 16,
            sequence_length: 5,
            produces_pooled_output: false,
        }
    }

    fn active_test_device() -> burn_tensor::Device<ActiveBurnBackend> {
        let config = BurnBackendConfig::new("/models", "/output");
        active_device(config.device())
    }
}
