//! Burn-native SDXL CLIP module structs.
//!
//! The `ClipTextEncoderWeights` structs remain as the current production
//! loader/forward compatibility surface. The `*Module` structs define the
//! migration target: backend-generic Burn modules composed from Burn's own
//! reusable `nn` layers and loadable through the typed runtime store seam.

use burn::module::{Module, Param};
use burn_core as burn;
use burn_nn::{
    Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    modules::attention::{
        MhaInput, MultiHeadAttention, MultiHeadAttentionConfig, generate_autoregressive_mask,
    },
};
use burn_tensor::{Int, Shape, Tensor, TensorData, activation, backend::Backend};

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

    pub fn load_weights(
        self,
        clip_l: &ClipTextEncoderWeights,
        open_clip_g: &ClipTextEncoderWeights,
        device: &B::Device,
    ) -> Self {
        Self {
            clip_l: self.clip_l.load_weights(clip_l, device),
            open_clip_g: self.open_clip_g.load_weights(open_clip_g, device),
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
        let text_projection = profile
            .produces_pooled_output
            .then(|| LinearConfig::new(width, width).init(device));

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

    pub fn load_weights(self, weights: &ClipTextEncoderWeights, device: &B::Device) -> Self {
        let text_projection =
            self.text_projection
                .zip(
                    (!weights.text_projection_weight.data.is_empty()).then_some((
                        &weights.text_projection_weight.data,
                        &weights.text_projection_bias.data,
                    )),
                );

        Self {
            token_embedding: load_embedding(
                self.token_embedding,
                &weights.token_embedding.data,
                device,
            ),
            position_embedding: load_embedding(
                self.position_embedding,
                &weights.position_embedding.data,
                device,
            ),
            final_layer_norm: load_layer_norm(
                self.final_layer_norm,
                &weights.final_layer_norm_weight.data,
                &weights.final_layer_norm_bias.data,
                device,
            ),
            text_projection: text_projection
                .map(|(module, (weight, bias))| load_linear(module, weight, Some(bias), device)),
            blocks: self
                .blocks
                .into_iter()
                .zip(&weights.blocks)
                .map(|(block, weights)| block.load_weights(weights, device))
                .collect(),
        }
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

    pub(crate) fn load_weights(self, weights: &ClipTransformerWeights, device: &B::Device) -> Self {
        let width = self.attention.d_model;
        let (query_weight, key_weight, value_weight) =
            split_fused_qkv_weight(&weights.attn_in_proj_weight.data, width);
        let (query_bias, key_bias, value_bias) =
            split_fused_qkv_bias(&weights.attn_in_proj_bias.data, width);

        Self {
            ln_1: load_layer_norm(
                self.ln_1,
                &weights.ln_1_weight.data,
                &weights.ln_1_bias.data,
                device,
            ),
            attention: MultiHeadAttention {
                query: load_linear(
                    self.attention.query,
                    &query_weight,
                    Some(&query_bias),
                    device,
                ),
                key: load_linear(self.attention.key, &key_weight, Some(&key_bias), device),
                value: load_linear(
                    self.attention.value,
                    &value_weight,
                    Some(&value_bias),
                    device,
                ),
                output: load_linear(
                    self.attention.output,
                    &weights.attn_out_proj_weight.data,
                    Some(&weights.attn_out_proj_bias.data),
                    device,
                ),
                ..self.attention
            },
            ln_2: load_layer_norm(
                self.ln_2,
                &weights.ln_2_weight.data,
                &weights.ln_2_bias.data,
                device,
            ),
            mlp_fc1: load_linear(
                self.mlp_fc1,
                &weights.mlp_fc1_weight.data,
                Some(&weights.mlp_fc1_bias.data),
                device,
            ),
            mlp_fc2: load_linear(
                self.mlp_fc2,
                &weights.mlp_fc2_weight.data,
                Some(&weights.mlp_fc2_bias.data),
                device,
            ),
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

fn load_layer_norm<B: Backend>(
    mut module: LayerNorm<B>,
    gamma: &[f32],
    beta: &[f32],
    device: &B::Device,
) -> LayerNorm<B> {
    module.gamma = param_from_1d(gamma, device);
    module.beta = Some(param_from_1d(beta, device));
    module
}

fn load_embedding<B: Backend>(
    mut module: Embedding<B>,
    weight: &[f32],
    device: &B::Device,
) -> Embedding<B> {
    let [n_embedding, d_model] = module.weight.shape().dims();
    module.weight = param_from_2d(weight, n_embedding, d_model, device);
    module
}

fn load_linear<B: Backend>(
    module: Linear<B>,
    pytorch_weight: &[f32],
    bias: Option<&[f32]>,
    device: &B::Device,
) -> Linear<B> {
    let [d_input, d_output] = module.weight.shape().dims();
    Linear {
        weight: param_from_2d(
            &transpose_row_major(pytorch_weight, d_output, d_input),
            d_input,
            d_output,
            device,
        ),
        bias: bias.map(|data| param_from_1d(data, device)),
    }
}

fn split_fused_qkv_weight(fused: &[f32], width: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let projection_len = width * width;
    (
        fused[0..projection_len].to_vec(),
        fused[projection_len..projection_len * 2].to_vec(),
        fused[projection_len * 2..projection_len * 3].to_vec(),
    )
}

fn split_fused_qkv_bias(fused: &[f32], width: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        fused[0..width].to_vec(),
        fused[width..width * 2].to_vec(),
        fused[width * 2..width * 3].to_vec(),
    )
}

fn transpose_row_major(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0; data.len()];
    for row in 0..rows {
        for col in 0..cols {
            transposed[col * rows + row] = data[row * cols + col];
        }
    }
    transposed
}

fn param_from_2d<B: Backend>(
    data: &[f32],
    rows: usize,
    cols: usize,
    device: &B::Device,
) -> Param<Tensor<B, 2>> {
    let shape = Shape::new([rows, cols]);
    Param::from_data(TensorData::new(data.to_vec(), shape), device)
}

fn param_from_1d<B: Backend>(data: &[f32], device: &B::Device) -> Param<Tensor<B, 1>> {
    Param::from_data(
        TensorData::new(data.to_vec(), Shape::new([data.len()])),
        device,
    )
}

/// Weight data loaded from safetensors — pre-allocated f32 buffers
/// that can be converted to Burn tensors on demand.
#[derive(Debug, Clone)]
pub struct ClipWeightData {
    /// Buffer of f32 values.
    pub data: Vec<f32>,
}

/// Single transformer block weights.
#[derive(Debug, Clone)]
pub struct ClipTransformerWeights {
    pub ln_1_weight: ClipWeightData,
    pub ln_1_bias: ClipWeightData,
    pub ln_2_weight: ClipWeightData,
    pub ln_2_bias: ClipWeightData,
    pub attn_in_proj_weight: ClipWeightData,
    pub attn_in_proj_bias: ClipWeightData,
    pub attn_out_proj_weight: ClipWeightData,
    pub attn_out_proj_bias: ClipWeightData,
    pub mlp_fc1_weight: ClipWeightData,
    pub mlp_fc1_bias: ClipWeightData,
    pub mlp_fc2_weight: ClipWeightData,
    pub mlp_fc2_bias: ClipWeightData,
}

/// Complete CLIP text encoder weights — the safetensors content
/// indexed by the ClipTextEncoderProfile key-space.
#[derive(Debug, Clone)]
pub struct ClipTextEncoderWeights {
    pub token_embedding: ClipWeightData,
    pub position_embedding: ClipWeightData,
    pub final_layer_norm_weight: ClipWeightData,
    pub final_layer_norm_bias: ClipWeightData,
    pub text_projection_weight: ClipWeightData,
    pub text_projection_bias: ClipWeightData,
    pub blocks: Vec<ClipTransformerWeights>,
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
    fn transformer_block_loads_fused_clip_attention_into_burn_qkv_linears() {
        type B = ActiveBurnBackend;

        let device = active_test_device();
        let profile = tiny_attention_profile();
        let block = super::ClipTransformerBlockModule::<B>::init(&profile, &device);
        let weights = tiny_block_weights();

        let loaded = block.load_weights(&weights, &device);

        assert_param_2d(&loaded.attention.query.weight, [1.0, 3.0, 2.0, 4.0]);
        assert_param_1d(
            loaded.attention.query.bias.as_ref().expect("query bias"),
            [101.0, 102.0],
        );
        assert_param_2d(&loaded.attention.key.weight, [5.0, 7.0, 6.0, 8.0]);
        assert_param_1d(
            loaded.attention.key.bias.as_ref().expect("key bias"),
            [103.0, 104.0],
        );
        assert_param_2d(&loaded.attention.value.weight, [9.0, 11.0, 10.0, 12.0]);
        assert_param_1d(
            loaded.attention.value.bias.as_ref().expect("value bias"),
            [105.0, 106.0],
        );
        assert_param_2d(
            &loaded.attention.output.weight,
            [201.0, 203.0, 202.0, 204.0],
        );
        assert_param_1d(
            loaded.attention.output.bias.as_ref().expect("output bias"),
            [301.0, 302.0],
        );
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

    fn tiny_block_weights() -> super::ClipTransformerWeights {
        super::ClipTransformerWeights {
            ln_1_weight: weight([1.0, 1.0]),
            ln_1_bias: weight([0.0, 0.0]),
            ln_2_weight: weight([1.0, 1.0]),
            ln_2_bias: weight([0.0, 0.0]),
            attn_in_proj_weight: weight([
                1.0, 2.0, // query row 0
                3.0, 4.0, // query row 1
                5.0, 6.0, // key row 0
                7.0, 8.0, // key row 1
                9.0, 10.0, // value row 0
                11.0, 12.0, // value row 1
            ]),
            attn_in_proj_bias: weight([101.0, 102.0, 103.0, 104.0, 105.0, 106.0]),
            attn_out_proj_weight: weight([201.0, 202.0, 203.0, 204.0]),
            attn_out_proj_bias: weight([301.0, 302.0]),
            mlp_fc1_weight: weight([
                401.0, 402.0, 403.0, 404.0, 405.0, 406.0, 407.0, 408.0, 409.0, 410.0, 411.0, 412.0,
                413.0, 414.0, 415.0, 416.0,
            ]),
            mlp_fc1_bias: weight([501.0, 502.0, 503.0, 504.0, 505.0, 506.0, 507.0, 508.0]),
            mlp_fc2_weight: weight([
                601.0, 602.0, 603.0, 604.0, 605.0, 606.0, 607.0, 608.0, 609.0, 610.0, 611.0, 612.0,
                613.0, 614.0, 615.0, 616.0,
            ]),
            mlp_fc2_bias: weight([701.0, 702.0]),
        }
    }

    fn weight<const N: usize>(data: [f32; N]) -> super::ClipWeightData {
        super::ClipWeightData {
            data: data.to_vec(),
        }
    }

    fn active_test_device() -> burn_tensor::Device<ActiveBurnBackend> {
        let config = BurnBackendConfig::new("/models", "/output");
        active_device(config.device())
    }

    fn assert_param_2d<const N: usize>(
        param: &burn_core::module::Param<burn_tensor::Tensor<ActiveBurnBackend, 2>>,
        expected: [f32; N],
    ) {
        assert_eq!(
            param.val().into_data().to_vec::<f32>().expect("f32 data"),
            expected
        );
    }

    fn assert_param_1d<const N: usize>(
        param: &burn_core::module::Param<burn_tensor::Tensor<ActiveBurnBackend, 1>>,
        expected: [f32; N],
    ) {
        assert_eq!(
            param.val().into_data().to_vec::<f32>().expect("f32 data"),
            expected
        );
    }
}
