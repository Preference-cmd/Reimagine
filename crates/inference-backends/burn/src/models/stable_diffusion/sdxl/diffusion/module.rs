//! Burn-native SDXL UNet module structs and legacy diffusion weight buffers.
//!
//! `DiffusionUNetWeights` remains the old safetensors projection surface while
//! `SdxlUnet<B>` is the Module migration target for 14j.

use burn::module::Module;
use burn_core as burn;
use burn_nn::{
    Linear, LinearConfig, PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
    modules::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    norm::{GroupNorm, GroupNormConfig},
};
use burn_tensor::{Int, Tensor, activation, backend::Backend};

/// Reimagine-owned SDXL UNet topology facts.
#[derive(Debug, Clone)]
pub struct SdxlUnetTopology {
    profile: SdxlUnetTopologyProfile,
    pub latent_channels: usize,
    pub model_channels: usize,
    pub time_input_dim: usize,
    pub time_hidden_dim: usize,
    pub conditioning_dim: usize,
    pub pooled_conditioning_dim: usize,
    pub time_ids_dim: usize,
    pub down_blocks: Vec<SdxlStageSpec>,
    pub middle_blocks: Vec<SdxlStageSpec>,
    pub up_blocks: Vec<SdxlStageSpec>,
}

/// Stable topology names used by loaders, diagnostics, and follow-up issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdxlUnetTopologyProfile {
    TinySdxlE2e,
    SdxlBase,
}

impl SdxlUnetTopologyProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TinySdxlE2e => "tiny_sdxl_e2e",
            Self::SdxlBase => "sdxl_base",
        }
    }

    pub fn topology(self) -> SdxlUnetTopology {
        match self {
            Self::TinySdxlE2e => SdxlUnetTopology::tiny(),
            Self::SdxlBase => SdxlUnetTopology::sdxl_base(),
        }
    }

    pub fn is_module_graph_supported(self) -> bool {
        matches!(self, Self::TinySdxlE2e)
    }
}

impl SdxlUnetTopology {
    pub fn tiny() -> Self {
        Self {
            profile: SdxlUnetTopologyProfile::TinySdxlE2e,
            latent_channels: 4,
            model_channels: 4,
            time_input_dim: 4,
            time_hidden_dim: 8,
            conditioning_dim: 16,
            pooled_conditioning_dim: 8,
            time_ids_dim: 6,
            down_blocks: vec![SdxlStageSpec {
                res_blocks: vec![SdxlResBlockSpec {
                    in_channels: 4,
                    out_channels: 4,
                    num_groups: 2,
                }],
                self_attention_blocks: vec![SdxlAttentionBlockSpec {
                    channels: 4,
                    num_heads: 2,
                    num_groups: 2,
                }],
                cross_attention_blocks: vec![SdxlCrossAttentionBlockSpec {
                    channels: 4,
                    context_dim: 16,
                    num_heads: 2,
                    num_groups: 2,
                }],
            }],
            middle_blocks: Vec::new(),
            up_blocks: Vec::new(),
        }
    }

    pub fn sdxl_base() -> Self {
        let stage = |channels: usize, heads: usize| SdxlStageSpec {
            res_blocks: vec![
                SdxlResBlockSpec {
                    in_channels: channels,
                    out_channels: channels,
                    num_groups: 32,
                },
                SdxlResBlockSpec {
                    in_channels: channels,
                    out_channels: channels,
                    num_groups: 32,
                },
            ],
            self_attention_blocks: vec![SdxlAttentionBlockSpec {
                channels,
                num_heads: heads,
                num_groups: 32,
            }],
            cross_attention_blocks: vec![SdxlCrossAttentionBlockSpec {
                channels,
                context_dim: 2048,
                num_heads: heads,
                num_groups: 32,
            }],
        };

        Self {
            profile: SdxlUnetTopologyProfile::SdxlBase,
            latent_channels: 4,
            model_channels: 320,
            time_input_dim: 320,
            time_hidden_dim: 1280,
            conditioning_dim: 2048,
            pooled_conditioning_dim: 1280,
            time_ids_dim: 6,
            down_blocks: vec![stage(320, 5), stage(640, 10), stage(1280, 20)],
            middle_blocks: vec![stage(1280, 20)],
            up_blocks: vec![stage(1280, 20), stage(640, 10), stage(320, 5)],
        }
    }

    pub fn profile(&self) -> SdxlUnetTopologyProfile {
        self.profile
    }

    pub fn name(&self) -> &'static str {
        self.profile.as_str()
    }

    pub fn res_block_count(&self) -> usize {
        self.stages().map(|stage| stage.res_blocks.len()).sum()
    }

    pub fn self_attention_block_count(&self) -> usize {
        self.stages()
            .map(|stage| stage.self_attention_blocks.len())
            .sum()
    }

    pub fn cross_attention_block_count(&self) -> usize {
        self.stages()
            .map(|stage| stage.cross_attention_blocks.len())
            .sum()
    }

    fn stages(&self) -> impl Iterator<Item = &SdxlStageSpec> {
        self.down_blocks
            .iter()
            .chain(self.middle_blocks.iter())
            .chain(self.up_blocks.iter())
    }
}

#[derive(Debug, Clone)]
pub struct SdxlStageSpec {
    pub res_blocks: Vec<SdxlResBlockSpec>,
    pub self_attention_blocks: Vec<SdxlAttentionBlockSpec>,
    pub cross_attention_blocks: Vec<SdxlCrossAttentionBlockSpec>,
}

#[derive(Debug, Clone)]
pub struct SdxlResBlockSpec {
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_groups: usize,
}

#[derive(Debug, Clone)]
pub struct SdxlAttentionBlockSpec {
    pub channels: usize,
    pub num_heads: usize,
    pub num_groups: usize,
}

#[derive(Debug, Clone)]
pub struct SdxlCrossAttentionBlockSpec {
    pub channels: usize,
    pub context_dim: usize,
    pub num_heads: usize,
    pub num_groups: usize,
}

/// Minimal Burn-native SDXL UNet graph.
///
/// This is the first 14j scaffold: it is intentionally small, but already uses
/// Burn `Module<B>` members and active-backend tensors. Later 14j slices expand
/// this into the full SDXL topology and delete the old ndarray helpers.
#[derive(Module, Debug)]
pub struct SdxlUnet<B: Backend> {
    pub conv_in: Conv2d<B>,
    time_embedding: SdxlTimeEmbedding<B>,
    added_conditioning: SdxlAddedConditioningProjection<B>,
    down_blocks: Vec<SdxlUnetStage<B>>,
    middle_blocks: Vec<SdxlUnetStage<B>>,
    up_blocks: Vec<SdxlUnetStage<B>>,
    pub conv_out: Conv2d<B>,
}

impl<B: Backend> SdxlUnet<B> {
    pub fn init(device: &B::Device) -> Self {
        Self::init_from_profile(SdxlUnetTopologyProfile::TinySdxlE2e, device)
    }

    pub fn init_from_profile(profile: SdxlUnetTopologyProfile, device: &B::Device) -> Self {
        Self::init_from_topology(&profile.topology(), device)
    }

    pub fn init_from_topology(topology: &SdxlUnetTopology, device: &B::Device) -> Self {
        Self {
            conv_in: Conv2dConfig::new([topology.latent_channels, topology.model_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            time_embedding: SdxlTimeEmbedding::init(
                topology.time_input_dim,
                topology.time_hidden_dim,
                device,
            ),
            added_conditioning: SdxlAddedConditioningProjection::init(
                topology.pooled_conditioning_dim,
                topology.time_ids_dim,
                topology.time_hidden_dim,
                device,
            ),
            down_blocks: topology
                .down_blocks
                .iter()
                .map(|spec| SdxlUnetStage::init(spec, topology.time_hidden_dim, device))
                .collect(),
            middle_blocks: topology
                .middle_blocks
                .iter()
                .map(|spec| SdxlUnetStage::init(spec, topology.time_hidden_dim, device))
                .collect(),
            up_blocks: topology
                .up_blocks
                .iter()
                .map(|spec| SdxlUnetStage::init(spec, topology.time_hidden_dim, device))
                .collect(),
            conv_out: Conv2dConfig::new(
                [topology.model_channels, topology.latent_channels],
                [3, 3],
            )
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device),
        }
    }

    #[cfg(test)]
    pub(crate) fn init_tiny(device: &B::Device) -> Self {
        Self::init(device)
    }

    pub fn forward(
        &self,
        latent: Tensor<B, 4>,
        timestep: Tensor<B, 1>,
        conditioning: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        let [batch, _, _, _] = latent.dims();
        let device = latent.device();
        let added_conditioning = SdxlAddedConditioning::new(
            Tensor::<B, 2>::zeros([batch, self.added_conditioning.pooled_dim()], &device),
            Tensor::<B, 2>::zeros([batch, self.added_conditioning.time_ids_dim()], &device),
        );
        self.forward_with_added_conditioning(latent, timestep, conditioning, added_conditioning)
    }

    pub fn forward_with_added_conditioning(
        &self,
        latent: Tensor<B, 4>,
        timestep: Tensor<B, 1>,
        conditioning: Tensor<B, 3>,
        added_conditioning: SdxlAddedConditioning<B>,
    ) -> Tensor<B, 4> {
        let [batch, _, _, _] = latent.dims();
        let timestep_embedding = sinusoidal_timestep_embedding(
            timestep,
            batch,
            self.time_embedding.input_dim(),
            &latent.device(),
        );
        let time_hidden = self.time_embedding.forward(timestep_embedding)
            + self.added_conditioning.forward(added_conditioning);
        let mut hidden = self.conv_in.forward(latent);
        for stage in &self.down_blocks {
            hidden = stage.forward(hidden, time_hidden.clone(), conditioning.clone());
        }
        for stage in &self.middle_blocks {
            hidden = stage.forward(hidden, time_hidden.clone(), conditioning.clone());
        }
        for stage in &self.up_blocks {
            hidden = stage.forward(hidden, time_hidden.clone(), conditioning.clone());
        }
        self.conv_out.forward(hidden)
    }

    pub fn input_block_count(&self) -> usize {
        self.stage_iter().map(|stage| stage.res_blocks.len()).sum()
    }

    pub fn attention_block_count(&self) -> usize {
        self.stage_iter()
            .map(|stage| stage.self_attention_blocks.len())
            .sum()
    }

    pub fn cross_attention_block_count(&self) -> usize {
        self.stage_iter()
            .map(|stage| stage.cross_attention_blocks.len())
            .sum()
    }

    pub fn cross_attention_context_dim(&self) -> Option<usize> {
        self.stage_iter()
            .flat_map(|stage| stage.cross_attention_blocks.iter())
            .next()
            .map(SdxlCrossAttentionBlock::context_dim)
    }

    pub fn time_embedding_dims(&self) -> [usize; 2] {
        self.time_embedding.dims()
    }

    pub fn added_conditioning_dims(&self) -> [usize; 3] {
        self.added_conditioning.dims()
    }

    pub fn down_block_count(&self) -> usize {
        self.down_blocks.len()
    }

    pub fn middle_block_count(&self) -> usize {
        self.middle_blocks.len()
    }

    pub fn up_block_count(&self) -> usize {
        self.up_blocks.len()
    }

    fn stage_iter(&self) -> impl Iterator<Item = &SdxlUnetStage<B>> {
        self.down_blocks
            .iter()
            .chain(self.middle_blocks.iter())
            .chain(self.up_blocks.iter())
    }
}

/// Burn-private SDXL added-conditioning tensors.
///
/// SDXL injects pooled text conditioning and six scalar time ids into the same
/// residual time path as the denoising timestep. Tiny fixtures keep the same
/// shape contract with a smaller pooled width.
#[derive(Debug, Clone)]
pub struct SdxlAddedConditioning<B: Backend> {
    pooled_text: Tensor<B, 2>,
    time_ids: Tensor<B, 2>,
}

impl<B: Backend> SdxlAddedConditioning<B> {
    pub fn new(pooled_text: Tensor<B, 2>, time_ids: Tensor<B, 2>) -> Self {
        Self {
            pooled_text,
            time_ids,
        }
    }
}

/// Burn-native projection from SDXL added-conditioning into time hidden width.
#[derive(Module, Debug)]
pub struct SdxlAddedConditioningProjection<B: Backend> {
    projection: Linear<B>,
    pooled_dim: usize,
    time_ids_dim: usize,
}

impl<B: Backend> SdxlAddedConditioningProjection<B> {
    pub fn init(
        pooled_dim: usize,
        time_ids_dim: usize,
        time_hidden_dim: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            projection: LinearConfig::new(pooled_dim + time_ids_dim, time_hidden_dim).init(device),
            pooled_dim,
            time_ids_dim,
        }
    }

    pub fn forward(&self, conditioning: SdxlAddedConditioning<B>) -> Tensor<B, 2> {
        let vector = Tensor::cat(vec![conditioning.pooled_text, conditioning.time_ids], 1);
        self.projection.forward(activation::silu(vector))
    }

    pub fn dims(&self) -> [usize; 3] {
        [
            self.pooled_dim(),
            self.time_ids_dim(),
            self.time_hidden_dim(),
        ]
    }

    fn pooled_dim(&self) -> usize {
        self.pooled_dim
    }

    fn time_ids_dim(&self) -> usize {
        self.time_ids_dim
    }

    fn time_hidden_dim(&self) -> usize {
        self.projection.weight.dims()[1]
    }
}

fn sinusoidal_timestep_embedding<B: Backend>(
    timestep: Tensor<B, 1>,
    batch: usize,
    width: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let [timestep_batch] = timestep.dims();
    let timestep = match timestep_batch {
        len if len == batch => timestep,
        1 => timestep.repeat_dim(0, batch),
        len => panic!("UNet timestep batch {len} does not match latent batch {batch}"),
    };
    let half = width / 2;
    let positions = Tensor::<B, 1, Int>::arange(0..half as i64, device).float();
    let exponent = positions * (-(10_000.0_f32).ln() / half as f32);
    let frequencies = exponent.exp().reshape([1, half]);
    let args = timestep.reshape([batch, 1]) * frequencies;
    let embedding = Tensor::cat(vec![args.clone().cos(), args.sin()], 1);

    if width.is_multiple_of(2) {
        embedding
    } else {
        Tensor::cat(vec![embedding, Tensor::zeros([batch, 1], device)], 1)
    }
}

/// Burn-native SDXL UNet stage scaffold.
#[derive(Module, Debug)]
pub struct SdxlUnetStage<B: Backend> {
    res_blocks: Vec<SdxlResBlock<B>>,
    self_attention_blocks: Vec<SdxlSelfAttentionBlock<B>>,
    cross_attention_blocks: Vec<SdxlCrossAttentionBlock<B>>,
}

impl<B: Backend> SdxlUnetStage<B> {
    pub fn init(spec: &SdxlStageSpec, time_hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            res_blocks: spec
                .res_blocks
                .iter()
                .map(|spec| {
                    SdxlResBlock::init_with_time_dim(
                        spec.in_channels,
                        spec.out_channels,
                        time_hidden_dim,
                        spec.num_groups,
                        device,
                    )
                })
                .collect(),
            self_attention_blocks: spec
                .self_attention_blocks
                .iter()
                .map(|spec| {
                    SdxlSelfAttentionBlock::init(
                        spec.channels,
                        spec.num_heads,
                        spec.num_groups,
                        device,
                    )
                })
                .collect(),
            cross_attention_blocks: spec
                .cross_attention_blocks
                .iter()
                .map(|spec| {
                    SdxlCrossAttentionBlock::init(
                        spec.channels,
                        spec.context_dim,
                        spec.num_heads,
                        spec.num_groups,
                        device,
                    )
                })
                .collect(),
        }
    }

    pub fn forward(
        &self,
        mut hidden: Tensor<B, 4>,
        time_hidden: Tensor<B, 2>,
        conditioning: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        for block in &self.res_blocks {
            hidden = block.forward_with_time(hidden, time_hidden.clone());
        }
        for block in &self.self_attention_blocks {
            hidden = block.forward(hidden);
        }
        for block in &self.cross_attention_blocks {
            hidden = block.forward(hidden, conditioning.clone());
        }
        hidden
    }
}

/// Burn-native SDXL time embedding MLP scaffold.
#[derive(Module, Debug)]
pub struct SdxlTimeEmbedding<B: Backend> {
    pub linear_1: Linear<B>,
    pub linear_2: Linear<B>,
}

impl<B: Backend> SdxlTimeEmbedding<B> {
    pub fn init(input_dim: usize, hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            linear_1: LinearConfig::new(input_dim, hidden_dim).init(device),
            linear_2: LinearConfig::new(hidden_dim, hidden_dim).init(device),
        }
    }

    pub fn forward(&self, embedding: Tensor<B, 2>) -> Tensor<B, 2> {
        let hidden = activation::silu(self.linear_1.forward(embedding));
        self.linear_2.forward(hidden)
    }

    pub fn dims(&self) -> [usize; 2] {
        [self.input_dim(), self.hidden_dim()]
    }

    fn input_dim(&self) -> usize {
        self.linear_1.weight.dims()[0]
    }

    fn hidden_dim(&self) -> usize {
        self.linear_2.weight.dims()[1]
    }
}

/// Burn-native SDXL residual block scaffold.
#[derive(Module, Debug)]
pub struct SdxlResBlock<B: Backend> {
    pub norm_1: GroupNorm<B>,
    pub conv_1: Conv2d<B>,
    pub time_projection: Linear<B>,
    pub norm_2: GroupNorm<B>,
    pub conv_2: Conv2d<B>,
    skip: Option<Conv2d<B>>,
}

impl<B: Backend> SdxlResBlock<B> {
    pub fn init(
        in_channels: usize,
        out_channels: usize,
        num_groups: usize,
        device: &B::Device,
    ) -> Self {
        Self::init_with_time_dim(in_channels, out_channels, out_channels, num_groups, device)
    }

    pub fn init_with_time_dim(
        in_channels: usize,
        out_channels: usize,
        time_dim: usize,
        num_groups: usize,
        device: &B::Device,
    ) -> Self {
        let conv3 = |channels: [usize; 2]| {
            Conv2dConfig::new(channels, [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device)
        };
        let skip = (in_channels != out_channels)
            .then(|| Conv2dConfig::new([in_channels, out_channels], [1, 1]).init(device));

        Self {
            norm_1: GroupNormConfig::new(num_groups, in_channels).init(device),
            conv_1: conv3([in_channels, out_channels]),
            time_projection: LinearConfig::new(time_dim, out_channels).init(device),
            norm_2: GroupNormConfig::new(num_groups, out_channels).init(device),
            conv_2: conv3([out_channels, out_channels]),
            skip,
        }
    }

    pub fn forward(&self, hidden: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, _, _, _] = hidden.dims();
        let time_hidden = Tensor::<B, 2>::zeros([batch, self.time_dim()], &hidden.device());
        self.forward_with_time(hidden, time_hidden)
    }

    pub fn forward_with_time(
        &self,
        hidden: Tensor<B, 4>,
        time_hidden: Tensor<B, 2>,
    ) -> Tensor<B, 4> {
        let residual = match &self.skip {
            Some(skip) => skip.forward(hidden.clone()),
            None => hidden.clone(),
        };
        let hidden = self
            .conv_1
            .forward(activation::silu(self.norm_1.forward(hidden)));
        let [batch, channels, _, _] = hidden.dims();
        let time_hidden = self
            .time_projection
            .forward(activation::silu(time_hidden))
            .reshape([batch, channels, 1, 1]);
        let hidden = hidden + time_hidden;
        let hidden = self
            .conv_2
            .forward(activation::silu(self.norm_2.forward(hidden)));
        hidden + residual
    }

    pub fn uses_skip_projection(&self) -> bool {
        self.skip.is_some()
    }

    pub fn uses_time_projection(&self) -> bool {
        true
    }

    fn time_dim(&self) -> usize {
        self.time_projection.weight.dims()[0]
    }
}

/// Burn-native SDXL spatial self-attention scaffold.
#[derive(Module, Debug)]
pub struct SdxlSelfAttentionBlock<B: Backend> {
    pub norm: GroupNorm<B>,
    pub attention: MultiHeadAttention<B>,
}

impl<B: Backend> SdxlSelfAttentionBlock<B> {
    pub fn init(channels: usize, num_heads: usize, num_groups: usize, device: &B::Device) -> Self {
        Self {
            norm: GroupNormConfig::new(num_groups, channels).init(device),
            attention: MultiHeadAttentionConfig::new(channels, num_heads)
                .with_dropout(0.0)
                .init(device),
        }
    }

    pub fn forward(&self, hidden: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = hidden.clone();
        let [batch, channels, height, width] = hidden.dims();
        let hidden = self
            .norm
            .forward(hidden)
            .swap_dims(1, 2)
            .swap_dims(2, 3)
            .reshape([batch, height * width, channels]);
        let hidden = self
            .attention
            .forward(MhaInput::self_attn(hidden))
            .context
            .reshape([batch, height, width, channels])
            .swap_dims(2, 3)
            .swap_dims(1, 2);

        hidden + residual
    }

    pub fn head_count(&self) -> usize {
        self.attention.n_heads
    }
}

/// Burn-native SDXL cross-attention scaffold.
#[derive(Module, Debug)]
pub struct SdxlCrossAttentionBlock<B: Backend> {
    pub norm: GroupNorm<B>,
    pub context_key: Linear<B>,
    pub context_value: Linear<B>,
    pub attention: MultiHeadAttention<B>,
}

impl<B: Backend> SdxlCrossAttentionBlock<B> {
    pub fn init(
        channels: usize,
        context_dim: usize,
        num_heads: usize,
        num_groups: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            norm: GroupNormConfig::new(num_groups, channels).init(device),
            context_key: LinearConfig::new(context_dim, channels).init(device),
            context_value: LinearConfig::new(context_dim, channels).init(device),
            attention: MultiHeadAttentionConfig::new(channels, num_heads)
                .with_dropout(0.0)
                .init(device),
        }
    }

    pub fn forward(&self, hidden: Tensor<B, 4>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let residual = hidden.clone();
        let [batch, channels, height, width] = hidden.dims();
        let query = self
            .norm
            .forward(hidden)
            .swap_dims(1, 2)
            .swap_dims(2, 3)
            .reshape([batch, height * width, channels]);
        let key = self.context_key.forward(context.clone());
        let value = self.context_value.forward(context);
        let hidden = self
            .attention
            .forward(MhaInput::new(query, key, value))
            .context
            .reshape([batch, height, width, channels])
            .swap_dims(2, 3)
            .swap_dims(1, 2);

        hidden + residual
    }

    pub fn context_dim(&self) -> usize {
        self.context_key.weight.dims()[0]
    }
}

/// Weight data buffer — pre-allocated f32 values from safetensors.
#[derive(Debug, Clone)]
pub struct DiffusionWeightData {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// Complete set of SDXL UNet weights loaded from the diffusion
/// component safetensors file.
///
/// V1 captures the key tensor families needed for the euler/normal
/// sampling loop. The struct is deliberately flat; a full UNet module
/// graph is deferred to when Burn's `#[derive(Module)]` supports
/// `B: Backend` with Vec fields.
#[derive(Debug, Clone)]
pub struct DiffusionUNetWeights {
    pub conv_in_weight: DiffusionWeightData,
    pub conv_in_bias: DiffusionWeightData,
    pub time_embed_0_weight: DiffusionWeightData,
    pub time_embed_0_bias: DiffusionWeightData,
    pub time_embed_2_weight: DiffusionWeightData,
    pub time_embed_2_bias: DiffusionWeightData,
    // Input blocks (down-sampling)
    pub input_blocks: Vec<DiffusionBlockWeights>,
    // Middle block
    pub middle_block: Option<DiffusionBlockWeights>,
    // Output blocks (up-sampling)
    pub output_blocks: Vec<DiffusionBlockWeights>,
    pub out_0_weight: DiffusionWeightData,
    pub out_0_bias: DiffusionWeightData,
}

/// Weights for one diffusion block (input, middle, or output).
#[derive(Debug, Clone)]
pub struct DiffusionBlockWeights {
    pub conv_weight: DiffusionWeightData,
    pub conv_bias: DiffusionWeightData,
    // Optional attention/transformer weights
    pub attn_q_weight: Option<DiffusionWeightData>,
    pub attn_k_weight: Option<DiffusionWeightData>,
    pub attn_v_weight: Option<DiffusionWeightData>,
    pub attn_out_weight: Option<DiffusionWeightData>,
}

#[cfg(test)]
mod tests {
    use burn_core::module::Module;
    use burn_store::{ModuleSnapshot, SafetensorsStore};
    use burn_tensor::Tensor;

    use crate::active_backend::{ActiveBurnBackend, active_device};
    use crate::config::BurnBackendConfig;
    use crate::runtime::BurnRuntime;

    #[test]
    fn sdxl_unet_module_forward_preserves_latent_noise_shape() {
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let module = super::SdxlUnet::<ActiveBurnBackend>::init_tiny(runtime.device());
        let latent = Tensor::<ActiveBurnBackend, 4>::zeros([2, 4, 8, 8], runtime.device());
        let timestep = Tensor::<ActiveBurnBackend, 1>::zeros([2], runtime.device());
        let conditioning = Tensor::<ActiveBurnBackend, 3>::zeros([2, 3, 16], runtime.device());

        let output = module.forward(latent, timestep, conditioning);

        assert_eq!(output.dims(), [2, 4, 8, 8]);
    }

    #[test]
    fn sdxl_unet_forward_uses_supplied_timestep() {
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let module = super::SdxlUnet::<ActiveBurnBackend>::init_tiny(runtime.device());
        let latent = Tensor::<ActiveBurnBackend, 4>::zeros([1, 4, 8, 8], runtime.device());
        let conditioning = Tensor::<ActiveBurnBackend, 3>::zeros([1, 3, 16], runtime.device());
        let output_a = module.forward(
            latent.clone(),
            Tensor::<ActiveBurnBackend, 1>::zeros([1], runtime.device()),
            conditioning.clone(),
        );
        let output_b = module.forward(
            latent,
            Tensor::<ActiveBurnBackend, 1>::ones([1], runtime.device()),
            conditioning,
        );

        let values_a = output_a.to_data().to_vec::<f32>().expect("output a");
        let values_b = output_b.to_data().to_vec::<f32>().expect("output b");

        assert_ne!(values_a, values_b);
    }

    #[test]
    fn sdxl_unet_forward_uses_added_conditioning_projection() {
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let module = super::SdxlUnet::<ActiveBurnBackend>::init_tiny(runtime.device());
        let latent = Tensor::<ActiveBurnBackend, 4>::zeros([1, 4, 8, 8], runtime.device());
        let timestep = Tensor::<ActiveBurnBackend, 1>::zeros([1], runtime.device());
        let conditioning = Tensor::<ActiveBurnBackend, 3>::zeros([1, 3, 16], runtime.device());
        let output_a = module.forward_with_added_conditioning(
            latent.clone(),
            timestep.clone(),
            conditioning.clone(),
            super::SdxlAddedConditioning::new(
                Tensor::<ActiveBurnBackend, 2>::zeros([1, 8], runtime.device()),
                Tensor::<ActiveBurnBackend, 2>::zeros([1, 6], runtime.device()),
            ),
        );
        let output_b = module.forward_with_added_conditioning(
            latent,
            timestep,
            conditioning,
            super::SdxlAddedConditioning::new(
                Tensor::<ActiveBurnBackend, 2>::ones([1, 8], runtime.device()),
                Tensor::<ActiveBurnBackend, 2>::ones([1, 6], runtime.device()),
            ),
        );

        let values_a = output_a.to_data().to_vec::<f32>().expect("output a");
        let values_b = output_b.to_data().to_vec::<f32>().expect("output b");

        assert_ne!(values_a, values_b);
    }

    #[test]
    fn sdxl_unet_module_saves_and_loads_through_burn_store() {
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let module = super::SdxlUnet::<ActiveBurnBackend>::init_tiny(runtime.device());
        let mut save_store = SafetensorsStore::from_bytes(None);
        module
            .save_into(&mut save_store)
            .expect("UNet skeleton should save into burn-store");
        let bytes = save_store
            .get_bytes()
            .expect("saved UNet bytes should be readable");
        let mut load_store = SafetensorsStore::from_bytes(Some(bytes));
        let mut loaded = super::SdxlUnet::<ActiveBurnBackend>::init_tiny(runtime.device());

        let result = runtime
            .load_module_store(&mut loaded, &mut load_store)
            .expect("UNet skeleton should load from burn-store");

        assert!(result.errors.is_empty(), "unexpected load errors: {result}");
        assert_eq!(module.num_params(), loaded.num_params());
    }

    #[test]
    fn sdxl_resblock_preserves_spatial_shape_and_projects_channels() {
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let block = super::SdxlResBlock::<ActiveBurnBackend>::init(4, 8, 2, runtime.device());
        let hidden = Tensor::<ActiveBurnBackend, 4>::zeros([2, 4, 8, 8], runtime.device());

        let output = block.forward(hidden);

        assert_eq!(output.dims(), [2, 8, 8, 8]);
        assert!(block.uses_skip_projection());
    }

    #[test]
    fn sdxl_resblock_projects_time_embedding_into_hidden_channels() {
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let block = super::SdxlResBlock::<ActiveBurnBackend>::init(4, 8, 2, runtime.device());
        let hidden = Tensor::<ActiveBurnBackend, 4>::zeros([2, 4, 8, 8], runtime.device());
        let time_hidden = Tensor::<ActiveBurnBackend, 2>::zeros([2, 8], runtime.device());

        let output = block.forward_with_time(hidden, time_hidden);

        assert_eq!(output.dims(), [2, 8, 8, 8]);
        assert!(block.uses_time_projection());
    }

    #[test]
    fn sdxl_self_attention_block_preserves_spatial_shape() {
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let block =
            super::SdxlSelfAttentionBlock::<ActiveBurnBackend>::init(4, 2, 2, runtime.device());
        let hidden = Tensor::<ActiveBurnBackend, 4>::zeros([2, 4, 3, 5], runtime.device());

        let output = block.forward(hidden);

        assert_eq!(output.dims(), [2, 4, 3, 5]);
        assert_eq!(block.head_count(), 2);
    }

    #[test]
    fn sdxl_cross_attention_block_projects_text_context() {
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let block = super::SdxlCrossAttentionBlock::<ActiveBurnBackend>::init(
            4,
            16,
            2,
            2,
            runtime.device(),
        );
        let hidden = Tensor::<ActiveBurnBackend, 4>::zeros([2, 4, 3, 5], runtime.device());
        let context = Tensor::<ActiveBurnBackend, 3>::zeros([2, 7, 16], runtime.device());

        let output = block.forward(hidden, context);

        assert_eq!(output.dims(), [2, 4, 3, 5]);
        assert_eq!(block.context_dim(), 16);
    }

    #[test]
    fn sdxl_unet_scaffold_contains_burn_native_resblocks() {
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let module = super::SdxlUnet::<ActiveBurnBackend>::init_tiny(runtime.device());

        assert_eq!(module.input_block_count(), 1);
    }

    #[test]
    fn sdxl_unet_scaffold_contains_burn_native_attention_blocks() {
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let module = super::SdxlUnet::<ActiveBurnBackend>::init_tiny(runtime.device());

        assert_eq!(module.attention_block_count(), 1);
    }

    #[test]
    fn sdxl_unet_scaffold_contains_burn_native_cross_attention_blocks() {
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let module = super::SdxlUnet::<ActiveBurnBackend>::init_tiny(runtime.device());

        assert_eq!(module.cross_attention_block_count(), 1);
    }

    #[test]
    fn sdxl_unet_scaffold_initializes_from_topology_profile() {
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let topology = super::SdxlUnetTopology::tiny();
        let module =
            super::SdxlUnet::<ActiveBurnBackend>::init_from_topology(&topology, runtime.device());

        assert_eq!(topology.latent_channels, 4);
        assert_eq!(module.down_block_count(), 1);
        assert_eq!(module.middle_block_count(), 0);
        assert_eq!(module.up_block_count(), 0);
        assert_eq!(module.input_block_count(), topology.res_block_count());
        assert_eq!(
            module.cross_attention_context_dim(),
            Some(topology.conditioning_dim)
        );
    }

    #[test]
    fn sdxl_unet_topology_profiles_have_stable_names() {
        assert_eq!(
            super::SdxlUnetTopologyProfile::TinySdxlE2e.as_str(),
            "tiny_sdxl_e2e"
        );
        assert_eq!(
            super::SdxlUnetTopologyProfile::SdxlBase.as_str(),
            "sdxl_base"
        );
        assert!(super::SdxlUnetTopologyProfile::TinySdxlE2e.is_module_graph_supported());
        assert!(!super::SdxlUnetTopologyProfile::SdxlBase.is_module_graph_supported());

        let tiny = super::SdxlUnetTopologyProfile::TinySdxlE2e.topology();
        assert_eq!(tiny.name(), "tiny_sdxl_e2e");
        assert_eq!(tiny.conditioning_dim, 16);
        assert_eq!(tiny.pooled_conditioning_dim, 8);
        assert_eq!(tiny.time_ids_dim, 6);

        let full = super::SdxlUnetTopologyProfile::SdxlBase.topology();
        assert_eq!(full.name(), "sdxl_base");
        assert_eq!(full.conditioning_dim, 2048);
        assert_eq!(full.pooled_conditioning_dim, 1280);
        assert_eq!(full.time_ids_dim, 6);
    }

    #[test]
    fn sdxl_base_topology_records_down_middle_up_stage_shape() {
        let topology = super::SdxlUnetTopology::sdxl_base();

        assert_eq!(topology.name(), "sdxl_base");
        assert_eq!(topology.latent_channels, 4);
        assert_eq!(topology.model_channels, 320);
        assert_eq!(topology.time_hidden_dim, 1280);
        assert_eq!(topology.conditioning_dim, 2048);
        assert_eq!(topology.pooled_conditioning_dim, 1280);
        assert_eq!(topology.time_ids_dim, 6);
        assert_eq!(topology.down_blocks.len(), 3);
        assert_eq!(topology.middle_blocks.len(), 1);
        assert_eq!(topology.up_blocks.len(), 3);
        assert!(topology.res_block_count() > 0);
        assert!(topology.cross_attention_block_count() > 0);
    }

    #[test]
    fn sdxl_time_embedding_projects_to_hidden_width() {
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let module = super::SdxlTimeEmbedding::<ActiveBurnBackend>::init(4, 8, runtime.device());
        let embedding = Tensor::<ActiveBurnBackend, 2>::zeros([2, 4], runtime.device());

        let output = module.forward(embedding);

        assert_eq!(output.dims(), [2, 8]);
    }

    #[test]
    fn sdxl_unet_scaffold_contains_burn_native_time_embedding() {
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let module = super::SdxlUnet::<ActiveBurnBackend>::init_tiny(runtime.device());

        assert_eq!(module.time_embedding_dims(), [4, 8]);
    }

    #[test]
    fn sdxl_unet_scaffold_contains_burn_native_added_conditioning_projection() {
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let module = super::SdxlUnet::<ActiveBurnBackend>::init_tiny(runtime.device());

        assert_eq!(module.added_conditioning_dims(), [8, 6, 8]);
    }
}
