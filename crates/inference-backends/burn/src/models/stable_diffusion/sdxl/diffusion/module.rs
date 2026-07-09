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

use super::diffusers_blocks::{SdxlDownsample2d, SdxlSpatialTransformer, SdxlUpsample2d};

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
    /// Diffusers `mid_block` (single object, not a list).
    pub mid_block: Option<SdxlMidBlockSpec>,
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
        matches!(self, Self::TinySdxlE2e | Self::SdxlBase)
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
                role: SdxlStageRole::Down,
                resnets: vec![SdxlResBlockSpec {
                    in_channels: 4,
                    out_channels: 4,
                    num_groups: 2,
                }],
                self_attn_blocks: vec![SdxlSelfAttentionBlockSpec {
                    channels: 4,
                    num_heads: 2,
                    num_groups: 2,
                }],
                cross_attn_blocks: vec![SdxlCrossAttentionBlockSpec {
                    channels: 4,
                    context_dim: 16,
                    num_heads: 2,
                    num_groups: 2,
                }],
                attentions: Vec::new(),
                skip_policy: SdxlSkipPolicy::None,
                sampling: SdxlSamplingOp::None,
            }],
            mid_block: None,
            up_blocks: Vec::new(),
        }
    }

    pub fn sdxl_base() -> Self {
        let res = |in_channels, out_channels| SdxlResBlockSpec {
            in_channels,
            out_channels,
            num_groups: 32,
        };
        let attn = |channels, heads, layers| SdxlSpatialTransformerSpec {
            channels,
            context_dim: 2048,
            num_heads: heads,
            num_layers: layers,
            num_groups: 32,
        };
        let down = |in_ch, out_ch, heads, layers, has_ds| SdxlStageSpec {
            role: SdxlStageRole::Down,
            resnets: vec![res(in_ch, out_ch), res(out_ch, out_ch)],
            self_attn_blocks: Vec::new(),
            cross_attn_blocks: Vec::new(),
            attentions: if layers == 0 {
                Vec::new()
            } else {
                vec![attn(out_ch, heads, layers), attn(out_ch, heads, layers)]
            },
            skip_policy: SdxlSkipPolicy::Push,
            sampling: if has_ds {
                SdxlSamplingOp::Downsample2d
            } else {
                SdxlSamplingOp::None
            },
        };
        let mid = SdxlMidBlockSpec {
            resnets: vec![res(1280, 1280), res(1280, 1280)],
            attentions: vec![attn(1280, 20, 10)],
        };
        let up = |prev, out, skip_chs: &[usize], heads, layers, has_us| SdxlStageSpec {
            role: SdxlStageRole::Up,
            resnets: skip_chs
                .iter()
                .enumerate()
                .map(|(index, &sk)| {
                    let base = if index == 0 { prev } else { out };
                    res(base + sk, out)
                })
                .collect(),
            self_attn_blocks: Vec::new(),
            cross_attn_blocks: Vec::new(),
            attentions: if layers == 0 {
                Vec::new()
            } else {
                vec![
                    attn(out, heads, layers),
                    attn(out, heads, layers),
                    attn(out, heads, layers),
                ]
            },
            skip_policy: SdxlSkipPolicy::Pop,
            sampling: if has_us {
                SdxlSamplingOp::Upsample2d
            } else {
                SdxlSamplingOp::None
            },
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
            down_blocks: vec![
                down(320, 320, 5, 0, true),
                down(320, 640, 10, 2, true),
                down(640, 1280, 20, 10, false),
            ],
            mid_block: Some(mid),
            up_blocks: vec![
                up(1280, 1280, &[1280, 1280, 640], 20, 10, true),
                up(1280, 640, &[640, 640, 320], 10, 2, true),
                up(640, 320, &[320, 320, 320], 5, 0, false),
            ],
        }
    }

    pub fn profile(&self) -> SdxlUnetTopologyProfile {
        self.profile
    }

    pub fn name(&self) -> &'static str {
        self.profile.as_str()
    }

    pub fn res_block_count(&self) -> usize {
        self.down_blocks
            .iter()
            .map(|stage| stage.resnets.len())
            .chain(
                self.mid_block
                    .iter()
                    .map(|mid| mid.resnets.len()),
            )
            .chain(self.up_blocks.iter().map(|stage| stage.resnets.len()))
            .sum()
    }

    pub fn self_attention_block_count(&self) -> usize {
        self.down_blocks
            .iter()
            .chain(self.up_blocks.iter())
            .map(|stage| {
                stage.self_attn_blocks.len()
                    + stage
                        .attentions
                        .iter()
                        .map(|a| a.num_layers)
                        .sum::<usize>()
            })
            .sum::<usize>()
            + self
                .mid_block
                .as_ref()
                .map(|mid| {
                    mid.attentions
                        .iter()
                        .map(|a| a.num_layers)
                        .sum::<usize>()
                })
                .unwrap_or(0)
    }

    pub fn cross_attention_block_count(&self) -> usize {
        // Diffusers spatial transformers each contain cross-attn (attn2).
        self.down_blocks
            .iter()
            .chain(self.up_blocks.iter())
            .map(|stage| {
                stage.cross_attn_blocks.len()
                    + stage
                        .attentions
                        .iter()
                        .map(|a| a.num_layers)
                        .sum::<usize>()
            })
            .sum::<usize>()
            + self
                .mid_block
                .as_ref()
                .map(|mid| {
                    mid.attentions
                        .iter()
                        .map(|a| a.num_layers)
                        .sum::<usize>()
                })
                .unwrap_or(0)
    }

    pub fn spatial_transformer_count(&self) -> usize {
        self.down_blocks
            .iter()
            .map(|stage| stage.attentions.len())
            .chain(
                self.mid_block
                    .iter()
                    .map(|mid| mid.attentions.len()),
            )
            .chain(self.up_blocks.iter().map(|stage| stage.attentions.len()))
            .sum()
    }
}

#[derive(Debug, Clone)]
pub struct SdxlStageSpec {
    pub role: SdxlStageRole,
    pub resnets: Vec<SdxlResBlockSpec>,
    /// Legacy tiny path (non-diffusers). Empty when `attentions` is used.
    pub self_attn_blocks: Vec<SdxlSelfAttentionBlockSpec>,
    pub cross_attn_blocks: Vec<SdxlCrossAttentionBlockSpec>,
    /// Diffusers `attentions.N` spatial transformers.
    pub attentions: Vec<SdxlSpatialTransformerSpec>,
    pub skip_policy: SdxlSkipPolicy,
    pub sampling: SdxlSamplingOp,
}

/// Diffusers `mid_block` (resnets + attentions, no skip/sampling).
#[derive(Debug, Clone)]
pub struct SdxlMidBlockSpec {
    pub resnets: Vec<SdxlResBlockSpec>,
    pub attentions: Vec<SdxlSpatialTransformerSpec>,
}

#[derive(Debug, Clone)]
pub struct SdxlSpatialTransformerSpec {
    pub channels: usize,
    pub context_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub num_groups: usize,
}

impl SdxlStageSpec {
    pub fn role(&self) -> SdxlStageRole {
        self.role
    }

    pub fn pushes_skip(&self) -> bool {
        matches!(self.skip_policy, SdxlSkipPolicy::Push)
    }

    pub fn pops_skip(&self) -> bool {
        matches!(self.skip_policy, SdxlSkipPolicy::Pop)
    }

    pub fn has_downsample(&self) -> bool {
        matches!(self.sampling, SdxlSamplingOp::Downsample2d)
    }

    pub fn has_upsample(&self) -> bool {
        matches!(self.sampling, SdxlSamplingOp::Upsample2d)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdxlStageRole {
    Down,
    Middle,
    Up,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdxlSkipPolicy {
    None,
    Push,
    Pop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdxlSamplingOp {
    None,
    Downsample2d,
    Upsample2d,
}

#[derive(Debug, Clone)]
pub struct SdxlResBlockSpec {
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_groups: usize,
}

#[derive(Debug, Clone)]
pub struct SdxlSelfAttentionBlockSpec {
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
/// Snapshot keyspace follows the package/diffusers dialect:
/// `down_blocks`, `mid_block`, `up_blocks`, `time_embedding`, `conv_norm_out`.
#[derive(Module, Debug)]
pub struct SdxlUnet<B: Backend> {
    pub conv_in: Conv2d<B>,
    time_embedding: SdxlTimeEmbedding<B>,
    added_conditioning: SdxlAddedConditioningProjection<B>,
    down_blocks: Vec<SdxlUnetStage<B>>,
    mid_block: Option<SdxlMidBlock<B>>,
    up_blocks: Vec<SdxlUnetStage<B>>,
    conv_norm_out: GroupNorm<B>,
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
            mid_block: topology.mid_block.as_ref().map(|spec| {
                SdxlMidBlock::init(spec, topology.time_hidden_dim, device)
            }),
            up_blocks: topology
                .up_blocks
                .iter()
                .map(|spec| SdxlUnetStage::init(spec, topology.time_hidden_dim, device))
                .collect(),
            conv_norm_out: GroupNormConfig::new(32.min(topology.model_channels), topology.model_channels)
                .init(device),
            conv_out: Conv2dConfig::new(
                [topology.model_channels, topology.latent_channels],
                [3, 3],
            )
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device),
        }
    }

    #[cfg(test)]
    pub(crate) fn topology_profile(&self) -> SdxlUnetTopologyProfile {
        if self.down_blocks.len() == 3 && self.mid_block.is_some() && self.up_blocks.len() == 3
        {
            SdxlUnetTopologyProfile::SdxlBase
        } else {
            SdxlUnetTopologyProfile::TinySdxlE2e
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
        // conv_in output feeds into the skip stack — it is the first residual
        // contribution the up path will receive when it first pops a skip tensor.
        let mut skip_stack = vec![hidden.clone()];
        for stage in &self.down_blocks {
            let output = stage.forward(hidden, time_hidden.clone(), conditioning.clone());
            hidden = output.hidden;
            skip_stack.extend(output.skips);
        }
        if let Some(mid) = &self.mid_block {
            hidden = mid.forward(hidden, time_hidden.clone(), conditioning.clone());
        }
        for stage in &self.up_blocks {
            hidden = stage.forward_up(
                hidden,
                time_hidden.clone(),
                conditioning.clone(),
                &mut skip_stack,
            );
        }
        let hidden = activation::silu(self.conv_norm_out.forward(hidden));
        self.conv_out.forward(hidden)
    }

    pub fn input_block_count(&self) -> usize {
        self.stage_iter().map(|stage| stage.resnets.len()).sum::<usize>()
            + self
                .mid_block
                .as_ref()
                .map(|mid| mid.resnets.len())
                .unwrap_or(0)
    }

    pub fn attention_block_count(&self) -> usize {
        self.stage_iter()
            .map(|stage| stage.self_attn_blocks.len())
            .sum()
    }

    pub fn cross_attention_block_count(&self) -> usize {
        self.stage_iter()
            .map(|stage| stage.cross_attn_blocks.len())
            .sum()
    }

    pub fn cross_attention_context_dim(&self) -> Option<usize> {
        self.stage_iter()
            .flat_map(|stage| stage.cross_attn_blocks.iter())
            .next()
            .map(SdxlCrossAttentionBlock::context_dim)
            .or_else(|| {
                self.stage_iter()
                    .flat_map(|stage| stage.attentions.iter())
                    .next()
                    .map(SdxlSpatialTransformer::context_dim)
            })
            .or_else(|| {
                self.mid_block
                    .as_ref()
                    .and_then(|mid| mid.attentions.first())
                    .map(SdxlSpatialTransformer::context_dim)
            })
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
        usize::from(self.mid_block.is_some())
    }

    pub fn up_block_count(&self) -> usize {
        self.up_blocks.len()
    }

    pub fn down_path_pushes_skip(&self) -> bool {
        self.down_blocks.iter().any(SdxlUnetStage::pushes_skip)
    }

    pub fn down_path_has_downsample(&self) -> bool {
        self.down_blocks.iter().any(SdxlUnetStage::has_downsample)
    }

    pub fn middle_path_has_no_skip_mutation(&self) -> bool {
        true
    }

    pub fn up_path_pops_skip(&self) -> bool {
        self.up_blocks.iter().any(SdxlUnetStage::pops_skip)
    }

    pub fn up_path_has_upsample(&self) -> bool {
        self.up_blocks.iter().any(SdxlUnetStage::has_upsample)
    }

    fn stage_iter(&self) -> impl Iterator<Item = &SdxlUnetStage<B>> {
        self.down_blocks.iter().chain(self.up_blocks.iter())
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

    pub(crate) fn shapes(&self) -> [[usize; 2]; 2] {
        [self.pooled_text.dims(), self.time_ids.dims()]
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
    role: SdxlStageRole,
    resnets: Vec<SdxlResBlock<B>>,
    self_attn_blocks: Vec<SdxlSelfAttentionBlock<B>>,
    cross_attn_blocks: Vec<SdxlCrossAttentionBlock<B>>,
    attentions: Vec<SdxlSpatialTransformer<B>>,
    skip_policy: SdxlSkipPolicy,
    sampling: SdxlSamplingOp,
    downsamplers: Vec<SdxlDownsample2d<B>>,
    upsamplers: Vec<SdxlUpsample2d<B>>,
}

impl<B: Backend> SdxlUnetStage<B> {
    pub fn init(spec: &SdxlStageSpec, time_hidden_dim: usize, device: &B::Device) -> Self {
        let output_channels = spec
            .resnets
            .last()
            .map(|block| block.out_channels)
            .expect("SDXL UNet stage requires at least one residual block");
        let downsamplers = if spec.has_downsample() {
            vec![SdxlDownsample2d::init(output_channels, device)]
        } else {
            Vec::new()
        };
        let upsamplers = if spec.has_upsample() {
            vec![SdxlUpsample2d::init(output_channels, device)]
        } else {
            Vec::new()
        };
        Self {
            role: spec.role,
            resnets: spec
                .resnets
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
            self_attn_blocks: spec
                .self_attn_blocks
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
            cross_attn_blocks: spec
                .cross_attn_blocks
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
            attentions: spec
                .attentions
                .iter()
                .map(|spec| {
                    SdxlSpatialTransformer::init(
                        spec.channels,
                        spec.context_dim,
                        spec.num_heads,
                        spec.num_layers,
                        spec.num_groups,
                        device,
                    )
                })
                .collect(),
            skip_policy: spec.skip_policy,
            sampling: spec.sampling,
            downsamplers,
            upsamplers,
        }
    }

    fn forward(
        &self,
        mut hidden: Tensor<B, 4>,
        time_hidden: Tensor<B, 2>,
        conditioning: Tensor<B, 3>,
    ) -> SdxlStageOutput<B> {
        let mut skips = Vec::new();
        // Diffusers CrossAttnDownBlock2D interleaves resnet[i] then attention[i]
        // before optional downsample. When attentions is empty, only resnets run.
        if self.attentions.is_empty() {
            for block in &self.resnets {
                hidden = block.forward_with_time(hidden, time_hidden.clone());
                if self.pushes_skip() {
                    skips.push(hidden.clone());
                }
            }
            for block in &self.self_attn_blocks {
                hidden = block.forward(hidden);
            }
            for block in &self.cross_attn_blocks {
                hidden = block.forward(hidden, conditioning.clone());
            }
        } else {
            assert_eq!(
                self.resnets.len(),
                self.attentions.len(),
                "diffusers down/up stage resnets and attentions must align"
            );
            for (resnet, attention) in self.resnets.iter().zip(self.attentions.iter()) {
                hidden = resnet.forward_with_time(hidden, time_hidden.clone());
                hidden = attention.forward(hidden, conditioning.clone());
                // Diffusers emits skip after resnet+attention pair.
                if self.pushes_skip() {
                    skips.push(hidden.clone());
                }
            }
        }
        let hidden = if let Some(downsample) = self.downsamplers.first() {
            let hidden = downsample.forward(hidden);
            if self.pushes_skip() {
                skips.push(hidden.clone());
            }
            hidden
        } else {
            hidden
        };
        SdxlStageOutput { hidden, skips }
    }

    pub fn forward_up(
        &self,
        hidden: Tensor<B, 4>,
        time_hidden: Tensor<B, 2>,
        conditioning: Tensor<B, 3>,
        skip_stack: &mut Vec<Tensor<B, 4>>,
    ) -> Tensor<B, 4> {
        let mut hidden = hidden;
        // Diffusers CrossAttnUpBlock2D: for each resnet, cat skip then resnet then attention.
        if self.attentions.is_empty() {
            for resnet in &self.resnets {
                if self.pops_skip() {
                    hidden = Tensor::cat(
                        vec![
                            hidden,
                            skip_stack
                                .pop()
                                .expect("SDXL UNet up stage expected a skip tensor"),
                        ],
                        1,
                    );
                }
                hidden = resnet.forward_with_time(hidden, time_hidden.clone());
            }
            for block in &self.self_attn_blocks {
                hidden = block.forward(hidden);
            }
            for block in &self.cross_attn_blocks {
                hidden = block.forward(hidden, conditioning.clone());
            }
        } else {
            assert_eq!(
                self.resnets.len(),
                self.attentions.len(),
                "diffusers up stage resnets and attentions must align"
            );
            for (resnet, attention) in self.resnets.iter().zip(self.attentions.iter()) {
                if self.pops_skip() {
                    hidden = Tensor::cat(
                        vec![
                            hidden,
                            skip_stack
                                .pop()
                                .expect("SDXL UNet up stage expected a skip tensor"),
                        ],
                        1,
                    );
                }
                hidden = resnet.forward_with_time(hidden, time_hidden.clone());
                hidden = attention.forward(hidden, conditioning.clone());
            }
        }
        if let Some(upsample) = self.upsamplers.first() {
            upsample.forward(hidden)
        } else {
            hidden
        }
    }

    pub fn pushes_skip(&self) -> bool {
        matches!(self.skip_policy, SdxlSkipPolicy::Push)
    }

    pub fn role(&self) -> SdxlStageRole {
        self.role
    }

    pub fn pops_skip(&self) -> bool {
        matches!(self.skip_policy, SdxlSkipPolicy::Pop)
    }

    pub fn has_downsample(&self) -> bool {
        matches!(self.sampling, SdxlSamplingOp::Downsample2d)
    }

    pub fn has_upsample(&self) -> bool {
        matches!(self.sampling, SdxlSamplingOp::Upsample2d)
    }
}

/// Diffusers `mid_block`: resnet0 → attention → resnet1.
#[derive(Module, Debug)]
pub struct SdxlMidBlock<B: Backend> {
    resnets: Vec<SdxlResBlock<B>>,
    attentions: Vec<SdxlSpatialTransformer<B>>,
}

impl<B: Backend> SdxlMidBlock<B> {
    pub fn init(spec: &SdxlMidBlockSpec, time_hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            resnets: spec
                .resnets
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
            attentions: spec
                .attentions
                .iter()
                .map(|spec| {
                    SdxlSpatialTransformer::init(
                        spec.channels,
                        spec.context_dim,
                        spec.num_heads,
                        spec.num_layers,
                        spec.num_groups,
                        device,
                    )
                })
                .collect(),
        }
    }

    pub fn forward(
        &self,
        hidden: Tensor<B, 4>,
        time_hidden: Tensor<B, 2>,
        conditioning: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        let mut hidden = self.resnets[0].forward_with_time(hidden, time_hidden.clone());
        for attention in &self.attentions {
            hidden = attention.forward(hidden, conditioning.clone());
        }
        for resnet in self.resnets.iter().skip(1) {
            hidden = resnet.forward_with_time(hidden, time_hidden.clone());
        }
        hidden
    }
}

struct SdxlStageOutput<B: Backend> {
    hidden: Tensor<B, 4>,
    skips: Vec<Tensor<B, 4>>,
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
    pub norm1: GroupNorm<B>,
    pub conv1: Conv2d<B>,
    pub time_emb_proj: Linear<B>,
    pub norm2: GroupNorm<B>,
    pub conv2: Conv2d<B>,
    conv_shortcut: Option<Conv2d<B>>,
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
        let conv_shortcut = (in_channels != out_channels)
            .then(|| Conv2dConfig::new([in_channels, out_channels], [1, 1]).init(device));

        Self {
            norm1: GroupNormConfig::new(num_groups, in_channels).init(device),
            conv1: conv3([in_channels, out_channels]),
            time_emb_proj: LinearConfig::new(time_dim, out_channels).init(device),
            norm2: GroupNormConfig::new(num_groups, out_channels).init(device),
            conv2: conv3([out_channels, out_channels]),
            conv_shortcut,
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
        let residual = match &self.conv_shortcut {
            Some(conv) => conv.forward(hidden.clone()),
            None => hidden.clone(),
        };
        let hidden = self
            .conv1
            .forward(activation::silu(self.norm1.forward(hidden)));
        let [batch, channels, _, _] = hidden.dims();
        let time_hidden = self
            .time_emb_proj
            .forward(activation::silu(time_hidden))
            .reshape([batch, channels, 1, 1]);
        let hidden = hidden + time_hidden;
        let hidden = self
            .conv2
            .forward(activation::silu(self.norm2.forward(hidden)));
        hidden + residual
    }

    pub fn uses_skip_projection(&self) -> bool {
        self.conv_shortcut.is_some()
    }

    pub fn uses_time_projection(&self) -> bool {
        true
    }

    fn time_dim(&self) -> usize {
        self.time_emb_proj.weight.dims()[0]
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
    pub to_k: Linear<B>,
    pub to_v: Linear<B>,
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
            to_k: LinearConfig::new(context_dim, channels).init(device),
            to_v: LinearConfig::new(context_dim, channels).init(device),
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
        let key = self.to_k.forward(context.clone());
        let value = self.to_v.forward(context);
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
        self.to_k.weight.dims()[0]
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
    fn sdxl_unet_module_keeps_full_profile_stage_execution_plan() {
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let module = super::SdxlUnet::<ActiveBurnBackend>::init_from_profile(
            super::SdxlUnetTopologyProfile::SdxlBase,
            runtime.device(),
        );

        assert!(module.down_path_pushes_skip());
        assert!(module.down_path_has_downsample());
        assert!(module.middle_path_has_no_skip_mutation());
        assert!(module.up_path_pops_skip());
        assert!(module.up_path_has_upsample());
    }

    #[test]
    fn sdxl_unet_forward_uses_skip_stack_and_sampling_plan() {
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let topology = full_style_small_topology();
        let module =
            super::SdxlUnet::<ActiveBurnBackend>::init_from_topology(&topology, runtime.device());
        let latent = Tensor::<ActiveBurnBackend, 4>::zeros([1, 4, 8, 8], runtime.device());
        let timestep = Tensor::<ActiveBurnBackend, 1>::ones([1], runtime.device());
        let conditioning = Tensor::<ActiveBurnBackend, 3>::zeros([1, 4, 8], runtime.device());
        let added = super::SdxlAddedConditioning::new(
            Tensor::<ActiveBurnBackend, 2>::zeros([1, 8], runtime.device()),
            Tensor::<ActiveBurnBackend, 2>::zeros([1, 6], runtime.device()),
        );

        let output = module.forward_with_added_conditioning(latent, timestep, conditioning, added);

        assert_eq!(output.dims(), [1, 4, 8, 8]);
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
        assert!(super::SdxlUnetTopologyProfile::SdxlBase.is_module_graph_supported());

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
        assert!(topology.mid_block.is_some());
        assert_eq!(topology.up_blocks.len(), 3);
        assert!(topology.res_block_count() > 0);
        assert!(topology.spatial_transformer_count() > 0);
        assert!(topology.cross_attention_block_count() > 0);
    }

    #[test]
    fn sdxl_base_topology_records_executable_skip_and_sampling_semantics() {
        let topology = super::SdxlUnetTopology::sdxl_base();

        assert!(
            topology
                .down_blocks
                .iter()
                .any(super::SdxlStageSpec::pushes_skip),
            "full UNet down path must own skip production before the guard can be removed"
        );
        assert!(
            topology
                .down_blocks
                .iter()
                .any(super::SdxlStageSpec::has_downsample),
            "full UNet down path must record downsample stages before the guard can be removed"
        );
        assert!(
            topology
                .mid_block
                .is_some(),
            "middle block must exist for full SDXL base"
        );
        assert!(
            topology
                .up_blocks
                .iter()
                .any(super::SdxlStageSpec::pops_skip),
            "full UNet up path must own skip consumption before the guard can be removed"
        );
        assert!(
            topology
                .up_blocks
                .iter()
                .any(super::SdxlStageSpec::has_upsample),
            "full UNet up path must record upsample stages before the guard can be removed"
        );
        assert!(
            super::SdxlUnetTopologyProfile::SdxlBase.is_module_graph_supported(),
            "15e requires the full-profile Module graph guard to be open"
        );
    }

    #[test]
    fn sdxl_base_topology_channel_plan_balances_skip_stack() {
        let topology = super::SdxlUnetTopology::sdxl_base();
        let mut hidden_channels = topology.model_channels;
        // conv_in contributes the first skip tensor at model channel width.
        let mut skip_channels = vec![topology.model_channels];

        for stage in &topology.down_blocks {
            assert_eq!(stage.role(), super::SdxlStageRole::Down);
            assert_eq!(stage.resnets[0].in_channels, hidden_channels);
            for resnet in &stage.resnets {
                hidden_channels = resnet.out_channels;
                if stage.pushes_skip() {
                    skip_channels.push(hidden_channels);
                }
            }
            if stage.pushes_skip() && stage.has_downsample() {
                skip_channels.push(hidden_channels);
            }
        }

        if let Some(mid) = &topology.mid_block {
            assert_eq!(mid.resnets[0].in_channels, hidden_channels);
            hidden_channels = mid
                .resnets
                .last()
                .expect("middle stage should have resblocks")
                .out_channels;
        }

        for stage in &topology.up_blocks {
            assert_eq!(stage.role(), super::SdxlStageRole::Up);
            for (index, resnet) in stage.resnets.iter().enumerate() {
                let skip = skip_channels
                    .pop()
                    .expect("up residual should have a matching down-path skip");
                let base_channels = if index == 0 {
                    hidden_channels
                } else {
                    stage.resnets[index - 1].out_channels
                };
                assert_eq!(
                    resnet.in_channels,
                    base_channels + skip,
                    "up residual {index} channel plan mismatch"
                );
                hidden_channels = resnet.out_channels;
            }
        }

        assert!(
            skip_channels.is_empty(),
            "leftover skip channels: {skip_channels:?}"
        );
        assert_eq!(hidden_channels, topology.model_channels);
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

    fn full_style_small_topology() -> super::SdxlUnetTopology {
        let res = |in_channels, out_channels| super::SdxlResBlockSpec {
            in_channels,
            out_channels,
            num_groups: 2,
        };
        let attn = |channels| super::SdxlSelfAttentionBlockSpec {
            channels,
            num_heads: 2,
            num_groups: 2,
        };
        let cross = |channels| super::SdxlCrossAttentionBlockSpec {
            channels,
            context_dim: 8,
            num_heads: 2,
            num_groups: 2,
        };
        let stage = |role, resnets, channels, skip_policy, sampling| super::SdxlStageSpec {
            role,
            resnets,
            self_attn_blocks: vec![attn(channels)],
            cross_attn_blocks: vec![cross(channels)],
            attentions: Vec::new(),
            skip_policy,
            sampling,
        };
        // Up block helper: first residual uses prev_channels + skip, later use
        // hidden_out + skip — matching SGM/diffusers stage transitions.
        let up_block = |role,
                        prev_channels: usize,
                        hidden_out: usize,
                        skip_chs: &[usize],
                        sampling| super::SdxlStageSpec {
            role,
            resnets: skip_chs
                .iter()
                .enumerate()
                .map(|(index, &sk)| {
                    let base = if index == 0 {
                        prev_channels
                    } else {
                        hidden_out
                    };
                    res(base + sk, hidden_out)
                })
                .collect(),
            self_attn_blocks: vec![attn(hidden_out)],
            cross_attn_blocks: vec![cross(hidden_out)],
            attentions: Vec::new(),
            skip_policy: super::SdxlSkipPolicy::Pop,
            sampling,
        };

        super::SdxlUnetTopology {
            profile: super::SdxlUnetTopologyProfile::SdxlBase,
            latent_channels: 4,
            model_channels: 4,
            time_input_dim: 4,
            time_hidden_dim: 8,
            conditioning_dim: 8,
            pooled_conditioning_dim: 8,
            time_ids_dim: 6,
            down_blocks: vec![
                stage(
                    super::SdxlStageRole::Down,
                    vec![res(4, 4)],
                    4,
                    super::SdxlSkipPolicy::Push,
                    super::SdxlSamplingOp::Downsample2d,
                ),
                stage(
                    super::SdxlStageRole::Down,
                    vec![res(4, 8)],
                    8,
                    super::SdxlSkipPolicy::Push,
                    super::SdxlSamplingOp::None,
                ),
            ],
            mid_block: Some(super::SdxlMidBlockSpec {
                resnets: vec![res(8, 8)],
                attentions: Vec::new(),
            }),
            up_blocks: vec![
                // skips produced: conv_in(4), down0 res(4), down0 ds(4), down1 res(8)
                up_block(
                    super::SdxlStageRole::Up,
                    8,
                    8,
                    &[8, 4],
                    super::SdxlSamplingOp::Upsample2d,
                ),
                up_block(
                    super::SdxlStageRole::Up,
                    8,
                    4,
                    &[4, 4],
                    super::SdxlSamplingOp::None,
                ),
            ],
        }
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
