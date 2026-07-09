use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use safetensors::tensor::{Dtype, SafeTensors};

use super::component::{BurnSdxlComponentRole, BurnTensorDType};
use super::contract::BurnDTypePolicy;
use super::conversion::{
    BURN_SDXL_CONVERSION_REPORT_FILE, BurnSdxlConversionError, BurnSdxlConversionReport,
    BurnSdxlSyntheticComponent, BurnSyntheticTensor, BurnTensorSource, SyntheticSdxlConversionPlan,
};
use super::source_layout::{BurnSdxlSourceSet, DIFFUSERS_STYLE_SPLIT_SAFETENSORS};
use super::writer::{write_conversion_report, write_synthetic_sdxl_components};
use crate::text_encoder::clip::ClipTextEncoderProfile;

const DIFFUSION_MAPPINGS: &[TensorMapping] = &[
    TensorMapping::new("model.diffusion.conv_in.weight", "conv_in.weight"),
    TensorMapping::new("model.diffusion.conv_in.bias", "conv_in.bias"),
    TensorMapping::new("conv_in.weight", "conv_in.weight"),
    TensorMapping::new("conv_in.bias", "conv_in.bias"),
    TensorMapping::new(
        "model.diffusion.time_embed.0.weight",
        "time_embedding.linear_1.weight",
    ),
    TensorMapping::new(
        "model.diffusion.time_embed.0.bias",
        "time_embedding.linear_1.bias",
    ),
    TensorMapping::new(
        "time_embedding.linear_1.weight",
        "time_embedding.linear_1.weight",
    ),
    TensorMapping::new(
        "time_embedding.linear_1.bias",
        "time_embedding.linear_1.bias",
    ),
    TensorMapping::new(
        "model.diffusion.time_embed.2.weight",
        "time_embedding.linear_2.weight",
    ),
    TensorMapping::new(
        "model.diffusion.time_embed.2.bias",
        "time_embedding.linear_2.bias",
    ),
    TensorMapping::new(
        "time_embedding.linear_2.weight",
        "time_embedding.linear_2.weight",
    ),
    TensorMapping::new(
        "time_embedding.linear_2.bias",
        "time_embedding.linear_2.bias",
    ),
    TensorMapping::new(
        "model.diffusion.input_blocks.1.0.in_layers.2.weight",
        "down_blocks.0.resnets.0.conv1.weight",
    ),
    TensorMapping::new(
        "down_blocks.0.resnets.0.conv1.weight",
        "down_blocks.0.resnets.0.conv1.weight",
    ),
    TensorMapping::new(
        "model.diffusion.input_blocks.1.0.in_layers.2.bias",
        "down_blocks.0.resnets.0.conv1.bias",
    ),
    TensorMapping::new(
        "down_blocks.0.resnets.0.conv1.bias",
        "down_blocks.0.resnets.0.conv1.bias",
    ),
    TensorMapping::new(
        "model.diffusion.input_blocks.1.0.emb_layers.1.weight",
        "down_blocks.0.resnets.0.time_emb_proj.weight",
    ),
    TensorMapping::new(
        "down_blocks.0.resnets.0.time_emb_proj.weight",
        "down_blocks.0.resnets.0.time_emb_proj.weight",
    ),
    TensorMapping::new(
        "model.diffusion.input_blocks.1.0.emb_layers.1.bias",
        "down_blocks.0.resnets.0.time_emb_proj.bias",
    ),
    TensorMapping::new(
        "down_blocks.0.resnets.0.time_emb_proj.bias",
        "down_blocks.0.resnets.0.time_emb_proj.bias",
    ),
    TensorMapping::new(
        "model.diffusion.input_blocks.1.0.out_layers.3.weight",
        "down_blocks.0.resnets.0.conv2.weight",
    ),
    TensorMapping::new(
        "down_blocks.0.resnets.0.conv2.weight",
        "down_blocks.0.resnets.0.conv2.weight",
    ),
    TensorMapping::new(
        "model.diffusion.input_blocks.1.0.out_layers.3.bias",
        "down_blocks.0.resnets.0.conv2.bias",
    ),
    TensorMapping::new(
        "down_blocks.0.resnets.0.conv2.bias",
        "down_blocks.0.resnets.0.conv2.bias",
    ),
    TensorMapping::optional(
        "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
        "down_blocks.0.self_attn_blocks.0.attention.query.weight",
    ),
    TensorMapping::optional(
        "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn1.to_k.weight",
        "down_blocks.0.self_attn_blocks.0.attention.key.weight",
    ),
    TensorMapping::optional(
        "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn1.to_v.weight",
        "down_blocks.0.self_attn_blocks.0.attention.value.weight",
    ),
    TensorMapping::optional(
        "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.weight",
        "down_blocks.0.self_attn_blocks.0.attention.output.weight",
    ),
    TensorMapping::optional(
        "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn2.to_q.weight",
        "down_blocks.0.cross_attn_blocks.0.attention.query.weight",
    ),
    TensorMapping::optional(
        "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
        "down_blocks.0.cross_attn_blocks.0.to_k.weight",
    ),
    TensorMapping::optional(
        "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn2.to_v.weight",
        "down_blocks.0.cross_attn_blocks.0.to_v.weight",
    ),
    TensorMapping::optional(
        "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.weight",
        "down_blocks.0.cross_attn_blocks.0.attention.output.weight",
    ),
    TensorMapping::new("model.diffusion.out.0.weight", "conv_out.weight"),
    TensorMapping::new("model.diffusion.out.0.bias", "conv_out.bias"),
    TensorMapping::new("conv_out.weight", "conv_out.weight"),
    TensorMapping::new("conv_out.bias", "conv_out.bias"),
];
const VAE_MAPPINGS: &[TensorMapping] = &[
    // decoder.conv_in → conv_in
    TensorMapping::new("decoder.conv_in.weight", "conv_in.weight"),
    TensorMapping::new("decoder.conv_in.bias", "conv_in.bias"),
    // mid_block resnets — accept two source layouts:
    //
    // 1. Old converter: decoder.residual_blocks.N.norm_1 / conv_1 / norm_2 / conv_2
    // 2. Diffusers-native: decoder.mid_block.resnets.N.norm1 / conv1 / norm2 / conv2
    //
    // Both map to the same diffusers target key in the target.
    // --------------------------
    // decoder.residual_blocks.0
    TensorMapping::new("decoder.residual_blocks.0.norm_1.weight", "mid_block.resnets.0.norm1.weight"),
    TensorMapping::new("decoder.residual_blocks.0.norm_1.bias", "mid_block.resnets.0.norm1.bias"),
    TensorMapping::new("decoder.residual_blocks.0.conv_1.weight", "mid_block.resnets.0.conv1.weight"),
    TensorMapping::new("decoder.residual_blocks.0.conv_1.bias", "mid_block.resnets.0.conv1.bias"),
    TensorMapping::new("decoder.residual_blocks.0.norm_2.weight", "mid_block.resnets.0.norm2.weight"),
    TensorMapping::new("decoder.residual_blocks.0.norm_2.bias", "mid_block.resnets.0.norm2.bias"),
    TensorMapping::new("decoder.residual_blocks.0.conv_2.weight", "mid_block.resnets.0.conv2.weight"),
    TensorMapping::new("decoder.residual_blocks.0.conv_2.bias", "mid_block.resnets.0.conv2.bias"),
    // decoder.residual_blocks.1
    TensorMapping::new("decoder.residual_blocks.1.norm_1.weight", "mid_block.resnets.1.norm1.weight"),
    TensorMapping::new("decoder.residual_blocks.1.norm_1.bias", "mid_block.resnets.1.norm1.bias"),
    TensorMapping::new("decoder.residual_blocks.1.conv_1.weight", "mid_block.resnets.1.conv1.weight"),
    TensorMapping::new("decoder.residual_blocks.1.conv_1.bias", "mid_block.resnets.1.conv1.bias"),
    TensorMapping::new("decoder.residual_blocks.1.norm_2.weight", "mid_block.resnets.1.norm2.weight"),
    TensorMapping::new("decoder.residual_blocks.1.norm_2.bias", "mid_block.resnets.1.norm2.bias"),
    TensorMapping::new("decoder.residual_blocks.1.conv_2.weight", "mid_block.resnets.1.conv2.weight"),
    TensorMapping::new("decoder.residual_blocks.1.conv_2.bias", "mid_block.resnets.1.conv2.bias"),
    // --------------------------
    // decoder.mid_block.resnets.0 (diffusers-native source)
    TensorMapping::new("decoder.mid_block.resnets.0.norm1.weight", "mid_block.resnets.0.norm1.weight"),
    TensorMapping::new("decoder.mid_block.resnets.0.norm1.bias", "mid_block.resnets.0.norm1.bias"),
    TensorMapping::new("decoder.mid_block.resnets.0.conv1.weight", "mid_block.resnets.0.conv1.weight"),
    TensorMapping::new("decoder.mid_block.resnets.0.conv1.bias", "mid_block.resnets.0.conv1.bias"),
    TensorMapping::new("decoder.mid_block.resnets.0.norm2.weight", "mid_block.resnets.0.norm2.weight"),
    TensorMapping::new("decoder.mid_block.resnets.0.norm2.bias", "mid_block.resnets.0.norm2.bias"),
    TensorMapping::new("decoder.mid_block.resnets.0.conv2.weight", "mid_block.resnets.0.conv2.weight"),
    TensorMapping::new("decoder.mid_block.resnets.0.conv2.bias", "mid_block.resnets.0.conv2.bias"),
    // decoder.mid_block.resnets.1
    TensorMapping::new("decoder.mid_block.resnets.1.norm1.weight", "mid_block.resnets.1.norm1.weight"),
    TensorMapping::new("decoder.mid_block.resnets.1.norm1.bias", "mid_block.resnets.1.norm1.bias"),
    TensorMapping::new("decoder.mid_block.resnets.1.conv1.weight", "mid_block.resnets.1.conv1.weight"),
    TensorMapping::new("decoder.mid_block.resnets.1.conv1.bias", "mid_block.resnets.1.conv1.bias"),
    TensorMapping::new("decoder.mid_block.resnets.1.norm2.weight", "mid_block.resnets.1.norm2.weight"),
    TensorMapping::new("decoder.mid_block.resnets.1.norm2.bias", "mid_block.resnets.1.norm2.bias"),
    TensorMapping::new("decoder.mid_block.resnets.1.conv2.weight", "mid_block.resnets.1.conv2.weight"),
    TensorMapping::new("decoder.mid_block.resnets.1.conv2.bias", "mid_block.resnets.1.conv2.bias"),
    // conv_out
    TensorMapping::new("decoder.conv_out.weight", "conv_out.weight"),
    TensorMapping::new("decoder.conv_out.bias", "conv_out.bias"),
];
const TEXT_ENCODER_MAPPINGS: &[TensorMapping] = &[
    TensorMapping::new(
        "transformer.text_model.embeddings.token_embedding.weight",
        "model.text_encoder.token_embedding.weight",
    ),
    TensorMapping::new(
        "transformer.text_model.final_layer_norm.weight",
        "model.text_encoder.final_layer_norm.gamma",
    ),
];
const TEXT_ENCODER_2_MAPPINGS: &[TensorMapping] = &[
    TensorMapping::new(
        "transformer.text_model.embeddings.token_embedding.weight",
        "model.text_encoder_2.token_embedding.weight",
    ),
    TensorMapping::new(
        "transformer.text_model.final_layer_norm.weight",
        "model.text_encoder_2.final_layer_norm.gamma",
    ),
];

pub(crate) fn map_diffusers_style_split_source(
    source_set: &BurnSdxlSourceSet,
    output_dir: impl AsRef<Path>,
) -> Result<BurnSdxlConversionReport, BurnSdxlConversionError> {
    let output_dir = output_dir.as_ref();
    let mapped = map_split_source_to_components(source_set)?;
    let plan = SyntheticSdxlConversionPlan {
        source_identity: source_set.root().display().to_string(),
        components: mapped.components,
    };
    let mut report = write_synthetic_sdxl_components(&plan, output_dir)?;
    report.source_layout = DIFFUSERS_STYLE_SPLIT_SAFETENSORS.to_owned();
    report.ignored_tensor_families = mapped.ignored_tensor_families;
    write_conversion_report(&report, output_dir.join(BURN_SDXL_CONVERSION_REPORT_FILE))?;
    Ok(report)
}

struct MappedSplitSource {
    components: Vec<BurnSdxlSyntheticComponent>,
    ignored_tensor_families: Vec<String>,
}

fn map_split_source_to_components(
    source_set: &BurnSdxlSourceSet,
) -> Result<MappedSplitSource, BurnSdxlConversionError> {
    let source_files = [
        SourceComponent::new(
            BurnSdxlComponentRole::Diffusion,
            source_set.diffusion_path(),
            DIFFUSION_MAPPINGS,
        ),
        SourceComponent::new(
            BurnSdxlComponentRole::Vae,
            source_set.vae_path(),
            VAE_MAPPINGS,
        ),
        SourceComponent::new(
            BurnSdxlComponentRole::TextEncoder,
            source_set.text_encoder_path(),
            TEXT_ENCODER_MAPPINGS,
        ),
        SourceComponent::new(
            BurnSdxlComponentRole::TextEncoder2,
            source_set.text_encoder_2_path(),
            TEXT_ENCODER_2_MAPPINGS,
        ),
    ];

    let mut components = Vec::new();
    let mut ignored_tensor_families = Vec::new();
    for source in source_files {
        let mapped = map_source_component(source)?;
        components.push(mapped.component);
        ignored_tensor_families.extend(mapped.ignored_tensor_families);
    }
    Ok(MappedSplitSource {
        components,
        ignored_tensor_families,
    })
}

#[derive(Debug, Clone, Copy)]
struct TensorMapping {
    source_key: &'static str,
    target_key: &'static str,
    required: bool,
}

impl TensorMapping {
    const fn new(source_key: &'static str, target_key: &'static str) -> Self {
        Self {
            source_key,
            target_key,
            required: true,
        }
    }

    const fn optional(source_key: &'static str, target_key: &'static str) -> Self {
        Self {
            source_key,
            target_key,
            required: false,
        }
    }
}

#[derive(Debug, Clone)]
struct SourceComponent {
    role: BurnSdxlComponentRole,
    path: PathBuf,
    mappings: &'static [TensorMapping],
}

impl SourceComponent {
    const fn new(
        role: BurnSdxlComponentRole,
        path: PathBuf,
        mappings: &'static [TensorMapping],
    ) -> Self {
        Self {
            role,
            path,
            mappings,
        }
    }
}

struct MappedSourceComponent {
    component: BurnSdxlSyntheticComponent,
    ignored_tensor_families: Vec<String>,
}

fn map_source_component(
    source: SourceComponent,
) -> Result<MappedSourceComponent, BurnSdxlConversionError> {
    let bytes = fs::read(&source.path).map_err(|err| BurnSdxlConversionError::Io {
        path: source.path.clone(),
        source: err,
    })?;
    if bytes.is_empty() {
        return Err(BurnSdxlConversionError::InvalidComponentSet {
            reason: format!("empty source component `{}`", source.path.display()),
        });
    }
    let safetensors = SafeTensors::deserialize(&bytes).map_err(|err| {
        BurnSdxlConversionError::SafetensorsReadBack {
            path: source.path.clone(),
            source: err,
        }
    })?;

    let source_keys = safetensors.names().into_iter().collect::<BTreeSet<_>>();
    if source_keys.is_empty() {
        return Err(BurnSdxlConversionError::InvalidComponentSet {
            reason: format!("empty source component `{}`", source.path.display()),
        });
    }

    let expected_keys = source
        .mappings
        .iter()
        .map(|mapping| mapping.source_key)
        .collect::<BTreeSet<_>>();
    let unknown_keys = source_keys
        .iter()
        .filter(|key| !expected_keys.contains(**key))
        .map(|key| (*key).to_owned())
        .collect::<Vec<_>>();
    let unsupported_keys = unknown_keys
        .iter()
        .filter(|key| !is_known_extra_source_tensor(source.role, key))
        .cloned()
        .collect::<Vec<_>>();
    if !unsupported_keys.is_empty() {
        return Err(BurnSdxlConversionError::InvalidComponentSet {
            reason: format!(
                "unsupported source tensor(s) in `{}`: {}",
                source.path.display(),
                unsupported_keys.join(", ")
            ),
        });
    }

    let mapped_key_count = source_keys
        .iter()
        .filter(|key| expected_keys.contains(**key))
        .count();
    if mapped_key_count == 0 {
        return Err(BurnSdxlConversionError::InvalidComponentSet {
            reason: format!(
                "empty source component `{}` contains no mapped tensors",
                source.path.display()
            ),
        });
    }

    let mut source_by_key = BTreeMap::new();
    for name in source_keys {
        let tensor = safetensors.tensor(name).map_err(|err| {
            BurnSdxlConversionError::SafetensorsReadBack {
                path: source.path.clone(),
                source: err,
            }
        })?;
        source_by_key.insert(
            name.to_owned(),
            SourceTensor {
                shape: tensor.shape().to_vec(),
                dtype: burn_dtype(tensor.dtype()),
                data: tensor.data().to_vec(),
            },
        );
    }

    if let Some(component) = map_hf_native_text_encoder_component(source.role, &source_by_key)? {
        return Ok(MappedSourceComponent {
            component,
            ignored_tensor_families: unknown_keys
                .into_iter()
                .filter(|key| is_known_extra_source_tensor(source.role, key))
                .map(|key| format!("{}:{key}", source.role.as_str()))
                .collect(),
        });
    }

    let mut mappings_by_target = BTreeMap::<&str, Vec<&TensorMapping>>::new();
    for mapping in source.mappings {
        mappings_by_target
            .entry(mapping.target_key)
            .or_default()
            .push(mapping);
    }
    let tensors = mappings_by_target
        .into_iter()
        .filter_map(|(target_key, source_key_candidates)| {
            let source_tensor = source_key_candidates.iter().find_map(|mapping| {
                source_by_key
                    .get(mapping.source_key)
                    .map(|tensor| (mapping.source_key, tensor))
            });
            let Some((_source_key, source_tensor)) = source_tensor else {
                if source_key_candidates
                    .iter()
                    .all(|mapping| !mapping.required)
                {
                    return None;
                }
                let tried = source_key_candidates
                    .iter()
                    .map(|mapping| mapping.source_key)
                    .collect::<Vec<_>>();
                return Some(Err(BurnSdxlConversionError::InvalidComponentSet {
                    reason: format!(
                        "missing source tensor for target `{}` in `{}`; tried {}",
                        target_key,
                        source.path.display(),
                        tried.join(", ")
                    ),
                }));
            };
            Some(Ok(BurnSyntheticTensor {
                key: target_key.to_owned(),
                shape: source_tensor.shape.clone(),
                dtype: source_tensor.dtype.clone(),
                source: BurnTensorSource::Data(source_tensor.data.clone()),
            }))
        })
        .collect::<Result<Vec<_>, BurnSdxlConversionError>>()?;

    Ok(MappedSourceComponent {
        component: BurnSdxlSyntheticComponent {
            role: source.role,
            dtype_policy: BurnDTypePolicy::Fp32,
            tensors,
        },
        ignored_tensor_families: unknown_keys
            .into_iter()
            .map(|key| format!("{}:{key}", source.role.as_str()))
            .collect(),
    })
}

fn is_known_extra_source_tensor(role: BurnSdxlComponentRole, key: &str) -> bool {
    match role {
        BurnSdxlComponentRole::Diffusion => {
            key.starts_with("down_blocks.")
                || key.starts_with("mid_block.")
                || key.starts_with("up_blocks.")
                || key.starts_with("add_embedding.")
                || key.starts_with("conv_norm_out.")
        }
        BurnSdxlComponentRole::Vae => {
            key.starts_with("decoder.")
                || key.starts_with("encoder.")
                || key.starts_with("quant_conv.")
                || key.starts_with("post_quant_conv.")
        }
        BurnSdxlComponentRole::TextEncoder | BurnSdxlComponentRole::TextEncoder2 => {
            key.starts_with("transformer.text_model.") || key.starts_with("text_projection.")
        }
    }
}

fn map_hf_native_text_encoder_component(
    role: BurnSdxlComponentRole,
    source_by_key: &BTreeMap<String, SourceTensor>,
) -> Result<Option<BurnSdxlSyntheticComponent>, BurnSdxlConversionError> {
    let Some((profile, source_prefix)) = text_encoder_profile_and_prefix(role) else {
        return Ok(None);
    };
    if !source_by_key.contains_key("transformer.text_model.embeddings.position_embedding.weight") {
        return Ok(None);
    }

    let mut tensors = Vec::new();
    push_direct_tensor(
        &mut tensors,
        source_by_key,
        "transformer.text_model.embeddings.token_embedding.weight",
        profile.token_embedding_key(),
    )?;
    push_direct_tensor(
        &mut tensors,
        source_by_key,
        "transformer.text_model.embeddings.position_embedding.weight",
        profile.position_embedding_key(),
    )?;
    push_direct_tensor(
        &mut tensors,
        source_by_key,
        "transformer.text_model.final_layer_norm.weight",
        profile.final_layer_norm_weight_key(),
    )?;
    push_direct_tensor(
        &mut tensors,
        source_by_key,
        "transformer.text_model.final_layer_norm.bias",
        profile.final_layer_norm_bias_key(),
    )?;
    if let Some(target) = profile.text_projection_weight_key() {
        push_direct_tensor(
            &mut tensors,
            source_by_key,
            "text_projection.weight",
            target,
        )?;
    }
    if let Some(target) = profile.text_projection_bias_key()
        && source_by_key.contains_key("text_projection.bias")
    {
        push_direct_tensor(&mut tensors, source_by_key, "text_projection.bias", target)?;
    }

    for layer in 0..profile.num_layers {
        let source_layer = format!("{source_prefix}.{layer}");
        push_fused_qkv_tensor(
            &mut tensors,
            source_by_key,
            &source_layer,
            profile.attn_in_proj_weight_key(layer),
            "weight",
        )?;
        push_fused_qkv_tensor(
            &mut tensors,
            source_by_key,
            &source_layer,
            profile.attn_in_proj_bias_key(layer),
            "bias",
        )?;
        push_direct_tensor(
            &mut tensors,
            source_by_key,
            &format!("{source_layer}.self_attn.out_proj.weight"),
            profile.attn_out_proj_weight_key(layer),
        )?;
        push_direct_tensor(
            &mut tensors,
            source_by_key,
            &format!("{source_layer}.self_attn.out_proj.bias"),
            profile.attn_out_proj_bias_key(layer),
        )?;
        push_direct_tensor(
            &mut tensors,
            source_by_key,
            &format!("{source_layer}.layer_norm1.weight"),
            profile.ln_1_weight_key(layer),
        )?;
        push_direct_tensor(
            &mut tensors,
            source_by_key,
            &format!("{source_layer}.layer_norm1.bias"),
            profile.ln_1_bias_key(layer),
        )?;
        push_direct_tensor(
            &mut tensors,
            source_by_key,
            &format!("{source_layer}.layer_norm2.weight"),
            profile.ln_2_weight_key(layer),
        )?;
        push_direct_tensor(
            &mut tensors,
            source_by_key,
            &format!("{source_layer}.layer_norm2.bias"),
            profile.ln_2_bias_key(layer),
        )?;
        push_direct_tensor(
            &mut tensors,
            source_by_key,
            &format!("{source_layer}.mlp.fc1.weight"),
            profile.mlp_fc1_weight_key(layer),
        )?;
        push_direct_tensor(
            &mut tensors,
            source_by_key,
            &format!("{source_layer}.mlp.fc1.bias"),
            profile.mlp_fc1_bias_key(layer),
        )?;
        push_direct_tensor(
            &mut tensors,
            source_by_key,
            &format!("{source_layer}.mlp.fc2.weight"),
            profile.mlp_fc2_weight_key(layer),
        )?;
        push_direct_tensor(
            &mut tensors,
            source_by_key,
            &format!("{source_layer}.mlp.fc2.bias"),
            profile.mlp_fc2_bias_key(layer),
        )?;
    }

    Ok(Some(BurnSdxlSyntheticComponent {
        role,
        dtype_policy: BurnDTypePolicy::Fp32,
        tensors,
    }))
}

fn text_encoder_profile_and_prefix(
    role: BurnSdxlComponentRole,
) -> Option<(ClipTextEncoderProfile, &'static str)> {
    match role {
        BurnSdxlComponentRole::TextEncoder => Some((
            ClipTextEncoderProfile::sdxl_clip_l(),
            "transformer.text_model.encoder.layers",
        )),
        BurnSdxlComponentRole::TextEncoder2 => Some((
            ClipTextEncoderProfile::sdxl_open_clip_g(),
            "transformer.text_model.encoder.layers",
        )),
        BurnSdxlComponentRole::Diffusion | BurnSdxlComponentRole::Vae => None,
    }
}

fn push_direct_tensor(
    tensors: &mut Vec<BurnSyntheticTensor>,
    source_by_key: &BTreeMap<String, SourceTensor>,
    source_key: &str,
    target_key: String,
) -> Result<(), BurnSdxlConversionError> {
    let source = required_source_tensor(source_by_key, source_key)?;
    tensors.push(BurnSyntheticTensor {
        key: target_key,
        shape: source.shape.clone(),
        dtype: source.dtype.clone(),
        source: BurnTensorSource::Data(source.data.clone()),
    });
    Ok(())
}

fn push_fused_qkv_tensor(
    tensors: &mut Vec<BurnSyntheticTensor>,
    source_by_key: &BTreeMap<String, SourceTensor>,
    source_layer: &str,
    target_key: String,
    suffix: &str,
) -> Result<(), BurnSdxlConversionError> {
    let q = required_source_tensor(
        source_by_key,
        &format!("{source_layer}.self_attn.q_proj.{suffix}"),
    )?;
    let k = required_source_tensor(
        source_by_key,
        &format!("{source_layer}.self_attn.k_proj.{suffix}"),
    )?;
    let v = required_source_tensor(
        source_by_key,
        &format!("{source_layer}.self_attn.v_proj.{suffix}"),
    )?;
    if q.dtype != k.dtype || q.dtype != v.dtype || q.shape != k.shape || q.shape != v.shape {
        return Err(BurnSdxlConversionError::InvalidComponentSet {
            reason: format!("incompatible q/k/v tensors for target `{target_key}`"),
        });
    }
    let mut shape = q.shape.clone();
    let Some(first_dim) = shape.first_mut() else {
        return Err(BurnSdxlConversionError::InvalidComponentSet {
            reason: format!("empty q/k/v shape for target `{target_key}`"),
        });
    };
    *first_dim =
        first_dim
            .checked_mul(3)
            .ok_or_else(|| BurnSdxlConversionError::InvalidComponentSet {
                reason: format!("q/k/v shape overflow for target `{target_key}`"),
            })?;
    let mut data = Vec::with_capacity(q.data.len() + k.data.len() + v.data.len());
    data.extend_from_slice(&q.data);
    data.extend_from_slice(&k.data);
    data.extend_from_slice(&v.data);
    tensors.push(BurnSyntheticTensor {
        key: target_key,
        shape,
        dtype: q.dtype.clone(),
        source: BurnTensorSource::Data(data),
    });
    Ok(())
}

fn required_source_tensor<'a>(
    source_by_key: &'a BTreeMap<String, SourceTensor>,
    source_key: &str,
) -> Result<&'a SourceTensor, BurnSdxlConversionError> {
    source_by_key
        .get(source_key)
        .ok_or_else(|| BurnSdxlConversionError::InvalidComponentSet {
            reason: format!("missing source tensor `{source_key}`"),
        })
}

#[derive(Debug, Clone)]
struct SourceTensor {
    shape: Vec<usize>,
    dtype: BurnTensorDType,
    data: Vec<u8>,
}

fn burn_dtype(dtype: Dtype) -> BurnTensorDType {
    match dtype {
        Dtype::F32 => BurnTensorDType::F32,
        Dtype::F16 => BurnTensorDType::F16,
        Dtype::BF16 => BurnTensorDType::Bf16,
        other => BurnTensorDType::Unsupported(format!("{other:?}")),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::borrow::Cow;
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;

    use safetensors::tensor::{Dtype, View, serialize_to_file};

    use super::super::component::{BurnSdxlComponentRole, BurnTensorDType};
    use super::super::source_layout::BurnSdxlSourceSet;
    use super::super::validation::{
        validate_component_inventory, validate_component_inventory_full,
    };
    use super::super::writer::inspect_component_safetensors;
    use super::map_diffusers_style_split_source;

    #[derive(Debug, Clone)]
    struct TestTensorView {
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl TestTensorView {
        fn f32(shape: Vec<usize>) -> Self {
            let len = shape.iter().product::<usize>() * 4;
            Self {
                dtype: Dtype::F32,
                shape,
                data: vec![0; len],
            }
        }
    }

    impl View for TestTensorView {
        fn dtype(&self) -> Dtype {
            self.dtype
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }

        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    fn write_source_file<K: AsRef<str>>(path: &Path, tensors: &[(K, Vec<usize>)]) {
        fs::create_dir_all(path.parent().expect("source path has parent")).unwrap();
        let views = tensors
            .iter()
            .map(|(name, shape)| (name.as_ref().to_owned(), TestTensorView::f32(shape.clone())))
            .collect::<Vec<_>>();
        serialize_to_file(views, Some(HashMap::new()), path).unwrap();
    }

    pub(crate) fn write_complete_split_source(root: &Path) {
        write_full_profile_diffusion_source(root);
        write_complete_non_diffusion_source(root);
    }

    fn write_complete_non_diffusion_source(root: &Path) {
        write_source_file(
            &root.join("vae/model.safetensors"),
            &[
                ("decoder.conv_in.weight", vec![4, 4, 3, 3]),
                ("decoder.conv_in.bias", vec![4]),
                ("decoder.mid_block.resnets.0.norm1.weight", vec![4]),
                ("decoder.mid_block.resnets.0.norm1.bias", vec![4]),
                ("decoder.mid_block.resnets.0.conv1.weight", vec![4, 4, 3, 3]),
                ("decoder.mid_block.resnets.0.conv1.bias", vec![4]),
                ("decoder.mid_block.resnets.0.norm2.weight", vec![4]),
                ("decoder.mid_block.resnets.0.norm2.bias", vec![4]),
                ("decoder.mid_block.resnets.0.conv2.weight", vec![4, 4, 3, 3]),
                ("decoder.mid_block.resnets.0.conv2.bias", vec![4]),
                ("decoder.mid_block.resnets.1.norm1.weight", vec![4]),
                ("decoder.mid_block.resnets.1.norm1.bias", vec![4]),
                ("decoder.mid_block.resnets.1.conv1.weight", vec![4, 4, 3, 3]),
                ("decoder.mid_block.resnets.1.conv1.bias", vec![4]),
                ("decoder.mid_block.resnets.1.norm2.weight", vec![4]),
                ("decoder.mid_block.resnets.1.norm2.bias", vec![4]),
                ("decoder.mid_block.resnets.1.conv2.weight", vec![4, 4, 3, 3]),
                ("decoder.mid_block.resnets.1.conv2.bias", vec![4]),
                ("decoder.conv_out.weight", vec![1, 1, 1, 1]),
                ("decoder.conv_out.bias", vec![1]),
            ],
        );
        for role_dir in ["text_encoder", "text_encoder_2"] {
            write_source_file(
                &root.join(role_dir).join("model.safetensors"),
                &[
                    (
                        "transformer.text_model.embeddings.token_embedding.weight",
                        vec![1, 1],
                    ),
                    ("transformer.text_model.final_layer_norm.weight", vec![1]),
                ],
            );
        }
    }

    fn write_partial_hf_native_diffusion_source(root: &Path) {
        write_source_file(
            &root.join("unet/model.safetensors"),
            &[
                ("conv_in.weight", vec![1, 1, 1, 1]),
                ("time_embedding.linear_1.weight", vec![1, 1]),
            ],
        );
        write_complete_non_diffusion_source(root);
    }

    fn write_full_profile_diffusion_source(root: &Path) {
        write_source_file(
            &root.join("unet/model.safetensors"),
            &[
                ("model.diffusion.conv_in.weight", vec![320, 4, 3, 3]),
                ("model.diffusion.conv_in.bias", vec![320]),
                ("model.diffusion.time_embed.0.weight", vec![1280, 320]),
                ("model.diffusion.time_embed.0.bias", vec![1280]),
                ("model.diffusion.time_embed.2.weight", vec![1280, 1280]),
                ("model.diffusion.time_embed.2.bias", vec![1280]),
                (
                    "model.diffusion.input_blocks.1.0.in_layers.2.weight",
                    vec![320, 320, 3, 3],
                ),
                (
                    "model.diffusion.input_blocks.1.0.in_layers.2.bias",
                    vec![320],
                ),
                (
                    "model.diffusion.input_blocks.1.0.emb_layers.1.weight",
                    vec![320, 1280],
                ),
                (
                    "model.diffusion.input_blocks.1.0.emb_layers.1.bias",
                    vec![320],
                ),
                (
                    "model.diffusion.input_blocks.1.0.out_layers.3.weight",
                    vec![320, 320, 3, 3],
                ),
                (
                    "model.diffusion.input_blocks.1.0.out_layers.3.bias",
                    vec![320],
                ),
                (
                    "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
                    vec![320, 320],
                ),
                (
                    "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn1.to_k.weight",
                    vec![320, 320],
                ),
                (
                    "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn1.to_v.weight",
                    vec![320, 320],
                ),
                (
                    "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.weight",
                    vec![320, 320],
                ),
                (
                    "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn2.to_q.weight",
                    vec![320, 320],
                ),
                (
                    "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
                    vec![320, 2048],
                ),
                (
                    "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn2.to_v.weight",
                    vec![320, 2048],
                ),
                (
                    "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.weight",
                    vec![320, 320],
                ),
                ("model.diffusion.out.0.weight", vec![4, 320, 3, 3]),
                ("model.diffusion.out.0.bias", vec![4]),
            ],
        );
    }

    fn write_hf_native_split_source(root: &Path) {
        write_source_file(
            &root.join("unet/model.safetensors"),
            &[
                ("conv_in.weight", vec![320, 4, 3, 3]),
                ("conv_in.bias", vec![320]),
                ("time_embedding.linear_1.weight", vec![1280, 320]),
                ("time_embedding.linear_1.bias", vec![1280]),
                ("time_embedding.linear_2.weight", vec![1280, 1280]),
                ("time_embedding.linear_2.bias", vec![1280]),
                ("down_blocks.0.resnets.0.conv1.weight", vec![320, 320, 3, 3]),
                ("down_blocks.0.resnets.0.conv1.bias", vec![320]),
                (
                    "down_blocks.0.resnets.0.time_emb_proj.weight",
                    vec![320, 1280],
                ),
                ("down_blocks.0.resnets.0.time_emb_proj.bias", vec![320]),
                ("down_blocks.0.resnets.0.conv2.weight", vec![320, 320, 3, 3]),
                ("down_blocks.0.resnets.0.conv2.bias", vec![320]),
                ("conv_out.weight", vec![4, 320, 3, 3]),
                ("conv_out.bias", vec![4]),
                ("down_blocks.0.resnets.1.conv1.weight", vec![320, 320, 3, 3]),
            ],
        );
        write_source_file(
            &root.join("vae/model.safetensors"),
            &[
                ("decoder.conv_in.weight", vec![4, 4, 3, 3]),
                ("decoder.conv_in.bias", vec![4]),
                ("decoder.mid_block.resnets.0.norm1.weight", vec![4]),
                ("decoder.mid_block.resnets.0.norm1.bias", vec![4]),
                ("decoder.mid_block.resnets.0.conv1.weight", vec![4, 4, 3, 3]),
                ("decoder.mid_block.resnets.0.conv1.bias", vec![4]),
                ("decoder.mid_block.resnets.0.norm2.weight", vec![4]),
                ("decoder.mid_block.resnets.0.norm2.bias", vec![4]),
                ("decoder.mid_block.resnets.0.conv2.weight", vec![4, 4, 3, 3]),
                ("decoder.mid_block.resnets.0.conv2.bias", vec![4]),
                ("decoder.mid_block.resnets.1.norm1.weight", vec![4]),
                ("decoder.mid_block.resnets.1.norm1.bias", vec![4]),
                ("decoder.mid_block.resnets.1.conv1.weight", vec![4, 4, 3, 3]),
                ("decoder.mid_block.resnets.1.conv1.bias", vec![4]),
                ("decoder.mid_block.resnets.1.norm2.weight", vec![4]),
                ("decoder.mid_block.resnets.1.norm2.bias", vec![4]),
                ("decoder.mid_block.resnets.1.conv2.weight", vec![4, 4, 3, 3]),
                ("decoder.mid_block.resnets.1.conv2.bias", vec![4]),
                ("decoder.conv_out.weight", vec![3, 4, 3, 3]),
                ("decoder.conv_out.bias", vec![3]),
                (
                    "decoder.up_blocks.0.resnets.0.conv1.weight",
                    vec![4, 4, 3, 3],
                ),
            ],
        );
        for (role_dir, layer_count, has_projection) in
            [("text_encoder", 12, false), ("text_encoder_2", 32, true)]
        {
            write_source_file(
                &root.join(role_dir).join("model.safetensors"),
                &hf_native_text_encoder_tensors(layer_count, has_projection),
            );
        }
    }

    fn hf_native_text_encoder_tensors(
        layer_count: u32,
        has_projection: bool,
    ) -> Vec<(String, Vec<usize>)> {
        let mut tensors = vec![
            (
                "transformer.text_model.embeddings.token_embedding.weight".to_owned(),
                vec![1, 1],
            ),
            (
                "transformer.text_model.embeddings.position_embedding.weight".to_owned(),
                vec![1, 1],
            ),
            (
                "transformer.text_model.final_layer_norm.weight".to_owned(),
                vec![1],
            ),
            (
                "transformer.text_model.final_layer_norm.bias".to_owned(),
                vec![1],
            ),
        ];
        if has_projection {
            tensors.extend([
                ("text_projection.weight".to_owned(), vec![1, 1]),
                ("text_projection.bias".to_owned(), vec![1]),
            ]);
        }
        for layer in 0..layer_count {
            let prefix = format!("transformer.text_model.encoder.layers.{layer}");
            tensors.extend([
                (format!("{prefix}.self_attn.q_proj.weight"), vec![1, 1]),
                (format!("{prefix}.self_attn.k_proj.weight"), vec![1, 1]),
                (format!("{prefix}.self_attn.v_proj.weight"), vec![1, 1]),
                (format!("{prefix}.self_attn.q_proj.bias"), vec![1]),
                (format!("{prefix}.self_attn.k_proj.bias"), vec![1]),
                (format!("{prefix}.self_attn.v_proj.bias"), vec![1]),
                (format!("{prefix}.self_attn.out_proj.weight"), vec![1, 1]),
                (format!("{prefix}.self_attn.out_proj.bias"), vec![1]),
                (format!("{prefix}.layer_norm1.weight"), vec![1]),
                (format!("{prefix}.layer_norm1.bias"), vec![1]),
                (format!("{prefix}.layer_norm2.weight"), vec![1]),
                (format!("{prefix}.layer_norm2.bias"), vec![1]),
                (format!("{prefix}.mlp.fc1.weight"), vec![1, 1]),
                (format!("{prefix}.mlp.fc1.bias"), vec![1]),
                (format!("{prefix}.mlp.fc2.weight"), vec![1, 1]),
                (format!("{prefix}.mlp.fc2.bias"), vec![1]),
            ]);
        }
        tensors
    }

    #[test]
    fn maps_hf_native_split_source_and_records_ignored_extra_tensors() {
        let source = tempfile::tempdir().expect("source temp dir");
        let output = tempfile::tempdir().expect("output temp dir");
        write_hf_native_split_source(source.path());
        let source_set =
            BurnSdxlSourceSet::diffusers_style_split_safetensors(source.path().to_path_buf());

        let report =
            map_diffusers_style_split_source(&source_set, output.path()).expect("map source");

        assert_eq!(report.output_components.len(), 4);
        assert!(
            report
                .ignored_tensor_families
                .iter()
                .any(|family| family.contains("down_blocks.0.resnets.1.conv1.weight")),
            "{:?}",
            report.ignored_tensor_families
        );
        assert!(
            report
                .ignored_tensor_families
                .iter()
                .any(|family| family.contains("decoder.up_blocks.0.resnets.0.conv1.weight")),
            "{:?}",
            report.ignored_tensor_families
        );

        let diffusion =
            inspect_component_safetensors(output.path().join("diffusion/model.safetensors"))
                .expect("inspect mapped diffusion");
        let diffusion_keys = diffusion
            .inventory
            .iter()
            .map(|entry| entry.key.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        for expected in [
            "conv_in.weight",
            "time_embedding.linear_1.weight",
            "down_blocks.0.resnets.0.conv1.weight",
            "down_blocks.0.resnets.0.time_emb_proj.weight",
            "down_blocks.0.resnets.0.conv2.weight",
            "conv_out.weight",
        ] {
            assert!(diffusion_keys.contains(expected), "missing `{expected}`");
        }

        let vae = inspect_component_safetensors(output.path().join("vae/model.safetensors"))
            .expect("inspect mapped VAE");
        let vae_keys = vae
            .inventory
            .iter()
            .map(|entry| entry.key.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        for expected in [
            "conv_in.weight",
            "mid_block.resnets.0.conv1.weight",
            "mid_block.resnets.1.conv2.weight",
            "conv_out.weight",
        ] {
            assert!(vae_keys.contains(expected), "missing `{expected}`");
        }
    }

    #[test]
    fn maps_hf_native_text_encoders_to_full_burn_inventory() {
        let source = tempfile::tempdir().expect("source temp dir");
        let output = tempfile::tempdir().expect("output temp dir");
        write_hf_native_split_source(source.path());
        let source_set =
            BurnSdxlSourceSet::diffusers_style_split_safetensors(source.path().to_path_buf());

        map_diffusers_style_split_source(&source_set, output.path()).expect("map source");

        for role in [
            BurnSdxlComponentRole::TextEncoder,
            BurnSdxlComponentRole::TextEncoder2,
        ] {
            let inspected = inspect_component_safetensors(
                output.path().join(role.as_str()).join("model.safetensors"),
            )
            .expect("inspect mapped text encoder");
            validate_component_inventory_full(&inspected.metadata, &inspected.inventory)
                .expect("mapped text encoder should satisfy full inventory");
        }
    }

    #[test]
    fn rejects_partial_hf_native_diffusion_source_before_writing_output() {
        let source = tempfile::tempdir().expect("source temp dir");
        let output = tempfile::tempdir().expect("output temp dir");
        write_partial_hf_native_diffusion_source(source.path());
        let source_set =
            BurnSdxlSourceSet::diffusers_style_split_safetensors(source.path().to_path_buf());

        let err = map_diffusers_style_split_source(&source_set, output.path())
            .expect_err("partial HF-native diffusion source keys should fail");

        assert!(err.to_string().contains("missing source tensor for target"));
        assert_eq!(fs::read_dir(output.path()).unwrap().count(), 0);
    }

    #[test]
    fn maps_diffusers_style_split_source_through_burn_writer() {
        let source = tempfile::tempdir().expect("source temp dir");
        let output = tempfile::tempdir().expect("output temp dir");
        write_complete_split_source(source.path());
        let source_set =
            BurnSdxlSourceSet::diffusers_style_split_safetensors(source.path().to_path_buf());

        let report =
            map_diffusers_style_split_source(&source_set, output.path()).expect("map source");

        assert_eq!(report.source_layout, "diffusers_style_split_safetensors");
        assert_eq!(report.output_components.len(), 4);
        assert_eq!(report.mapped_tensor_count, 46);
        assert!(report.diagnostics.is_empty());
        let report_json = fs::read_to_string(output.path().join("conversion-report.json"))
            .expect("conversion report");
        let report_from_disk: super::super::conversion::BurnSdxlConversionReport =
            serde_json::from_str(&report_json).expect("report json");
        assert_eq!(report_from_disk, report);

        for role in BurnSdxlComponentRole::all() {
            let path = output.path().join(role.as_str()).join("model.safetensors");
            assert!(path.is_file());
            let inspected = inspect_component_safetensors(&path).expect("inspect mapped output");
            let validation =
                validate_component_inventory(&inspected.metadata, &inspected.inventory)
                    .expect("mapped output validates");
            assert_eq!(validation.component_role, role);
            let expected_matched_required_tensors = match role {
                BurnSdxlComponentRole::Vae => 20,
                _ => 2,
            };
            assert_eq!(
                validation.matched_required_tensors.len(),
                expected_matched_required_tensors
            );
            assert!(
                inspected
                    .inventory
                    .iter()
                    .all(|entry| matches!(entry.dtype, BurnTensorDType::F32))
            );
        }
    }

    #[test]
    fn mapped_diffusion_component_uses_runtime_loader_snapshot_names() {
        let source = tempfile::tempdir().expect("source temp dir");
        let output = tempfile::tempdir().expect("output temp dir");
        write_complete_split_source(source.path());
        let source_set =
            BurnSdxlSourceSet::diffusers_style_split_safetensors(source.path().to_path_buf());
        map_diffusers_style_split_source(&source_set, output.path()).expect("map source");
        let inspected =
            inspect_component_safetensors(output.path().join("diffusion/model.safetensors"))
                .expect("inspect mapped diffusion");
        let keys = inspected
            .inventory
            .iter()
            .map(|entry| entry.key.as_str())
            .collect::<std::collections::BTreeSet<_>>();

        for expected in [
            "conv_in.weight",
            "time_embedding.linear_1.weight",
            "time_embedding.linear_2.weight",
            "down_blocks.0.resnets.0.conv1.weight",
            "down_blocks.0.resnets.0.time_emb_proj.weight",
            "down_blocks.0.resnets.0.conv2.weight",
            "down_blocks.0.self_attn_blocks.0.attention.query.weight",
            "down_blocks.0.self_attn_blocks.0.attention.key.weight",
            "down_blocks.0.self_attn_blocks.0.attention.value.weight",
            "down_blocks.0.self_attn_blocks.0.attention.output.weight",
            "down_blocks.0.cross_attn_blocks.0.attention.query.weight",
            "down_blocks.0.cross_attn_blocks.0.to_k.weight",
            "down_blocks.0.cross_attn_blocks.0.to_v.weight",
            "down_blocks.0.cross_attn_blocks.0.attention.output.weight",
            "conv_out.weight",
        ] {
            assert!(keys.contains(expected), "missing mapped key `{expected}`");
        }
        assert!(
            !keys.contains("model.diffusion.time_embed.0.weight"),
            "source-style keys should not be written into mapped components"
        );
    }

    #[test]
    fn mapped_vae_component_uses_runtime_loader_snapshot_names() {
        let source = tempfile::tempdir().expect("source temp dir");
        let output = tempfile::tempdir().expect("output temp dir");
        write_complete_split_source(source.path());
        let source_set =
            BurnSdxlSourceSet::diffusers_style_split_safetensors(source.path().to_path_buf());
        map_diffusers_style_split_source(&source_set, output.path()).expect("map source");
        let inspected = inspect_component_safetensors(output.path().join("vae/model.safetensors"))
            .expect("inspect mapped VAE");
        let keys = inspected
            .inventory
            .iter()
            .map(|entry| entry.key.as_str())
            .collect::<std::collections::BTreeSet<_>>();

        for expected in [
            "conv_in.weight",
            "conv_in.bias",
            "mid_block.resnets.0.conv1.weight",
            "mid_block.resnets.1.conv2.weight",
            "conv_out.weight",
            "conv_out.bias",
        ] {
            assert!(keys.contains(expected), "missing mapped key `{expected}`");
        }
        for source_style in [
            "decoder.conv_in.weight",
            "decoder.residual_blocks.0.conv_1.weight",
            "decoder.mid_block.resnets.0.conv1.weight",
            "model.vae.decoder.conv_out.weight",
            "model.vae.decoder.conv_out.bias",
            "model.vae.encoder.conv_in.weight",
        ] {
            assert!(
                !keys.contains(source_style),
                "source-style key `{source_style}` should not be written into mapped components"
            );
        }
    }

    #[test]
    fn rejects_missing_required_source_file_before_writing_output() {
        let source = tempfile::tempdir().expect("source temp dir");
        let output = tempfile::tempdir().expect("output temp dir");
        write_complete_split_source(source.path());
        fs::remove_file(source.path().join("text_encoder_2/model.safetensors")).unwrap();
        let source_set =
            BurnSdxlSourceSet::diffusers_style_split_safetensors(source.path().to_path_buf());

        let err = map_diffusers_style_split_source(&source_set, output.path())
            .expect_err("missing source file should fail");

        assert!(err.to_string().contains("text_encoder_2/model.safetensors"));
        assert_eq!(fs::read_dir(output.path()).unwrap().count(), 0);
    }

    #[test]
    fn rejects_unknown_source_tensor_before_writing_output() {
        let source = tempfile::tempdir().expect("source temp dir");
        let output = tempfile::tempdir().expect("output temp dir");
        write_complete_split_source(source.path());
        write_source_file(
            &source.path().join("unet/model.safetensors"),
            &[
                ("model.diffusion.conv_in.weight", vec![320, 4, 3, 3]),
                ("model.diffusion.conv_in.bias", vec![320]),
                ("surprise.block.weight", vec![1]),
            ],
        );
        let source_set =
            BurnSdxlSourceSet::diffusers_style_split_safetensors(source.path().to_path_buf());

        let err = map_diffusers_style_split_source(&source_set, output.path())
            .expect_err("unknown source tensor should fail");

        assert!(err.to_string().contains("surprise.block.weight"));
        assert_eq!(fs::read_dir(output.path()).unwrap().count(), 0);
    }

    #[test]
    fn rejects_invalid_safetensors_source_before_writing_output() {
        let source = tempfile::tempdir().expect("source temp dir");
        let output = tempfile::tempdir().expect("output temp dir");
        write_complete_split_source(source.path());
        fs::write(
            source.path().join("vae/model.safetensors"),
            b"not safetensors",
        )
        .unwrap();
        let source_set =
            BurnSdxlSourceSet::diffusers_style_split_safetensors(source.path().to_path_buf());

        let err = map_diffusers_style_split_source(&source_set, output.path())
            .expect_err("invalid source should fail");

        assert!(err.to_string().contains("vae/model.safetensors"));
        assert_eq!(fs::read_dir(output.path()).unwrap().count(), 0);
    }

    #[test]
    fn rejects_empty_component_before_writing_output() {
        let source = tempfile::tempdir().expect("source temp dir");
        let output = tempfile::tempdir().expect("output temp dir");
        write_complete_split_source(source.path());
        write_source_file::<&str>(&source.path().join("unet/model.safetensors"), &[]);
        let source_set =
            BurnSdxlSourceSet::diffusers_style_split_safetensors(source.path().to_path_buf());

        let err = map_diffusers_style_split_source(&source_set, output.path())
            .expect_err("empty component should fail");

        assert!(err.to_string().contains("unet/model.safetensors"));
        assert_eq!(fs::read_dir(output.path()).unwrap().count(), 0);
    }

    #[test]
    fn rejects_missing_representative_tensor_before_writing_output() {
        let source = tempfile::tempdir().expect("source temp dir");
        let output = tempfile::tempdir().expect("output temp dir");
        write_complete_split_source(source.path());
        write_source_file(
            &source.path().join("unet/model.safetensors"),
            &[
                ("model.diffusion.conv_in.weight", vec![320, 4, 3, 3]),
                ("model.diffusion.conv_in.bias", vec![320]),
            ],
        );
        let source_set =
            BurnSdxlSourceSet::diffusers_style_split_safetensors(source.path().to_path_buf());

        let err = map_diffusers_style_split_source(&source_set, output.path())
            .expect_err("missing representative tensor should fail");

        assert!(err.to_string().contains("missing source tensor for target"));
        assert_eq!(fs::read_dir(output.path()).unwrap().count(), 0);
    }

    #[test]
    fn rejects_unsupported_source_dtype_before_writing_output() {
        let source = tempfile::tempdir().expect("source temp dir");
        let output = tempfile::tempdir().expect("output temp dir");
        write_complete_split_source(source.path());
        write_source_file(
            &source.path().join("text_encoder/model.safetensors"),
            &[
                (
                    "transformer.text_model.embeddings.token_embedding.weight",
                    vec![1, 1],
                ),
                ("transformer.text_model.final_layer_norm.weight", vec![1]),
            ],
        );
        fs::create_dir_all(source.path().join("text_encoder")).unwrap();
        let unsupported = vec![
            (
                "transformer.text_model.embeddings.token_embedding.weight".to_owned(),
                TestTensorView {
                    dtype: Dtype::I64,
                    shape: vec![1, 1],
                    data: vec![0; 8],
                },
            ),
            (
                "transformer.text_model.final_layer_norm.weight".to_owned(),
                TestTensorView::f32(vec![1]),
            ),
        ];
        serialize_to_file(
            unsupported,
            Some(HashMap::new()),
            &source.path().join("text_encoder/model.safetensors"),
        )
        .unwrap();
        let source_set =
            BurnSdxlSourceSet::diffusers_style_split_safetensors(source.path().to_path_buf());

        let err = map_diffusers_style_split_source(&source_set, output.path())
            .expect_err("unsupported dtype should fail");

        assert!(err.to_string().contains("unsupported dtype"));
        assert_eq!(fs::read_dir(output.path()).unwrap().count(), 0);
    }
}
