use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};

use super::checkpoint_import::{SdxlConvertedComponent, SdxlIgnoredFamily};
use super::checkpoint_mapping::{SdxlTensorMappingError, map_sdxl_checkpoint_tensor};
use super::unet_target_keys::{SdxlUnetTargetFamily, classify_sdxl_unet_target_key};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SdxlComponentWritePlan {
    tensors: BTreeMap<SdxlConvertedComponent, Vec<(String, String)>>,
    ignored_families: Vec<SdxlIgnoredFamily>,
}

impl SdxlComponentWritePlan {
    pub(crate) fn from_safetensors(
        safetensors: &SafeTensors<'_>,
    ) -> Result<Self, SdxlCheckpointWriterError> {
        let mut tensors: BTreeMap<SdxlConvertedComponent, Vec<(String, String)>> = BTreeMap::new();
        let mut ignored_families: Vec<SdxlIgnoredFamily> = Vec::new();

        for name in safetensors.names() {
            match map_sdxl_checkpoint_tensor(name) {
                Ok(mapped) => {
                    tensors
                        .entry(mapped.component)
                        .or_default()
                        .push((mapped.target_name, name.to_owned()));
                }
                Err(SdxlTensorMappingError::Ignored { name: _, reason }) => {
                    // Collect unique ignored family reasons. Multiple
                    // tensors from the same family share the same
                    // reason string; deduplicate by reason.
                    if !ignored_families.iter().any(|entry| entry.reason == reason) {
                        let family = name.split('.').take(3).collect::<Vec<_>>().join(".");
                        ignored_families.push(SdxlIgnoredFamily {
                            family: format!("{family}.*"),
                            reason,
                        });
                    }
                }
                Err(error) => {
                    return Err(SdxlCheckpointWriterError::UnsupportedMapping {
                        source_name: name.to_owned(),
                        reason: error.to_string(),
                    });
                }
            }
        }

        Ok(Self {
            tensors,
            ignored_families,
        })
    }

    pub(crate) fn ignored_families(&self) -> &[SdxlIgnoredFamily] {
        &self.ignored_families
    }

    pub(crate) fn tensor_count(&self, component: SdxlConvertedComponent) -> usize {
        self.tensors
            .get(&component)
            .map(Vec::len)
            .unwrap_or_default()
    }

    fn entries(&self, component: SdxlConvertedComponent) -> Option<&[(String, String)]> {
        self.tensors.get(&component).map(Vec::as_slice)
    }

    fn validate_complete(&self) -> Result<(), SdxlCheckpointWriterError> {
        for component in SdxlConvertedComponent::all() {
            if self.tensor_count(component) == 0 {
                return Err(SdxlCheckpointWriterError::MissingComponent {
                    component: component.manifest_key(),
                });
            }
            validate_required_targets(component, self.entries(component).unwrap_or_default())?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SdxlCheckpointWriterError {
    ReadSource {
        path: PathBuf,
        reason: String,
    },
    ParseSource {
        path: PathBuf,
        reason: String,
    },
    UnsupportedMapping {
        source_name: String,
        reason: String,
    },
    MissingComponent {
        component: &'static str,
    },
    MissingRequiredTarget {
        component: &'static str,
        target: &'static str,
    },
    TensorRead {
        source_name: String,
        reason: String,
    },
    TensorView {
        target_name: String,
        reason: String,
    },
    CreateDirectory {
        path: PathBuf,
        reason: String,
    },
    WriteComponent {
        path: PathBuf,
        reason: String,
    },
}

impl std::fmt::Display for SdxlCheckpointWriterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadSource { path, reason } => write!(
                f,
                "failed to read SDXL checkpoint source {}: {reason}",
                path.display()
            ),
            Self::ParseSource { path, reason } => write!(
                f,
                "failed to parse SDXL checkpoint source {} as safetensors: {reason}",
                path.display()
            ),
            Self::UnsupportedMapping {
                source_name,
                reason,
            } => write!(
                f,
                "unsupported SDXL checkpoint tensor mapping for `{source_name}`: {reason}"
            ),
            Self::MissingComponent { component } => {
                write!(
                    f,
                    "SDXL checkpoint import produced no tensors for component `{component}`"
                )
            }
            Self::MissingRequiredTarget { component, target } => {
                write!(
                    f,
                    "SDXL checkpoint import component `{component}` is missing required Candle example target `{target}`"
                )
            }
            Self::TensorRead {
                source_name,
                reason,
            } => write!(f, "failed to read source tensor `{source_name}`: {reason}"),
            Self::TensorView {
                target_name,
                reason,
            } => write!(
                f,
                "failed to build target tensor view `{target_name}`: {reason}"
            ),
            Self::CreateDirectory { path, reason } => write!(
                f,
                "failed to create SDXL checkpoint import directory {}: {reason}",
                path.display()
            ),
            Self::WriteComponent { path, reason } => write!(
                f,
                "failed to write SDXL checkpoint component {}: {reason}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for SdxlCheckpointWriterError {}

fn validate_required_targets(
    component: SdxlConvertedComponent,
    entries: &[(String, String)],
) -> Result<(), SdxlCheckpointWriterError> {
    match component {
        SdxlConvertedComponent::Unet => validate_required_unet_targets(entries),
        SdxlConvertedComponent::ClipL | SdxlConvertedComponent::ClipG => {
            validate_required_named_targets(component, entries, REQUIRED_CLIP_TARGETS)
        }
        SdxlConvertedComponent::Vae => {
            validate_required_named_targets(component, entries, REQUIRED_VAE_TARGETS)
        }
    }
}

fn validate_required_unet_targets(
    entries: &[(String, String)],
) -> Result<(), SdxlCheckpointWriterError> {
    for family in REQUIRED_UNET_FAMILIES {
        if entries
            .iter()
            .any(|(target, _)| classify_sdxl_unet_target_key(target) == Some(*family))
        {
            continue;
        }
        return Err(SdxlCheckpointWriterError::MissingRequiredTarget {
            component: SdxlConvertedComponent::Unet.manifest_key(),
            target: family.required_label(),
        });
    }
    Ok(())
}

fn validate_required_named_targets(
    component: SdxlConvertedComponent,
    entries: &[(String, String)],
    required: &[&'static str],
) -> Result<(), SdxlCheckpointWriterError> {
    for target in required {
        if entries.iter().any(|(candidate, _)| candidate == target) {
            continue;
        }
        return Err(SdxlCheckpointWriterError::MissingRequiredTarget {
            component: component.manifest_key(),
            target,
        });
    }
    Ok(())
}

const REQUIRED_UNET_FAMILIES: &[SdxlUnetTargetFamily] = &[
    SdxlUnetTargetFamily::ConvIn,
    SdxlUnetTargetFamily::TimeEmbedding,
    SdxlUnetTargetFamily::DownBlockResnet,
    SdxlUnetTargetFamily::DownBlockAttention,
    SdxlUnetTargetFamily::DownBlockDownsample,
    SdxlUnetTargetFamily::MidBlockResnet,
    SdxlUnetTargetFamily::MidBlockAttention,
    SdxlUnetTargetFamily::UpBlockResnet,
    SdxlUnetTargetFamily::UpBlockAttention,
    SdxlUnetTargetFamily::UpBlockUpsample,
    SdxlUnetTargetFamily::ConvNormOut,
    SdxlUnetTargetFamily::ConvOut,
];

const REQUIRED_CLIP_TARGETS: &[&str] = &[
    "transformer.text_model.embeddings.token_embedding.weight",
    "transformer.text_model.embeddings.position_embedding.weight",
    "transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
    "transformer.text_model.encoder.layers.0.self_attn.k_proj.weight",
    "transformer.text_model.encoder.layers.0.self_attn.v_proj.weight",
    "transformer.text_model.encoder.layers.0.self_attn.out_proj.weight",
    "transformer.text_model.encoder.layers.0.layer_norm1.weight",
    "transformer.text_model.final_layer_norm.weight",
];

const REQUIRED_VAE_TARGETS: &[&str] = &[
    "encoder.conv_in.weight",
    "decoder.conv_in.weight",
    "decoder.conv_out.weight",
    "quant_conv.weight",
    "post_quant_conv.weight",
];

impl SdxlUnetTargetFamily {
    fn required_label(self) -> &'static str {
        match self {
            Self::ConvIn => "conv_in.*",
            Self::TimeEmbedding => "time_embedding.*",
            Self::DownBlockResnet => "down_blocks.*.resnets.*",
            Self::DownBlockAttention => "down_blocks.*.attentions.*",
            Self::DownBlockDownsample => "down_blocks.*.downsamplers.0.conv.*",
            Self::MidBlockResnet => "mid_block.resnets.*",
            Self::MidBlockAttention => "mid_block.attentions.*",
            Self::UpBlockResnet => "up_blocks.*.resnets.*",
            Self::UpBlockAttention => "up_blocks.*.attentions.*",
            Self::UpBlockUpsample => "up_blocks.*.upsamplers.0.conv.*",
            Self::ConvNormOut => "conv_norm_out.*",
            Self::ConvOut => "conv_out.*",
        }
    }
}

pub(crate) fn write_sdxl_checkpoint_components(
    source_path: &Path,
    conversion_dir: &Path,
    component_relative_path: impl Fn(SdxlConvertedComponent) -> String,
) -> Result<SdxlComponentWritePlan, SdxlCheckpointWriterError> {
    let bytes = fs::read(source_path).map_err(|error| SdxlCheckpointWriterError::ReadSource {
        path: source_path.to_path_buf(),
        reason: error.to_string(),
    })?;
    let safetensors = SafeTensors::deserialize(&bytes).map_err(|error| {
        SdxlCheckpointWriterError::ParseSource {
            path: source_path.to_path_buf(),
            reason: error.to_string(),
        }
    })?;
    let plan = SdxlComponentWritePlan::from_safetensors(&safetensors)?;
    plan.validate_complete()?;

    for component in SdxlConvertedComponent::all() {
        let Some(entries) = plan.entries(component) else {
            continue;
        };
        let output_path = conversion_dir.join(component_relative_path(component));
        write_component_file(&safetensors, entries, &output_path)?;
    }

    Ok(plan)
}

fn write_component_file(
    safetensors: &SafeTensors<'_>,
    entries: &[(String, String)],
    output_path: &Path,
) -> Result<(), SdxlCheckpointWriterError> {
    let parent = output_path
        .parent()
        .expect("component output path has parent directory");
    fs::create_dir_all(parent).map_err(|error| SdxlCheckpointWriterError::CreateDirectory {
        path: parent.to_path_buf(),
        reason: error.to_string(),
    })?;

    let mut views = Vec::with_capacity(entries.len());
    for (target_name, source_name) in entries {
        let view = safetensors.tensor(source_name).map_err(|error| {
            SdxlCheckpointWriterError::TensorRead {
                source_name: source_name.clone(),
                reason: error.to_string(),
            }
        })?;
        views.push(OwnedTensorView::from_tensor_view(
            target_name.clone(),
            view,
        )?);
    }

    safetensors::serialize_to_file(
        views.iter().map(|view| (view.name.as_str(), view)),
        None,
        output_path,
    )
    .map_err(|error| SdxlCheckpointWriterError::WriteComponent {
        path: output_path.to_path_buf(),
        reason: error.to_string(),
    })
}

#[derive(Debug, Clone)]
struct OwnedTensorView {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl OwnedTensorView {
    fn from_tensor_view(
        target_name: String,
        view: TensorView<'_>,
    ) -> Result<Self, SdxlCheckpointWriterError> {
        let data = view.data().to_vec();
        TensorView::new(view.dtype(), view.shape().to_vec(), &data).map_err(|error| {
            SdxlCheckpointWriterError::TensorView {
                target_name: target_name.clone(),
                reason: error.to_string(),
            }
        })?;
        Ok(Self {
            name: target_name,
            dtype: view.dtype(),
            shape: view.shape().to_vec(),
            data,
        })
    }
}

impl safetensors::View for &OwnedTensorView {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        std::borrow::Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor};

    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "reimagine-sdxl-checkpoint-writer-{name}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn write_safetensors(path: &Path, names: &[&str]) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut tensors = HashMap::new();
        for (idx, name) in names.iter().enumerate() {
            let tensor = Tensor::from_vec(vec![idx as f32], (1,), &Device::Cpu).unwrap();
            tensors.insert((*name).to_owned(), tensor);
        }
        candle_core::safetensors::save(&tensors, path).unwrap();
    }

    fn complete_diffusers_split_names() -> Vec<&'static str> {
        let mut names = vec![
            "conv_in.weight",
            "time_embedding.linear_1.weight",
            "down_blocks.0.resnets.0.norm1.weight",
            "down_blocks.1.attentions.0.proj_in.weight",
            "down_blocks.0.downsamplers.0.conv.weight",
            "mid_block.resnets.0.norm1.weight",
            "mid_block.attentions.0.proj_in.weight",
            "up_blocks.0.resnets.0.conv_shortcut.weight",
            "up_blocks.0.attentions.0.proj_in.weight",
            "up_blocks.0.upsamplers.0.conv.weight",
            "conv_norm_out.weight",
            "conv_out.weight",
        ];
        names.extend(required_clip_source_names("conditioner.embedders.0."));
        names.extend(required_clip_source_names("conditioner.embedders.1.model."));
        names.extend(required_vae_source_names());
        names
    }

    fn complete_original_checkpoint_names() -> Vec<&'static str> {
        let mut names = vec![
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.time_embed.0.weight",
            "model.diffusion_model.input_blocks.1.0.in_layers.0.weight",
            "model.diffusion_model.input_blocks.4.1.proj_in.weight",
            "model.diffusion_model.input_blocks.3.0.op.weight",
            "model.diffusion_model.middle_block.0.in_layers.0.weight",
            "model.diffusion_model.middle_block.1.proj_in.weight",
            "model.diffusion_model.output_blocks.0.0.skip_connection.weight",
            "model.diffusion_model.output_blocks.0.1.proj_in.weight",
            "model.diffusion_model.output_blocks.2.2.conv.weight",
            "model.diffusion_model.out.0.weight",
            "model.diffusion_model.out.2.weight",
            "model.diffusion_model.label_emb.0.0.weight",
        ];
        names.extend(required_clip_source_names("conditioner.embedders.0."));
        names.extend(required_clip_source_names("conditioner.embedders.1.model."));
        names.extend(required_vae_source_names());
        names
    }

    fn required_clip_source_names(prefix: &'static str) -> Vec<&'static str> {
        REQUIRED_CLIP_TARGETS
            .iter()
            .map(|target| Box::leak(format!("{prefix}{target}").into_boxed_str()) as &'static str)
            .collect()
    }

    fn required_vae_source_names() -> Vec<&'static str> {
        REQUIRED_VAE_TARGETS
            .iter()
            .map(|target| {
                Box::leak(format!("first_stage_model.{target}").into_boxed_str()) as &'static str
            })
            .collect()
    }

    #[test]
    fn write_components_splits_supported_tensor_families() {
        let dir = temp_dir("split");
        let source = dir.join("source.safetensors");
        let output = dir.join("converted");
        write_safetensors(&source, &complete_diffusers_split_names());

        let plan = write_sdxl_checkpoint_components(&source, &output, |component| {
            component.relative_path().to_owned()
        })
        .unwrap();

        assert_eq!(plan.tensor_count(SdxlConvertedComponent::Unet), 12);
        assert_eq!(plan.tensor_count(SdxlConvertedComponent::ClipL), 8);
        assert_eq!(plan.tensor_count(SdxlConvertedComponent::ClipG), 8);
        assert_eq!(plan.tensor_count(SdxlConvertedComponent::Vae), 5);
        assert!(output.join("unet/model.safetensors").is_file());
        assert!(output.join("text_encoder/model.safetensors").is_file());
        assert!(output.join("text_encoder_2/model.safetensors").is_file());
        assert!(output.join("vae/model.safetensors").is_file());

        let unet =
            candle_core::safetensors::load(output.join("unet/model.safetensors"), &Device::Cpu)
                .unwrap();
        assert!(unet.contains_key("conv_in.weight"));

        let clip_l = candle_core::safetensors::load(
            output.join("text_encoder/model.safetensors"),
            &Device::Cpu,
        )
        .unwrap();
        assert!(clip_l.contains_key("transformer.text_model.embeddings.token_embedding.weight"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn original_unet_keys_are_mapped_and_written_successfully() {
        let dir = temp_dir("original-unet-mapped");
        let source = dir.join("source.safetensors");
        let output = dir.join("converted");
        write_safetensors(&source, &complete_original_checkpoint_names());

        let plan = write_sdxl_checkpoint_components(&source, &output, |component| {
            component.relative_path().to_owned()
        })
        .expect("original UNet keys should now map successfully");

        assert_eq!(plan.tensor_count(SdxlConvertedComponent::Unet), 12);
        assert!(output.join("unet/model.safetensors").is_file());
        assert!(output.join("text_encoder/model.safetensors").is_file());
        assert!(output.join("text_encoder_2/model.safetensors").is_file());
        assert!(output.join("vae/model.safetensors").is_file());

        // Verify the mapped target key names in the written UNet file.
        let unet =
            candle_core::safetensors::load(output.join("unet/model.safetensors"), &Device::Cpu)
                .unwrap();
        assert!(
            unet.contains_key("conv_in.weight"),
            "input_blocks.0.0.weight should map to conv_in.weight"
        );
        assert!(
            unet.contains_key("mid_block.attentions.0.proj_in.weight"),
            "middle_block.1.proj_in.weight should map"
        );
        assert!(
            unet.contains_key("up_blocks.0.resnets.0.conv_shortcut.weight"),
            "output_blocks.0.0.skip_connection.weight should map"
        );
        assert!(
            unet.contains_key("time_embedding.linear_1.weight"),
            "time_embed.0.weight should map"
        );
        assert!(
            unet.contains_key("conv_out.weight"),
            "out.2.weight should map"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn unsupported_original_unet_block_index_fails_before_writing() {
        let dir = temp_dir("unsupported-block-idx");
        let source = dir.join("source.safetensors");
        let output = dir.join("converted");
        write_safetensors(
            &source,
            &[
                "model.diffusion_model.input_blocks.99.0.weight",
                "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
                "conditioner.embedders.1.model.transformer.text_model.embeddings.token_embedding.weight",
                "first_stage_model.decoder.conv_in.weight",
            ],
        );

        let err = write_sdxl_checkpoint_components(&source, &output, |component| {
            component.relative_path().to_owned()
        })
        .unwrap_err();

        assert!(matches!(
            err,
            SdxlCheckpointWriterError::UnsupportedMapping { .. }
        ));
        assert!(!output.exists());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn missing_component_fails_before_writing_components() {
        let dir = temp_dir("missing-component");
        let source = dir.join("source.safetensors");
        let output = dir.join("converted");
        let mut names = complete_diffusers_split_names();
        names.retain(|name| !name.starts_with("first_stage_model."));
        write_safetensors(&source, &names);

        let err = write_sdxl_checkpoint_components(&source, &output, |component| {
            component.relative_path().to_owned()
        })
        .unwrap_err();

        assert!(matches!(
            err,
            SdxlCheckpointWriterError::MissingComponent { component: "vae" }
        ));
        assert!(!output.exists());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn incomplete_unet_target_surface_fails_before_writing_components() {
        let dir = temp_dir("incomplete-unet");
        let source = dir.join("source.safetensors");
        let output = dir.join("converted");
        let mut names = complete_diffusers_split_names();
        names.retain(|name| *name != "down_blocks.1.attentions.0.proj_in.weight");
        write_safetensors(&source, &names);

        let err = write_sdxl_checkpoint_components(&source, &output, |component| {
            component.relative_path().to_owned()
        })
        .unwrap_err();

        assert!(matches!(
            err,
            SdxlCheckpointWriterError::MissingRequiredTarget {
                component: "unet",
                target: "down_blocks.*.attentions.*"
            }
        ));
        assert!(!output.exists());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_writer_fixture_uses_f32_tensors() {
        let dir = temp_dir("fixture-dtype");
        let source = dir.join("source.safetensors");
        write_safetensors(&source, &["conv_in.weight"]);
        let tensors = candle_core::safetensors::load(&source, &Device::Cpu).unwrap();
        assert_eq!(tensors["conv_in.weight"].dtype(), DType::F32);
        let _ = fs::remove_dir_all(dir);
    }
}
