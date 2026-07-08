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
    tensors: BTreeMap<SdxlConvertedComponent, Vec<WriteEntry>>,
    ignored_families: Vec<SdxlIgnoredFamily>,
}

/// A single entry in the write plan: a source tensor copied (or
/// sliced) into a target name.
#[derive(Debug, Clone, PartialEq, Eq)]
struct WriteEntry {
    target_name: String,
    source_name: String,
    /// Row range to slice from the source tensor. None means copy
    /// the full tensor (1:1).
    source_row_range: Option<(usize, usize)>,
}

impl SdxlComponentWritePlan {
    pub(crate) fn from_safetensors(
        safetensors: &SafeTensors<'_>,
    ) -> Result<Self, SdxlCheckpointWriterError> {
        let mut tensors: BTreeMap<SdxlConvertedComponent, Vec<WriteEntry>> = BTreeMap::new();
        let mut ignored_families: Vec<SdxlIgnoredFamily> = Vec::new();

        for name in safetensors.names() {
            match map_sdxl_checkpoint_tensor(name) {
                Ok(mapped_vec) => {
                    for mapped in mapped_vec {
                        tensors
                            .entry(mapped.component)
                            .or_default()
                            .push(WriteEntry {
                                target_name: mapped.target_name,
                                source_name: name.to_owned(),
                                source_row_range: mapped.source_row_range,
                            });
                    }
                }
                Err(SdxlTensorMappingError::Ignored { name: _, reason }) => {
                    // Collect unique ignored family reasons.
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

    fn entries(&self, component: SdxlConvertedComponent) -> Option<&[WriteEntry]> {
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
    RowSlice {
        target_name: String,
        source_shape: Vec<usize>,
        range: (usize, usize),
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
            Self::RowSlice {
                target_name,
                source_shape,
                range,
                reason,
            } => write!(
                f,
                "failed to slice source tensor for `{target_name}`: {reason} (source shape {source_shape:?}, range {range:?})",
            ),
        }
    }
}

impl std::error::Error for SdxlCheckpointWriterError {}

fn validate_required_targets(
    component: SdxlConvertedComponent,
    entries: &[WriteEntry],
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

fn validate_required_unet_targets(entries: &[WriteEntry]) -> Result<(), SdxlCheckpointWriterError> {
    for family in REQUIRED_UNET_FAMILIES {
        if entries
            .iter()
            .any(|entry| classify_sdxl_unet_target_key(&entry.target_name) == Some(*family))
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
    entries: &[WriteEntry],
    required: &[&'static str],
) -> Result<(), SdxlCheckpointWriterError> {
    for target in required {
        if entries.iter().any(|entry| entry.target_name == *target) {
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
    "encoder.conv_out.weight",
    "encoder.conv_norm_out.weight",
    "decoder.conv_in.weight",
    "decoder.conv_out.weight",
    "decoder.conv_norm_out.weight",
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
    entries: &[WriteEntry],
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
    for entry in entries {
        let view = safetensors.tensor(&entry.source_name).map_err(|error| {
            SdxlCheckpointWriterError::TensorRead {
                source_name: entry.source_name.clone(),
                reason: error.to_string(),
            }
        })?;

        if let Some((row_start, row_end)) = entry.source_row_range {
            views.push(OwnedTensorView::from_tensor_view_slice(
                entry.target_name.clone(),
                view,
                row_start,
                row_end,
            )?);
        } else {
            views.push(OwnedTensorView::from_tensor_view(
                entry.target_name.clone(),
                view,
            )?);
        }
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

    /// Creates a view that copies a row range from the source tensor.
    ///
    /// The source tensor is assumed to have shape `[total_rows, embed_dim]`
    /// (or `[total_rows]` for bias vectors). `row_start` is the first row
    /// to include, `row_end` is the first row *after* the slice
    /// (so `row_end - row_start` rows are kept).
    fn from_tensor_view_slice(
        target_name: String,
        view: TensorView<'_>,
        row_start: usize,
        row_end: usize,
    ) -> Result<Self, SdxlCheckpointWriterError> {
        let shape = view.shape().to_vec();
        let dtype = view.dtype();
        let elem_size = match dtype {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
            _ => {
                return Err(SdxlCheckpointWriterError::RowSlice {
                    target_name,
                    source_shape: shape,
                    range: (row_start, row_end),
                    reason: format!("unsupported dtype {dtype:?} for row slicing"),
                });
            }
        };

        let total_rows =
            shape
                .first()
                .copied()
                .ok_or_else(|| SdxlCheckpointWriterError::RowSlice {
                    target_name: target_name.clone(),
                    source_shape: shape.clone(),
                    range: (row_start, row_end),
                    reason: "cannot slice a 0-dimensional tensor".to_owned(),
                })?;

        if row_end > total_rows || row_start >= row_end {
            return Err(SdxlCheckpointWriterError::RowSlice {
                target_name,
                source_shape: shape,
                range: (row_start, row_end),
                reason: "row range is out of bounds".to_owned(),
            });
        }

        let num_rows = row_end - row_start;
        let cols_per_row: usize = shape.iter().skip(1).copied().product::<usize>().max(1);
        let row_bytes = cols_per_row * elem_size;

        let src_data = view.data();
        let start_byte = row_start * row_bytes;
        let end_byte = row_end * row_bytes;

        if end_byte > src_data.len() {
            return Err(SdxlCheckpointWriterError::RowSlice {
                target_name: target_name.clone(),
                source_shape: shape,
                range: (row_start, row_end),
                reason: "byte range exceeds source data length".to_owned(),
            });
        }

        let sliced = src_data[start_byte..end_byte].to_vec();

        // Compute output shape: [num_rows, ...rest].
        let mut out_shape = shape;
        out_shape[0] = num_rows;

        TensorView::new(dtype, out_shape.clone(), &sliced).map_err(|error| {
            SdxlCheckpointWriterError::TensorView {
                target_name: target_name.clone(),
                reason: error.to_string(),
            }
        })?;

        Ok(Self {
            name: target_name,
            dtype,
            shape: out_shape,
            data: sliced,
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

    fn write_multi_safetensors(path: &Path, tensors: &[(&str, Vec<f32>, &[usize])]) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut map = HashMap::new();
        for (name, data, shape) in tensors {
            let t = Tensor::from_vec(data.clone(), *shape, &Device::Cpu).unwrap();
            map.insert(name.to_string(), t);
        }
        candle_core::safetensors::save(&map, path).unwrap();
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
        assert_eq!(plan.tensor_count(SdxlConvertedComponent::Vae), 8);
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

    // ---- OpenCLIP-G Writer Integration Tests ----

    #[test]
    fn openclipg_keys_write_to_valid_components() {
        let dir = temp_dir("openclipg-write");
        let source = dir.join("source.safetensors");
        let output = dir.join("converted");
        fs::create_dir_all(&dir).unwrap();

        // Build all tensors with correct shapes, then write once.
        let mut all_tensors: Vec<(&str, Vec<f32>, &[usize])> = Vec::new();

        // Minimal Unet (1D `[1]` — no slicing needed).
        for name in &[
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
        ] {
            all_tensors.push((name, vec![0.0f32; 1], &[1usize]));
        }

        // Minimal ClipL (with reasonable shapes for validate).
        all_tensors.push((
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
            vec![0.0f32; 768 * 256],
            &[256usize, 768],
        ));
        all_tensors.push((
            "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight",
            vec![0.0f32; 77 * 768],
            &[77usize, 768],
        ));
        for name in &[
            "transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
            "transformer.text_model.encoder.layers.0.self_attn.k_proj.weight",
            "transformer.text_model.encoder.layers.0.self_attn.v_proj.weight",
            "transformer.text_model.encoder.layers.0.self_attn.out_proj.weight",
        ] {
            let full: String = format!("conditioner.embedders.0.{name}");
            all_tensors.push((
                Box::leak(full.into_boxed_str()),
                vec![0.0f32; 768 * 768],
                &[768usize, 768],
            ));
        }
        all_tensors.push((
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm1.weight",
            vec![0.0f32; 768],
            &[768usize],
        ));
        all_tensors.push((
            "conditioner.embedders.0.transformer.text_model.final_layer_norm.weight",
            vec![0.0f32; 768],
            &[768usize],
        ));

        // Minimal Vae.
        for name in &[
            "first_stage_model.encoder.conv_in.weight",
            "first_stage_model.encoder.conv_out.weight",
            "first_stage_model.encoder.conv_norm_out.weight",
            "first_stage_model.decoder.conv_in.weight",
            "first_stage_model.decoder.conv_out.weight",
            "first_stage_model.decoder.conv_norm_out.weight",
            "first_stage_model.quant_conv.weight",
            "first_stage_model.post_quant_conv.weight",
        ] {
            all_tensors.push((name, vec![0.0f32; 1], &[1usize]));
        }

        // OpenCLIP-G ClipG — must be shaped for row slicing.
        all_tensors.push((
            "conditioner.embedders.1.model.token_embedding.weight",
            vec![0.0f32; 49408 * 1280],
            &[49408usize, 1280],
        ));
        all_tensors.push((
            "conditioner.embedders.1.model.positional_embedding",
            vec![0.0f32; 77 * 1280],
            &[77usize, 1280],
        ));
        all_tensors.push((
            "conditioner.embedders.1.model.ln_final.weight",
            vec![1.0f32; 1280],
            &[1280usize],
        ));
        all_tensors.push((
            "conditioner.embedders.1.model.text_projection",
            vec![0.0f32; 1280 * 1280],
            &[1280usize, 1280],
        ));
        all_tensors.push((
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
            vec![0.0f32; 3840 * 1280],
            &[3840usize, 1280],
        ));
        all_tensors.push((
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.out_proj.weight",
            vec![0.0f32; 1280 * 1280],
            &[1280usize, 1280],
        ));
        all_tensors.push((
            "conditioner.embedders.1.model.transformer.resblocks.0.ln_1.weight",
            vec![1.0f32; 1280],
            &[1280usize],
        ));
        all_tensors.push((
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_bias",
            vec![0.0f32; 3840],
            &[3840usize],
        ));
        all_tensors.push((
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.out_proj.bias",
            vec![0.0f32; 1280],
            &[1280usize],
        ));
        all_tensors.push((
            "conditioner.embedders.1.model.transformer.resblocks.0.ln_1.bias",
            vec![1.0f32; 1280],
            &[1280usize],
        ));
        all_tensors.push((
            "conditioner.embedders.1.model.transformer.resblocks.0.mlp.c_fc.weight",
            vec![0.0f32; 5120 * 1280],
            &[5120usize, 1280],
        ));
        all_tensors.push((
            "conditioner.embedders.1.model.transformer.resblocks.0.mlp.c_proj.weight",
            vec![0.0f32; 1280 * 5120],
            &[1280usize, 5120],
        ));
        all_tensors.push((
            "conditioner.embedders.1.model.transformer.resblocks.10.attn.in_proj_weight",
            vec![0.0f32; 3840 * 1280],
            &[3840usize, 1280],
        ));
        all_tensors.push((
            "conditioner.embedders.1.model.transformer.resblocks.10.attn.out_proj.weight",
            vec![0.0f32; 1280 * 1280],
            &[1280usize, 1280],
        ));
        all_tensors.push((
            "conditioner.embedders.1.model.transformer.resblocks.10.ln_1.weight",
            vec![1.0f32; 1280],
            &[1280usize],
        ));

        write_multi_safetensors(&source, &all_tensors);

        let plan = write_sdxl_checkpoint_components(&source, &output, |component| {
            component.relative_path().to_owned()
        })
        .expect("OpenCLIP-G keys should produce a valid write plan");

        // in_proj_weight splits into 3 per layer; in_proj_bias into 3.
        // For 2 layers with both in_proj_weight and in_proj_bias:
        // layer 0: 3 (weight) + 3 (bias) + out_proj.weight + out_proj.bias + ln_1.weight + ln_1.bias + mlp.c_fc.weight + mlp.c_proj.weight = 12
        // layer 10: 3 (weight) + out_proj.weight + ln_1.weight = 5
        // top-level: token_embedding (1) + positional_embedding (1)
        // + ln_final.weight (1) + text_projection.weight (1) = 4
        // Total ClipG = 12 + 5 + 4 = 21
        assert_eq!(plan.tensor_count(SdxlConvertedComponent::ClipG), 21);
        assert!(output.join("text_encoder_2/model.safetensors").is_file());
        let clip_g = candle_core::safetensors::load(
            output.join("text_encoder_2/model.safetensors"),
            &Device::Cpu,
        )
        .unwrap();
        assert!(
            clip_g.contains_key("text_projection.weight"),
            "OpenCLIP-G text_projection should be preserved for Burn packaging"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn openclipg_in_proj_weight_slice_produces_correct_shapes() {
        let dir = temp_dir("openclipg-slice");
        let source = dir.join("source.safetensors");
        let output = dir.join("converted");
        fs::create_dir_all(&dir).unwrap();

        // Write a real fused in_proj_weight like OpenCLIP-G uses:
        // shape [3840, 1280], 32-bit floats, sequential values as a
        // crude sentinel.
        let mut data = Vec::with_capacity(3840 * 1280);
        for i in 0..(3840 * 1280) {
            data.push(i as f32);
        }
        write_multi_safetensors(
            &source,
            &[
                // ClipG OpenCLIP-G tensors
                (
                    "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
                    data.clone(),
                    &[3840usize, 1280],
                ),
                (
                    "conditioner.embedders.1.model.transformer.resblocks.0.attn.out_proj.weight",
                    vec![0.0f32; 1280 * 1280],
                    &[1280usize, 1280],
                ),
                (
                    "conditioner.embedders.1.model.transformer.resblocks.0.ln_1.weight",
                    vec![1.0f32; 1280],
                    &[1280usize],
                ),
                (
                    "conditioner.embedders.1.model.token_embedding.weight",
                    vec![0.0f32; 49408 * 1280],
                    &[49408usize, 1280],
                ),
                (
                    "conditioner.embedders.1.model.positional_embedding",
                    vec![0.0f32; 77 * 1280],
                    &[77usize, 1280],
                ),
                (
                    "conditioner.embedders.1.model.ln_final.weight",
                    vec![1.0f32; 1280],
                    &[1280usize],
                ),
                // Minimal Unet (validate_complete needs all families).
                ("conv_in.weight", vec![0.0f32; 16], &[4usize, 4]),
                ("time_embedding.linear_1.weight", vec![0.0f32; 4], &[4usize]),
                (
                    "down_blocks.0.resnets.0.norm1.weight",
                    vec![0.0f32; 4],
                    &[4usize],
                ),
                (
                    "down_blocks.1.attentions.0.proj_in.weight",
                    vec![0.0f32; 4],
                    &[4usize],
                ),
                (
                    "down_blocks.0.downsamplers.0.conv.weight",
                    vec![0.0f32; 4],
                    &[4usize],
                ),
                (
                    "mid_block.resnets.0.norm1.weight",
                    vec![0.0f32; 4],
                    &[4usize],
                ),
                (
                    "mid_block.attentions.0.proj_in.weight",
                    vec![0.0f32; 4],
                    &[4usize],
                ),
                (
                    "up_blocks.0.resnets.0.conv_shortcut.weight",
                    vec![0.0f32; 4],
                    &[4usize],
                ),
                (
                    "up_blocks.0.attentions.0.proj_in.weight",
                    vec![0.0f32; 4],
                    &[4usize],
                ),
                (
                    "up_blocks.0.upsamplers.0.conv.weight",
                    vec![0.0f32; 4],
                    &[4usize],
                ),
                ("conv_norm_out.weight", vec![0.0f32; 4], &[4usize]),
                ("conv_out.weight", vec![0.0f32; 4], &[4usize]),
                // Minimal ClipL.
                (
                    "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
                    vec![0.0f32; 768 * 49408],
                    &[49408usize, 768],
                ),
                (
                    "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight",
                    vec![0.0f32; 77 * 768],
                    &[77usize, 768],
                ),
                (
                    "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
                    vec![0.0f32; 768 * 768],
                    &[768usize, 768],
                ),
                (
                    "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight",
                    vec![0.0f32; 768 * 768],
                    &[768usize, 768],
                ),
                (
                    "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.v_proj.weight",
                    vec![0.0f32; 768 * 768],
                    &[768usize, 768],
                ),
                (
                    "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.out_proj.weight",
                    vec![0.0f32; 768 * 768],
                    &[768usize, 768],
                ),
                (
                    "conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm1.weight",
                    vec![0.0f32; 768],
                    &[768usize],
                ),
                (
                    "conditioner.embedders.0.transformer.text_model.final_layer_norm.weight",
                    vec![0.0f32; 768],
                    &[768usize],
                ),
                // Minimal Vae.
                (
                    "first_stage_model.encoder.conv_in.weight",
                    vec![0.0f32; 4],
                    &[4usize],
                ),
                (
                    "first_stage_model.encoder.conv_out.weight",
                    vec![0.0f32; 4],
                    &[4usize],
                ),
                (
                    "first_stage_model.encoder.conv_norm_out.weight",
                    vec![0.0f32; 4],
                    &[4usize],
                ),
                (
                    "first_stage_model.decoder.conv_in.weight",
                    vec![0.0f32; 4],
                    &[4usize],
                ),
                (
                    "first_stage_model.decoder.conv_out.weight",
                    vec![0.0f32; 4],
                    &[4usize],
                ),
                (
                    "first_stage_model.decoder.conv_norm_out.weight",
                    vec![0.0f32; 4],
                    &[4usize],
                ),
                (
                    "first_stage_model.quant_conv.weight",
                    vec![0.0f32; 4],
                    &[4usize],
                ),
                (
                    "first_stage_model.post_quant_conv.weight",
                    vec![0.0f32; 4],
                    &[4usize],
                ),
            ],
        );

        let plan = write_sdxl_checkpoint_components(&source, &output, |component| {
            component.relative_path().to_owned()
        })
        .expect("OpenCLIP-G with shaped tensors should write successfully");

        // Layer 0 has in_proj_weight (→3) + out_proj.weight (1) + ln_1.weight (1) + top-level (3) = 8
        assert_eq!(plan.tensor_count(SdxlConvertedComponent::ClipG), 8);

        // Read back the ClipG file and verify shapes.
        let clip_g = candle_core::safetensors::load(
            output.join("text_encoder_2/model.safetensors"),
            &Device::Cpu,
        )
        .expect("should read back ClipG safetensors");

        // q/k/v each should be [1280, 1280].
        let q_proj = clip_g
            .get("transformer.text_model.encoder.layers.0.self_attn.q_proj.weight")
            .expect("q_proj.weight should exist");
        assert_eq!(q_proj.shape().dims(), &[1280, 1280]);
        let k_proj = clip_g
            .get("transformer.text_model.encoder.layers.0.self_attn.k_proj.weight")
            .expect("k_proj.weight should exist");
        assert_eq!(k_proj.shape().dims(), &[1280, 1280]);
        let v_proj = clip_g
            .get("transformer.text_model.encoder.layers.0.self_attn.v_proj.weight")
            .expect("v_proj.weight should exist");
        assert_eq!(v_proj.shape().dims(), &[1280, 1280]);

        // Verify split content: row 0 of q_proj should equal row 0 of source,
        // row 0 of k_proj should equal row 1280, row 0 of v_proj should equal row 2560.
        let q_row0: Vec<f32> = q_proj.get(0).unwrap().to_vec1().unwrap();
        let k_row0: Vec<f32> = k_proj.get(0).unwrap().to_vec1().unwrap();
        let v_row0: Vec<f32> = v_proj.get(0).unwrap().to_vec1().unwrap();

        // We know source data was sequential: element (r,c) = (r * 1280 + c) as f32.
        let expected_q_row0: Vec<f32> = (0u32..1280).map(|i| i as f32).collect();
        let expected_k_row0: Vec<f32> = ((1280 * 1280) as u32..(1280 * 1280 + 1280) as u32)
            .map(|i| i as f32)
            .collect();
        let expected_v_row0: Vec<f32> = ((2560 * 1280) as u32..(2560 * 1280 + 1280) as u32)
            .map(|i| i as f32)
            .collect();

        assert_eq!(q_row0, expected_q_row0, "q_proj row 0 data mismatch");
        assert_eq!(k_row0, expected_k_row0, "k_proj row 0 data mismatch");
        assert_eq!(v_row0, expected_v_row0, "v_proj row 0 data mismatch");

        // out_proj.weight should be [1280, 1280].
        let out_proj = clip_g
            .get("transformer.text_model.encoder.layers.0.self_attn.out_proj.weight")
            .expect("out_proj.weight should exist");
        assert_eq!(out_proj.shape().dims(), &[1280, 1280]);

        let _ = fs::remove_dir_all(dir);
    }

    // ---- Original SDXL VAE compvis → diffusers mapping tests ----

    /// Minimal set of UNet + CLIP source keys (diffusers layout) plus a
    /// full set of compvis VAE source keys. Used to exercise the compvis
    /// VAE mapping end-to-end against the validator and writer without
    /// requiring UNet/CLIP real weights.
    fn compvis_vae_full_split_names() -> Vec<&'static str> {
        let mut names: Vec<&'static str> = vec![
            // Minimal UNet (matches one of each required family).
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
        names.extend(compvis_vae_source_names());
        names
    }

    /// Minimal set of compvis VAE source keys covering each supported
    /// mapping category (resnet block, downsampler, mid resnet, mid
    /// attention q/k/v + norm + proj_out, norm_out → conv_norm_out,
    /// quant_conv / post_quant_conv, decoder upsample / up resnet).
    fn compvis_vae_source_names() -> Vec<&'static str> {
        vec![
            "first_stage_model.encoder.conv_in.weight",
            "first_stage_model.encoder.conv_out.weight",
            "first_stage_model.encoder.norm_out.weight",
            "first_stage_model.encoder.down.0.block.0.norm1.weight",
            "first_stage_model.encoder.down.0.block.0.conv1.weight",
            "first_stage_model.encoder.down.0.block.0.norm2.weight",
            "first_stage_model.encoder.down.0.block.0.conv2.weight",
            "first_stage_model.encoder.down.1.block.0.nin_shortcut.weight",
            "first_stage_model.encoder.down.0.downsample.conv.weight",
            "first_stage_model.encoder.mid.block_1.norm1.weight",
            "first_stage_model.encoder.mid.block_2.conv2.weight",
            "first_stage_model.encoder.mid.attn_1.norm.weight",
            "first_stage_model.encoder.mid.attn_1.q.weight",
            "first_stage_model.encoder.mid.attn_1.k.bias",
            "first_stage_model.encoder.mid.attn_1.v.weight",
            "first_stage_model.encoder.mid.attn_1.proj_out.weight",
            "first_stage_model.decoder.conv_in.weight",
            "first_stage_model.decoder.conv_out.weight",
            "first_stage_model.decoder.norm_out.weight",
            "first_stage_model.decoder.up.0.block.0.norm1.weight",
            "first_stage_model.decoder.up.1.block.0.nin_shortcut.weight",
            "first_stage_model.decoder.up.1.upsample.conv.weight",
            "first_stage_model.decoder.mid.block_1.norm1.weight",
            "first_stage_model.decoder.mid.attn_1.norm.weight",
            "first_stage_model.quant_conv.weight",
            "first_stage_model.post_quant_conv.weight",
        ]
    }

    #[test]
    fn compvis_vae_keys_are_written_with_diffusers_target_names() {
        let dir = temp_dir("compvis-vae-write");
        let source = dir.join("source.safetensors");
        let output = dir.join("converted");
        write_safetensors(&source, &compvis_vae_full_split_names());

        let plan = write_sdxl_checkpoint_components(&source, &output, |component| {
            component.relative_path().to_owned()
        })
        .expect("compvis VAE keys should map and write successfully");

        // 26 compvis VAE source keys all produce a single diffusers target.
        assert_eq!(plan.tensor_count(SdxlConvertedComponent::Vae), 26);
        assert!(output.join("vae/model.safetensors").is_file());

        let vae =
            candle_core::safetensors::load(output.join("vae/model.safetensors"), &Device::Cpu)
                .expect("should read back VAE safetensors");

        // compvis `encoder.down.0.block.0.norm1.weight` →
        // diffusers `encoder.down_blocks.0.resnets.0.norm1.weight`
        assert!(vae.contains_key("encoder.down_blocks.0.resnets.0.norm1.weight"));
        assert!(vae.contains_key("encoder.down_blocks.0.resnets.0.conv1.weight"));
        assert!(vae.contains_key("encoder.down_blocks.0.resnets.0.norm2.weight"));
        assert!(vae.contains_key("encoder.down_blocks.0.resnets.0.conv2.weight"));
        // compvis nin_shortcut maps to Candle/Diffusers conv_shortcut.
        assert!(vae.contains_key("encoder.down_blocks.1.resnets.0.conv_shortcut.weight"));
        // downsample → downsamplers.0
        assert!(vae.contains_key("encoder.down_blocks.0.downsamplers.0.conv.weight"));
        // mid.block_1 → mid_block.resnets.0, mid.block_2 → mid_block.resnets.1
        assert!(vae.contains_key("encoder.mid_block.resnets.0.norm1.weight"));
        assert!(vae.contains_key("encoder.mid_block.resnets.1.conv2.weight"));
        // mid.attn_1.norm → attentions.0.group_norm
        assert!(vae.contains_key("encoder.mid_block.attentions.0.group_norm.weight"));
        // mid.attn_1.q/k/v → attentions.0.to_q/k/v
        assert!(vae.contains_key("encoder.mid_block.attentions.0.to_q.weight"));
        assert!(vae.contains_key("encoder.mid_block.attentions.0.to_k.bias"));
        assert!(vae.contains_key("encoder.mid_block.attentions.0.to_v.weight"));
        // mid.attn_1.proj_out → attentions.0.to_out.0
        assert!(vae.contains_key("encoder.mid_block.attentions.0.to_out.0.weight"));
        // norm_out → conv_norm_out
        assert!(vae.contains_key("encoder.conv_norm_out.weight"));
        // decoder up blocks are reversed by Candle's AutoEncoderKL construction.
        assert!(vae.contains_key("decoder.up_blocks.3.resnets.0.norm1.weight"));
        assert!(vae.contains_key("decoder.up_blocks.2.resnets.0.conv_shortcut.weight"));
        assert!(vae.contains_key("decoder.up_blocks.2.upsamplers.0.conv.weight"));
        assert!(vae.contains_key("decoder.mid_block.resnets.0.norm1.weight"));
        assert!(vae.contains_key("decoder.mid_block.attentions.0.group_norm.weight"));
        assert!(vae.contains_key("decoder.conv_norm_out.weight"));
        // quant_conv / post_quant_conv unchanged
        assert!(vae.contains_key("quant_conv.weight"));
        assert!(vae.contains_key("post_quant_conv.weight"));

        // Verify the compvis-style keys are NOT in the output file.
        assert!(!vae.contains_key("encoder.down.0.block.0.norm1.weight"));
        assert!(!vae.contains_key("encoder.norm_out.weight"));
        assert!(!vae.contains_key("encoder.mid.attn_1.proj_out.weight"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn compvis_vae_optional_resnet_subkey_absence_still_passes() {
        // `encoder.down.0.block.0.norm1.weight` is a representative
        // compvis resnet subkey. It maps to a diffusers target but that
        // target is NOT in REQUIRED_VAE_TARGETS, so its absence must
        // not fail the writer — the import produces one fewer VAE
        // tensor and the resulting safetensors file still validates.
        let dir = temp_dir("compvis-vae-optional-missing");
        let source = dir.join("source.safetensors");
        let output = dir.join("converted");
        let names: Vec<&'static str> = compvis_vae_full_split_names()
            .into_iter()
            .filter(|n| *n != "first_stage_model.encoder.down.0.block.0.norm1.weight")
            .collect();
        write_safetensors(&source, &names);

        let plan = write_sdxl_checkpoint_components(&source, &output, |component| {
            component.relative_path().to_owned()
        })
        .expect("writer accepts when optional resnet subkeys are missing");
        assert_eq!(plan.tensor_count(SdxlConvertedComponent::Vae), 25);

        let vae =
            candle_core::safetensors::load(output.join("vae/model.safetensors"), &Device::Cpu)
                .expect("should read back VAE safetensors");
        assert!(!vae.contains_key("encoder.down_blocks.0.resnets.0.norm1.weight"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn compvis_vae_fails_fast_when_norm_out_renames_are_missing() {
        let dir = temp_dir("compvis-vae-norm-out-missing");
        let source = dir.join("source.safetensors");
        let output = dir.join("converted");
        // Drop the encoder.norm_out.weight key — `encoder.conv_norm_out.weight`
        // is a REQUIRED_VAE_TARGETS so the validator must reject this input.
        let names: Vec<&'static str> = compvis_vae_full_split_names()
            .into_iter()
            .filter(|n| *n != "first_stage_model.encoder.norm_out.weight")
            .collect();
        write_safetensors(&source, &names);

        let err = write_sdxl_checkpoint_components(&source, &output, |component| {
            component.relative_path().to_owned()
        })
        .unwrap_err();

        assert!(matches!(
            err,
            SdxlCheckpointWriterError::MissingRequiredTarget {
                component: "vae",
                target: "encoder.conv_norm_out.weight"
            }
        ));

        let _ = fs::remove_dir_all(dir);
    }
}
