use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};

use super::checkpoint_import::SdxlConvertedComponent;
use super::checkpoint_mapping::{SdxlTensorMappingError, map_sdxl_checkpoint_tensor};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SdxlComponentWritePlan {
    tensors: BTreeMap<SdxlConvertedComponent, Vec<(String, String)>>,
}

impl SdxlComponentWritePlan {
    pub(crate) fn from_safetensors(
        safetensors: &SafeTensors<'_>,
    ) -> Result<Self, SdxlCheckpointWriterError> {
        let mut tensors: BTreeMap<SdxlConvertedComponent, Vec<(String, String)>> = BTreeMap::new();

        for name in safetensors.names() {
            match map_sdxl_checkpoint_tensor(name) {
                Ok(mapped) => {
                    tensors
                        .entry(mapped.component)
                        .or_default()
                        .push((mapped.target_name, name.to_owned()));
                }
                Err(SdxlTensorMappingError::Ignored) => {}
                Err(error) => {
                    return Err(SdxlCheckpointWriterError::UnsupportedMapping {
                        source_name: name.to_owned(),
                        reason: error.to_string(),
                    });
                }
            }
        }

        Ok(Self { tensors })
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
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SdxlCheckpointWriterError {
    ReadSource { path: PathBuf, reason: String },
    ParseSource { path: PathBuf, reason: String },
    UnsupportedMapping { source_name: String, reason: String },
    MissingComponent { component: &'static str },
    TensorRead { source_name: String, reason: String },
    TensorView { target_name: String, reason: String },
    CreateDirectory { path: PathBuf, reason: String },
    WriteComponent { path: PathBuf, reason: String },
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

    #[test]
    fn write_components_splits_supported_tensor_families() {
        let dir = temp_dir("split");
        let source = dir.join("source.safetensors");
        let output = dir.join("converted");
        write_safetensors(
            &source,
            &[
                "conv_in.weight",
                "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
                "conditioner.embedders.1.model.transformer.text_model.embeddings.token_embedding.weight",
                "first_stage_model.decoder.conv_in.weight",
            ],
        );

        let plan = write_sdxl_checkpoint_components(&source, &output, |component| {
            component.relative_path().to_owned()
        })
        .unwrap();

        assert_eq!(plan.tensor_count(SdxlConvertedComponent::Unet), 1);
        assert_eq!(plan.tensor_count(SdxlConvertedComponent::ClipL), 1);
        assert_eq!(plan.tensor_count(SdxlConvertedComponent::ClipG), 1);
        assert_eq!(plan.tensor_count(SdxlConvertedComponent::Vae), 1);
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
    fn original_unet_mapping_fails_before_writing_components() {
        let dir = temp_dir("unsupported-unet");
        let source = dir.join("source.safetensors");
        let output = dir.join("converted");
        write_safetensors(
            &source,
            &[
                "model.diffusion_model.input_blocks.0.0.weight",
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
        write_safetensors(&source, &["conv_in.weight"]);

        let err = write_sdxl_checkpoint_components(&source, &output, |component| {
            component.relative_path().to_owned()
        })
        .unwrap_err();

        assert!(matches!(
            err,
            SdxlCheckpointWriterError::MissingComponent { component: "vae" }
                | SdxlCheckpointWriterError::MissingComponent {
                    component: "text_encoder"
                }
                | SdxlCheckpointWriterError::MissingComponent {
                    component: "text_encoder_2"
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
