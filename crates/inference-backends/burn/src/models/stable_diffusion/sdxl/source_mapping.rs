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

const DIFFUSION_MAPPINGS: &[TensorMapping] = &[
    TensorMapping::new("model.diffusion.conv_in.weight", "conv_in.weight"),
    TensorMapping::new("model.diffusion.conv_in.bias", "conv_in.bias"),
    TensorMapping::new(
        "model.diffusion.time_embed.0.weight",
        "time_embedding.linear_1.weight",
    ),
    TensorMapping::new(
        "model.diffusion.time_embed.0.bias",
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
        "model.diffusion.input_blocks.1.0.in_layers.2.weight",
        "down_blocks.0.res_blocks.0.conv_1.weight",
    ),
    TensorMapping::new(
        "model.diffusion.input_blocks.1.0.in_layers.2.bias",
        "down_blocks.0.res_blocks.0.conv_1.bias",
    ),
    TensorMapping::new(
        "model.diffusion.input_blocks.1.0.emb_layers.1.weight",
        "down_blocks.0.res_blocks.0.time_projection.weight",
    ),
    TensorMapping::new(
        "model.diffusion.input_blocks.1.0.emb_layers.1.bias",
        "down_blocks.0.res_blocks.0.time_projection.bias",
    ),
    TensorMapping::new(
        "model.diffusion.input_blocks.1.0.out_layers.3.weight",
        "down_blocks.0.res_blocks.0.conv_2.weight",
    ),
    TensorMapping::new(
        "model.diffusion.input_blocks.1.0.out_layers.3.bias",
        "down_blocks.0.res_blocks.0.conv_2.bias",
    ),
    TensorMapping::new("model.diffusion.out.0.weight", "conv_out.weight"),
    TensorMapping::new("model.diffusion.out.0.bias", "conv_out.bias"),
];
const VAE_MAPPINGS: &[TensorMapping] = &[
    TensorMapping::new("encoder.conv_in.weight", "model.vae.encoder.conv_in.weight"),
    TensorMapping::new(
        "decoder.conv_out.weight",
        "model.vae.decoder.conv_out.weight",
    ),
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
    let components = map_split_source_to_components(source_set)?;
    let plan = SyntheticSdxlConversionPlan {
        source_identity: source_set.root().display().to_string(),
        components,
    };
    let mut report = write_synthetic_sdxl_components(&plan, output_dir)?;
    report.source_layout = DIFFUSERS_STYLE_SPLIT_SAFETENSORS.to_owned();
    write_conversion_report(&report, output_dir.join(BURN_SDXL_CONVERSION_REPORT_FILE))?;
    Ok(report)
}

fn map_split_source_to_components(
    source_set: &BurnSdxlSourceSet,
) -> Result<Vec<BurnSdxlSyntheticComponent>, BurnSdxlConversionError> {
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

    source_files.into_iter().map(map_source_component).collect()
}

#[derive(Debug, Clone, Copy)]
struct TensorMapping {
    source_key: &'static str,
    target_key: &'static str,
}

impl TensorMapping {
    const fn new(source_key: &'static str, target_key: &'static str) -> Self {
        Self {
            source_key,
            target_key,
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

fn map_source_component(
    source: SourceComponent,
) -> Result<BurnSdxlSyntheticComponent, BurnSdxlConversionError> {
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
    if !unknown_keys.is_empty() {
        return Err(BurnSdxlConversionError::InvalidComponentSet {
            reason: format!(
                "unsupported source tensor(s) in `{}`: {}",
                source.path.display(),
                unknown_keys.join(", ")
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

    let tensors = source
        .mappings
        .iter()
        .map(|mapping| {
            let source_tensor = source_by_key.get(mapping.source_key).ok_or_else(|| {
                BurnSdxlConversionError::InvalidComponentSet {
                    reason: format!(
                        "missing source tensor `{}` in `{}`",
                        mapping.source_key,
                        source.path.display()
                    ),
                }
            })?;
            Ok(BurnSyntheticTensor {
                key: mapping.target_key.to_owned(),
                shape: source_tensor.shape.clone(),
                dtype: source_tensor.dtype.clone(),
                source: BurnTensorSource::Data(source_tensor.data.clone()),
            })
        })
        .collect::<Result<Vec<_>, BurnSdxlConversionError>>()?;

    Ok(BurnSdxlSyntheticComponent {
        role: source.role,
        dtype_policy: BurnDTypePolicy::Fp32,
        tensors,
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
    use super::super::validation::validate_component_inventory;
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

    fn write_source_file(path: &Path, tensors: &[(&str, Vec<usize>)]) {
        fs::create_dir_all(path.parent().expect("source path has parent")).unwrap();
        let views = tensors
            .iter()
            .map(|(name, shape)| ((*name).to_owned(), TestTensorView::f32(shape.clone())))
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
                ("encoder.conv_in.weight", vec![1, 1, 1, 1]),
                ("decoder.conv_out.weight", vec![1, 1, 1, 1]),
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

    fn write_legacy_representative_split_source(root: &Path) {
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
                ("model.diffusion.out.0.weight", vec![4, 320, 3, 3]),
                ("model.diffusion.out.0.bias", vec![4]),
            ],
        );
    }

    #[test]
    fn rejects_legacy_diffusion_representative_source_keys_before_writing_output() {
        let source = tempfile::tempdir().expect("source temp dir");
        let output = tempfile::tempdir().expect("output temp dir");
        write_legacy_representative_split_source(source.path());
        let source_set =
            BurnSdxlSourceSet::diffusers_style_split_safetensors(source.path().to_path_buf());

        let err = map_diffusers_style_split_source(&source_set, output.path())
            .expect_err("legacy diffusion source keys should fail");

        assert!(err.to_string().contains("conv_in.weight"));
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
        assert_eq!(report.mapped_tensor_count, 20);
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
            assert_eq!(validation.matched_required_tensors.len(), 2);
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
            "down_blocks.0.res_blocks.0.conv_1.weight",
            "down_blocks.0.res_blocks.0.time_projection.weight",
            "down_blocks.0.res_blocks.0.conv_2.weight",
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
        write_source_file(&source.path().join("unet/model.safetensors"), &[]);
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

        assert!(
            err.to_string()
                .contains("model.diffusion.time_embed.0.weight")
        );
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
