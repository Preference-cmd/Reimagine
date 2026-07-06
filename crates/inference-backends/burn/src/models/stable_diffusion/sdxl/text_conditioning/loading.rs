//! Burn-store owned CLIP text encoder loading from safetensors component files.
//!
//! Component-backed SDXL `text.encode` loads the CLIP-L and OpenCLIP-G
//! [`SdxlTextEncoders`] Module graph through burn-store. Raw f32 buffer
//! loaders are intentionally not part of this production boundary.

use burn_store::ApplyResult;
use burn_tensor::backend::Backend;

use crate::error::BurnBackendError;
use crate::models::stable_diffusion::sdxl::load_diagnostics::{
    SdxlLoadPolicy, validate_apply_result as validate_sdxl_apply_result,
};
use crate::models::stable_diffusion::sdxl::loaded::BurnLoadedSdxlBundle;
use crate::models::stable_diffusion::sdxl::text_conditioning::store::sdxl_clip_store_from_path;
use crate::runtime::BurnRuntime;
use crate::text_encoder::clip::ClipTextEncoderProfile;

use super::module::SdxlTextEncoders;

/// Load both SDXL text encoders as Burn-native modules through burn-store.
#[allow(dead_code)]
pub(crate) fn load_text_encoder_modules<B: Backend>(
    runtime: &BurnRuntime<B>,
    bundle: &BurnLoadedSdxlBundle,
) -> Result<SdxlTextEncoders<B>, BurnBackendError> {
    let (clip_l_profile, open_clip_g_profile) = text_encoder_profiles_for_bundle(bundle);
    load_text_encoder_modules_from_profiles(runtime, bundle, &clip_l_profile, &open_clip_g_profile)
}

fn text_encoder_profiles_for_bundle(
    bundle: &BurnLoadedSdxlBundle,
) -> (ClipTextEncoderProfile, ClipTextEncoderProfile) {
    if bundle.uses_tiny_sdxl_e2e_text_profiles() {
        return (
            ClipTextEncoderProfile::tiny_sdxl_clip_l(),
            ClipTextEncoderProfile::tiny_sdxl_open_clip_g(),
        );
    }

    (
        ClipTextEncoderProfile::sdxl_clip_l(),
        ClipTextEncoderProfile::sdxl_open_clip_g(),
    )
}

#[allow(dead_code)]
fn load_text_encoder_modules_from_profiles<B: Backend>(
    runtime: &BurnRuntime<B>,
    bundle: &BurnLoadedSdxlBundle,
    clip_l_profile: &ClipTextEncoderProfile,
    open_clip_g_profile: &ClipTextEncoderProfile,
) -> Result<SdxlTextEncoders<B>, BurnBackendError> {
    let (primary_path, secondary_path) = bundle.text_encoder_component_paths()?;
    let mut module = SdxlTextEncoders::<B>::init_from_profiles(
        clip_l_profile,
        open_clip_g_profile,
        runtime.device(),
    );

    let mut primary_store = sdxl_clip_store_from_path(primary_path);
    let primary_result = runtime
        .load_module_store(&mut module, &mut primary_store)
        .map_err(|source| BurnBackendError::InvalidRequest(source.to_string()))?;
    validate_apply_result("text_encoder", &primary_result)?;

    let mut secondary_store = sdxl_clip_store_from_path(secondary_path);
    let secondary_result = runtime
        .load_module_store(&mut module, &mut secondary_store)
        .map_err(|source| BurnBackendError::InvalidRequest(source.to_string()))?;
    validate_apply_result("text_encoder_2", &secondary_result)?;

    Ok(module)
}

#[allow(dead_code)]
fn validate_apply_result(component: &str, result: &ApplyResult) -> Result<(), BurnBackendError> {
    validate_sdxl_apply_result(clip_load_policy(component), result)
}

fn clip_load_policy(component: &str) -> SdxlLoadPolicy {
    match component {
        "text_encoder" => SdxlLoadPolicy::new("text_encoder")
            .with_required_prefixes(&["clip_l."])
            .with_generated_snapshot_contains(&[
                ".attention.query.",
                ".attention.key.",
                ".attention.value.",
            ])
            .with_remapped_key_patterns(&[
                "model.text_encoder -> clip_l",
                ".transformer.resblocks. -> .blocks.",
                ".attn.in_proj_* -> generated q/k/v snapshots",
            ]),
        "text_encoder_2" => SdxlLoadPolicy::new("text_encoder_2")
            .with_required_prefixes(&["open_clip_g."])
            .with_optional_snapshots(&[
                "open_clip_g.text_projection.weight",
                "open_clip_g.text_projection.bias",
            ])
            .with_generated_snapshot_contains(&[
                ".attention.query.",
                ".attention.key.",
                ".attention.value.",
            ])
            .with_remapped_key_patterns(&[
                "model.text_encoder_2 -> open_clip_g",
                ".transformer.resblocks. -> .blocks.",
                ".attn.in_proj_* -> generated q/k/v snapshots",
            ]),
        _ => SdxlLoadPolicy::new("text_encoder"),
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::fs;
    use std::path::Path;

    use crate::active_backend::{ActiveBurnBackend, active_device};
    use crate::config::BurnBackendConfig;
    use burn_tensor::{Int, Tensor};
    use reimagine_core::model::ModelId;
    use reimagine_inference::BackendPayloadKey;
    use safetensors::tensor::{Dtype, View, serialize_to_file};

    use crate::models::stable_diffusion::sdxl::component::BurnSdxlComponentRole;
    use crate::models::stable_diffusion::sdxl::loaded::BurnLoadedSdxlBundle;
    use crate::runtime::BurnRuntime;
    use crate::text_encoder::clip::{ClipTextEncoderProfile, ClipTextEncoderVariant};

    #[test]
    fn load_text_encoder_modules_loads_component_paths_through_burn_store() {
        type B = ActiveBurnBackend;

        let temp = tempfile::tempdir().expect("temp dir");
        let primary_path = temp.path().join("text_encoder.safetensors");
        let secondary_path = temp.path().join("text_encoder_2.safetensors");
        write_text_encoder_component(&primary_path, "model.text_encoder");
        write_text_encoder_component(&secondary_path, "model.text_encoder_2");
        let bundle = BurnLoadedSdxlBundle::for_test_only(
            ModelId::new("unit-sdxl"),
            BackendPayloadKey::new("clip"),
        )
        .with_test_components(vec![
            (BurnSdxlComponentRole::TextEncoder, primary_path),
            (BurnSdxlComponentRole::TextEncoder2, secondary_path),
        ]);
        let runtime = BurnRuntime::<B>::new(active_test_device());
        let clip_l_profile = tiny_profile(ClipTextEncoderVariant::ClipL, false);
        let open_clip_g_profile = tiny_profile(ClipTextEncoderVariant::OpenClipG, true);

        let module = super::load_text_encoder_modules_from_profiles(
            &runtime,
            &bundle,
            &clip_l_profile,
            &open_clip_g_profile,
        )
        .expect("text encoder modules should load through burn-store");

        assert_param_2d(
            &module.clip_l.blocks()[0].attention.query.weight,
            [1.0, 3.0, 2.0, 4.0],
        );
        assert_param_2d(
            &module.clip_l.blocks()[0].attention.key.weight,
            [5.0, 7.0, 6.0, 8.0],
        );
        assert_param_2d(
            &module.open_clip_g.blocks()[0].attention.value.weight,
            [9.0, 11.0, 10.0, 12.0],
        );
    }

    #[test]
    fn loaded_text_encoder_modules_can_run_burn_native_forward() {
        type B = ActiveBurnBackend;

        let temp = tempfile::tempdir().expect("temp dir");
        let primary_path = temp.path().join("text_encoder.safetensors");
        let secondary_path = temp.path().join("text_encoder_2.safetensors");
        write_text_encoder_component(&primary_path, "model.text_encoder");
        write_text_encoder_component(&secondary_path, "model.text_encoder_2");
        let bundle = BurnLoadedSdxlBundle::for_test_only(
            ModelId::new("unit-sdxl"),
            BackendPayloadKey::new("clip"),
        )
        .with_test_components(vec![
            (BurnSdxlComponentRole::TextEncoder, primary_path),
            (BurnSdxlComponentRole::TextEncoder2, secondary_path),
        ]);
        let runtime = BurnRuntime::<B>::new(active_test_device());
        let clip_l_profile = tiny_profile(ClipTextEncoderVariant::ClipL, false);
        let open_clip_g_profile = tiny_profile(ClipTextEncoderVariant::OpenClipG, true);
        let module = super::load_text_encoder_modules_from_profiles(
            &runtime,
            &bundle,
            &clip_l_profile,
            &open_clip_g_profile,
        )
        .expect("text encoder modules should load through burn-store");
        let token_ids = Tensor::<B, 2, Int>::from_ints([[1, 2, 3, 4, 5]], runtime.device());

        let clip_l = module.clip_l.forward(token_ids.clone());
        let clip_g = module.open_clip_g.forward(token_ids);

        assert_eq!(clip_l.hidden.dims(), [1, 5, 2]);
        assert!(clip_l.pooled.is_none());
        assert_eq!(clip_g.hidden.dims(), [1, 5, 2]);
        assert_eq!(clip_g.pooled.expect("pooled output").dims(), [1, 2]);
    }

    #[test]
    fn load_text_encoder_modules_rejects_missing_required_snapshots_with_policy_report() {
        type B = ActiveBurnBackend;

        let temp = tempfile::tempdir().expect("temp dir");
        let primary_path = temp.path().join("text_encoder.safetensors");
        let secondary_path = temp.path().join("text_encoder_2.safetensors");
        write_incomplete_text_encoder_component(&primary_path, "model.text_encoder");
        write_text_encoder_component(&secondary_path, "model.text_encoder_2");
        let bundle = BurnLoadedSdxlBundle::for_test_only(
            ModelId::new("unit-sdxl"),
            BackendPayloadKey::new("clip"),
        )
        .with_test_components(vec![
            (BurnSdxlComponentRole::TextEncoder, primary_path),
            (BurnSdxlComponentRole::TextEncoder2, secondary_path),
        ]);
        let runtime = BurnRuntime::<B>::new(active_test_device());
        let clip_l_profile = tiny_profile(ClipTextEncoderVariant::ClipL, false);
        let open_clip_g_profile = tiny_profile(ClipTextEncoderVariant::OpenClipG, true);

        let err = super::load_text_encoder_modules_from_profiles(
            &runtime,
            &bundle,
            &clip_l_profile,
            &open_clip_g_profile,
        )
        .expect_err("missing required CLIP snapshots should fail validation");
        let message = err.to_string();

        assert!(message.contains("required snapshot missing"), "{message}");
        assert!(message.contains("component_role=text_encoder"), "{message}");
        assert!(message.contains("partial load policy"), "{message}");
    }

    fn write_text_encoder_component(path: &Path, prefix: &str) {
        let tensors = vec![
            tensor_view(
                &format!("{prefix}.token_embedding.weight"),
                vec![16, 2],
                (0..32).map(|value| value as f32 / 100.0).collect(),
            ),
            tensor_view(
                &format!("{prefix}.position_embedding.weight"),
                vec![5, 2],
                (0..10).map(|value| value as f32 / 50.0).collect(),
            ),
            tensor_view(
                &format!("{prefix}.final_layer_norm.gamma"),
                vec![2],
                vec![1.0, 1.0],
            ),
            tensor_view(
                &format!("{prefix}.final_layer_norm.beta"),
                vec![2],
                vec![0.0, 0.0],
            ),
            tensor_view(
                &format!("{prefix}.text_projection.weight"),
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            ),
            tensor_view(
                &format!("{prefix}.text_projection.bias"),
                vec![2],
                vec![0.0, 0.0],
            ),
            tensor_view(
                &format!("{prefix}.transformer.resblocks.0.ln_1.weight"),
                vec![2],
                vec![1.0, 1.0],
            ),
            tensor_view(
                &format!("{prefix}.transformer.resblocks.0.ln_1.bias"),
                vec![2],
                vec![0.0, 0.0],
            ),
            tensor_view(
                &format!("{prefix}.transformer.resblocks.0.ln_2.weight"),
                vec![2],
                vec![1.0, 1.0],
            ),
            tensor_view(
                &format!("{prefix}.transformer.resblocks.0.ln_2.bias"),
                vec![2],
                vec![0.0, 0.0],
            ),
            tensor_view(
                &format!("{prefix}.transformer.resblocks.0.attn.in_proj_weight"),
                vec![6, 2],
                vec![
                    1.0, 2.0, // query row 0
                    3.0, 4.0, // query row 1
                    5.0, 6.0, // key row 0
                    7.0, 8.0, // key row 1
                    9.0, 10.0, // value row 0
                    11.0, 12.0, // value row 1
                ],
            ),
            tensor_view(
                &format!("{prefix}.transformer.resblocks.0.attn.in_proj_bias"),
                vec![6],
                vec![101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            ),
            tensor_view(
                &format!("{prefix}.transformer.resblocks.0.attn.out_proj.weight"),
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            ),
            tensor_view(
                &format!("{prefix}.transformer.resblocks.0.attn.out_proj.bias"),
                vec![2],
                vec![0.0, 0.0],
            ),
            tensor_view(
                &format!("{prefix}.transformer.resblocks.0.mlp.fc1.weight"),
                vec![8, 2],
                vec![
                    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13,
                    0.14, 0.15, 0.16,
                ],
            ),
            tensor_view(
                &format!("{prefix}.transformer.resblocks.0.mlp.fc1.bias"),
                vec![8],
                vec![0.0; 8],
            ),
            tensor_view(
                &format!("{prefix}.transformer.resblocks.0.mlp.fc2.weight"),
                vec![2, 8],
                vec![
                    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13,
                    0.14, 0.15, 0.16,
                ],
            ),
            tensor_view(
                &format!("{prefix}.transformer.resblocks.0.mlp.fc2.bias"),
                vec![2],
                vec![0.0, 0.0],
            ),
        ];
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("component parent dir");
        }
        serialize_to_file(tensors, None, path).expect("write component safetensors");
    }

    fn write_incomplete_text_encoder_component(path: &Path, prefix: &str) {
        let tensors = vec![
            tensor_view(
                &format!("{prefix}.token_embedding.weight"),
                vec![16, 2],
                (0..32).map(|value| value as f32 / 100.0).collect(),
            ),
            tensor_view(
                &format!("{prefix}.position_embedding.weight"),
                vec![5, 2],
                (0..10).map(|value| value as f32 / 50.0).collect(),
            ),
        ];
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("component parent dir");
        }
        serialize_to_file(tensors, None, path).expect("write incomplete component safetensors");
    }

    fn tensor_view(path: &str, shape: Vec<usize>, values: Vec<f32>) -> (String, TestTensorView) {
        let data = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        (path.to_string(), TestTensorView { shape, data })
    }

    #[derive(Debug, Clone)]
    struct TestTensorView {
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl View for TestTensorView {
        fn dtype(&self) -> Dtype {
            Dtype::F32
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

    fn tiny_profile(
        variant: ClipTextEncoderVariant,
        produces_pooled_output: bool,
    ) -> ClipTextEncoderProfile {
        ClipTextEncoderProfile {
            variant,
            target_prefix: "test.text_encoder".to_string(),
            num_layers: 1,
            width: 2,
            heads: 1,
            inner_width: 8,
            vocab_size: 16,
            sequence_length: 5,
            produces_pooled_output,
        }
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

    fn active_test_device() -> burn_tensor::Device<ActiveBurnBackend> {
        let config = BurnBackendConfig::new("/models", "/output");
        active_device(config.device())
    }
}
