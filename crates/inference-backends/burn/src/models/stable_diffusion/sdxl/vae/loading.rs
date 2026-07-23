//! Load SDXL VAE decoder weights through burn-store.

use burn_store::{ApplyResult, KeyRemapper, PyTorchToBurnAdapter, SafetensorsStore};
use burn_tensor::backend::Backend;

use crate::error::BurnBackendError;
use crate::models::stable_diffusion::sdxl::load_diagnostics::{
    SdxlLoadPolicy, validate_apply_result as validate_sdxl_apply_result,
};
use crate::runtime::BurnRuntime;

use super::module::SdxlVaeDecoder;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SdxlVaeDecoderLoadProfile {
    TinySdxlE2e,
    SdxlBase,
}

/// Load an SDXL VAE decoder Module from a VAE component through burn-store.
#[cfg(test)]
pub(crate) fn load_vae_decoder_module_from_path<B: Backend>(
    runtime: &BurnRuntime<B>,
    module: &mut SdxlVaeDecoder<B>,
    path: impl Into<std::path::PathBuf>,
) -> Result<ApplyResult, BurnBackendError> {
    load_vae_decoder_module_from_path_with_profile(
        runtime,
        module,
        path,
        SdxlVaeDecoderLoadProfile::SdxlBase,
    )
}

pub(crate) fn load_vae_decoder_module_from_path_with_profile<B: Backend>(
    runtime: &BurnRuntime<B>,
    module: &mut SdxlVaeDecoder<B>,
    path: impl Into<std::path::PathBuf>,
    profile: SdxlVaeDecoderLoadProfile,
) -> Result<ApplyResult, BurnBackendError> {
    let mut store = sdxl_vae_store_from_path(path);
    let result = runtime
        .load_module_store(module, &mut store)
        .map_err(|source| BurnBackendError::InvalidRequest(source.to_string()))?;
    validate_apply_result(profile, &result)?;
    Ok(result)
}

fn sdxl_vae_store_from_path(path: impl Into<std::path::PathBuf>) -> SafetensorsStore {
    SafetensorsStore::from_file(path)
        .remap(sdxl_vae_key_remapper())
        .with_from_adapter(PyTorchToBurnAdapter)
        .allow_partial(true)
        .validate(true)
}

fn sdxl_vae_key_remapper() -> KeyRemapper {
    // Package dialect is diffusers AutoencoderKL decoder keys. Loader remapper
    // is only a compatibility layer:
    // - strip optional `model.vae.decoder.` prefixes
    // - accept pre-diffusers-unification Burn singular field names
    //
    // GroupNorm weight/bias → gamma/beta is handled by PyTorchToBurnAdapter.
    KeyRemapper::new()
        .add_pattern(r"^model\.vae\.decoder\.conv_in\.", "conv_in.")
        .expect("static VAE decoder conv_in remapping regex should compile")
        .add_pattern(r"^model\.vae\.decoder\.mid_block\.", "mid_block.")
        .expect("static VAE decoder mid_block remapping regex should compile")
        .add_pattern(r"^model\.vae\.decoder\.up_blocks\.", "up_blocks.")
        .expect("static VAE decoder up_blocks remapping regex should compile")
        .add_pattern(r"^model\.vae\.decoder\.conv_norm_out\.", "conv_norm_out.")
        .expect("static VAE decoder conv_norm_out remapping regex should compile")
        .add_pattern(r"^model\.vae\.decoder\.conv_out\.", "conv_out.")
        .expect("static VAE decoder conv_out remapping regex should compile")
        // Legacy Burn-singular dialect (pre package-dialect unification).
        .add_pattern(r"^mid_block\.attention\.", "mid_block.attentions.0.")
        .expect("static legacy VAE mid attention remapping regex should compile")
        .add_pattern(
            r"^up_blocks\.([0-9]+)\.upsampler\.",
            "up_blocks.$1.upsamplers.0.",
        )
        .expect("static legacy VAE upsampler remapping regex should compile")
        .add_pattern(r"\.to_out\.(weight|bias)$", ".to_out.0.$1")
        .expect("static legacy VAE to_out remapping regex should compile")
}

fn validate_apply_result(
    profile: SdxlVaeDecoderLoadProfile,
    result: &ApplyResult,
) -> Result<(), BurnBackendError> {
    validate_sdxl_apply_result(vae_load_policy(profile), result)
}

fn vae_load_policy(profile: SdxlVaeDecoderLoadProfile) -> SdxlLoadPolicy {
    match profile {
        SdxlVaeDecoderLoadProfile::TinySdxlE2e => SdxlLoadPolicy::new("vae")
            .with_required_snapshots(&["conv_out.weight", "conv_out.bias"])
            .with_remapped_key_patterns(&["model.vae.decoder.conv_out -> conv_out"]),
        SdxlVaeDecoderLoadProfile::SdxlBase => SdxlLoadPolicy::new("vae")
            .with_required_snapshots(&[
                "conv_in.weight",
                "conv_in.bias",
                "mid_block.resnets.0.conv1.weight",
                "mid_block.resnets.0.conv1.bias",
                "mid_block.resnets.0.norm1.gamma",
                "mid_block.resnets.0.norm1.beta",
                "mid_block.resnets.0.norm2.gamma",
                "mid_block.resnets.0.norm2.beta",
                "mid_block.resnets.0.conv2.weight",
                "mid_block.resnets.0.conv2.bias",
                "mid_block.resnets.1.conv1.weight",
                "mid_block.resnets.1.conv1.bias",
                "mid_block.resnets.1.norm1.gamma",
                "mid_block.resnets.1.norm1.beta",
                "mid_block.resnets.1.norm2.gamma",
                "mid_block.resnets.1.norm2.beta",
                "mid_block.resnets.1.conv2.weight",
                "mid_block.resnets.1.conv2.bias",
                "mid_block.attentions.0.group_norm.gamma",
                "mid_block.attentions.0.group_norm.beta",
                "mid_block.attentions.0.to_q.weight",
                "mid_block.attentions.0.to_q.bias",
                "mid_block.attentions.0.to_k.weight",
                "mid_block.attentions.0.to_k.bias",
                "mid_block.attentions.0.to_v.weight",
                "mid_block.attentions.0.to_v.bias",
                "mid_block.attentions.0.to_out.0.weight",
                "mid_block.attentions.0.to_out.0.bias",
                // up_blocks.0: 3 resnets @ 512, upsampler
                "up_blocks.0.resnets.0.conv1.weight",
                "up_blocks.0.resnets.0.conv1.bias",
                "up_blocks.0.resnets.0.norm1.gamma",
                "up_blocks.0.resnets.0.norm1.beta",
                "up_blocks.0.resnets.0.norm2.gamma",
                "up_blocks.0.resnets.0.norm2.beta",
                "up_blocks.0.resnets.0.conv2.weight",
                "up_blocks.0.resnets.0.conv2.bias",
                "up_blocks.0.resnets.1.conv1.weight",
                "up_blocks.0.resnets.1.conv1.bias",
                "up_blocks.0.resnets.1.norm1.gamma",
                "up_blocks.0.resnets.1.norm1.beta",
                "up_blocks.0.resnets.1.norm2.gamma",
                "up_blocks.0.resnets.1.norm2.beta",
                "up_blocks.0.resnets.1.conv2.weight",
                "up_blocks.0.resnets.1.conv2.bias",
                "up_blocks.0.resnets.2.conv1.weight",
                "up_blocks.0.resnets.2.conv1.bias",
                "up_blocks.0.resnets.2.norm1.gamma",
                "up_blocks.0.resnets.2.norm1.beta",
                "up_blocks.0.resnets.2.norm2.gamma",
                "up_blocks.0.resnets.2.norm2.beta",
                "up_blocks.0.resnets.2.conv2.weight",
                "up_blocks.0.resnets.2.conv2.bias",
                "up_blocks.0.upsamplers.0.conv.weight",
                "up_blocks.0.upsamplers.0.conv.bias",
                // up_blocks.1: 3 resnets @ 512, upsampler
                "up_blocks.1.resnets.0.conv1.weight",
                "up_blocks.1.resnets.0.conv1.bias",
                "up_blocks.1.resnets.0.norm1.gamma",
                "up_blocks.1.resnets.0.norm1.beta",
                "up_blocks.1.resnets.0.norm2.gamma",
                "up_blocks.1.resnets.0.norm2.beta",
                "up_blocks.1.resnets.0.conv2.weight",
                "up_blocks.1.resnets.0.conv2.bias",
                "up_blocks.1.resnets.1.conv1.weight",
                "up_blocks.1.resnets.1.conv1.bias",
                "up_blocks.1.resnets.1.norm1.gamma",
                "up_blocks.1.resnets.1.norm1.beta",
                "up_blocks.1.resnets.1.norm2.gamma",
                "up_blocks.1.resnets.1.norm2.beta",
                "up_blocks.1.resnets.1.conv2.weight",
                "up_blocks.1.resnets.1.conv2.bias",
                "up_blocks.1.resnets.2.conv1.weight",
                "up_blocks.1.resnets.2.conv1.bias",
                "up_blocks.1.resnets.2.norm1.gamma",
                "up_blocks.1.resnets.2.norm1.beta",
                "up_blocks.1.resnets.2.norm2.gamma",
                "up_blocks.1.resnets.2.norm2.beta",
                "up_blocks.1.resnets.2.conv2.weight",
                "up_blocks.1.resnets.2.conv2.bias",
                "up_blocks.1.upsamplers.0.conv.weight",
                "up_blocks.1.upsamplers.0.conv.bias",
                // up_blocks.2: 3 resnets, first has skip 512→256, upsampler
                "up_blocks.2.resnets.0.conv1.weight",
                "up_blocks.2.resnets.0.conv1.bias",
                "up_blocks.2.resnets.0.norm1.gamma",
                "up_blocks.2.resnets.0.norm1.beta",
                "up_blocks.2.resnets.0.norm2.gamma",
                "up_blocks.2.resnets.0.norm2.beta",
                "up_blocks.2.resnets.0.conv2.weight",
                "up_blocks.2.resnets.0.conv2.bias",
                "up_blocks.2.resnets.0.conv_shortcut.weight",
                "up_blocks.2.resnets.0.conv_shortcut.bias",
                "up_blocks.2.resnets.1.conv1.weight",
                "up_blocks.2.resnets.1.conv1.bias",
                "up_blocks.2.resnets.1.norm1.gamma",
                "up_blocks.2.resnets.1.norm1.beta",
                "up_blocks.2.resnets.1.norm2.gamma",
                "up_blocks.2.resnets.1.norm2.beta",
                "up_blocks.2.resnets.1.conv2.weight",
                "up_blocks.2.resnets.1.conv2.bias",
                "up_blocks.2.resnets.2.conv1.weight",
                "up_blocks.2.resnets.2.conv1.bias",
                "up_blocks.2.resnets.2.norm1.gamma",
                "up_blocks.2.resnets.2.norm1.beta",
                "up_blocks.2.resnets.2.norm2.gamma",
                "up_blocks.2.resnets.2.norm2.beta",
                "up_blocks.2.resnets.2.conv2.weight",
                "up_blocks.2.resnets.2.conv2.bias",
                "up_blocks.2.upsamplers.0.conv.weight",
                "up_blocks.2.upsamplers.0.conv.bias",
                // up_blocks.3: 3 resnets, first has skip 256→128, NO upsampler
                "up_blocks.3.resnets.0.conv1.weight",
                "up_blocks.3.resnets.0.conv1.bias",
                "up_blocks.3.resnets.0.norm1.gamma",
                "up_blocks.3.resnets.0.norm1.beta",
                "up_blocks.3.resnets.0.norm2.gamma",
                "up_blocks.3.resnets.0.norm2.beta",
                "up_blocks.3.resnets.0.conv2.weight",
                "up_blocks.3.resnets.0.conv2.bias",
                "up_blocks.3.resnets.0.conv_shortcut.weight",
                "up_blocks.3.resnets.0.conv_shortcut.bias",
                "up_blocks.3.resnets.1.conv1.weight",
                "up_blocks.3.resnets.1.conv1.bias",
                "up_blocks.3.resnets.1.norm1.gamma",
                "up_blocks.3.resnets.1.norm1.beta",
                "up_blocks.3.resnets.1.norm2.gamma",
                "up_blocks.3.resnets.1.norm2.beta",
                "up_blocks.3.resnets.1.conv2.weight",
                "up_blocks.3.resnets.1.conv2.bias",
                "up_blocks.3.resnets.2.conv1.weight",
                "up_blocks.3.resnets.2.conv1.bias",
                "up_blocks.3.resnets.2.norm1.gamma",
                "up_blocks.3.resnets.2.norm1.beta",
                "up_blocks.3.resnets.2.norm2.gamma",
                "up_blocks.3.resnets.2.norm2.beta",
                "up_blocks.3.resnets.2.conv2.weight",
                "up_blocks.3.resnets.2.conv2.bias",
                "conv_norm_out.gamma",
                "conv_norm_out.beta",
                "conv_out.weight",
                "conv_out.bias",
            ])
            .with_remapped_key_patterns(&[
                "model.vae.decoder.conv_in -> conv_in",
                "model.vae.decoder.mid_block -> mid_block",
                "model.vae.decoder.up_blocks -> up_blocks",
                "model.vae.decoder.conv_norm_out -> conv_norm_out",
                "model.vae.decoder.conv_out -> conv_out",
                "legacy mid_block.attention -> mid_block.attentions.0",
                "legacy up_blocks.N.upsampler -> up_blocks.N.upsamplers.0",
                "legacy to_out -> to_out.0",
            ]),
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use burn_store::ApplyResult;
    use burn_tensor::Tensor;

    use crate::active_backend::{ActiveBurnBackend, active_device};
    use crate::config::BurnBackendConfig;
    use crate::models::stable_diffusion::sdxl::load_diagnostics::format_apply_report;
    use crate::runtime::BurnRuntime;

    use super::SdxlVaeDecoder;

    #[test]
    fn load_vae_decoder_module_from_path_applies_decoder_snapshots_through_burn_store() {
        let temp = tempfile::tempdir().expect("temp dir");
        let vae_path = temp.path().join("vae.safetensors");
        write_tiny_vae_component(&vae_path);
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let mut module = SdxlVaeDecoder::<ActiveBurnBackend>::init(runtime.device());

        let result = super::load_vae_decoder_module_from_path_with_profile(
            &runtime,
            &mut module,
            &vae_path,
            super::SdxlVaeDecoderLoadProfile::SdxlBase,
        )
        .expect("tiny VAE decoder should load through burn-store");

        assert!(result.errors.is_empty(), "unexpected load errors: {result}");
        assert!(result.applied.contains(&"conv_out.weight".to_string()));
        let output = module.forward(Tensor::<ActiveBurnBackend, 4>::zeros(
            [1, 4, 4, 4],
            runtime.device(),
        ));
        assert_eq!(output.shape().dims(), [1, 3, 32, 32]);
    }

    #[test]
    fn load_vae_decoder_module_from_path_rejects_missing_required_snapshots_with_policy_report() {
        let temp = tempfile::tempdir().expect("temp dir");
        let vae_path = temp.path().join("missing-vae.safetensors");
        write_missing_required_vae_component(&vae_path);
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let mut module = SdxlVaeDecoder::<ActiveBurnBackend>::init(runtime.device());

        let err = super::load_vae_decoder_module_from_path_with_profile(
            &runtime,
            &mut module,
            &vae_path,
            super::SdxlVaeDecoderLoadProfile::SdxlBase,
        )
        .expect_err("missing VAE snapshots should fail validation");
        let message = err.to_string();

        assert!(
            message.contains("required snapshot missing")
                && message.contains("component_role=vae")
                && message.contains("conv_out.bias")
                && message.contains("partial load policy"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_vae_decoder_module_from_path_accepts_mapped_runtime_snapshot_names() {
        let temp = tempfile::tempdir().expect("temp dir");
        let vae_path = temp.path().join("mapped-vae.safetensors");
        write_mapped_vae_component(&vae_path);
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let mut module = SdxlVaeDecoder::<ActiveBurnBackend>::init(runtime.device());

        let result = super::load_vae_decoder_module_from_path(&runtime, &mut module, &vae_path)
            .expect("mapped VAE decoder snapshots should load through burn-store");

        assert!(result.errors.is_empty(), "unexpected load errors: {result}");
        for expected in ["conv_out.weight", "conv_out.bias"] {
            assert!(
                result.applied.contains(&expected.to_owned()),
                "missing applied VAE snapshot `{expected}` in: {result}"
            );
        }
    }

    #[test]
    fn full_vae_decoder_policy_rejects_15d6_conv_out_only_components() {
        let report = format_apply_report(
            super::vae_load_policy(super::SdxlVaeDecoderLoadProfile::SdxlBase),
            &ApplyResult {
                applied: vec!["conv_out.weight".to_owned(), "conv_out.bias".to_owned()],
                skipped: Vec::new(),
                missing: Vec::new(),
                unused: Vec::new(),
                errors: Vec::new(),
            },
        );

        for expected in [
            "required snapshot missing: conv_in.weight",
            "required snapshot missing: up_blocks.0.resnets.0.conv1.weight",
            "required snapshot missing: up_blocks.2.resnets.0.conv_shortcut.bias",
        ] {
            assert!(
                report.contains(expected),
                "missing `{expected}` in:\n{report}"
            );
        }
    }

    #[test]
    fn tiny_vae_decoder_policy_accepts_conv_out_only_fixture_components() {
        let report = format_apply_report(
            super::vae_load_policy(super::SdxlVaeDecoderLoadProfile::TinySdxlE2e),
            &ApplyResult {
                applied: vec!["conv_out.weight".to_owned(), "conv_out.bias".to_owned()],
                skipped: Vec::new(),
                missing: Vec::new(),
                unused: Vec::new(),
                errors: Vec::new(),
            },
        );

        assert!(!report.contains("required snapshot missing"), "{report}");
        assert!(
            !report.contains("conv_in"),
            "tiny fixture policy should not require full-profile decoder snapshots:\n{report}"
        );
    }

    fn write_tiny_vae_component(path: &std::path::Path) {
        // Build a compatible SdxlBase fixture for the test load.
        let mut tensors = full_sdxl_vae_fixture("model.vae.decoder.");
        tensors.extend([
            tensor_view(
                "model.vae.decoder.conv_out.weight",
                // Conv2d weight: [out_channels=3, in_channels=128, 3, 3]
                vec![3usize, 128, 3, 3],
                vec![0.0f32; 3 * 128 * 3 * 3],
            ),
            tensor_view(
                "model.vae.decoder.conv_out.bias",
                vec![3usize],
                vec![0.0f32; 3],
            ),
        ]);
        safetensors::tensor::serialize_to_file(tensors, None, path)
            .expect("write full VAE safetensors");
    }

    fn write_mapped_vae_component(path: &std::path::Path) {
        // Build with no prefix (direct Burn snapshot names) + conv_out
        let mut tensors = full_sdxl_vae_fixture("");
        tensors.extend([
            tensor_view(
                "conv_out.weight",
                vec![3usize, 128, 3, 3],
                vec![0.0f32; 3 * 128 * 3 * 3],
            ),
            tensor_view("conv_out.bias", vec![3usize], vec![0.0f32; 3]),
        ]);
        safetensors::tensor::serialize_to_file(tensors, None, path)
            .expect("write mapped VAE safetensors");
    }

    fn write_missing_required_vae_component(path: &std::path::Path) {
        let tensors = vec![tensor_view(
            "model.vae.decoder.conv_out.weight",
            vec![3usize, 128, 3, 3],
            vec![0.0f32; 3 * 128 * 3 * 3],
        )];
        safetensors::tensor::serialize_to_file(tensors, None, path)
            .expect("write incomplete VAE safetensors");
    }

    fn full_vae_decoder_tensors(prefix: &str) -> Vec<(String, TestTensorView)> {
        let conv_in_prefix = if prefix.is_empty() {
            "conv_in".to_owned()
        } else {
            format!("{prefix}conv_in")
        };
        let mut tensors = vec![
            // Conv2d weight: Burn layout [out_channels, in_channels, H, W] = [512, 4, 3, 3]
            tensor_view(
                &format!("{conv_in_prefix}.weight"),
                vec![512usize, 4, 3, 3],
                vec![0.0f32; 512 * 4 * 3 * 3],
            ),
            // bias shape: [out_channels=512]
            tensor_view(
                &format!("{conv_in_prefix}.bias"),
                vec![512usize],
                vec![0.0f32; 512],
            ),
        ];
        for index in 0..2 {
            for norm in ["norm1", "norm2"] {
                tensors.push(tensor_view(
                    &format!("{prefix}mid_block.resnets.{index}.{norm}.gamma"),
                    vec![512usize],
                    vec![1.0f32; 512],
                ));
                tensors.push(tensor_view(
                    &format!("{prefix}mid_block.resnets.{index}.{norm}.beta"),
                    vec![512usize],
                    vec![0.0f32; 512],
                ));
            }
            for conv in ["conv1", "conv2"] {
                // Conv2d weight: [out_channels, in_channels, 3, 3] = [512, 512, 3, 3]
                tensors.push(tensor_view(
                    &format!("{prefix}mid_block.resnets.{index}.{conv}.weight"),
                    vec![512usize, 512, 3, 3],
                    vec![0.0f32; 512 * 512 * 3 * 3],
                ));
                tensors.push(tensor_view(
                    &format!("{prefix}mid_block.resnets.{index}.{conv}.bias"),
                    vec![512usize],
                    vec![0.0f32; 512],
                ));
            }
        }
        tensors
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

    impl safetensors::tensor::View for TestTensorView {
        fn dtype(&self) -> safetensors::tensor::Dtype {
            safetensors::tensor::Dtype::F32
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

    // Helper to generate full-profile VAE decoder fixture tensors.
    // Used by load_vae_decoder_module_from_path_applies_decoder_snapshots_through_burn_store
    // to build a complete SdxlBase profile fixture.
    fn full_sdxl_vae_fixture(prefix: &str) -> Vec<(String, TestTensorView)> {
        // Build the same structure as the old full_vae_decoder_tensors but with all
        // mid_block attention and up_block tensors required by SdxlBase policy.
        let mut tensors = full_vae_decoder_tensors(prefix);
        // mid_block attention
        for key in [
            &format!("{prefix}mid_block.attentions.0.group_norm.gamma"),
            &format!("{prefix}mid_block.attentions.0.group_norm.beta"),
            &format!("{prefix}mid_block.attentions.0.to_q.weight"),
            &format!("{prefix}mid_block.attentions.0.to_q.bias"),
            &format!("{prefix}mid_block.attentions.0.to_k.weight"),
            &format!("{prefix}mid_block.attentions.0.to_k.bias"),
            &format!("{prefix}mid_block.attentions.0.to_v.weight"),
            &format!("{prefix}mid_block.attentions.0.to_v.bias"),
            &format!("{prefix}mid_block.attentions.0.to_out.0.weight"),
            &format!("{prefix}mid_block.attentions.0.to_out.0.bias"),
        ] {
            if key.contains("weight") {
                tensors.push(tensor_view(
                    key,
                    vec![512, 512, 1, 1],
                    vec![0.0f32; 512 * 512],
                ));
            } else if key.contains(".bias") {
                tensors.push(tensor_view(key, vec![512], vec![0.0f32; 512]));
            } else {
                tensors.push(tensor_view(key, vec![512], vec![1.0f32; 512]));
            }
        }
        // up_blocks.0 and .1: all 512 channels, 3 resnets each, upsampler
        for block in 0..=1 {
            for resnet in 0..3 {
                for norm in ["norm1", "norm2"] {
                    let k = &format!("{prefix}up_blocks.{block}.resnets.{resnet}.{norm}.gamma");
                    tensors.push(tensor_view(k, vec![512], vec![1.0f32; 512]));
                    let k = &format!("{prefix}up_blocks.{block}.resnets.{resnet}.{norm}.beta");
                    tensors.push(tensor_view(k, vec![512], vec![0.0f32; 512]));
                }
                for conv in ["conv1", "conv2"] {
                    let k = &format!("{prefix}up_blocks.{block}.resnets.{resnet}.{conv}.weight");
                    tensors.push(tensor_view(
                        k,
                        vec![512, 512, 3, 3],
                        vec![0.0f32; 512 * 512 * 3 * 3],
                    ));
                    let k = &format!("{prefix}up_blocks.{block}.resnets.{resnet}.{conv}.bias");
                    tensors.push(tensor_view(k, vec![512], vec![0.0f32; 512]));
                }
            }
            let k = &format!("{prefix}up_blocks.{block}.upsamplers.0.conv.weight");
            tensors.push(tensor_view(
                k,
                vec![512, 512, 3, 3],
                vec![0.0f32; 512 * 512 * 3 * 3],
            ));
            let k = &format!("{prefix}up_blocks.{block}.upsamplers.0.conv.bias");
            tensors.push(tensor_view(k, vec![512], vec![0.0f32; 512]));
        }
        // up_blocks.2: 512→256 transition, first resnet has conv_shortcut
        let res0 = 0;
        for norm in ["norm1", "norm2"] {
            let k = &format!("{prefix}up_blocks.2.resnets.{res0}.{norm}.gamma");
            // first resnet: norm1 at 512, norm2 at 256
            tensors.push(tensor_view(
                k,
                if norm == "norm1" {
                    vec![512]
                } else {
                    vec![256]
                },
                if norm == "norm1" {
                    vec![1.0f32; 512]
                } else {
                    vec![1.0f32; 256]
                },
            ));
            let k = &format!("{prefix}up_blocks.2.resnets.{res0}.{norm}.beta");
            tensors.push(tensor_view(
                k,
                if norm == "norm1" {
                    vec![512]
                } else {
                    vec![256]
                },
                if norm == "norm1" {
                    vec![0.0f32; 512]
                } else {
                    vec![0.0f32; 256]
                },
            ));
        }
        // conv1: [out=256, in=512, 3, 3]
        tensors.push(tensor_view(
            &format!("{prefix}up_blocks.2.resnets.{res0}.conv1.weight"),
            vec![256, 512, 3, 3],
            vec![0.0f32; 256 * 512 * 3 * 3],
        ));
        tensors.push(tensor_view(
            &format!("{prefix}up_blocks.2.resnets.{res0}.conv1.bias"),
            vec![256],
            vec![0.0f32; 256],
        ));
        // conv2: [out=256, in=256, 3, 3]
        tensors.push(tensor_view(
            &format!("{prefix}up_blocks.2.resnets.{res0}.conv2.weight"),
            vec![256, 256, 3, 3],
            vec![0.0f32; 256 * 256 * 3 * 3],
        ));
        tensors.push(tensor_view(
            &format!("{prefix}up_blocks.2.resnets.{res0}.conv2.bias"),
            vec![256],
            vec![0.0f32; 256],
        ));
        // conv_shortcut: [out=256, in=512, 1, 1]
        tensors.push(tensor_view(
            &format!("{prefix}up_blocks.2.resnets.{res0}.conv_shortcut.weight"),
            vec![256, 512, 1, 1],
            vec![0.0f32; 256 * 512],
        ));
        tensors.push(tensor_view(
            &format!("{prefix}up_blocks.2.resnets.{res0}.conv_shortcut.bias"),
            vec![256],
            vec![0.0f32; 256],
        ));
        // resnets 1 and 2: all 256
        for resnet in 1..3 {
            for norm in ["norm1", "norm2"] {
                let k = &format!("{prefix}up_blocks.2.resnets.{resnet}.{norm}.gamma");
                tensors.push(tensor_view(k, vec![256], vec![1.0f32; 256]));
                let k = &format!("{prefix}up_blocks.2.resnets.{resnet}.{norm}.beta");
                tensors.push(tensor_view(k, vec![256], vec![0.0f32; 256]));
            }
            for conv in ["conv1", "conv2"] {
                let k = &format!("{prefix}up_blocks.2.resnets.{resnet}.{conv}.weight");
                tensors.push(tensor_view(
                    k,
                    vec![256, 256, 3, 3],
                    vec![0.0f32; 256 * 256 * 3 * 3],
                ));
                let k = &format!("{prefix}up_blocks.2.resnets.{resnet}.{conv}.bias");
                tensors.push(tensor_view(k, vec![256], vec![0.0f32; 256]));
            }
        }
        // up_blocks.2 upsampler: out=256 (write Burn [256, 256, 3, 3])
        tensors.push(tensor_view(
            &format!("{prefix}up_blocks.2.upsamplers.0.conv.weight"),
            vec![256, 256, 3, 3],
            vec![0.0f32; 256 * 256 * 3 * 3],
        ));
        tensors.push(tensor_view(
            &format!("{prefix}up_blocks.2.upsamplers.0.conv.bias"),
            vec![256],
            vec![0.0f32; 256],
        ));
        // up_blocks.3: 256→128 transition, first resnet has conv_shortcut
        // resnet 0: norm1 at 256, norm2 at 128; conv1: in=256, out=128; conv2: 128→128; skip: 256→128
        let res0 = 0;
        for norm in ["norm1", "norm2"] {
            let k = &format!("{prefix}up_blocks.3.resnets.{res0}.{norm}.gamma");
            tensors.push(tensor_view(
                k,
                if norm == "norm1" {
                    vec![256]
                } else {
                    vec![128]
                },
                if norm == "norm1" {
                    vec![1.0f32; 256]
                } else {
                    vec![1.0f32; 128]
                },
            ));
            let k = &format!("{prefix}up_blocks.3.resnets.{res0}.{norm}.beta");
            tensors.push(tensor_view(
                k,
                if norm == "norm1" {
                    vec![256]
                } else {
                    vec![128]
                },
                if norm == "norm1" {
                    vec![0.0f32; 256]
                } else {
                    vec![0.0f32; 128]
                },
            ));
        }
        tensors.push(tensor_view(
            &format!("{prefix}up_blocks.3.resnets.{res0}.conv1.weight"),
            vec![128, 256, 3, 3],
            vec![0.0f32; 128 * 256 * 3 * 3],
        ));
        tensors.push(tensor_view(
            &format!("{prefix}up_blocks.3.resnets.{res0}.conv1.bias"),
            vec![128],
            vec![0.0f32; 128],
        ));
        tensors.push(tensor_view(
            &format!("{prefix}up_blocks.3.resnets.{res0}.conv2.weight"),
            vec![128, 128, 3, 3],
            vec![0.0f32; 128 * 128 * 3 * 3],
        ));
        tensors.push(tensor_view(
            &format!("{prefix}up_blocks.3.resnets.{res0}.conv2.bias"),
            vec![128],
            vec![0.0f32; 128],
        ));
        tensors.push(tensor_view(
            &format!("{prefix}up_blocks.3.resnets.{res0}.conv_shortcut.weight"),
            vec![128, 256, 1, 1],
            vec![0.0f32; 128 * 256],
        ));
        tensors.push(tensor_view(
            &format!("{prefix}up_blocks.3.resnets.{res0}.conv_shortcut.bias"),
            vec![128],
            vec![0.0f32; 128],
        ));
        // resnets 1, 2: all 128
        for resnet in 1..3 {
            for norm in ["norm1", "norm2"] {
                let k = &format!("{prefix}up_blocks.3.resnets.{resnet}.{norm}.gamma");
                tensors.push(tensor_view(k, vec![128], vec![1.0f32; 128]));
                let k = &format!("{prefix}up_blocks.3.resnets.{resnet}.{norm}.beta");
                tensors.push(tensor_view(k, vec![128], vec![0.0f32; 128]));
            }
            for conv in ["conv1", "conv2"] {
                let k = &format!("{prefix}up_blocks.3.resnets.{resnet}.{conv}.weight");
                tensors.push(tensor_view(
                    k,
                    vec![128, 128, 3, 3],
                    vec![0.0f32; 128 * 128 * 3 * 3],
                ));
                let k = &format!("{prefix}up_blocks.3.resnets.{resnet}.{conv}.bias");
                tensors.push(tensor_view(k, vec![128], vec![0.0f32; 128]));
            }
        }
        // conv_norm_out: GroupNorm 32 groups, 128 channels
        tensors.push(tensor_view(
            &format!("{prefix}conv_norm_out.gamma"),
            vec![128],
            vec![1.0f32; 128],
        ));
        tensors.push(tensor_view(
            &format!("{prefix}conv_norm_out.beta"),
            vec![128],
            vec![0.0f32; 128],
        ));
        tensors
    }
}
