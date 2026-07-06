//! Load SDXL VAE decoder weights through burn-store.

use burn_store::{ApplyResult, KeyRemapper, PyTorchToBurnAdapter, SafetensorsStore};
use burn_tensor::backend::Backend;

use crate::error::BurnBackendError;
use crate::models::stable_diffusion::sdxl::load_diagnostics::{
    SdxlLoadPolicy, validate_apply_result as validate_sdxl_apply_result,
};
use crate::runtime::BurnRuntime;

use super::module::SdxlVaeDecoder;

/// Load an SDXL VAE decoder Module from a VAE component through burn-store.
pub(crate) fn load_vae_decoder_module_from_path<B: Backend>(
    runtime: &BurnRuntime<B>,
    module: &mut SdxlVaeDecoder<B>,
    path: impl Into<std::path::PathBuf>,
) -> Result<ApplyResult, BurnBackendError> {
    let mut store = sdxl_vae_store_from_path(path);
    let result = runtime
        .load_module_store(module, &mut store)
        .map_err(|source| BurnBackendError::InvalidRequest(source.to_string()))?;
    validate_apply_result("vae", &result)?;
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
    KeyRemapper::new()
        .add_pattern(r"^model\.vae\.decoder\.conv_out\.", "conv_out.")
        .expect("static VAE decoder conv_out remapping regex should compile")
}

fn validate_apply_result(component: &str, result: &ApplyResult) -> Result<(), BurnBackendError> {
    validate_sdxl_apply_result(vae_load_policy(component), result)
}

fn vae_load_policy(component: &str) -> SdxlLoadPolicy {
    match component {
        "vae" => SdxlLoadPolicy::new("vae")
            .with_required_snapshots(&["conv_out.weight", "conv_out.bias"])
            .with_remapped_key_patterns(&["model.vae.decoder.conv_out -> conv_out"]),
        _ => SdxlLoadPolicy::new("vae"),
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use burn_tensor::Tensor;

    use crate::active_backend::{ActiveBurnBackend, active_device};
    use crate::config::BurnBackendConfig;
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

        let result = super::load_vae_decoder_module_from_path(&runtime, &mut module, &vae_path)
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

        let err = super::load_vae_decoder_module_from_path(&runtime, &mut module, &vae_path)
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

    fn write_tiny_vae_component(path: &std::path::Path) {
        let tensors = vec![
            tensor_view(
                "model.vae.decoder.conv_out.weight",
                vec![3usize, 4, 3, 3],
                vec![0.0f32; 3 * 4 * 3 * 3],
            ),
            tensor_view(
                "model.vae.decoder.conv_out.bias",
                vec![3usize],
                vec![0.0f32; 3],
            ),
        ];
        safetensors::tensor::serialize_to_file(tensors, None, path)
            .expect("write tiny VAE safetensors");
    }

    fn write_missing_required_vae_component(path: &std::path::Path) {
        let tensors = vec![tensor_view(
            "model.vae.decoder.conv_out.weight",
            vec![3usize, 4, 3, 3],
            vec![0.0f32; 3 * 4 * 3 * 3],
        )];
        safetensors::tensor::serialize_to_file(tensors, None, path)
            .expect("write incomplete VAE safetensors");
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
}
