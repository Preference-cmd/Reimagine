//! Load SDXL diffusion UNet weights from Burn-native safetensors
//! component files using the bundle's component paths.

use std::fs;

use burn_store::{ApplyResult, KeyRemapper, PyTorchToBurnAdapter, SafetensorsStore};
use burn_tensor::backend::Backend;
use safetensors::tensor::SafeTensors;

use crate::error::BurnBackendError;
use crate::models::stable_diffusion::sdxl::load_diagnostics::{
    SdxlLoadPolicy, validate_apply_result as validate_sdxl_apply_result,
};
use crate::models::stable_diffusion::sdxl::{BurnLoadedModelBundle, BurnSdxlComponentRole};
use crate::runtime::BurnRuntime;

use super::module::{
    DiffusionBlockWeights, DiffusionUNetWeights, DiffusionWeightData, SdxlUnet,
    SdxlUnetTopologyProfile,
};

/// Load an SDXL UNet Module from a diffusion component through burn-store.
#[allow(dead_code)]
pub(crate) fn load_unet_module_from_path<B: Backend>(
    runtime: &BurnRuntime<B>,
    module: &mut SdxlUnet<B>,
    path: impl Into<std::path::PathBuf>,
) -> Result<ApplyResult, BurnBackendError> {
    load_unet_module_from_path_with_profile(
        runtime,
        module,
        path,
        SdxlUnetTopologyProfile::TinySdxlE2e,
    )
}

pub(crate) fn load_unet_module_from_path_with_profile<B: Backend>(
    runtime: &BurnRuntime<B>,
    module: &mut SdxlUnet<B>,
    path: impl Into<std::path::PathBuf>,
    profile: SdxlUnetTopologyProfile,
) -> Result<ApplyResult, BurnBackendError> {
    let mut store = sdxl_unet_store_from_path(path);
    let result = runtime
        .load_module_store(module, &mut store)
        .map_err(|source| BurnBackendError::InvalidRequest(source.to_string()))?;
    validate_apply_result(profile, &result)?;
    Ok(result)
}

#[allow(dead_code)]
fn sdxl_unet_store_from_path(path: impl Into<std::path::PathBuf>) -> SafetensorsStore {
    SafetensorsStore::from_file(path)
        .remap(sdxl_unet_key_remapper())
        .with_from_adapter(PyTorchToBurnAdapter)
        .allow_partial(true)
        .validate(true)
}

#[allow(dead_code)]
fn sdxl_unet_key_remapper() -> KeyRemapper {
    KeyRemapper::new()
        .add_pattern(r"^model\.diffusion\.conv_in\.", "conv_in.")
        .expect("static diffusion conv_in remapping regex should compile")
        .add_pattern(
            r"^model\.diffusion\.time_embed\.0\.",
            "time_embedding.linear_1.",
        )
        .expect("static diffusion time embedding linear_1 regex should compile")
        .add_pattern(
            r"^model\.diffusion\.time_embed\.2\.",
            "time_embedding.linear_2.",
        )
        .expect("static diffusion time embedding linear_2 regex should compile")
        .add_pattern(
            r"^model\.diffusion\.input_blocks\.1\.0\.in_layers\.2\.",
            "down_blocks.0.res_blocks.0.conv_1.",
        )
        .expect("static diffusion first resblock conv_1 regex should compile")
        .add_pattern(
            r"^model\.diffusion\.input_blocks\.1\.0\.emb_layers\.1\.",
            "down_blocks.0.res_blocks.0.time_projection.",
        )
        .expect("static diffusion first resblock time projection regex should compile")
        .add_pattern(
            r"^model\.diffusion\.input_blocks\.1\.0\.out_layers\.3\.",
            "down_blocks.0.res_blocks.0.conv_2.",
        )
        .expect("static diffusion first resblock conv_2 regex should compile")
        .add_pattern(
            r"^model\.diffusion\.input_blocks\.1\.1\.transformer_blocks\.0\.attn1\.to_q\.",
            "down_blocks.0.self_attention_blocks.0.attention.query.",
        )
        .expect("static diffusion first self-attention query regex should compile")
        .add_pattern(
            r"^model\.diffusion\.input_blocks\.1\.1\.transformer_blocks\.0\.attn1\.to_k\.",
            "down_blocks.0.self_attention_blocks.0.attention.key.",
        )
        .expect("static diffusion first self-attention key regex should compile")
        .add_pattern(
            r"^model\.diffusion\.input_blocks\.1\.1\.transformer_blocks\.0\.attn1\.to_v\.",
            "down_blocks.0.self_attention_blocks.0.attention.value.",
        )
        .expect("static diffusion first self-attention value regex should compile")
        .add_pattern(
            r"^model\.diffusion\.input_blocks\.1\.1\.transformer_blocks\.0\.attn1\.to_out\.0\.",
            "down_blocks.0.self_attention_blocks.0.attention.output.",
        )
        .expect("static diffusion first self-attention output regex should compile")
        .add_pattern(
            r"^model\.diffusion\.input_blocks\.1\.1\.transformer_blocks\.0\.attn2\.to_q\.",
            "down_blocks.0.cross_attention_blocks.0.attention.query.",
        )
        .expect("static diffusion first cross-attention query regex should compile")
        .add_pattern(
            r"^model\.diffusion\.input_blocks\.1\.1\.transformer_blocks\.0\.attn2\.to_k\.",
            "down_blocks.0.cross_attention_blocks.0.context_key.",
        )
        .expect("static diffusion first cross-attention key regex should compile")
        .add_pattern(
            r"^model\.diffusion\.input_blocks\.1\.1\.transformer_blocks\.0\.attn2\.to_v\.",
            "down_blocks.0.cross_attention_blocks.0.context_value.",
        )
        .expect("static diffusion first cross-attention value regex should compile")
        .add_pattern(
            r"^model\.diffusion\.input_blocks\.1\.1\.transformer_blocks\.0\.attn2\.to_out\.0\.",
            "down_blocks.0.cross_attention_blocks.0.attention.output.",
        )
        .expect("static diffusion first cross-attention output regex should compile")
        .add_pattern(r"^model\.diffusion\.out\.0\.", "conv_out.")
        .expect("static diffusion output conv remapping regex should compile")
}

#[allow(dead_code)]
fn validate_apply_result(
    profile: SdxlUnetTopologyProfile,
    result: &ApplyResult,
) -> Result<(), BurnBackendError> {
    validate_sdxl_apply_result(diffusion_load_policy_for_profile(profile), result)
}

fn diffusion_load_policy_for_profile(profile: SdxlUnetTopologyProfile) -> SdxlLoadPolicy {
    diffusion_load_policy_for_component("diffusion", profile)
}

fn diffusion_load_policy_for_component(
    component: &str,
    profile: SdxlUnetTopologyProfile,
) -> SdxlLoadPolicy {
    match component {
        "diffusion" => match profile {
            SdxlUnetTopologyProfile::TinySdxlE2e => tiny_diffusion_load_policy(),
            SdxlUnetTopologyProfile::SdxlBase => sdxl_base_diffusion_load_policy(),
        },
        _ => SdxlLoadPolicy::new("diffusion"),
    }
}

fn tiny_diffusion_load_policy() -> SdxlLoadPolicy {
    SdxlLoadPolicy::new("diffusion")
        .with_required_snapshots(&[
            "conv_in.weight",
            "conv_in.bias",
            "conv_out.weight",
            "conv_out.bias",
        ])
        .with_remapped_key_patterns(&[
            "model.diffusion.conv_in -> conv_in",
            "model.diffusion.out.0 -> conv_out",
        ])
}

fn sdxl_base_diffusion_load_policy() -> SdxlLoadPolicy {
    SdxlLoadPolicy::new("diffusion")
        .with_required_snapshots(&[
            "conv_in.weight",
            "conv_in.bias",
            "time_embedding.linear_1.weight",
            "time_embedding.linear_1.bias",
            "time_embedding.linear_2.weight",
            "time_embedding.linear_2.bias",
            "down_blocks.0.res_blocks.0.conv_1.weight",
            "down_blocks.0.res_blocks.0.conv_1.bias",
            "down_blocks.0.res_blocks.0.time_projection.weight",
            "down_blocks.0.res_blocks.0.time_projection.bias",
            "down_blocks.0.res_blocks.0.conv_2.weight",
            "down_blocks.0.res_blocks.0.conv_2.bias",
            "conv_out.weight",
            "conv_out.bias",
        ])
        .with_optional_snapshots(&[
            "down_blocks.0.res_blocks.0.skip.weight",
            "down_blocks.0.res_blocks.0.skip.bias",
            "down_blocks.0.self_attention_blocks.0.attention.query.weight",
            "down_blocks.0.self_attention_blocks.0.attention.key.weight",
            "down_blocks.0.self_attention_blocks.0.attention.value.weight",
            "down_blocks.0.self_attention_blocks.0.attention.output.weight",
            "down_blocks.0.self_attention_blocks.0.norm.gamma",
            "down_blocks.0.self_attention_blocks.0.norm.beta",
            "down_blocks.0.self_attention_blocks.0.attention.query.bias",
            "down_blocks.0.self_attention_blocks.0.attention.key.bias",
            "down_blocks.0.self_attention_blocks.0.attention.value.bias",
            "down_blocks.0.self_attention_blocks.0.attention.output.bias",
            "down_blocks.0.cross_attention_blocks.0.attention.query.weight",
            "down_blocks.0.cross_attention_blocks.0.context_key.weight",
            "down_blocks.0.cross_attention_blocks.0.context_value.weight",
            "down_blocks.0.cross_attention_blocks.0.attention.output.weight",
            "down_blocks.0.cross_attention_blocks.0.norm.gamma",
            "down_blocks.0.cross_attention_blocks.0.norm.beta",
            "down_blocks.0.cross_attention_blocks.0.attention.query.bias",
            "down_blocks.0.cross_attention_blocks.0.context_key.bias",
            "down_blocks.0.cross_attention_blocks.0.context_value.bias",
            "down_blocks.0.cross_attention_blocks.0.attention.output.bias",
        ])
        .with_generated_snapshot_contains(&[
            ".attention.query.",
            ".attention.key.",
            ".attention.value.",
        ])
        .with_remapped_key_patterns(&[
            "model.diffusion.conv_in -> conv_in",
            "model.diffusion.time_embed.0 -> time_embedding.linear_1",
            "model.diffusion.time_embed.2 -> time_embedding.linear_2",
            "model.diffusion.input_blocks -> down_blocks",
            "model.diffusion.out.0 -> conv_out",
        ])
}

/// Load diffusion UNet weights from the bundle's diffusion component
/// file. The bundle owns the resolved component path; this loader
/// reads the safetensors file and projects the keys into the weight
/// struct.
#[allow(dead_code)]
pub fn load_diffusion_weights(
    bundle: &BurnLoadedModelBundle,
) -> Result<DiffusionUNetWeights, BurnBackendError> {
    let sdxl = match bundle {
        BurnLoadedModelBundle::StableDiffusionSdxl(bundle) => bundle.as_ref(),
    };

    let component = sdxl
        .components()
        .iter()
        .find(|c| c.component_role == BurnSdxlComponentRole::Diffusion)
        .ok_or_else(|| BurnBackendError::MissingComponent("diffusion".to_owned()))?;

    let bytes = fs::read(&component.source_path).map_err(|e| BurnBackendError::ComponentRead {
        path: component.source_path.clone(),
        message: e.to_string(),
    })?;

    let safetensors =
        SafeTensors::deserialize(&bytes).map_err(|e| BurnBackendError::ComponentRead {
            path: component.source_path.clone(),
            message: e.to_string(),
        })?;

    // V1: build a minimal UNet weights struct with representative tensors
    // The full key-space projection is a follow-up deepening.
    let conv_in_weight = load_tensor(&safetensors, "model.diffusion.conv_in.weight")?;
    let conv_in_bias = load_tensor(&safetensors, "model.diffusion.conv_in.bias")?;
    let time_embed_0_weight = load_tensor(&safetensors, "model.diffusion.time_embed.0.weight")?;
    let time_embed_0_bias = load_tensor(&safetensors, "model.diffusion.time_embed.0.bias")?;
    let time_embed_2_weight = load_tensor(&safetensors, "model.diffusion.time_embed.2.weight")?;
    let time_embed_2_bias = load_tensor(&safetensors, "model.diffusion.time_embed.2.bias")?;

    // For V1, input/output blocks are loaded from known keys
    let mut input_blocks = Vec::new();
    for i in 0..12 {
        let prefix = format!("model.diffusion.input_blocks.{i}");
        if let Ok(w) = load_tensor_opt(&safetensors, &format!("{prefix}.0.weight")) {
            let b =
                load_tensor_opt(&safetensors, &format!("{prefix}.0.bias")).unwrap_or_else(|_| {
                    DiffusionWeightData {
                        data: vec![],
                        shape: vec![],
                    }
                });
            input_blocks.push(DiffusionBlockWeights {
                conv_weight: w,
                conv_bias: b,
                attn_q_weight: None,
                attn_k_weight: None,
                attn_v_weight: None,
                attn_out_weight: None,
            });
        }
    }

    let out_0_weight = load_tensor(&safetensors, "model.diffusion.out.0.weight")?;
    let out_0_bias = load_tensor(&safetensors, "model.diffusion.out.0.bias")?;

    Ok(DiffusionUNetWeights {
        conv_in_weight,
        conv_in_bias,
        time_embed_0_weight,
        time_embed_0_bias,
        time_embed_2_weight,
        time_embed_2_bias,
        input_blocks,
        middle_block: None,
        output_blocks: Vec::new(),
        out_0_weight,
        out_0_bias,
    })
}

#[allow(dead_code)]
fn load_tensor(
    safetensors: &SafeTensors,
    key: &str,
) -> Result<DiffusionWeightData, BurnBackendError> {
    let tensor = safetensors
        .tensor(key)
        .map_err(|_| BurnBackendError::ComponentRead {
            path: Default::default(),
            message: format!("missing diffusion tensor key `{key}`"),
        })?;
    let data = tensor.data().to_vec();
    let shape = tensor.shape().to_vec();
    // Convert bytes to f32
    let f32_data: Vec<f32> = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(DiffusionWeightData {
        data: f32_data,
        shape,
    })
}

#[allow(dead_code)]
fn load_tensor_opt(
    safetensors: &SafeTensors,
    key: &str,
) -> Result<DiffusionWeightData, BurnBackendError> {
    load_tensor(safetensors, key)
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use burn_store::ApplyResult;
    use burn_tensor::Tensor;

    use crate::active_backend::{ActiveBurnBackend, active_device};
    use crate::config::BurnBackendConfig;
    use crate::models::stable_diffusion::sdxl::diffusion::module::{
        SdxlUnet, SdxlUnetTopology, SdxlUnetTopologyProfile,
    };
    use crate::models::stable_diffusion::sdxl::load_diagnostics::format_apply_report;
    use crate::models::stable_diffusion::sdxl::source_layout::BurnSdxlSourceSet;
    use crate::models::stable_diffusion::sdxl::source_mapping::map_diffusers_style_split_source;
    use crate::runtime::BurnRuntime;

    #[test]
    fn load_unet_module_from_path_applies_diffusion_snapshots_through_burn_store() {
        let temp = tempfile::tempdir().expect("temp dir");
        let diffusion_path = temp.path().join("diffusion.safetensors");
        write_tiny_diffusion_component(&diffusion_path);
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let mut module = SdxlUnet::<ActiveBurnBackend>::init_from_topology(
            &SdxlUnetTopology::tiny(),
            runtime.device(),
        );

        let result = super::load_unet_module_from_path(&runtime, &mut module, &diffusion_path)
            .expect("tiny diffusion module should load through burn-store");

        assert!(result.errors.is_empty(), "unexpected load errors: {result}");
        assert!(result.applied.contains(&"conv_in.weight".to_string()));
        let output = module.forward(
            Tensor::<ActiveBurnBackend, 4>::zeros([1, 4, 4, 4], runtime.device()),
            Tensor::<ActiveBurnBackend, 1>::zeros([1], runtime.device()),
            Tensor::<ActiveBurnBackend, 3>::zeros([1, 3, 16], runtime.device()),
        );
        assert_eq!(output.dims(), [1, 4, 4, 4]);
    }

    #[test]
    fn load_unet_module_from_path_rejects_shape_incompatible_snapshots() {
        let temp = tempfile::tempdir().expect("temp dir");
        let diffusion_path = temp.path().join("bad-diffusion.safetensors");
        write_shape_incompatible_diffusion_component(&diffusion_path);
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let mut module = SdxlUnet::<ActiveBurnBackend>::init_from_topology(
            &SdxlUnetTopology::tiny(),
            runtime.device(),
        );

        let err = super::load_unet_module_from_path(&runtime, &mut module, &diffusion_path)
            .expect_err("shape-incompatible diffusion snapshots should fail validation");
        let message = err.to_string();

        assert!(
            message.contains("ShapeMismatch") && message.contains("conv_in.weight"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_unet_module_from_path_rejects_missing_required_snapshots() {
        let temp = tempfile::tempdir().expect("temp dir");
        let diffusion_path = temp.path().join("missing-diffusion.safetensors");
        write_missing_required_diffusion_component(&diffusion_path);
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let mut module = SdxlUnet::<ActiveBurnBackend>::init_from_topology(
            &SdxlUnetTopology::tiny(),
            runtime.device(),
        );

        let err = super::load_unet_module_from_path(&runtime, &mut module, &diffusion_path)
            .expect_err("missing diffusion snapshots should fail validation");
        let message = err.to_string();

        assert!(
            message.contains("required snapshot missing")
                && message.contains("component_role=diffusion")
                && message.contains("conv_out.weight")
                && message.contains("unexpected source snapshot")
                && message.contains("model.diffusion.extra.weight")
                && message.contains("partial load policy"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn full_sdxl_unet_loader_uses_profile_policy_report() {
        let temp = tempfile::tempdir().expect("temp dir");
        let diffusion_path = temp.path().join("tiny-source-full-policy.safetensors");
        write_tiny_diffusion_component(&diffusion_path);
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let mut module = SdxlUnet::<ActiveBurnBackend>::init_from_topology(
            &SdxlUnetTopology::tiny(),
            runtime.device(),
        );

        let err = super::load_unet_module_from_path_with_profile(
            &runtime,
            &mut module,
            &diffusion_path,
            SdxlUnetTopologyProfile::SdxlBase,
        )
        .expect_err("full profile policy should reject tiny-only snapshots");
        let message = err.to_string();

        for expected in [
            "required snapshot missing: time_embedding.linear_1.weight",
            "required snapshot missing: down_blocks.0.res_blocks.0.conv_1.weight",
            "remapped source key pattern: model.diffusion.input_blocks -> down_blocks",
        ] {
            assert!(
                message.contains(expected),
                "missing `{expected}` in:\n{message}"
            );
        }
    }

    #[test]
    fn full_sdxl_unet_loader_applies_first_resblock_time_tranche() {
        let temp = tempfile::tempdir().expect("temp dir");
        let diffusion_path = temp.path().join("first-resblock-time.safetensors");
        write_first_resblock_time_diffusion_component(&diffusion_path);
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let mut module = SdxlUnet::<ActiveBurnBackend>::init_from_topology(
            &SdxlUnetTopology::sdxl_base(),
            runtime.device(),
        );

        let result = super::load_unet_module_from_path_with_profile(
            &runtime,
            &mut module,
            &diffusion_path,
            SdxlUnetTopologyProfile::SdxlBase,
        )
        .expect("first full-profile resblock/time tranche should load through burn-store");

        for expected in [
            "time_embedding.linear_1.weight",
            "time_embedding.linear_1.bias",
            "time_embedding.linear_2.weight",
            "time_embedding.linear_2.bias",
            "down_blocks.0.res_blocks.0.conv_1.weight",
            "down_blocks.0.res_blocks.0.conv_1.bias",
            "down_blocks.0.res_blocks.0.time_projection.weight",
            "down_blocks.0.res_blocks.0.time_projection.bias",
            "down_blocks.0.res_blocks.0.conv_2.weight",
            "down_blocks.0.res_blocks.0.conv_2.bias",
        ] {
            assert!(
                result.applied.contains(&expected.to_owned()),
                "missing applied snapshot `{expected}` in: {result}"
            );
        }
    }

    #[test]
    fn full_sdxl_unet_loader_applies_first_attention_tranche() {
        let temp = tempfile::tempdir().expect("temp dir");
        let diffusion_path = temp.path().join("first-attention.safetensors");
        write_first_attention_diffusion_component(&diffusion_path);
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let mut module = SdxlUnet::<ActiveBurnBackend>::init_from_topology(
            &SdxlUnetTopology::sdxl_base(),
            runtime.device(),
        );

        let result = super::load_unet_module_from_path_with_profile(
            &runtime,
            &mut module,
            &diffusion_path,
            SdxlUnetTopologyProfile::SdxlBase,
        )
        .expect("first full-profile attention tranche should load through burn-store");

        for expected in [
            "down_blocks.0.self_attention_blocks.0.attention.query.weight",
            "down_blocks.0.self_attention_blocks.0.attention.key.weight",
            "down_blocks.0.self_attention_blocks.0.attention.value.weight",
            "down_blocks.0.self_attention_blocks.0.attention.output.weight",
            "down_blocks.0.cross_attention_blocks.0.attention.query.weight",
            "down_blocks.0.cross_attention_blocks.0.context_key.weight",
            "down_blocks.0.cross_attention_blocks.0.context_value.weight",
            "down_blocks.0.cross_attention_blocks.0.attention.output.weight",
        ] {
            assert!(
                result.applied.contains(&expected.to_owned()),
                "missing applied snapshot `{expected}` in: {result}"
            );
        }
    }

    #[test]
    fn mapped_diffusion_component_loads_through_full_profile_unet_loader() {
        let source = tempfile::tempdir().expect("source temp dir");
        let output = tempfile::tempdir().expect("output temp dir");
        crate::models::stable_diffusion::sdxl::source_mapping::tests::write_complete_split_source(
            source.path(),
        );
        let source_set =
            BurnSdxlSourceSet::diffusers_style_split_safetensors(source.path().to_path_buf());
        map_diffusers_style_split_source(&source_set, output.path()).expect("map source");
        let diffusion_path = output.path().join("diffusion/model.safetensors");
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));
        let mut module = SdxlUnet::<ActiveBurnBackend>::init_from_topology(
            &SdxlUnetTopology::sdxl_base(),
            runtime.device(),
        );

        let result = super::load_unet_module_from_path_with_profile(
            &runtime,
            &mut module,
            &diffusion_path,
            SdxlUnetTopologyProfile::SdxlBase,
        )
        .expect("mapped diffusion component should load through full-profile UNet loader");

        for expected in [
            "time_embedding.linear_1.weight",
            "time_embedding.linear_2.weight",
            "down_blocks.0.res_blocks.0.conv_1.weight",
            "down_blocks.0.res_blocks.0.time_projection.weight",
            "down_blocks.0.res_blocks.0.conv_2.weight",
            "down_blocks.0.self_attention_blocks.0.attention.query.weight",
            "down_blocks.0.cross_attention_blocks.0.context_key.weight",
        ] {
            assert!(
                result.applied.contains(&expected.to_owned()),
                "missing applied snapshot `{expected}` in: {result}"
            );
        }
    }

    #[test]
    fn full_sdxl_unet_policy_reports_required_block_families_without_deferred_topology() {
        let report = format_apply_report(
            super::diffusion_load_policy_for_profile(SdxlUnetTopologyProfile::SdxlBase),
            &ApplyResult {
                applied: Vec::new(),
                skipped: Vec::new(),
                missing: Vec::new(),
                unused: Vec::new(),
                errors: Vec::new(),
            },
        );

        for expected in [
            "required snapshot missing: time_embedding.linear_1.weight",
            "required snapshot missing: down_blocks.0.res_blocks.0.conv_1.weight",
            "remapped source key pattern: model.diffusion.input_blocks -> down_blocks",
        ] {
            assert!(
                report.contains(expected),
                "missing `{expected}` in:\n{report}"
            );
        }
        assert!(!report.contains("deferred snapshot family"), "{report}");
    }

    #[test]
    fn full_sdxl_unet_policy_has_no_deferred_topology_families_after_15e() {
        let report = format_apply_report(
            super::diffusion_load_policy_for_profile(SdxlUnetTopologyProfile::SdxlBase),
            &ApplyResult {
                applied: Vec::new(),
                skipped: Vec::new(),
                missing: Vec::new(),
                unused: Vec::new(),
                errors: Vec::new(),
            },
        );

        assert!(
            !report.contains("deferred snapshot family"),
            "full-profile UNet policy must not report deferred topology families once 15e enables the graph:\n{report}"
        );
    }

    #[test]
    fn tiny_unet_policy_keeps_scaffold_only_required_snapshots() {
        let report = format_apply_report(
            super::diffusion_load_policy_for_profile(SdxlUnetTopologyProfile::TinySdxlE2e),
            &ApplyResult {
                applied: vec![
                    "conv_in.weight".to_owned(),
                    "conv_in.bias".to_owned(),
                    "conv_out.weight".to_owned(),
                    "conv_out.bias".to_owned(),
                ],
                skipped: Vec::new(),
                missing: Vec::new(),
                unused: Vec::new(),
                errors: Vec::new(),
            },
        );

        assert!(
            !report.contains("time_embedding.linear_1.weight"),
            "{report}"
        );
        assert!(!report.contains("deferred snapshot family"), "{report}");
    }

    #[test]
    fn full_sdxl_unet_policy_classifies_optional_and_generated_snapshots() {
        let report = format_apply_report(
            super::diffusion_load_policy_for_profile(SdxlUnetTopologyProfile::SdxlBase),
            &ApplyResult {
                applied: vec![
                    "down_blocks.0.self_attention_blocks.0.attention.query.weight".to_owned(),
                    "down_blocks.0.self_attention_blocks.0.attention.key.weight".to_owned(),
                    "down_blocks.0.self_attention_blocks.0.attention.value.weight".to_owned(),
                ],
                skipped: Vec::new(),
                missing: Vec::new(),
                unused: Vec::new(),
                errors: Vec::new(),
            },
        );

        for expected in [
            "optional snapshot missing: down_blocks.0.res_blocks.0.skip.weight",
            "generated snapshot: down_blocks.0.self_attention_blocks.0.attention.query.weight",
            "generated snapshot: down_blocks.0.self_attention_blocks.0.attention.key.weight",
            "generated snapshot: down_blocks.0.self_attention_blocks.0.attention.value.weight",
            "remapped source key pattern: model.diffusion.input_blocks -> down_blocks",
        ] {
            assert!(
                report.contains(expected),
                "missing `{expected}` in:\n{report}"
            );
        }
    }

    fn write_tiny_diffusion_component(path: &std::path::Path) {
        let tensors = vec![
            tensor_view(
                "model.diffusion.conv_in.weight",
                vec![4, 4, 3, 3],
                vec![0.01; 4 * 4 * 3 * 3],
            ),
            tensor_view("model.diffusion.conv_in.bias", vec![4], vec![0.0; 4]),
            tensor_view(
                "model.diffusion.out.0.weight",
                vec![4, 4, 3, 3],
                vec![0.01; 4 * 4 * 3 * 3],
            ),
            tensor_view("model.diffusion.out.0.bias", vec![4], vec![0.0; 4]),
        ];
        safetensors::tensor::serialize_to_file(tensors, None, path)
            .expect("serialize tiny diffusion safetensors");
    }

    fn write_first_resblock_time_diffusion_component(path: &std::path::Path) {
        safetensors::tensor::serialize_to_file(first_resblock_time_diffusion_tensors(), None, path)
            .expect("serialize first resblock/time diffusion safetensors");
    }

    fn first_resblock_time_diffusion_tensors() -> Vec<(String, TestTensorView)> {
        vec![
            tensor_view(
                "model.diffusion.conv_in.weight",
                vec![320, 4, 3, 3],
                vec![0.01; 320 * 4 * 3 * 3],
            ),
            tensor_view("model.diffusion.conv_in.bias", vec![320], vec![0.0; 320]),
            tensor_view(
                "model.diffusion.time_embed.0.weight",
                vec![1280, 320],
                vec![0.01; 1280 * 320],
            ),
            tensor_view(
                "model.diffusion.time_embed.0.bias",
                vec![1280],
                vec![0.0; 1280],
            ),
            tensor_view(
                "model.diffusion.time_embed.2.weight",
                vec![1280, 1280],
                vec![0.01; 1280 * 1280],
            ),
            tensor_view(
                "model.diffusion.time_embed.2.bias",
                vec![1280],
                vec![0.0; 1280],
            ),
            tensor_view(
                "model.diffusion.input_blocks.1.0.in_layers.2.weight",
                vec![320, 320, 3, 3],
                vec![0.01; 320 * 320 * 3 * 3],
            ),
            tensor_view(
                "model.diffusion.input_blocks.1.0.in_layers.2.bias",
                vec![320],
                vec![0.0; 320],
            ),
            tensor_view(
                "model.diffusion.input_blocks.1.0.emb_layers.1.weight",
                vec![320, 1280],
                vec![0.01; 320 * 1280],
            ),
            tensor_view(
                "model.diffusion.input_blocks.1.0.emb_layers.1.bias",
                vec![320],
                vec![0.0; 320],
            ),
            tensor_view(
                "model.diffusion.input_blocks.1.0.out_layers.3.weight",
                vec![320, 320, 3, 3],
                vec![0.01; 320 * 320 * 3 * 3],
            ),
            tensor_view(
                "model.diffusion.input_blocks.1.0.out_layers.3.bias",
                vec![320],
                vec![0.0; 320],
            ),
            tensor_view(
                "model.diffusion.out.0.weight",
                vec![4, 320, 3, 3],
                vec![0.01; 4 * 320 * 3 * 3],
            ),
            tensor_view("model.diffusion.out.0.bias", vec![4], vec![0.0; 4]),
        ]
    }

    fn write_first_attention_diffusion_component(path: &std::path::Path) {
        let mut tensors = first_resblock_time_diffusion_tensors();
        tensors.extend([
            tensor_view(
                "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
                vec![320, 320],
                vec![0.01; 320 * 320],
            ),
            tensor_view(
                "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn1.to_k.weight",
                vec![320, 320],
                vec![0.01; 320 * 320],
            ),
            tensor_view(
                "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn1.to_v.weight",
                vec![320, 320],
                vec![0.01; 320 * 320],
            ),
            tensor_view(
                "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.weight",
                vec![320, 320],
                vec![0.01; 320 * 320],
            ),
            tensor_view(
                "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn2.to_q.weight",
                vec![320, 320],
                vec![0.01; 320 * 320],
            ),
            tensor_view(
                "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
                vec![320, 2048],
                vec![0.01; 320 * 2048],
            ),
            tensor_view(
                "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn2.to_v.weight",
                vec![320, 2048],
                vec![0.01; 320 * 2048],
            ),
            tensor_view(
                "model.diffusion.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.weight",
                vec![320, 320],
                vec![0.01; 320 * 320],
            ),
        ]);
        safetensors::tensor::serialize_to_file(tensors, None, path)
            .expect("serialize first attention diffusion safetensors");
    }

    fn write_shape_incompatible_diffusion_component(path: &std::path::Path) {
        let tensors = vec![
            tensor_view(
                "model.diffusion.conv_in.weight",
                vec![4, 4],
                vec![0.01; 4 * 4],
            ),
            tensor_view("model.diffusion.conv_in.bias", vec![4], vec![0.0; 4]),
        ];
        safetensors::tensor::serialize_to_file(tensors, None, path)
            .expect("serialize invalid diffusion safetensors");
    }

    fn write_missing_required_diffusion_component(path: &std::path::Path) {
        let tensors = vec![
            tensor_view(
                "model.diffusion.conv_in.weight",
                vec![4, 4, 3, 3],
                vec![0.01; 4 * 4 * 3 * 3],
            ),
            tensor_view("model.diffusion.conv_in.bias", vec![4], vec![0.0; 4]),
            tensor_view("model.diffusion.extra.weight", vec![1], vec![1.0]),
        ];
        safetensors::tensor::serialize_to_file(tensors, None, path)
            .expect("serialize incomplete diffusion safetensors");
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
