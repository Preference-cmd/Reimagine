//! CLIP-specific burn-store adapters for SDXL text encoder weights.

use std::collections::BTreeMap;
use std::path::PathBuf;

use burn_store::{
    ApplyResult, KeyRemapper, ModuleAdapter, ModuleSnapshot, ModuleStore, PyTorchToBurnAdapter,
    SafetensorsStore, TensorSnapshot,
};
use burn_tensor::{Shape, TensorData, backend::Backend};

/// Build a burn-store loader for one SDXL CLIP component safetensors file.
#[allow(dead_code)]
pub(crate) fn sdxl_clip_store_from_path(
    path: impl Into<PathBuf>,
) -> SdxlClipStore<SafetensorsStore> {
    SdxlClipStore::new(sdxl_clip_safetensors_store(SafetensorsStore::from_file(
        path,
    )))
    .with_from_adapter(PyTorchToBurnAdapter)
}

#[cfg(test)]
fn sdxl_clip_store_from_bytes(bytes: Vec<u8>) -> SdxlClipStore<SafetensorsStore> {
    SdxlClipStore::new(sdxl_clip_safetensors_store(SafetensorsStore::from_bytes(
        Some(bytes),
    )))
    .with_from_adapter(PyTorchToBurnAdapter)
}

#[allow(dead_code)]
fn sdxl_clip_safetensors_store(store: SafetensorsStore) -> SafetensorsStore {
    store
        .remap(sdxl_clip_key_remapper())
        .allow_partial(true)
        .validate(true)
}

#[allow(dead_code)]
fn sdxl_clip_key_remapper() -> KeyRemapper {
    KeyRemapper::new()
        .add_pattern(r"^model\.text_encoder_2\.", "open_clip_g.")
        .expect("static OpenCLIP-G prefix remapping regex should compile")
        .add_pattern(r"^model\.text_encoder\.", "clip_l.")
        .expect("static CLIP-L prefix remapping regex should compile")
        .add_pattern(r"\.transformer\.resblocks\.", ".blocks.")
        .expect("static CLIP block remapping regex should compile")
        .add_pattern(r"\.attn\.out_proj\.", ".attention.output.")
        .expect("static CLIP output projection remapping regex should compile")
        .add_pattern(r"\.attn\.in_proj_", ".attention.in_proj_")
        .expect("static CLIP fused QKV remapping regex should compile")
        .add_pattern(r"\.mlp\.fc1\.", ".mlp_fc1.")
        .expect("static CLIP MLP fc1 remapping regex should compile")
        .add_pattern(r"\.mlp\.fc2\.", ".mlp_fc2.")
        .expect("static CLIP MLP fc2 remapping regex should compile")
        .add_pattern(r"\.final_layer_norm\.gamma$", ".final_layer_norm.weight")
        .expect("static final layer norm weight remapping regex should compile")
        .add_pattern(r"\.final_layer_norm\.beta$", ".final_layer_norm.bias")
        .expect("static final layer norm bias remapping regex should compile")
}

/// Store wrapper that expands CLIP/OpenCLIP fused QKV tensors before applying
/// snapshots to Burn-native `MultiHeadAttention` modules.
#[allow(dead_code)]
pub(crate) struct SdxlClipStore<S> {
    inner: S,
    from_adapter: Option<Box<dyn ModuleAdapter>>,
}

#[allow(dead_code)]
impl<S> SdxlClipStore<S> {
    pub(crate) fn new(inner: S) -> Self {
        Self {
            inner,
            from_adapter: None,
        }
    }

    pub(crate) fn with_from_adapter(mut self, adapter: impl ModuleAdapter + 'static) -> Self {
        self.from_adapter = Some(Box::new(adapter));
        self
    }
}

impl<S: ModuleStore> ModuleStore for SdxlClipStore<S> {
    type Error = S::Error;

    fn collect_from<B: Backend, M: ModuleSnapshot<B>>(
        &mut self,
        module: &M,
    ) -> Result<(), Self::Error> {
        self.inner.collect_from(module)
    }

    fn apply_to<B: Backend, M: ModuleSnapshot<B>>(
        &mut self,
        module: &mut M,
    ) -> Result<ApplyResult, Self::Error> {
        let snapshots = self.expanded_snapshots()?;
        Ok(module.apply(snapshots, None, self.from_adapter.clone(), false))
    }

    fn get_snapshot(&mut self, name: &str) -> Result<Option<&TensorSnapshot>, Self::Error> {
        self.inner.get_snapshot(name)
    }

    fn get_all_snapshots(&mut self) -> Result<&BTreeMap<String, TensorSnapshot>, Self::Error> {
        self.inner.get_all_snapshots()
    }

    fn keys(&mut self) -> Result<Vec<String>, Self::Error> {
        self.inner.keys()
    }
}

#[cfg(test)]
pub(crate) fn clip_load_report_for_test(component: &'static str, result: &ApplyResult) -> String {
    crate::models::stable_diffusion::sdxl::load_diagnostics::format_apply_report(
        crate::models::stable_diffusion::sdxl::load_diagnostics::SdxlLoadPolicy::new(component)
            .with_generated_snapshot_contains(&[
                ".attention.query.",
                ".attention.key.",
                ".attention.value.",
            ])
            .with_remapped_key_patterns(&[".attn.in_proj_* -> generated q/k/v snapshots"]),
        result,
    )
}

#[allow(dead_code)]
impl<S: ModuleStore> SdxlClipStore<S> {
    fn expanded_snapshots(&mut self) -> Result<Vec<TensorSnapshot>, S::Error> {
        let source = self.inner.get_all_snapshots()?;
        let mut snapshots: Vec<TensorSnapshot> = source.values().cloned().collect();

        for (path, snapshot) in source {
            if let Some(prefix) = path.strip_suffix(".in_proj_weight")
                && snapshot.shape.len() == 2
                && snapshot.shape[0] % 3 == 0
            {
                let width = snapshot.shape[0] / 3;
                for (name, offset) in [("query", 0), ("key", width), ("value", width * 2)] {
                    snapshots.push(split_snapshot_2d(
                        snapshot,
                        &format!("{}.{}.weight", prefix, name),
                        offset,
                        width,
                    ));
                }
            } else if let Some(prefix) = path.strip_suffix(".in_proj_bias")
                && snapshot.shape.len() == 1
                && snapshot.shape[0] % 3 == 0
            {
                let width = snapshot.shape[0] / 3;
                for (name, offset) in [("query", 0), ("key", width), ("value", width * 2)] {
                    snapshots.push(split_snapshot_1d(
                        snapshot,
                        &format!("{}.{}.bias", prefix, name),
                        offset,
                        width,
                    ));
                }
            }
        }

        Ok(snapshots)
    }
}

#[allow(dead_code)]
fn split_snapshot_2d(
    snapshot: &TensorSnapshot,
    path: &str,
    row_offset: usize,
    row_count: usize,
) -> TensorSnapshot {
    let cols = snapshot.shape[1];
    let path_stack = path.split('.').map(str::to_string).collect();
    let container_stack = snapshot.container_stack.clone().unwrap_or_default();
    let tensor_id = snapshot.tensor_id.unwrap_or_default();
    let snapshot = snapshot.clone();
    let dtype = snapshot.dtype;
    let data_snapshot = snapshot.clone();

    TensorSnapshot::from_closure(
        std::rc::Rc::new(move || {
            let data = data_snapshot.to_data()?;
            let values = data.to_vec::<f32>().map_err(|err| {
                burn_store::TensorSnapshotError::DataError(format!(
                    "fused qkv weight must be f32: {err:?}"
                ))
            })?;
            let start = row_offset * cols;
            let end = start + row_count * cols;
            Ok(TensorData::new(
                values[start..end].to_vec(),
                [row_count, cols],
            ))
        }),
        dtype,
        Shape::new([row_count, cols]),
        path_stack,
        container_stack,
        tensor_id,
    )
}

#[allow(dead_code)]
fn split_snapshot_1d(
    snapshot: &TensorSnapshot,
    path: &str,
    offset: usize,
    len: usize,
) -> TensorSnapshot {
    let path_stack = path.split('.').map(str::to_string).collect();
    let container_stack = snapshot.container_stack.clone().unwrap_or_default();
    let tensor_id = snapshot.tensor_id.unwrap_or_default();
    let snapshot = snapshot.clone();
    let dtype = snapshot.dtype;
    let data_snapshot = snapshot.clone();

    TensorSnapshot::from_closure(
        std::rc::Rc::new(move || {
            let data = data_snapshot.to_data()?;
            let values = data.to_vec::<f32>().map_err(|err| {
                burn_store::TensorSnapshotError::DataError(format!(
                    "fused qkv bias must be f32: {err:?}"
                ))
            })?;
            Ok(TensorData::new(
                values[offset..offset + len].to_vec(),
                [len],
            ))
        }),
        dtype,
        Shape::new([len]),
        path_stack,
        container_stack,
        tensor_id,
    )
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::collections::BTreeMap;

    use crate::active_backend::{ActiveBurnBackend, active_device};
    use crate::config::BurnBackendConfig;
    use burn_core::module::ParamId;
    use burn_store::{
        ApplyResult, ModuleSnapshot, ModuleStore, PyTorchToBurnAdapter, TensorSnapshot,
    };
    use burn_tensor::TensorData;

    use crate::models::stable_diffusion::sdxl::text_conditioning::module::SdxlTextEncoders;
    use crate::runtime::BurnRuntime;
    use crate::text_encoder::clip::{ClipTextEncoderProfile, ClipTextEncoderVariant};

    #[test]
    fn clip_store_expands_fused_qkv_into_burn_mha_snapshots() {
        type B = ActiveBurnBackend;

        let runtime = BurnRuntime::<B>::new(active_test_device());
        let clip_l_profile = tiny_profile(ClipTextEncoderVariant::ClipL, false);
        let open_clip_g_profile = tiny_profile(ClipTextEncoderVariant::OpenClipG, true);
        let mut module = SdxlTextEncoders::<B>::init_from_profiles(
            &clip_l_profile,
            &open_clip_g_profile,
            runtime.device(),
        );
        let mut store = super::SdxlClipStore::new(SnapshotStore::new(vec![
            snapshot_2d(
                "clip_l.blocks.0.attention.in_proj_weight",
                6,
                2,
                vec![
                    1.0, 2.0, // query row 0
                    3.0, 4.0, // query row 1
                    5.0, 6.0, // key row 0
                    7.0, 8.0, // key row 1
                    9.0, 10.0, // value row 0
                    11.0, 12.0, // value row 1
                ],
            ),
            snapshot_1d(
                "clip_l.blocks.0.attention.in_proj_bias",
                vec![101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            ),
        ]))
        .with_from_adapter(PyTorchToBurnAdapter);

        let result = runtime
            .load_module_store(&mut module, &mut store)
            .expect("fused qkv store should load into text encoder Module");

        assert!(
            result.errors.is_empty(),
            "unexpected store load errors: {result}"
        );
        assert!(
            result
                .applied
                .contains(&"clip_l.blocks.0.attention.query.weight".to_string())
        );
        assert_param_2d(
            &module.clip_l.blocks()[0].attention.query.weight,
            [1.0, 3.0, 2.0, 4.0],
        );
        assert_param_1d(
            module.clip_l.blocks()[0]
                .attention
                .query
                .bias
                .as_ref()
                .expect("query bias"),
            [101.0, 102.0],
        );
        assert_param_2d(
            &module.clip_l.blocks()[0].attention.key.weight,
            [5.0, 7.0, 6.0, 8.0],
        );
        assert_param_1d(
            module.clip_l.blocks()[0]
                .attention
                .key
                .bias
                .as_ref()
                .expect("key bias"),
            [103.0, 104.0],
        );
        assert_param_2d(
            &module.clip_l.blocks()[0].attention.value.weight,
            [9.0, 11.0, 10.0, 12.0],
        );
        assert_param_1d(
            module.clip_l.blocks()[0]
                .attention
                .value
                .bias
                .as_ref()
                .expect("value bias"),
            [105.0, 106.0],
        );

        let report = super::clip_load_report_for_test("text_encoder", &result);
        assert!(report.contains("generated snapshot"), "{report}");
        assert!(
            report.contains("clip_l.blocks.0.attention.query.weight"),
            "{report}"
        );
    }

    #[test]
    fn component_safetensors_store_remaps_sdxl_clip_keys_before_qkv_split() {
        type B = ActiveBurnBackend;

        let runtime = BurnRuntime::<B>::new(active_test_device());
        let clip_l_profile = tiny_profile(ClipTextEncoderVariant::ClipL, false);
        let open_clip_g_profile = tiny_profile(ClipTextEncoderVariant::OpenClipG, true);
        let mut module = SdxlTextEncoders::<B>::init_from_profiles(
            &clip_l_profile,
            &open_clip_g_profile,
            runtime.device(),
        );
        let bytes = safetensors_bytes(vec![
            tensor_view(
                "model.text_encoder.transformer.resblocks.0.attn.in_proj_weight",
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
                "model.text_encoder.transformer.resblocks.0.attn.in_proj_bias",
                vec![6],
                vec![101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            ),
            tensor_view(
                "model.text_encoder.transformer.resblocks.0.attn.out_proj.weight",
                vec![2, 2],
                vec![201.0, 202.0, 203.0, 204.0],
            ),
            tensor_view(
                "model.text_encoder.transformer.resblocks.0.attn.out_proj.bias",
                vec![2],
                vec![301.0, 302.0],
            ),
        ]);
        let mut store = super::sdxl_clip_store_from_bytes(bytes);

        let result = runtime
            .load_module_store(&mut module, &mut store)
            .expect("component-style SDXL CLIP store should load through burn-store");

        assert!(
            result.errors.is_empty(),
            "unexpected store load errors: {result}"
        );
        assert!(
            result
                .applied
                .contains(&"clip_l.blocks.0.attention.query.weight".to_string())
        );
        assert_param_2d(
            &module.clip_l.blocks()[0].attention.query.weight,
            [1.0, 3.0, 2.0, 4.0],
        );
        assert_param_2d(
            &module.clip_l.blocks()[0].attention.key.weight,
            [5.0, 7.0, 6.0, 8.0],
        );
        assert_param_2d(
            &module.clip_l.blocks()[0].attention.value.weight,
            [9.0, 11.0, 10.0, 12.0],
        );
        assert_param_2d(
            &module.clip_l.blocks()[0].attention.output.weight,
            [201.0, 203.0, 202.0, 204.0],
        );
    }

    struct SnapshotStore {
        snapshots: BTreeMap<String, TensorSnapshot>,
        from_adapter: Option<Box<dyn burn_store::ModuleAdapter>>,
    }

    impl SnapshotStore {
        fn new(snapshots: Vec<TensorSnapshot>) -> Self {
            Self {
                snapshots: snapshots
                    .into_iter()
                    .map(|snapshot| (snapshot.full_path(), snapshot))
                    .collect(),
                from_adapter: None,
            }
        }
    }

    impl ModuleStore for SnapshotStore {
        type Error = std::convert::Infallible;

        fn collect_from<B: burn_tensor::backend::Backend, M: ModuleSnapshot<B>>(
            &mut self,
            _module: &M,
        ) -> Result<(), Self::Error> {
            Ok(())
        }

        fn apply_to<B: burn_tensor::backend::Backend, M: ModuleSnapshot<B>>(
            &mut self,
            module: &mut M,
        ) -> Result<ApplyResult, Self::Error> {
            let snapshots = self.snapshots.values().cloned().collect();
            Ok(module.apply(snapshots, None, self.from_adapter.clone(), false))
        }

        fn get_snapshot(&mut self, name: &str) -> Result<Option<&TensorSnapshot>, Self::Error> {
            Ok(self.snapshots.get(name))
        }

        fn get_all_snapshots(&mut self) -> Result<&BTreeMap<String, TensorSnapshot>, Self::Error> {
            Ok(&self.snapshots)
        }

        fn keys(&mut self) -> Result<Vec<String>, Self::Error> {
            Ok(self.snapshots.keys().cloned().collect())
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

    fn snapshot_2d(path: &str, rows: usize, cols: usize, values: Vec<f32>) -> TensorSnapshot {
        TensorSnapshot::from_data(
            TensorData::new(values, [rows, cols]),
            path.split('.').map(str::to_string).collect(),
            vec![],
            ParamId::new(),
        )
    }

    fn snapshot_1d(path: &str, values: Vec<f32>) -> TensorSnapshot {
        TensorSnapshot::from_data(
            TensorData::new(values.clone(), [values.len()]),
            path.split('.').map(str::to_string).collect(),
            vec![],
            ParamId::new(),
        )
    }

    fn safetensors_bytes(tensors: Vec<(String, TestTensorView)>) -> Vec<u8> {
        safetensors::tensor::serialize(tensors, None).expect("serialize safetensors bytes")
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

    fn assert_param_2d<const N: usize>(
        param: &burn_core::module::Param<burn_tensor::Tensor<ActiveBurnBackend, 2>>,
        expected: [f32; N],
    ) {
        assert_eq!(
            param.val().into_data().to_vec::<f32>().expect("f32 data"),
            expected
        );
    }

    fn assert_param_1d<const N: usize>(
        param: &burn_core::module::Param<burn_tensor::Tensor<ActiveBurnBackend, 1>>,
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
