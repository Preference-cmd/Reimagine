//! CLIP-specific burn-store adapters for SDXL text encoder weights.

use std::collections::BTreeMap;
use std::path::PathBuf;

use burn_store::{
    ApplyResult, KeyRemapper, ModuleAdapter, ModuleSnapshot, ModuleStore, PyTorchToBurnAdapter,
    SafetensorsStore, TensorSnapshot,
};
use burn_tensor::{DType, Shape, TensorData, backend::Backend};

/// Detect whether a safetensors file contains diffusers-format CLIP keys
/// (separate `q_proj`/`k_proj`/`v_proj`) versus converted fused QKV keys
/// (`in_proj_weight`).
///
/// Reads the safetensors header to check key names without loading tensor data.
#[allow(dead_code)]
fn detect_diffusers_clip_format(path: &std::path::Path) -> bool {
    let Ok(bytes) = std::fs::read(path) else {
        return false;
    };
    if bytes.is_empty() {
        return false;
    }
    // The safetensors header is a JSON object at the start of the file.
    // We look for `q_proj` as a reliable indicator of diffusers CLIP format.
    // Converted files use `in_proj_weight`/`in_proj_bias` instead.
    let header_end = bytes.len().min(1024 * 1024); // 1 MiB header limit
    let header_bytes = &bytes[..header_end];
    // Search for the diffusers CLIP indicator key pattern.
    // SAFETY: we only search for ASCII patterns in the JSON header.
    let header_str = String::from_utf8_lossy(header_bytes);
    header_str.contains("q_proj")
}

/// Build a burn-store loader for one SDXL CLIP component safetensors file.
///
/// The `component_role` parameter determines the target prefix for
/// diffusers-format remapping: `"text_encoder"` maps to `clip_l.blocks.`,
/// `"text_encoder_2"` maps to `open_clip_g.blocks.`.
#[allow(dead_code)]
pub(crate) fn sdxl_clip_store_from_path(
    path: impl Into<PathBuf>,
    component_role: &str,
) -> SdxlClipStore<SafetensorsStore> {
    let path = path.into();
    let diffusers_format = detect_diffusers_clip_format(&path);
    SdxlClipStore::new(sdxl_clip_safetensors_store(
        SafetensorsStore::from_file(path),
        diffusers_format,
        component_role,
    ))
    .with_from_adapter(PyTorchToBurnAdapter)
    .with_diffusers_format(diffusers_format)
}

#[cfg(test)]
fn sdxl_clip_store_from_bytes(bytes: Vec<u8>) -> SdxlClipStore<SafetensorsStore> {
    // Detect format from raw bytes: check if any key contains `q_proj`
    let diffusers_format = {
        let header_end = bytes.len().min(1024 * 1024);
        let header_str = String::from_utf8_lossy(&bytes[..header_end]);
        header_str.contains("q_proj")
    };
    SdxlClipStore::new(sdxl_clip_safetensors_store(
        SafetensorsStore::from_bytes(Some(bytes)),
        diffusers_format,
        "text_encoder", // default for test; tests that need open_clip_g will use a variant
    ))
    .with_from_adapter(PyTorchToBurnAdapter)
    .with_diffusers_format(diffusers_format)
}

#[allow(dead_code)]
fn sdxl_clip_safetensors_store(
    store: SafetensorsStore,
    diffusers_format: bool,
    component_role: &str,
) -> SafetensorsStore {
    let remapper = if diffusers_format {
        diffusers_clip_key_remapper(component_role)
    } else {
        sdxl_clip_key_remapper()
    };
    store.remap(remapper).allow_partial(true).validate(true)
}

/// Key remapper for diffusers-format CLIP safetensors files.
///
/// Diffusers CLIP uses separate `q_proj`/`k_proj`/`v_proj` attention
/// projections instead of a single fused `in_proj_weight`. This remapper
/// maps diffusers CLIP key paths to Burn module snapshot key paths.
///
/// The `component_role` parameter determines the target prefix:
/// - `"text_encoder"` (CLIP-L) -> `clip_l.blocks.N.`
/// - `"text_encoder_2"` (OpenCLIP-G) -> `open_clip_g.blocks.N.`
///
/// Key transformations:
/// - `text_model.encoder.` prefix is stripped
/// - `layers.N.` is mapped to the component-specific `blocks.N.` prefix
/// - `self_attn.q_proj` -> `attention.query`
/// - `self_attn.k_proj` -> `attention.key`
/// - `self_attn.v_proj` -> `attention.value`
/// - `self_attn.out_proj` -> `attention.output`
/// - `mlp.fc1` -> `ffn.ff1`
/// - `mlp.fc2` -> `ffn.ff2`
/// - `layer_norm1` -> `layer_norm`
/// - `layer_norm2` -> `layer_norm_inner`
#[allow(dead_code)]
fn diffusers_clip_key_remapper(component_role: &str) -> KeyRemapper {
    let block_prefix = match component_role {
        "text_encoder" => "clip_l.",
        "text_encoder_2" => "open_clip_g.",
        _ => "clip_l.",
    };

    KeyRemapper::new()
        // Attention projection renames (applied before prefix strip).
        .add_pattern(r"\.self_attn\.q_proj\.", ".attention.query.")
        .expect("static diffusers CLIP q_proj remapping regex should compile")
        .add_pattern(r"\.self_attn\.k_proj\.", ".attention.key.")
        .expect("static diffusers CLIP k_proj remapping regex should compile")
        .add_pattern(r"\.self_attn\.v_proj\.", ".attention.value.")
        .expect("static diffusers CLIP v_proj remapping regex should compile")
        .add_pattern(r"\.self_attn\.out_proj\.", ".attention.output.")
        .expect("static diffusers CLIP out_proj remapping regex should compile")
        // MLP layer renames.
        .add_pattern(r"\.mlp\.fc1\.", ".ffn.ff1.")
        .expect("static diffusers CLIP mlp.fc1 remapping regex should compile")
        .add_pattern(r"\.mlp\.fc2\.", ".ffn.ff2.")
        .expect("static diffusers CLIP mlp.fc2 remapping regex should compile")
        // Layer norm renames.
        .add_pattern(r"\.layer_norm1\.", ".layer_norm.")
        .expect("static diffusers CLIP layer_norm1 remapping regex should compile")
        .add_pattern(r"\.layer_norm2\.", ".layer_norm_inner.")
        .expect("static diffusers CLIP layer_norm2 remapping regex should compile")
        // Strip the `text_model.encoder.` prefix and add component-specific
        // block prefix. `layers.N` becomes `clip_l.blocks.N` or
        // `open_clip_g.blocks.N`.
        .add_pattern(
            r"^text_model\.encoder\.layers\.(\d+)\.",
            format!("{block_prefix}blocks.$1."),
        )
        .expect("static diffusers CLIP prefix+block remapping regex should compile")
        // Final layer norm gamma/beta -> weight/bias.
        .add_pattern(r"\.final_layer_norm\.gamma$", ".final_layer_norm.weight")
        .expect("static diffusers CLIP final layer norm weight regex should compile")
        .add_pattern(r"\.final_layer_norm\.beta$", ".final_layer_norm.bias")
        .expect("static diffusers CLIP final layer norm bias regex should compile")
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
///
/// For diffusers-format source files (separate `q_proj`/`k_proj`/`v_proj`),
/// the QKV expansion is skipped because the source already contains separate
/// attention projections. The `diffusers_format` flag controls this behavior.
#[allow(dead_code)]
pub(crate) struct SdxlClipStore<S> {
    inner: S,
    from_adapter: Option<Box<dyn ModuleAdapter>>,
    diffusers_format: bool,
}

#[allow(dead_code)]
impl<S> SdxlClipStore<S> {
    pub(crate) fn new(inner: S) -> Self {
        Self {
            inner,
            from_adapter: None,
            diffusers_format: false,
        }
    }

    pub(crate) fn with_from_adapter(mut self, adapter: impl ModuleAdapter + 'static) -> Self {
        self.from_adapter = Some(Box::new(adapter));
        self
    }

    /// Mark this store as loading diffusers-format CLIP keys (separate
    /// q/k/v projections). When set, fused QKV expansion is skipped.
    pub(crate) fn with_diffusers_format(mut self, diffusers_format: bool) -> Self {
        self.diffusers_format = diffusers_format;
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

        // For diffusers format, the source already contains separate
        // q_proj/k_proj/v_proj projections. No QKV expansion is needed.
        if self.diffusers_format {
            return Ok(snapshots);
        }

        // For converted/legacy format, expand fused in_proj_weight/in_proj_bias
        // into separate query/key/value snapshots for Burn MultiHeadAttention.
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
            let values = slice_tensor_data_rows(&data, row_offset, row_count, cols)?;
            Ok(TensorData::from_bytes_vec(
                values,
                vec![row_count, cols],
                data.dtype,
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
            let values = slice_tensor_data_rows(&data, offset, len, 1)?;
            Ok(TensorData::from_bytes_vec(values, vec![len], data.dtype))
        }),
        dtype,
        Shape::new([len]),
        path_stack,
        container_stack,
        tensor_id,
    )
}

#[allow(dead_code)]
fn slice_tensor_data_rows(
    data: &TensorData,
    row_offset: usize,
    row_count: usize,
    cols: usize,
) -> Result<Vec<u8>, burn_store::TensorSnapshotError> {
    let element_size = dtype_size(data.dtype)?;
    let row_size = cols.checked_mul(element_size).ok_or_else(|| {
        burn_store::TensorSnapshotError::DataError("row byte size overflow".into())
    })?;
    let start = row_offset.checked_mul(row_size).ok_or_else(|| {
        burn_store::TensorSnapshotError::DataError("row offset byte size overflow".into())
    })?;
    let len = row_count.checked_mul(row_size).ok_or_else(|| {
        burn_store::TensorSnapshotError::DataError("row count byte size overflow".into())
    })?;
    let end = start.checked_add(len).ok_or_else(|| {
        burn_store::TensorSnapshotError::DataError("row slice byte range overflow".into())
    })?;
    let bytes = data.as_bytes();
    if end > bytes.len() {
        return Err(burn_store::TensorSnapshotError::DataError(format!(
            "row slice byte range {start}..{end} exceeds tensor byte length {}",
            bytes.len()
        )));
    }
    Ok(bytes[start..end].to_vec())
}

#[allow(dead_code)]
fn dtype_size(dtype: DType) -> Result<usize, burn_store::TensorSnapshotError> {
    match dtype {
        DType::F64 => Ok(8),
        DType::F32 => Ok(4),
        DType::F16 | DType::BF16 => Ok(2),
        DType::I64 | DType::U64 => Ok(8),
        DType::I32 | DType::U32 => Ok(4),
        DType::I16 | DType::U16 => Ok(2),
        DType::I8 | DType::U8 => Ok(1),
        DType::Bool(_) => Ok(1),
        other => Err(burn_store::TensorSnapshotError::DataError(format!(
            "unsupported fused qkv dtype {other:?}"
        ))),
    }
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

    #[test]
    fn fused_qkv_split_preserves_f16_tensor_bytes() {
        let values = (0_u16..12_u16)
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let data = TensorData::from_bytes_vec(values.clone(), vec![6, 2], burn_tensor::DType::F16);

        let slice = super::slice_tensor_data_rows(&data, 2, 2, 2).expect("slice f16 rows");

        assert_eq!(
            slice,
            values[2 * 2 * 2..4 * 2 * 2],
            "row slice should preserve raw f16 bytes"
        );
        let tensor = TensorData::from_bytes_vec(slice, vec![2, 2], burn_tensor::DType::F16);
        assert_eq!(tensor.dtype, burn_tensor::DType::F16);
        assert_eq!(tensor.shape.dims(), [2, 2]);
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

    // ── Diffusers CLIP format tests ──────────────────────────────

    #[test]
    fn diffusers_clip_store_remaps_qkv_to_burn_mha_keys() {
        type B = ActiveBurnBackend;

        let runtime = BurnRuntime::<B>::new(active_test_device());
        let clip_l_profile = tiny_profile(ClipTextEncoderVariant::ClipL, false);
        let open_clip_g_profile = tiny_profile(ClipTextEncoderVariant::OpenClipG, true);
        let mut module = SdxlTextEncoders::<B>::init_from_profiles(
            &clip_l_profile,
            &open_clip_g_profile,
            runtime.device(),
        );
        // Diffusers CLIP format: separate q_proj/k_proj/v_proj
        let bytes = safetensors_bytes(vec![
            tensor_view(
                "text_model.encoder.layers.0.self_attn.q_proj.weight",
                vec![2, 2],
                vec![1.0, 2.0, 3.0, 4.0],
            ),
            tensor_view(
                "text_model.encoder.layers.0.self_attn.q_proj.bias",
                vec![2],
                vec![101.0, 102.0],
            ),
            tensor_view(
                "text_model.encoder.layers.0.self_attn.k_proj.weight",
                vec![2, 2],
                vec![5.0, 6.0, 7.0, 8.0],
            ),
            tensor_view(
                "text_model.encoder.layers.0.self_attn.k_proj.bias",
                vec![2],
                vec![103.0, 104.0],
            ),
            tensor_view(
                "text_model.encoder.layers.0.self_attn.v_proj.weight",
                vec![2, 2],
                vec![9.0, 10.0, 11.0, 12.0],
            ),
            tensor_view(
                "text_model.encoder.layers.0.self_attn.v_proj.bias",
                vec![2],
                vec![105.0, 106.0],
            ),
            tensor_view(
                "text_model.encoder.layers.0.self_attn.out_proj.weight",
                vec![2, 2],
                vec![201.0, 202.0, 203.0, 204.0],
            ),
            tensor_view(
                "text_model.encoder.layers.0.self_attn.out_proj.bias",
                vec![2],
                vec![301.0, 302.0],
            ),
            tensor_view(
                "text_model.encoder.layers.0.mlp.fc1.weight",
                vec![8, 2],
                vec![
                    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13,
                    0.14, 0.15, 0.16,
                ],
            ),
            tensor_view(
                "text_model.encoder.layers.0.mlp.fc1.bias",
                vec![8],
                vec![0.0; 8],
            ),
            tensor_view(
                "text_model.encoder.layers.0.mlp.fc2.weight",
                vec![2, 8],
                vec![
                    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13,
                    0.14, 0.15, 0.16,
                ],
            ),
            tensor_view(
                "text_model.encoder.layers.0.mlp.fc2.bias",
                vec![2],
                vec![0.0; 2],
            ),
            tensor_view(
                "text_model.encoder.layers.0.layer_norm1.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            tensor_view(
                "text_model.encoder.layers.0.layer_norm1.bias",
                vec![2],
                vec![0.0, 0.0],
            ),
            tensor_view(
                "text_model.encoder.layers.0.layer_norm2.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            tensor_view(
                "text_model.encoder.layers.0.layer_norm2.bias",
                vec![2],
                vec![0.0, 0.0],
            ),
        ]);
        let mut store = super::sdxl_clip_store_from_bytes(bytes);

        let result = runtime
            .load_module_store(&mut module, &mut store)
            .expect("diffusers CLIP store should load through burn-store");

        assert!(
            result.errors.is_empty(),
            "unexpected store load errors: {result}"
        );
        // Verify Q/K/V remapping
        assert!(
            result
                .applied
                .contains(&"clip_l.blocks.0.attention.query.weight".to_string()),
            "missing query.weight in applied: {result}"
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
        // Verify out_proj remapping
        assert_param_2d(
            &module.clip_l.blocks()[0].attention.output.weight,
            [201.0, 203.0, 202.0, 204.0],
        );
    }

    #[test]
    fn diffusers_clip_store_remaps_mlp_and_layer_norm_keys() {
        // Test that mlp.fc1/fc2 and layer_norm1/layer_norm2 are correctly remapped
        // through the diffusers CLIP key remapper.
        let remapper = super::diffusers_clip_key_remapper("text_encoder");

        let test_cases: Vec<(&str, &str)> = vec![
            (
                "text_model.encoder.layers.0.self_attn.q_proj.weight",
                "clip_l.blocks.0.attention.query.weight",
            ),
            (
                "text_model.encoder.layers.0.self_attn.k_proj.weight",
                "clip_l.blocks.0.attention.key.weight",
            ),
            (
                "text_model.encoder.layers.0.self_attn.v_proj.weight",
                "clip_l.blocks.0.attention.value.weight",
            ),
            (
                "text_model.encoder.layers.0.self_attn.out_proj.weight",
                "clip_l.blocks.0.attention.output.weight",
            ),
            (
                "text_model.encoder.layers.0.mlp.fc1.weight",
                "clip_l.blocks.0.ffn.ff1.weight",
            ),
            (
                "text_model.encoder.layers.0.mlp.fc2.weight",
                "clip_l.blocks.0.ffn.ff2.weight",
            ),
            (
                "text_model.encoder.layers.0.layer_norm1.weight",
                "clip_l.blocks.0.layer_norm.weight",
            ),
            (
                "text_model.encoder.layers.0.layer_norm2.weight",
                "clip_l.blocks.0.layer_norm_inner.weight",
            ),
            (
                "text_model.encoder.layers.0.layer_norm1.bias",
                "clip_l.blocks.0.layer_norm.bias",
            ),
            (
                "text_model.encoder.layers.0.layer_norm2.bias",
                "clip_l.blocks.0.layer_norm_inner.bias",
            ),
        ];

        for (source, expected) in test_cases {
            let snapshot = snapshot_2d(source, 1, 1, vec![1.0]);
            let (remapped, _) = remapper.remap(vec![snapshot]);
            assert_eq!(
                remapped.len(),
                1,
                "remapper should produce exactly one snapshot for `{source}`"
            );
            let result_path = remapped[0].full_path();
            assert_eq!(
                result_path, expected,
                "remapping `{source}` should produce `{expected}`, got `{result_path}`"
            );
        }
    }

    #[test]
    fn detect_diffusers_clip_format_returns_true_for_q_proj_keys() {
        let bytes = safetensors_bytes(vec![tensor_view(
            "text_model.encoder.layers.0.self_attn.q_proj.weight",
            vec![2, 2],
            vec![1.0, 2.0, 3.0, 4.0],
        )]);
        let header_str = String::from_utf8_lossy(&bytes[..bytes.len().min(1024 * 1024)]);
        assert!(
            header_str.contains("q_proj"),
            "header should contain q_proj indicator"
        );
    }

    #[test]
    fn detect_diffusers_clip_format_returns_false_for_in_proj_weight() {
        let bytes = safetensors_bytes(vec![tensor_view(
            "model.text_encoder.transformer.resblocks.0.attn.in_proj_weight",
            vec![6, 2],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )]);
        let header_str = String::from_utf8_lossy(&bytes[..bytes.len().min(1024 * 1024)]);
        assert!(
            !header_str.contains("q_proj"),
            "header should not contain q_proj for fused format"
        );
    }

    #[test]
    fn diffusers_format_skips_qkv_expansion() {
        // When diffusers_format is true, the store should NOT try to split
        // in_proj_weight into separate q/k/v (since they're already separate).
        type B = ActiveBurnBackend;

        let runtime = BurnRuntime::<B>::new(active_test_device());
        let clip_l_profile = tiny_profile(ClipTextEncoderVariant::ClipL, false);
        let open_clip_g_profile = tiny_profile(ClipTextEncoderVariant::OpenClipG, true);
        let mut module = SdxlTextEncoders::<B>::init_from_profiles(
            &clip_l_profile,
            &open_clip_g_profile,
            runtime.device(),
        );
        // Diffusers format with separate Q/K/V - already in target shape
        let mut store = super::SdxlClipStore::new(SnapshotStore::new(vec![
            snapshot_2d(
                "clip_l.blocks.0.attention.query.weight",
                2,
                2,
                vec![1.0, 2.0, 3.0, 4.0],
            ),
            snapshot_2d(
                "clip_l.blocks.0.attention.key.weight",
                2,
                2,
                vec![5.0, 6.0, 7.0, 8.0],
            ),
            snapshot_2d(
                "clip_l.blocks.0.attention.value.weight",
                2,
                2,
                vec![9.0, 10.0, 11.0, 12.0],
            ),
        ]))
        .with_from_adapter(PyTorchToBurnAdapter)
        .with_diffusers_format(true);

        let result = runtime
            .load_module_store(&mut module, &mut store)
            .expect("diffusers format store should load without QKV expansion");

        assert!(
            result.errors.is_empty(),
            "unexpected store load errors: {result}"
        );
        // The snapshots should be applied directly, not split
        assert!(
            result
                .applied
                .contains(&"clip_l.blocks.0.attention.query.weight".to_string()),
            "query.weight should be applied directly in diffusers mode"
        );
        assert_param_2d(
            &module.clip_l.blocks()[0].attention.query.weight,
            [1.0, 3.0, 2.0, 4.0],
        );
    }
}
