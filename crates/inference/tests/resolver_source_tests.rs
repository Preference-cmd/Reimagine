use reimagine_core::model::{ModelId, ModelRole, ModelSeries, ModelVariant};
use reimagine_inference::{
    ModelFormat, ModelSourceKind, ResolvedInferenceModel, ResolvedInferenceModelSource,
    ResolvedInferenceModelSourceSet,
};

#[test]
fn checkpoint_bundle_source_roundtrip() {
    let src = ResolvedInferenceModelSource::new(
        ModelSourceKind::CheckpointBundle,
        ModelRole::CheckpointBundle,
        std::path::PathBuf::from("/models/sdxl_base.safetensors"),
        ModelFormat::SafeTensors,
    );
    assert_eq!(src.kind(), ModelSourceKind::CheckpointBundle);
    assert_eq!(src.role(), ModelRole::CheckpointBundle);
    assert_eq!(src.format(), ModelFormat::SafeTensors);
    assert_eq!(
        src.path(),
        std::path::Path::new("/models/sdxl_base.safetensors")
    );
    assert!(src.metadata().is_none());
}

#[test]
fn split_component_source() {
    let mut src = ResolvedInferenceModelSource::new(
        ModelSourceKind::SplitComponent,
        ModelRole::TextEncoder,
        std::path::PathBuf::from("/models/clip_l.safetensors"),
        ModelFormat::SafeTensors,
    );
    src = src.with_metadata("clip=clip_l");
    let mut set = ResolvedInferenceModelSourceSet::new(src.clone());
    let unet = ResolvedInferenceModelSource::new(
        ModelSourceKind::SplitComponent,
        ModelRole::DiffusionModel,
        std::path::PathBuf::from("/models/unet.safetensors"),
        ModelFormat::SafeTensors,
    )
    .with_metadata("component=unet");
    set = set.with_source(unet);
    assert_eq!(set.sources().len(), 2);
    assert_eq!(set.sources()[0].role(), ModelRole::TextEncoder);
    assert_eq!(set.sources()[1].role(), ModelRole::DiffusionModel);
    assert_eq!(set.sources()[1].metadata().unwrap(), "component=unet");
}

#[test]
fn source_set_serde_roundtrip() {
    let src = ResolvedInferenceModelSource::new(
        ModelSourceKind::CheckpointBundle,
        ModelRole::CheckpointBundle,
        std::path::PathBuf::from("/models/test.safetensors"),
        ModelFormat::SafeTensors,
    )
    .with_metadata("test=true");
    let set = ResolvedInferenceModelSourceSet::new(src);
    let json = serde_json::to_string(&set).unwrap();
    let back: ResolvedInferenceModelSourceSet = serde_json::from_str(&json).unwrap();
    assert_eq!(set.sources().len(), back.sources().len());
}

#[test]
fn resolved_model_with_source_set() {
    let source = ResolvedInferenceModelSource::new(
        ModelSourceKind::CheckpointBundle,
        ModelRole::CheckpointBundle,
        std::path::PathBuf::from("/models/sdxl.safetensors"),
        ModelFormat::SafeTensors,
    )
    .with_metadata("source=checkpoint");
    let source_set = ResolvedInferenceModelSourceSet::new(source);
    let model = ResolvedInferenceModel::new(
        ModelId::new("sdxl-base"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        std::path::PathBuf::from("/models/sdxl.safetensors"),
        ModelFormat::SafeTensors,
    )
    .with_source_set(source_set.clone());
    assert_eq!(
        model.source_path(),
        std::path::Path::new("/models/sdxl.safetensors")
    );
    assert_eq!(model.format(), ModelFormat::SafeTensors);
    let ss = model.source_set().unwrap();
    assert_eq!(ss.sources().len(), 1);
    assert_eq!(ss.sources()[0].kind(), ModelSourceKind::CheckpointBundle);
    assert!(ss.is_checkpoint_bundle());
}

#[test]
fn resolved_model_without_source_set_defaults_none() {
    let model = ResolvedInferenceModel::new(
        ModelId::new("test-model"),
        ModelSeries::new("test"),
        ModelVariant::new("v1"),
        ModelRole::CheckpointBundle,
        std::path::PathBuf::from("/models/test.gguf"),
        ModelFormat::Gguf,
    );
    assert!(model.source_set().is_none());
}

#[test]
fn to_checkpoint_bundle_source_set() {
    let model = ResolvedInferenceModel::new(
        ModelId::new("test"),
        ModelSeries::new("sd"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        std::path::PathBuf::from("/models/test.safetensors"),
        ModelFormat::SafeTensors,
    )
    .with_metadata("v1=base");
    let set = model.to_checkpoint_bundle_source_set();
    assert!(set.is_checkpoint_bundle());
    assert_eq!(set.sources()[0].role(), ModelRole::CheckpointBundle);
    assert_eq!(set.sources()[0].format(), ModelFormat::SafeTensors);
    assert_eq!(
        set.sources()[0].path(),
        std::path::Path::new("/models/test.safetensors")
    );
    assert_eq!(set.sources()[0].metadata(), Some("v1=base"));
}

#[test]
#[should_panic(expected = "ResolvedInferenceModelSourceSet cannot be empty")]
fn source_set_rejects_empty_source_list() {
    let _ = ResolvedInferenceModelSourceSet::from_sources(Vec::new());
}
