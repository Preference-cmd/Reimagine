use reimagine_core::model::{
    ArtifactRef, ModelId, ModelRole, NodeId, ParamValue, SlotId, TensorDType, TensorShape,
    WorkflowInputId,
};
use reimagine_inference::{
    Backend, BackendPayloadKey, BackendTensorHandle, ConditioningMetadata, ExecutionConditioning,
    ExecutionValue, RuntimeClipHandle, RuntimeImage, RuntimeLatent, RuntimeModelHandle,
    RuntimeVaeHandle,
};
use reimagine_runtime::{NodeInputs, OutputKey, RunInputs, RunValueStore};
use std::sync::Arc;

#[test]
fn runtime_values_can_express_the_minimal_sdxl_base_intermediates() {
    let backend = Backend::new("candle");
    let device = "metal:0";
    let checkpoint = ModelId::new("sdxl-base");

    let model = RuntimeModelHandle::new(
        checkpoint.clone(),
        ModelRole::DiffusionModel,
        backend.clone(),
        "sdxl-base/denoiser",
    )
    .with_device(device);
    let clip = RuntimeClipHandle::new(checkpoint.clone(), backend.clone(), "sdxl-base/clip")
        .with_device(device);
    let vae = RuntimeVaeHandle::new(checkpoint.clone(), backend.clone(), "sdxl-base/vae")
        .with_device(device);

    assert!(matches!(
        ExecutionValue::Model(model.clone()),
        ExecutionValue::Model(_)
    ));
    assert_eq!(model.model_id(), &checkpoint);
    assert_eq!(model.role(), ModelRole::DiffusionModel);
    assert_eq!(clip.backend().as_str(), "candle");
    assert_eq!(vae.device_label(), Some(device));

    let text_embedding = BackendTensorHandle::new(
        backend.clone(),
        BackendPayloadKey::new("positive/text"),
        TensorDType::F32,
        TensorShape::new(vec![1, 77, 2048]),
        device,
    );
    let pooled_embedding = BackendTensorHandle::new(
        backend.clone(),
        BackendPayloadKey::new("positive/pooled"),
        TensorDType::F32,
        TensorShape::new(vec![1, 1280]),
        device,
    );
    let metadata = ConditioningMetadata::new(1024, 1024)
        .with_crop(0, 0)
        .with_target_size(1024, 1024);
    let positive = ExecutionConditioning::new(text_embedding.clone(), metadata.clone())
        .with_pooled_embedding(pooled_embedding.clone());
    let negative = ExecutionConditioning::new(
        BackendTensorHandle::new(
            backend.clone(),
            BackendPayloadKey::new("negative/text"),
            TensorDType::F32,
            TensorShape::new(vec![1, 77, 2048]),
            device,
        ),
        metadata,
    );

    assert_eq!(positive.text_embedding(), &text_embedding);
    assert_eq!(positive.pooled_embedding(), Some(&pooled_embedding));
    assert_eq!(positive.metadata().target_width(), 1024);
    assert!(matches!(
        ExecutionValue::Conditioning(negative),
        ExecutionValue::Conditioning(_)
    ));

    let latent_tensor = BackendTensorHandle::new(
        backend.clone(),
        BackendPayloadKey::new("latent/noise"),
        TensorDType::F32,
        TensorShape::new(vec![1, 4, 128, 128]),
        device,
    );
    let latent = RuntimeLatent::new(latent_tensor.clone(), 1024, 1024, 1, 4);
    assert_eq!(latent.payload(), &latent_tensor);
    assert_eq!(latent.width(), 1024);
    assert!(matches!(
        ExecutionValue::Latent(latent),
        ExecutionValue::Latent(_)
    ));

    let image_tensor = BackendTensorHandle::new(
        backend,
        BackendPayloadKey::new("image/decoded"),
        TensorDType::F32,
        TensorShape::new(vec![1, 3, 1024, 1024]),
        device,
    );
    let image = RuntimeImage::new(image_tensor.clone(), 1024, 1024, 1, "rgb");
    assert_eq!(image.payload(), &image_tensor);
    assert_eq!(image.color_space(), "rgb");
    assert!(matches!(
        ExecutionValue::Image(image),
        ExecutionValue::Image(_)
    ));

    let artifact = ExecutionValue::Artifact(ArtifactRef::new("outputs/sdxl.png"));
    assert!(matches!(artifact, ExecutionValue::Artifact(_)));
}

#[test]
fn runtime_values_do_not_require_candle_types() {
    let value = ExecutionValue::Param(ParamValue::String("a cinematic lake".to_owned()));
    assert_eq!(
        value.as_param(),
        Some(&ParamValue::String("a cinematic lake".to_owned()))
    );

    let tensor = BackendTensorHandle::new(
        Backend::new("mock-backend"),
        BackendPayloadKey::new("tensor-1"),
        TensorDType::F32,
        TensorShape::new(vec![1, 4, 8, 8]),
        "cpu",
    );

    assert_eq!(tensor.backend().as_str(), "mock-backend");
    assert_eq!(tensor.payload_key().as_str(), "tensor-1");
    assert_eq!(tensor.dtype(), TensorDType::F32);
    assert_eq!(tensor.shape().dims(), &[1, 4, 8, 8]);
    assert_eq!(tensor.device_label(), "cpu");
}

#[test]
fn runtime_api_surfaces_accept_execution_value_directly() {
    let value = Arc::new(ExecutionValue::Param(ParamValue::String(
        "hello".to_owned(),
    )));

    let mut node_inputs = NodeInputs::new();
    node_inputs.insert(SlotId::new("text"), value.clone());
    assert_eq!(
        node_inputs
            .get(&SlotId::new("text"))
            .and_then(|input| input.as_param()),
        Some(&ParamValue::String("hello".to_owned()))
    );

    let mut run_inputs = RunInputs::new();
    run_inputs.insert(NodeId::new("node-a"), SlotId::new("text"), value.clone());
    run_inputs.insert_workflow_input(WorkflowInputId::new("prompt"), value.clone());
    assert_eq!(
        run_inputs
            .get(&NodeId::new("node-a"), &SlotId::new("text"))
            .and_then(|input| input.as_param()),
        Some(&ParamValue::String("hello".to_owned()))
    );
    assert_eq!(
        run_inputs
            .workflow_input(&WorkflowInputId::new("prompt"))
            .and_then(|input| input.as_param()),
        Some(&ParamValue::String("hello".to_owned()))
    );

    let mut store = RunValueStore::new();
    let key = OutputKey::new(NodeId::new("node-a"), SlotId::new("text"));
    store.insert(key.clone(), value);
    let stored = store.get(&key).expect("stored output");
    assert_eq!(
        stored.as_param(),
        Some(&ParamValue::String("hello".to_owned()))
    );
}

#[test]
fn run_value_store_collapse_keeps_value_and_retention_in_one_record() {
    use reimagine_runtime::ExecutionValueRetention;

    let mut store = RunValueStore::new();
    let key = OutputKey::new(NodeId::new("node-a"), SlotId::new("latent"));
    let value = Arc::new(ExecutionValue::Param(ParamValue::String("x".to_owned())));

    // The default `insert` keeps the V1 RunScoped contract.
    store.insert(key.clone(), value.clone());
    assert_eq!(
        store.retention(&key),
        Some(ExecutionValueRetention::RunScoped)
    );
    assert_eq!(store.len(), 1);
    assert!(!store.is_empty());

    // Re-inserting with an explicit retention replaces the record.
    store.insert_with_retention(
        key.clone(),
        value.clone(),
        ExecutionValueRetention::SingleUse,
    );
    assert_eq!(
        store.retention(&key),
        Some(ExecutionValueRetention::SingleUse)
    );
    assert_eq!(store.len(), 1);

    // Removing drops the value AND the retention in one step.
    let removed = store.remove(&key);
    assert!(removed.is_some());
    assert!(store.retention(&key).is_none());
    assert!(store.get(&key).is_none());
    assert!(store.is_empty());

    // `clear` drops every record and releases the runtime's references.
    let key_b = OutputKey::new(NodeId::new("node-b"), SlotId::new("text"));
    store.insert(key_b.clone(), value.clone());
    assert_eq!(store.len(), 1);
    store.clear();
    assert!(store.is_empty());
    assert!(!store.contains(&key_b));
}
