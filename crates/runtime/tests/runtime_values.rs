use reimagine_core::model::{
    ArtifactRef, ModelId, ModelRole, NodeId, ParamValue, SlotId, TensorDType, TensorShape,
    WorkflowInputId,
};
use reimagine_core::{
    BackendKind, BackendPayloadKey, BackendTensorHandle, ConditioningMetadata,
    ExecutionConditioning, ExecutionValue, RuntimeClipHandle, RuntimeImage, RuntimeLatent,
    RuntimeModelHandle, RuntimeVaeHandle,
};
use reimagine_runtime::{NodeInputs, OutputKey, RunInputs, RunValueStore};
use std::sync::Arc;

#[test]
fn runtime_values_can_express_the_minimal_sdxl_base_intermediates() {
    let backend = BackendKind::new("candle");
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
        BackendKind::new("mock-backend"),
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
