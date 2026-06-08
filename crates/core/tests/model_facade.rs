use reimagine_core::model::{
    ArtifactId, ArtifactRef, CommandBatchId, DiagnosticId, EdgeId, HistoryEntryId, InputSlotDef,
    ModelId, ModelRef, ModelRole, ModelSeries, ModelVariant, NodeDef, NodeEffect, NodeId,
    NodeValue, OutputSlotDef, ParamValue, ProposalId, RunId, SlotKind, TensorDType, TensorData,
    TensorShape, WorkflowId,
};

#[test]
fn shared_domain_types_are_available_from_the_model_facade() {
    let workflow_id = WorkflowId::new("workflow_demo");
    let node_id = NodeId::new("node_clip_encode");
    let edge_id = EdgeId::new("edge_prompt_to_sampler");
    let run_id = RunId::new("run_demo");
    let artifact_id = ArtifactId::new("artifact_id_demo");
    let diagnostic_id = DiagnosticId::new("diagnostic_demo");
    let history_entry_id = HistoryEntryId::new("history_demo");
    let command_batch_id = CommandBatchId::new("command_batch_demo");
    let proposal_id = ProposalId::new("proposal_demo");
    let model_id = ModelId::new("sdxl_base_demo");
    let artifact = ArtifactRef::new("artifact_preview");

    let model_input = InputSlotDef::new("model", SlotKind::Model)
        .dynamic(true)
        .required(true);
    let steps_input = InputSlotDef::new("steps", SlotKind::Integer)
        .required(true)
        .with_default_value(ParamValue::Integer(30));
    let output = OutputSlotDef::new("latent", SlotKind::Latent).required(true);

    let node = NodeDef::new("ksampler", "KSampler", "sampling")
        .with_effect(NodeEffect::Pure)
        .with_input_slot(model_input.clone())
        .with_input_slot(steps_input.clone())
        .with_output_slot(output.clone());

    assert_eq!(workflow_id.as_str(), "workflow_demo");
    assert_eq!(node_id.as_str(), "node_clip_encode");
    assert_eq!(edge_id.as_str(), "edge_prompt_to_sampler");
    assert_eq!(run_id.as_str(), "run_demo");
    assert_eq!(artifact_id.as_str(), "artifact_id_demo");
    assert_eq!(diagnostic_id.as_str(), "diagnostic_demo");
    assert_eq!(history_entry_id.as_str(), "history_demo");
    assert_eq!(command_batch_id.as_str(), "command_batch_demo");
    assert_eq!(proposal_id.as_str(), "proposal_demo");
    assert_eq!(model_id.as_str(), "sdxl_base_demo");
    assert_eq!(artifact.as_str(), "artifact_preview");
    assert_eq!(node.type_id().as_str(), "ksampler");
    assert_eq!(node.input_slots(), &[model_input, steps_input]);
    assert_eq!(node.output_slots(), &[output]);
}

#[test]
fn inference_contracts_use_tensor_data_from_the_shared_model_layer() {
    let tensor = TensorData::from_vec(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]);

    let input = reimagine_core::inference::NodeInput::Image(tensor.clone());
    let output = reimagine_core::inference::NodeOutput::Embedding(tensor.clone());
    let value = NodeValue::Tensor(tensor.clone());

    assert_eq!(tensor.numel(), 4);
    assert!(matches!(
        input,
        reimagine_core::inference::NodeInput::Image(ref image) if image == &tensor
    ));
    assert!(matches!(
        output,
        reimagine_core::inference::NodeOutput::Embedding(ref embedding) if embedding == &tensor
    ));
    assert_eq!(value, NodeValue::Tensor(tensor));
}

#[test]
fn runtime_facing_model_values_are_available_from_the_model_facade() {
    let model_id = ModelId::new("sdxl-base");
    let model_ref = ModelRef::new(
        model_id.clone(),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::DiffusionModel,
    );
    let shape = TensorShape::new(vec![1, 4, 128, 128]);
    let prompt = ParamValue::String("cinematic lake".to_owned());

    assert_eq!(model_ref.id(), &model_id);
    assert_eq!(model_ref.model_series().as_str(), "stable_diffusion");
    assert_eq!(model_ref.variant().as_str(), "sdxl");
    assert_eq!(model_ref.role(), ModelRole::DiffusionModel);
    assert_eq!(shape.dims(), &[1, 4, 128, 128]);
    assert_eq!(shape.rank(), 4);
    assert_eq!(shape.numel(), 65_536);
    assert_eq!(TensorDType::F32, TensorDType::F32);
    assert_eq!(prompt, ParamValue::String("cinematic lake".to_owned()));
}
