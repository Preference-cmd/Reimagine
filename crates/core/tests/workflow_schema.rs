use reimagine_core::model::{
    ModelId, ModelRef, ModelRole, ModelSeries, ModelVariant, NodeId, NodeTypeId, ParamValue,
    SlotId, WorkflowId, WorkflowInputId, WorkflowOutputId, WorkflowVersion,
};
use reimagine_core::workflow::{Endpoint, Workflow};

#[test]
fn sdxl_base_workflow_example_roundtrips_through_core_schema() {
    let json = include_str!("../../../examples/workflows/sdxl-base-workflow.json");

    let workflow: Workflow = serde_json::from_str(json).expect("parse sdxl workflow example");
    let serialized = serde_json::to_string_pretty(&workflow).expect("serialize workflow");
    let reparsed: Workflow = serde_json::from_str(&serialized).expect("reparse workflow");

    assert_eq!(workflow, reparsed);
    assert_eq!(workflow.schema_version().as_str(), "reimagine.workflow.v1");
    assert_eq!(workflow.id(), &WorkflowId::new("workflow_sdxl_base_demo"));
    assert_eq!(workflow.version(), WorkflowVersion::new(1));
    assert_eq!(workflow.nodes().len(), 9);
    assert_eq!(workflow.edges().len(), 11);

    let checkpoint = workflow
        .nodes()
        .iter()
        .find(|node| node.id() == &NodeId::new("node_checkpoint"))
        .expect("checkpoint node exists");
    assert_eq!(
        checkpoint.type_id(),
        &NodeTypeId::new("builtin.checkpoint_loader")
    );
    assert_eq!(
        checkpoint.params().get(&SlotId::new("checkpoint")),
        Some(&ParamValue::ModelRef(ModelRef::new(
            ModelId::new("sdxl-base-1.0"),
            ModelSeries::new("stable_diffusion"),
            ModelVariant::new("sdxl"),
            ModelRole::CheckpointBundle,
        )))
    );
}

#[test]
fn endpoint_json_uses_node_slot_and_workflow_interface_forms() {
    let node_endpoint: Endpoint =
        serde_json::from_str(r#"{ "node": "node_sampler", "slot": "latent" }"#).unwrap();
    assert_eq!(
        node_endpoint,
        Endpoint::node_slot(NodeId::new("node_sampler"), SlotId::new("latent"))
    );

    let workflow_input: Endpoint =
        serde_json::from_str(r#"{ "workflow_input": "positive_prompt" }"#).unwrap();
    assert_eq!(
        workflow_input,
        Endpoint::workflow_input(WorkflowInputId::new("positive_prompt"))
    );

    let workflow_output: Endpoint =
        serde_json::from_str(r#"{ "workflow_output": "image" }"#).unwrap();
    assert_eq!(
        workflow_output,
        Endpoint::workflow_output(WorkflowOutputId::new("image"))
    );
}
