use std::collections::HashMap;

use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::model::{
    InputSlotDef, ModelId, ModelRef, ModelRole, ModelSeries, ModelVariant, NodeCatalog, NodeDef,
    NodeEffect, NodeId, NodeTypeId, OutputSlotDef, ParamValue, SlotId, SlotKind, WorkflowOutputId,
};
use reimagine_core::readiness::{
    ExecutionInputSource, ExecutionWorkflowOutputSource, ExternalReadinessContext,
    ExternalReadinessProvider, ExternalReadinessSubject, RunTarget, RunTargetSelection,
    build_execution_plan,
};
use reimagine_core::workflow::{
    Endpoint, Workflow, WorkflowEdge, WorkflowInputDef, WorkflowInterface, WorkflowNode,
    WorkflowOutputDef,
};

#[test]
fn successful_sdxl_like_plan_is_built_for_default_targets() {
    let workflow: Workflow = serde_json::from_str(include_str!(
        "../../../examples/workflows/sdxl-base-workflow.json"
    ))
    .expect("parse sdxl workflow example");
    let catalog = sdxl_catalog();
    let provider = SnapshotProvider::with_ok(model_ref("sdxl-base-1.0"));

    let result = build_execution_plan(
        &workflow,
        &catalog,
        RunTargetSelection::AllDefaultTargets,
        Some(&provider),
    );

    assert!(result.report().diagnostics().is_empty());

    let plan = result.plan().expect("plan should be available");
    assert_eq!(
        plan.targets(),
        &[RunTarget::Node {
            node_id: NodeId::new("node_save_image"),
        }]
    );
    assert_eq!(plan.nodes().len(), 9);
    assert_eq!(plan.edges().len(), 11);
    assert_eq!(
        plan.stages()
            .iter()
            .map(|stage| {
                stage
                    .node_ids()
                    .iter()
                    .map(|node_id| node_id.as_str())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        vec![
            vec![
                "node_checkpoint",
                "node_positive_prompt",
                "node_negative_prompt",
                "node_latent",
            ],
            vec!["node_positive_encode", "node_negative_encode"],
            vec!["node_sampler"],
            vec!["node_vae_decode"],
            vec!["node_save_image"],
        ]
    );
}

#[test]
fn missing_required_input_blocks_terminal_target_execution() {
    let workflow = Workflow::new("workflow_missing_input", 1.into()).with_node(
        WorkflowNode::new("node_preview", "builtin.preview_image")
            .with_param("label", ParamValue::String("preview".to_owned())),
    );
    let catalog = preview_catalog();

    let result = build_execution_plan(
        &workflow,
        &catalog,
        RunTargetSelection::AllDefaultTargets,
        None,
    );

    assert!(result.plan().is_none());
    let codes = diagnostic_codes(&result);
    assert!(codes.contains(&"CORE/WORKFLOW_REQUIRED_INPUT_MISSING"));
}

#[test]
fn missing_external_readiness_entry_blocks_required_model_ref_subject() {
    let workflow: Workflow = serde_json::from_str(include_str!(
        "../../../examples/workflows/sdxl-base-workflow.json"
    ))
    .expect("parse sdxl workflow example");
    let catalog = sdxl_catalog();
    let provider = SnapshotProvider {
        entries: HashMap::new(),
    };

    let result = build_execution_plan(
        &workflow,
        &catalog,
        RunTargetSelection::AllDefaultTargets,
        Some(&provider),
    );

    assert!(result.plan().is_none());
    assert!(diagnostic_codes(&result).contains(&"CORE/WORKFLOW_EXTERNAL_READINESS_MISSING"));
}

#[test]
fn external_readiness_warnings_do_not_block_plan_construction() {
    let workflow: Workflow = serde_json::from_str(include_str!(
        "../../../examples/workflows/sdxl-base-workflow.json"
    ))
    .expect("parse sdxl workflow example");
    let catalog = sdxl_catalog();
    let provider = SnapshotProvider::with_warning(model_ref("sdxl-base-1.0"));

    let result = build_execution_plan(
        &workflow,
        &catalog,
        RunTargetSelection::AllDefaultTargets,
        Some(&provider),
    );

    assert!(result.plan().is_some());
    assert!(diagnostic_codes(&result).contains(&"MODEL_MANAGER/MODEL_SOURCE_STALE"));
}

#[test]
fn edged_model_ref_param_does_not_trigger_external_check_for_overridden_param() {
    let stale_ref = model_ref("stale");
    let workflow = Workflow::new("workflow_model_ref_override", 1.into())
        .with_node(
            WorkflowNode::new("node_model_ref", "builtin.model_ref")
                .with_param("value", ParamValue::ModelRef(model_ref("active"))),
        )
        .with_node(
            WorkflowNode::new("node_loader", "builtin.model_ref_passthrough")
                .with_param("value", ParamValue::ModelRef(stale_ref)),
        )
        .with_node(WorkflowNode::new(
            "node_model_target",
            "builtin.model_target",
        ))
        .with_edge(WorkflowEdge::new(
            "edge_ref_loader",
            Endpoint::node_slot("node_model_ref".into(), "value".into()),
            Endpoint::node_slot("node_loader".into(), "value".into()),
        ))
        .with_edge(WorkflowEdge::new(
            "edge_loader_target",
            Endpoint::node_slot("node_loader".into(), "value".into()),
            Endpoint::node_slot("node_model_target".into(), "value".into()),
        ));
    let catalog = model_ref_catalog();
    let provider = SnapshotProvider::with_ok(model_ref("active"));

    let result = build_execution_plan(
        &workflow,
        &catalog,
        RunTargetSelection::AllDefaultTargets,
        Some(&provider),
    );

    assert!(result.plan().is_some());
    assert!(result.report().diagnostics().is_empty());
}

#[test]
fn no_target_reports_non_contributing_pure_graphs() {
    let workflow = Workflow::new("workflow_no_target", 1.into())
        .with_node(
            WorkflowNode::new("node_prompt", "builtin.string")
                .with_param("value", ParamValue::String("hello".to_owned())),
        )
        .with_node(WorkflowNode::new("node_encode", "builtin.clip_text_encode"))
        .with_edge(WorkflowEdge::new(
            "edge_prompt",
            Endpoint::node_slot("node_prompt".into(), "value".into()),
            Endpoint::node_slot("node_encode".into(), "text".into()),
        ));
    let catalog = no_target_catalog();

    let result = build_execution_plan(
        &workflow,
        &catalog,
        RunTargetSelection::AllDefaultTargets,
        None,
    );

    assert!(result.plan().is_none());
    let codes = diagnostic_codes(&result);
    assert!(codes.contains(&"CORE/WORKFLOW_NO_TARGET"));
    assert!(codes.contains(&"CORE/WORKFLOW_NON_CONTRIBUTING_PURE_GRAPH"));
}

#[test]
fn executable_cycle_blocks_plan_construction() {
    let workflow = Workflow::new("workflow_cycle", 1.into())
        .with_node(WorkflowNode::new("node_a", "builtin.image_transform"))
        .with_node(WorkflowNode::new("node_b", "builtin.image_transform"))
        .with_node(
            WorkflowNode::new("node_preview", "builtin.preview_image")
                .with_param("label", ParamValue::String("preview".to_owned())),
        )
        .with_edge(WorkflowEdge::new(
            "edge_a_to_b",
            Endpoint::node_slot("node_a".into(), "image".into()),
            Endpoint::node_slot("node_b".into(), "image".into()),
        ))
        .with_edge(WorkflowEdge::new(
            "edge_b_to_a",
            Endpoint::node_slot("node_b".into(), "image".into()),
            Endpoint::node_slot("node_a".into(), "image".into()),
        ))
        .with_edge(WorkflowEdge::new(
            "edge_b_to_preview",
            Endpoint::node_slot("node_b".into(), "image".into()),
            Endpoint::node_slot("node_preview".into(), "image".into()),
        ));
    let catalog = preview_catalog();

    let result = build_execution_plan(
        &workflow,
        &catalog,
        RunTargetSelection::AllDefaultTargets,
        None,
    );

    assert!(result.plan().is_none());
    assert!(diagnostic_codes(&result).contains(&"CORE/WORKFLOW_EXECUTABLE_CYCLE"));
}

#[test]
fn preview_terminal_node_is_a_valid_explicit_target() {
    let workflow = preview_and_save_workflow();
    let catalog = preview_catalog();

    let result = build_execution_plan(
        &workflow,
        &catalog,
        RunTargetSelection::ExplicitTargets(vec![RunTarget::Node {
            node_id: NodeId::new("node_preview"),
        }]),
        None,
    );

    let plan = result.plan().expect("preview target should plan");
    assert_eq!(
        plan.targets(),
        &[RunTarget::Node {
            node_id: NodeId::new("node_preview"),
        }]
    );
    assert_eq!(
        plan.nodes()
            .iter()
            .map(|node| node.node_id().as_str())
            .collect::<Vec<_>>(),
        vec!["node_image_source", "node_preview"]
    );
}

#[test]
fn explicit_target_partial_execution_excludes_other_terminal_branches() {
    let workflow = preview_and_save_workflow();
    let catalog = preview_catalog();

    let result = build_execution_plan(
        &workflow,
        &catalog,
        RunTargetSelection::ExplicitTargets(vec![RunTarget::NodeOutput {
            node_id: NodeId::new("node_image_source"),
            slot_id: SlotId::new("image"),
        }]),
        None,
    );

    let plan = result.plan().expect("node output target should plan");
    assert_eq!(
        plan.nodes()
            .iter()
            .map(|node| node.node_id().as_str())
            .collect::<Vec<_>>(),
        vec!["node_image_source"]
    );
    assert!(plan.edges().is_empty());
}

#[test]
fn explicit_workflow_output_target_traces_its_producer() {
    let workflow = Workflow::new("workflow_output_target", 1.into())
        .with_interface(WorkflowInterface::new().with_output(WorkflowOutputDef::new(
            WorkflowOutputId::new("image"),
            "image".into(),
            SlotKind::Image,
        )))
        .with_node(WorkflowNode::new(
            "node_image_source",
            "builtin.image_source",
        ))
        .with_edge(WorkflowEdge::new(
            "edge_source_output",
            Endpoint::node_slot("node_image_source".into(), "image".into()),
            Endpoint::workflow_output(WorkflowOutputId::new("image")),
        ));
    let catalog = preview_catalog();

    let result = build_execution_plan(
        &workflow,
        &catalog,
        RunTargetSelection::ExplicitTargets(vec![RunTarget::WorkflowOutput {
            output_id: WorkflowOutputId::new("image"),
        }]),
        None,
    );

    let plan = result.plan().expect("workflow output target should plan");
    assert_eq!(
        plan.nodes()
            .iter()
            .map(|node| node.node_id().as_str())
            .collect::<Vec<_>>(),
        vec!["node_image_source"]
    );
    assert_eq!(
        plan.workflow_outputs()[0].source(),
        &ExecutionWorkflowOutputSource::NodeOutput {
            node_id: NodeId::new("node_image_source"),
            slot_id: SlotId::new("image"),
        }
    );
}

#[test]
fn explicit_workflow_output_target_can_passthrough_workflow_input() {
    let workflow = Workflow::new("workflow_output_passthrough", 1.into())
        .with_interface(
            WorkflowInterface::new()
                .with_input(WorkflowInputDef::new(
                    "input_image".into(),
                    "image".into(),
                    SlotKind::Image,
                ))
                .with_output(WorkflowOutputDef::new(
                    WorkflowOutputId::new("image"),
                    "image".into(),
                    SlotKind::Image,
                )),
        )
        .with_edge(WorkflowEdge::new(
            "edge_input_output",
            Endpoint::workflow_input("input_image".into()),
            Endpoint::workflow_output(WorkflowOutputId::new("image")),
        ));
    let catalog = preview_catalog();

    let result = build_execution_plan(
        &workflow,
        &catalog,
        RunTargetSelection::ExplicitTargets(vec![RunTarget::WorkflowOutput {
            output_id: WorkflowOutputId::new("image"),
        }]),
        None,
    );

    let plan = result.plan().expect("passthrough workflow output target");
    assert!(plan.nodes().is_empty());
    assert!(plan.stages().is_empty());
    assert_eq!(
        plan.workflow_outputs()[0].source(),
        &ExecutionWorkflowOutputSource::WorkflowInput {
            workflow_input_id: "input_image".into(),
        }
    );
}

#[test]
fn explicit_node_output_target_requires_existing_output_slot() {
    let workflow = preview_and_save_workflow();
    let catalog = preview_catalog();

    let result = build_execution_plan(
        &workflow,
        &catalog,
        RunTargetSelection::ExplicitTargets(vec![RunTarget::NodeOutput {
            node_id: NodeId::new("node_image_source"),
            slot_id: SlotId::new("missing"),
        }]),
        None,
    );

    assert!(result.plan().is_none());
    assert!(diagnostic_codes(&result).contains(&"CORE/WORKFLOW_TARGET_INVALID"));
}

#[test]
fn empty_explicit_target_selection_reports_invalid_target() {
    let workflow = preview_and_save_workflow();
    let catalog = preview_catalog();

    let result = build_execution_plan(
        &workflow,
        &catalog,
        RunTargetSelection::ExplicitTargets(vec![]),
        None,
    );

    assert!(result.plan().is_none());
    assert!(diagnostic_codes(&result).contains(&"CORE/WORKFLOW_TARGET_INVALID"));
}

#[test]
fn workflow_input_edge_satisfies_required_dynamic_input() {
    let workflow = Workflow::new("workflow_input_target", 1.into())
        .with_interface(WorkflowInterface::new().with_input(WorkflowInputDef::new(
            "input_image".into(),
            "image".into(),
            SlotKind::Image,
        )))
        .with_node(
            WorkflowNode::new("node_preview", "builtin.preview_image")
                .with_param("label", ParamValue::String("preview".to_owned())),
        )
        .with_edge(WorkflowEdge::new(
            "edge_input_preview",
            Endpoint::workflow_input("input_image".into()),
            Endpoint::node_slot("node_preview".into(), "image".into()),
        ));
    let catalog = preview_catalog();

    let result = build_execution_plan(
        &workflow,
        &catalog,
        RunTargetSelection::AllDefaultTargets,
        None,
    );

    let plan = result.plan().expect("workflow input should satisfy input");
    assert_eq!(plan.nodes().len(), 1);
    assert!(plan.edges().is_empty());
    assert_eq!(
        plan.nodes()[0].input_bindings()[0].source(),
        &ExecutionInputSource::WorkflowInput {
            edge_id: "edge_input_preview".into(),
            workflow_input_id: "input_image".into(),
        }
    );
}

#[test]
fn pure_required_output_must_be_consumed_or_exposed() {
    let workflow = Workflow::new("workflow_unconsumed_pure_output", 1.into())
        .with_node(WorkflowNode::new(
            "node_dual_image_source",
            "builtin.dual_image_source",
        ))
        .with_node(
            WorkflowNode::new("node_preview", "builtin.preview_image")
                .with_param("label", ParamValue::String("preview".to_owned())),
        )
        .with_edge(WorkflowEdge::new(
            "edge_image_preview",
            Endpoint::node_slot("node_dual_image_source".into(), "image".into()),
            Endpoint::node_slot("node_preview".into(), "image".into()),
        ));
    let catalog = preview_catalog();

    let result = build_execution_plan(
        &workflow,
        &catalog,
        RunTargetSelection::ExplicitTargets(vec![RunTarget::Node {
            node_id: NodeId::new("node_preview"),
        }]),
        None,
    );

    assert!(result.plan().is_none());
    assert!(diagnostic_codes(&result).contains(&"CORE/WORKFLOW_NON_CONTRIBUTING_PURE_GRAPH"));
}

#[test]
fn pure_required_output_is_valid_when_explicitly_targeted() {
    let workflow = Workflow::new("workflow_exposed_pure_output", 1.into()).with_node(
        WorkflowNode::new("node_image_source", "builtin.image_source"),
    );
    let catalog = preview_catalog();

    let result = build_execution_plan(
        &workflow,
        &catalog,
        RunTargetSelection::ExplicitTargets(vec![RunTarget::NodeOutput {
            node_id: NodeId::new("node_image_source"),
            slot_id: SlotId::new("image"),
        }]),
        None,
    );

    assert!(result.plan().is_some());
    assert!(result.report().diagnostics().is_empty());
}

#[test]
fn merged_multi_target_planning_executes_shared_upstream_once() {
    let workflow = preview_and_save_workflow();
    let catalog = preview_catalog();

    let result = build_execution_plan(
        &workflow,
        &catalog,
        RunTargetSelection::AllDefaultTargets,
        None,
    );

    let plan = result.plan().expect("default targets should plan");
    assert_eq!(
        plan.targets(),
        &[
            RunTarget::Node {
                node_id: NodeId::new("node_preview"),
            },
            RunTarget::Node {
                node_id: NodeId::new("node_save"),
            },
        ]
    );
    assert_eq!(
        plan.nodes()
            .iter()
            .map(|node| node.node_id().as_str())
            .collect::<Vec<_>>(),
        vec!["node_image_source", "node_preview", "node_save"]
    );
    assert_eq!(
        plan.stages()
            .iter()
            .map(|stage| {
                stage
                    .node_ids()
                    .iter()
                    .map(|node_id| node_id.as_str())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        vec![vec!["node_image_source"], vec!["node_preview", "node_save"]]
    );
}

#[test]
fn deterministic_stage_order_is_stable_across_runs() {
    let workflow = Workflow::new("workflow_deterministic", 1.into())
        .with_node(WorkflowNode::new("node_source_c", "builtin.image_source"))
        .with_node(WorkflowNode::new("node_source_a", "builtin.image_source"))
        .with_node(WorkflowNode::new("node_source_b", "builtin.image_source"))
        .with_node(
            WorkflowNode::new("node_merge", "builtin.image_merge")
                .with_param("mode", ParamValue::Select("overlay".to_owned())),
        )
        .with_node(
            WorkflowNode::new("node_preview", "builtin.preview_image")
                .with_param("label", ParamValue::String("preview".to_owned())),
        )
        .with_edge(WorkflowEdge::new(
            "edge_a",
            Endpoint::node_slot("node_source_a".into(), "image".into()),
            Endpoint::node_slot("node_merge".into(), "left".into()),
        ))
        .with_edge(WorkflowEdge::new(
            "edge_b",
            Endpoint::node_slot("node_source_b".into(), "image".into()),
            Endpoint::node_slot("node_merge".into(), "right".into()),
        ))
        .with_edge(WorkflowEdge::new(
            "edge_merge_preview",
            Endpoint::node_slot("node_merge".into(), "image".into()),
            Endpoint::node_slot("node_preview".into(), "image".into()),
        ));
    let catalog = preview_catalog();

    let first = build_execution_plan(
        &workflow,
        &catalog,
        RunTargetSelection::AllDefaultTargets,
        None,
    );
    let second = build_execution_plan(
        &workflow,
        &catalog,
        RunTargetSelection::AllDefaultTargets,
        None,
    );

    let first_plan = first.plan().expect("first plan");
    let second_plan = second.plan().expect("second plan");
    assert_eq!(first_plan.stages(), second_plan.stages());
    assert_eq!(
        first_plan
            .stages()
            .iter()
            .map(|stage| {
                stage
                    .node_ids()
                    .iter()
                    .map(|node_id| node_id.as_str())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        vec![
            vec!["node_source_a", "node_source_b"],
            vec!["node_merge"],
            vec!["node_preview"],
        ]
    );
}

struct TestCatalog {
    defs: HashMap<NodeTypeId, NodeDef>,
}

impl TestCatalog {
    fn new(defs: Vec<NodeDef>) -> Self {
        Self {
            defs: defs
                .into_iter()
                .map(|node_def| (node_def.type_id().clone(), node_def))
                .collect(),
        }
    }
}

impl NodeCatalog for TestCatalog {
    fn get(&self, type_id: &NodeTypeId) -> Option<&NodeDef> {
        self.defs.get(type_id)
    }
}

struct SnapshotProvider {
    entries: HashMap<ModelRef, Vec<Diagnostic>>,
}

impl SnapshotProvider {
    fn with_ok(model_ref: ModelRef) -> Self {
        Self {
            entries: HashMap::from([(model_ref, Vec::new())]),
        }
    }

    fn with_warning(model_ref: ModelRef) -> Self {
        Self {
            entries: HashMap::from([(
                model_ref.clone(),
                vec![Diagnostic::new(
                    "diag-model-source-stale".into(),
                    DiagnosticCode::new("MODEL_MANAGER/MODEL_SOURCE_STALE"),
                    DiagnosticSeverity::Warning,
                    DiagnosticSourceName::new("model-manager"),
                    "model source is stale",
                    DiagnosticTarget::new(DiagnosticTargetDomain::new("model"))
                        .with_id(model_ref.id().as_str().to_owned()),
                )],
            )]),
        }
    }
}

impl ExternalReadinessProvider for SnapshotProvider {
    fn diagnostics_for(
        &self,
        _context: &ExternalReadinessContext,
        subject: &ExternalReadinessSubject,
    ) -> Option<Vec<Diagnostic>> {
        match subject {
            ExternalReadinessSubject::ModelRef(model_ref) => self.entries.get(model_ref).cloned(),
        }
    }
}

fn diagnostic_codes<'a>(
    result: &'a reimagine_core::readiness::ExecutionPlanResult,
) -> Vec<&'a str> {
    result
        .report()
        .diagnostics()
        .iter()
        .map(|diagnostic| diagnostic.code().as_str())
        .collect()
}

fn preview_and_save_workflow() -> Workflow {
    Workflow::new("workflow_preview_and_save", 1.into())
        .with_node(WorkflowNode::new(
            "node_image_source",
            "builtin.image_source",
        ))
        .with_node(
            WorkflowNode::new("node_preview", "builtin.preview_image")
                .with_param("label", ParamValue::String("preview".to_owned())),
        )
        .with_node(
            WorkflowNode::new("node_save", "builtin.save_image")
                .with_param("filename_prefix", ParamValue::String("demo".to_owned())),
        )
        .with_edge(WorkflowEdge::new(
            "edge_source_preview",
            Endpoint::node_slot("node_image_source".into(), "image".into()),
            Endpoint::node_slot("node_preview".into(), "image".into()),
        ))
        .with_edge(WorkflowEdge::new(
            "edge_source_save",
            Endpoint::node_slot("node_image_source".into(), "image".into()),
            Endpoint::node_slot("node_save".into(), "image".into()),
        ))
}

fn preview_catalog() -> TestCatalog {
    TestCatalog::new(vec![
        NodeDef::new("builtin.image_source", "Image Source", "test")
            .with_output_slot(OutputSlotDef::new("image", SlotKind::Image).required(true)),
        NodeDef::new("builtin.dual_image_source", "Dual Image Source", "test")
            .with_output_slot(OutputSlotDef::new("image", SlotKind::Image).required(true))
            .with_output_slot(OutputSlotDef::new("mask", SlotKind::Image).required(true)),
        NodeDef::new("builtin.image_transform", "Image Transform", "test")
            .with_input_slot(
                InputSlotDef::new("image", SlotKind::Image)
                    .dynamic(true)
                    .required(true),
            )
            .with_output_slot(OutputSlotDef::new("image", SlotKind::Image).required(true)),
        NodeDef::new("builtin.image_merge", "Image Merge", "test")
            .with_input_slot(
                InputSlotDef::new("left", SlotKind::Image)
                    .dynamic(true)
                    .required(true),
            )
            .with_input_slot(
                InputSlotDef::new("right", SlotKind::Image)
                    .dynamic(true)
                    .required(true),
            )
            .with_input_slot(InputSlotDef::new("mode", SlotKind::Select).required(true))
            .with_output_slot(OutputSlotDef::new("image", SlotKind::Image).required(true)),
        NodeDef::new("builtin.preview_image", "Preview Image", "test")
            .with_effect(NodeEffect::SideEffect)
            .with_input_slot(
                InputSlotDef::new("image", SlotKind::Image)
                    .dynamic(true)
                    .required(true),
            )
            .with_input_slot(InputSlotDef::new("label", SlotKind::String).required(true)),
        NodeDef::new("builtin.save_image", "Save Image", "test")
            .with_effect(NodeEffect::SideEffect)
            .with_input_slot(
                InputSlotDef::new("image", SlotKind::Image)
                    .dynamic(true)
                    .required(true),
            )
            .with_input_slot(InputSlotDef::new("filename_prefix", SlotKind::String).required(true)),
    ])
}

fn no_target_catalog() -> TestCatalog {
    TestCatalog::new(vec![
        NodeDef::new("builtin.string", "String", "input")
            .with_input_slot(InputSlotDef::new("value", SlotKind::String).required(true))
            .with_output_slot(OutputSlotDef::new("value", SlotKind::String).required(true)),
        NodeDef::new(
            "builtin.clip_text_encode",
            "CLIP Text Encode",
            "conditioning",
        )
        .with_input_slot(
            InputSlotDef::new("text", SlotKind::String)
                .dynamic(true)
                .required(true),
        )
        .with_output_slot(
            OutputSlotDef::new("conditioning", SlotKind::Conditioning).required(true),
        ),
    ])
}

fn model_ref_catalog() -> TestCatalog {
    TestCatalog::new(vec![
        NodeDef::new("builtin.model_ref", "Model Ref", "test")
            .with_input_slot(
                InputSlotDef::new("value", SlotKind::ModelRef)
                    .required(true)
                    .with_default_value(ParamValue::ModelRef(model_ref("active"))),
            )
            .with_output_slot(OutputSlotDef::new("value", SlotKind::ModelRef).required(true)),
        NodeDef::new(
            "builtin.model_ref_passthrough",
            "Model Ref Passthrough",
            "test",
        )
        .with_input_slot(
            InputSlotDef::new("value", SlotKind::ModelRef)
                .dynamic(true)
                .required(true),
        )
        .with_output_slot(OutputSlotDef::new("value", SlotKind::ModelRef).required(true)),
        NodeDef::new("builtin.model_target", "Model Target", "test")
            .with_effect(NodeEffect::SideEffect)
            .with_input_slot(
                InputSlotDef::new("value", SlotKind::ModelRef)
                    .dynamic(true)
                    .required(true),
            ),
        NodeDef::new("builtin.debug_model", "Debug Model", "test")
            .with_output_slot(OutputSlotDef::new("value", SlotKind::ModelRef).required(true)),
    ])
}

fn sdxl_catalog() -> TestCatalog {
    TestCatalog::new(vec![
        NodeDef::new("builtin.checkpoint_loader", "Checkpoint Loader", "model")
            .with_input_slot(InputSlotDef::new("checkpoint", SlotKind::ModelRef).required(true))
            .with_output_slot(OutputSlotDef::new("model", SlotKind::Model).required(true))
            .with_output_slot(OutputSlotDef::new("clip", SlotKind::Clip).required(true))
            .with_output_slot(OutputSlotDef::new("vae", SlotKind::Vae).required(true)),
        NodeDef::new("builtin.string", "String", "input")
            .with_input_slot(InputSlotDef::new("value", SlotKind::String).required(true))
            .with_output_slot(OutputSlotDef::new("value", SlotKind::String).required(true)),
        NodeDef::new(
            "builtin.clip_text_encode",
            "CLIP Text Encode",
            "conditioning",
        )
        .with_input_slot(
            InputSlotDef::new("clip", SlotKind::Clip)
                .dynamic(true)
                .required(true),
        )
        .with_input_slot(
            InputSlotDef::new("text", SlotKind::String)
                .dynamic(true)
                .required(true),
        )
        .with_output_slot(
            OutputSlotDef::new("conditioning", SlotKind::Conditioning).required(true),
        ),
        NodeDef::new("builtin.empty_latent_image", "Empty Latent Image", "latent")
            .with_input_slot(InputSlotDef::new("width", SlotKind::Integer).required(true))
            .with_input_slot(InputSlotDef::new("height", SlotKind::Integer).required(true))
            .with_input_slot(InputSlotDef::new("batch_size", SlotKind::Integer).required(true))
            .with_output_slot(OutputSlotDef::new("latent", SlotKind::Latent).required(true)),
        NodeDef::new("builtin.ksampler", "KSampler", "sampling")
            .with_input_slot(
                InputSlotDef::new("model", SlotKind::Model)
                    .dynamic(true)
                    .required(true),
            )
            .with_input_slot(
                InputSlotDef::new("positive", SlotKind::Conditioning)
                    .dynamic(true)
                    .required(true),
            )
            .with_input_slot(
                InputSlotDef::new("negative", SlotKind::Conditioning)
                    .dynamic(true)
                    .required(true),
            )
            .with_input_slot(
                InputSlotDef::new("latent", SlotKind::Latent)
                    .dynamic(true)
                    .required(true),
            )
            .with_input_slot(InputSlotDef::new("seed", SlotKind::Seed).required(true))
            .with_input_slot(InputSlotDef::new("steps", SlotKind::Integer).required(true))
            .with_input_slot(InputSlotDef::new("cfg", SlotKind::Float).required(true))
            .with_input_slot(InputSlotDef::new("sampler", SlotKind::Select).required(true))
            .with_input_slot(InputSlotDef::new("scheduler", SlotKind::Select).required(true))
            .with_input_slot(InputSlotDef::new("denoise", SlotKind::Float).required(true))
            .with_output_slot(OutputSlotDef::new("latent", SlotKind::Latent).required(true)),
        NodeDef::new("builtin.vae_decode", "VAE Decode", "image")
            .with_input_slot(
                InputSlotDef::new("vae", SlotKind::Vae)
                    .dynamic(true)
                    .required(true),
            )
            .with_input_slot(
                InputSlotDef::new("latent", SlotKind::Latent)
                    .dynamic(true)
                    .required(true),
            )
            .with_output_slot(OutputSlotDef::new("image", SlotKind::Image).required(true)),
        NodeDef::new("builtin.save_image", "Save Image", "image")
            .with_effect(NodeEffect::SideEffect)
            .with_input_slot(
                InputSlotDef::new("image", SlotKind::Image)
                    .dynamic(true)
                    .required(true),
            )
            .with_input_slot(InputSlotDef::new("filename_prefix", SlotKind::String).required(true)),
    ])
}

fn model_ref(id: &str) -> ModelRef {
    ModelRef::new(
        ModelId::new(id),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
    )
}
