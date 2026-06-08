use std::collections::HashMap;

use reimagine_core::model::{
    InputSlotDef, NodeCatalog, NodeDef, NodeEffect, NodeTypeId, OutputSlotDef, SlotKind,
};
use reimagine_core::validation::validate_structure;
use reimagine_core::workflow::Workflow;

#[test]
fn structural_validation_accepts_sdxl_example_with_matching_catalog() {
    let workflow: Workflow = serde_json::from_str(include_str!(
        "../../../docs/architecture/examples/sdxl-base-workflow.json"
    ))
    .expect("parse sdxl workflow example");
    let catalog = sdxl_catalog();

    let report = validate_structure(&workflow, &catalog);

    assert!(report.diagnostics().is_empty());
}

#[test]
fn structural_validation_reports_representative_graph_errors() {
    let workflow: Workflow = serde_json::from_str(
        r#"
        {
          "schema_version": "reimagine.workflow.v1",
          "id": "workflow_invalid",
          "version": 1,
          "metadata": {},
          "interface": { "inputs": [], "outputs": [] },
          "nodes": [
            {
              "id": "node_a",
              "type_id": "builtin.source",
              "params": {}
            },
            {
              "id": "node_b",
              "type_id": "builtin.consumer",
              "params": {
                "dynamic_value": { "type": "string", "value": "not allowed" },
                "count": { "type": "string", "value": "wrong kind" }
              }
            },
            {
              "id": "node_unknown",
              "type_id": "builtin.unknown",
              "params": {}
            }
          ],
          "edges": [
            {
              "id": "edge_one",
              "from": { "node": "node_a", "slot": "value" },
              "to": { "node": "node_b", "slot": "dynamic_value" }
            },
            {
              "id": "edge_two",
              "from": { "node": "node_a", "slot": "value" },
              "to": { "node": "node_b", "slot": "dynamic_value" }
            },
            {
              "id": "edge_bad_slot",
              "from": { "node": "node_a", "slot": "missing" },
              "to": { "node": "node_b", "slot": "count" }
            }
          ],
          "layout": {
            "nodes": {
              "node_a": { "x": 0, "y": 0 },
              "node_missing": { "x": 100, "y": 0 }
            }
          }
        }
        "#,
    )
    .expect("parse invalid workflow");
    let catalog = representative_catalog();

    let report = validate_structure(&workflow, &catalog);
    let codes: Vec<&str> = report
        .diagnostics()
        .iter()
        .map(|diagnostic| diagnostic.code().as_str())
        .collect();

    assert!(codes.contains(&"CORE/WORKFLOW_NODE_TYPE_UNKNOWN"));
    assert!(codes.contains(&"CORE/WORKFLOW_PARAM_ON_DYNAMIC_SLOT"));
    assert!(codes.contains(&"CORE/WORKFLOW_PARAM_KIND_MISMATCH"));
    assert!(codes.contains(&"CORE/WORKFLOW_INPUT_EDGE_DUPLICATE"));
    assert!(codes.contains(&"CORE/WORKFLOW_ENDPOINT_SLOT_MISSING"));
    assert!(codes.contains(&"CORE/WORKFLOW_LAYOUT_NODE_MISSING"));
}

#[test]
fn structural_validation_reports_remaining_v1_structural_errors() {
    let workflow: Workflow = serde_json::from_str(
        r#"
        {
          "schema_version": "reimagine.workflow.v1",
          "id": "workflow_more_invalid",
          "version": 1,
          "metadata": {},
          "interface": { "inputs": [], "outputs": [] },
          "nodes": [
            {
              "id": "node_a",
              "type_id": "builtin.source",
              "params": {}
            },
            {
              "id": "node_b",
              "type_id": "builtin.consumer",
              "params": {
                "missing_param": { "type": "string", "value": "extra" }
              }
            },
            {
              "id": "node_a",
              "type_id": "builtin.source",
              "params": {}
            }
          ],
          "edges": [
            {
              "id": "edge_duplicate",
              "from": { "node": "node_a", "slot": "value" },
              "to": { "node": "node_b", "slot": "count" }
            },
            {
              "id": "edge_duplicate",
              "from": { "node": "node_missing", "slot": "value" },
              "to": { "node": "node_b", "slot": "count" }
            },
            {
              "id": "edge_bad_direction_from",
              "from": { "workflow_output": "image" },
              "to": { "node": "node_b", "slot": "count" }
            },
            {
              "id": "edge_bad_direction_to",
              "from": { "node": "node_a", "slot": "value" },
              "to": { "workflow_input": "prompt" }
            }
          ],
          "layout": { "nodes": {} }
        }
        "#,
    )
    .expect("parse invalid workflow");
    let catalog = representative_catalog();

    let report = validate_structure(&workflow, &catalog);
    let codes: Vec<&str> = report
        .diagnostics()
        .iter()
        .map(|diagnostic| diagnostic.code().as_str())
        .collect();

    assert!(codes.contains(&"CORE/WORKFLOW_NODE_ID_DUPLICATE"));
    assert!(codes.contains(&"CORE/WORKFLOW_EDGE_ID_DUPLICATE"));
    assert!(codes.contains(&"CORE/WORKFLOW_PARAM_SLOT_MISSING"));
    assert!(codes.contains(&"CORE/WORKFLOW_SLOT_KIND_MISMATCH"));
    assert!(codes.contains(&"CORE/WORKFLOW_ENDPOINT_NODE_MISSING"));
    assert!(codes.contains(&"CORE/WORKFLOW_ENDPOINT_DIRECTION_INVALID"));
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

fn representative_catalog() -> TestCatalog {
    TestCatalog::new(vec![
        NodeDef::new("builtin.source", "Source", "test")
            .with_output_slot(OutputSlotDef::new("value", SlotKind::String).required(true)),
        NodeDef::new("builtin.consumer", "Consumer", "test")
            .with_input_slot(
                InputSlotDef::new("dynamic_value", SlotKind::String)
                    .dynamic(true)
                    .required(true),
            )
            .with_input_slot(InputSlotDef::new("count", SlotKind::Integer).required(true)),
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
