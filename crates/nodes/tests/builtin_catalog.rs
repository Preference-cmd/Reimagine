use reimagine_core::model::{NodeCatalog, NodeEffect, NodeTypeId, ParamValue, SlotKind};
use reimagine_core::validation::validate_structure;
use reimagine_core::workflow::{Workflow, WorkflowNode};
use reimagine_nodes::{
    BUILTIN_CHECKPOINT_LOADER, BUILTIN_CLIP_TEXT_ENCODE, BUILTIN_EMPTY_LATENT_IMAGE,
    BUILTIN_KSAMPLER, BUILTIN_LOAD_IMAGE, BUILTIN_PREVIEW_IMAGE, BUILTIN_SAVE_IMAGE,
    BUILTIN_STRING, BUILTIN_VAE_DECODE, BUILTIN_VAE_ENCODE, BuiltinNodeCatalog, comfy_aliases,
};

#[test]
fn builtin_catalog_contains_v1_sdxl_node_defs() {
    let catalog = BuiltinNodeCatalog::v1();

    for type_id in [
        BUILTIN_STRING,
        BUILTIN_LOAD_IMAGE,
        BUILTIN_CHECKPOINT_LOADER,
        BUILTIN_CLIP_TEXT_ENCODE,
        BUILTIN_EMPTY_LATENT_IMAGE,
        BUILTIN_VAE_ENCODE,
        BUILTIN_KSAMPLER,
        BUILTIN_VAE_DECODE,
        BUILTIN_SAVE_IMAGE,
        BUILTIN_PREVIEW_IMAGE,
    ] {
        assert!(
            catalog.get(&NodeTypeId::new(type_id)).is_some(),
            "missing built-in node {type_id}"
        );
    }

    let checkpoint = catalog
        .get(&NodeTypeId::new(BUILTIN_CHECKPOINT_LOADER))
        .expect("checkpoint loader exists");
    assert_eq!(checkpoint.effect(), NodeEffect::Pure);
    assert_eq!(
        checkpoint.input_slot(&"checkpoint".into()).unwrap().kind(),
        SlotKind::ModelRef
    );
    assert!(
        !checkpoint
            .input_slot(&"checkpoint".into())
            .unwrap()
            .is_dynamic()
    );
    assert_eq!(
        checkpoint.output_slot(&"model".into()).unwrap().kind(),
        SlotKind::Model
    );
    assert_eq!(
        checkpoint.output_slot(&"clip".into()).unwrap().kind(),
        SlotKind::Clip
    );
    assert_eq!(
        checkpoint.output_slot(&"vae".into()).unwrap().kind(),
        SlotKind::Vae
    );

    let sampler = catalog
        .get(&NodeTypeId::new(BUILTIN_KSAMPLER))
        .expect("ksampler exists");
    assert!(sampler.input_slot(&"model".into()).unwrap().is_dynamic());
    assert!(sampler.input_slot(&"positive".into()).unwrap().is_dynamic());
    assert!(!sampler.input_slot(&"steps".into()).unwrap().is_dynamic());
    assert_eq!(
        sampler.input_slot(&"seed".into()).unwrap().kind(),
        SlotKind::Seed
    );

    let save = catalog
        .get(&NodeTypeId::new(BUILTIN_SAVE_IMAGE))
        .expect("save image exists");
    assert_eq!(save.effect(), NodeEffect::SideEffect);
    assert!(save.output_slots().is_empty());

    let preview = catalog
        .get(&NodeTypeId::new(BUILTIN_PREVIEW_IMAGE))
        .expect("preview image exists");
    assert_eq!(preview.effect(), NodeEffect::SideEffect);
    assert!(preview.output_slots().is_empty());

    let load_image = catalog
        .get(&NodeTypeId::new(BUILTIN_LOAD_IMAGE))
        .expect("load image exists");
    assert_eq!(load_image.effect(), NodeEffect::Pure);
    assert_eq!(
        load_image
            .input_slot(&"image".into())
            .expect("image input slot exists")
            .kind(),
        SlotKind::Path
    );
    assert_eq!(
        load_image
            .output_slot(&"image".into())
            .expect("image output slot exists")
            .kind(),
        SlotKind::Image
    );

    let vae_encode = catalog
        .get(&NodeTypeId::new(BUILTIN_VAE_ENCODE))
        .expect("vae encode exists");
    assert_eq!(vae_encode.effect(), NodeEffect::Pure);
    assert!(vae_encode.input_slot(&"vae".into()).unwrap().is_dynamic());
    assert!(vae_encode.input_slot(&"image".into()).unwrap().is_dynamic());
    assert_eq!(
        vae_encode.output_slot(&"latent".into()).unwrap().kind(),
        SlotKind::Latent
    );
}

#[test]
fn sdxl_base_workflow_example_validates_against_builtin_catalog() {
    let workflow: Workflow = serde_json::from_str(include_str!(
        "../../../docs/architecture/examples/sdxl-base-workflow.json"
    ))
    .expect("parse sdxl workflow example");

    let report = validate_structure(&workflow, &BuiltinNodeCatalog::v1());

    assert!(
        report.diagnostics().is_empty(),
        "unexpected diagnostics: {:?}",
        report.diagnostics()
    );
}

#[test]
fn dynamic_slots_still_reject_static_params_with_builtin_catalog() {
    let workflow = Workflow::new("workflow_dynamic_param", 1.into()).with_node(
        WorkflowNode::new("node_encode", BUILTIN_CLIP_TEXT_ENCODE)
            .with_param("clip", ParamValue::Null)
            .with_param("text", ParamValue::String("prompt".to_owned())),
    );

    let report = validate_structure(&workflow, &BuiltinNodeCatalog::v1());
    let codes: Vec<_> = report
        .diagnostics()
        .iter()
        .map(|diagnostic| diagnostic.code().as_str())
        .collect();

    assert!(codes.contains(&"CORE/WORKFLOW_PARAM_ON_DYNAMIC_SLOT"));
}

#[test]
fn comfy_aliases_map_import_names_to_canonical_type_ids() {
    let aliases = comfy_aliases();

    assert_eq!(
        aliases.get("CheckpointLoaderSimple"),
        Some(&BUILTIN_CHECKPOINT_LOADER)
    );
    assert_eq!(
        aliases.get("CLIPTextEncode"),
        Some(&BUILTIN_CLIP_TEXT_ENCODE)
    );
    assert_eq!(
        aliases.get("EmptyLatentImage"),
        Some(&BUILTIN_EMPTY_LATENT_IMAGE)
    );
    assert_eq!(aliases.get("KSampler"), Some(&BUILTIN_KSAMPLER));
    assert_eq!(aliases.get("VAEDecode"), Some(&BUILTIN_VAE_DECODE));
    assert_eq!(aliases.get("SaveImage"), Some(&BUILTIN_SAVE_IMAGE));
}
