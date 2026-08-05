use reimagine_core::model::{
    InputSlotDef, OutputSlotDef, ParamValue, SlotEditConstraint, SlotKind, SlotType,
};

#[test]
fn slot_type_maps_domain_for_all_kinds() {
    let cases = [
        (SlotKind::String, SlotType::Primitive),
        (SlotKind::Text, SlotType::Primitive),
        (SlotKind::Integer, SlotType::Primitive),
        (SlotKind::Float, SlotType::Primitive),
        (SlotKind::Bool, SlotType::Primitive),
        (SlotKind::Seed, SlotType::Primitive),
        (SlotKind::Select, SlotType::Primitive),
        (SlotKind::Path, SlotType::Primitive),
        (SlotKind::Null, SlotType::Primitive),
        (SlotKind::Latent, SlotType::Tensor),
        (SlotKind::ModelRef, SlotType::ModelHandle),
        (SlotKind::Model, SlotType::ModelHandle),
        (SlotKind::Clip, SlotType::ModelHandle),
        (SlotKind::Vae, SlotType::ModelHandle),
        (SlotKind::Conditioning, SlotType::Conditioning),
        (SlotKind::Image, SlotType::Artifact),
        (SlotKind::Artifact, SlotType::Artifact),
    ];
    for (kind, expected) in cases {
        assert_eq!(kind.slot_type(), expected, "slot_type({kind:?})");
    }
}

#[test]
fn edit_constraint_maps_ui_hints() {
    assert_eq!(
        SlotKind::Text.edit_constraint(),
        Some(SlotEditConstraint::Multiline)
    );
    assert_eq!(
        SlotKind::Select.edit_constraint(),
        Some(SlotEditConstraint::Select)
    );
    assert_eq!(
        SlotKind::Path.edit_constraint(),
        Some(SlotEditConstraint::FilePath)
    );
    for kind in [
        SlotKind::String,
        SlotKind::Integer,
        SlotKind::Float,
        SlotKind::Bool,
        SlotKind::Seed,
        SlotKind::ModelRef,
        SlotKind::Model,
        SlotKind::Clip,
        SlotKind::Vae,
        SlotKind::Latent,
        SlotKind::Conditioning,
        SlotKind::Image,
        SlotKind::Artifact,
        SlotKind::Null,
    ] {
        assert_eq!(kind.edit_constraint(), None, "edit_constraint({kind:?})");
    }
}

#[test]
fn from_slot_type_and_constraint_selects_nearest_kind() {
    let cases = [
        (
            (SlotType::Primitive, SlotEditConstraint::None),
            SlotKind::String,
        ),
        (
            (SlotType::Primitive, SlotEditConstraint::Optional),
            SlotKind::String,
        ),
        (
            (SlotType::Primitive, SlotEditConstraint::Multiline),
            SlotKind::Text,
        ),
        (
            (SlotType::Primitive, SlotEditConstraint::Select),
            SlotKind::Select,
        ),
        (
            (SlotType::Primitive, SlotEditConstraint::FilePath),
            SlotKind::Path,
        ),
        (
            (SlotType::Primitive, SlotEditConstraint::Range),
            SlotKind::Integer,
        ),
        (
            (SlotType::Tensor, SlotEditConstraint::None),
            SlotKind::Latent,
        ),
        (
            (SlotType::ModelHandle, SlotEditConstraint::None),
            SlotKind::Model,
        ),
        (
            (SlotType::Conditioning, SlotEditConstraint::None),
            SlotKind::Conditioning,
        ),
        (
            (SlotType::Artifact, SlotEditConstraint::None),
            SlotKind::Artifact,
        ),
    ];
    for (from, expected) in cases {
        assert_eq!(SlotKind::from(from), expected, "from {from:?}");
    }
}

#[test]
fn slot_kind_serialization_shape_is_unchanged() {
    assert_eq!(
        serde_json::to_string(&SlotKind::String).unwrap(),
        "\"String\""
    );
    assert_eq!(serde_json::to_string(&SlotKind::Text).unwrap(), "\"Text\"");
    assert_eq!(
        serde_json::to_string(&SlotKind::ModelRef).unwrap(),
        "\"ModelRef\""
    );
    assert_eq!(
        serde_json::to_string(&SlotKind::Conditioning).unwrap(),
        "\"Conditioning\""
    );
    assert_eq!(
        serde_json::to_string(&SlotKind::Artifact).unwrap(),
        "\"Artifact\""
    );

    let input: InputSlotDef = serde_json::from_value(serde_json::json!({
        "id": "value",
        "kind": "String",
        "dynamic": false,
        "required": true,
        "default_value": null,
        "constraints": [],
        "ui": {}
    }))
    .expect("deserialize input slot def");
    assert_eq!(input.kind(), SlotKind::String);
    assert!(input.is_required());
    assert!(input.constraints().is_empty());

    let output: OutputSlotDef = serde_json::from_value(serde_json::json!({
        "id": "latent",
        "kind": "Latent",
        "required": true
    }))
    .expect("deserialize output slot def");
    assert_eq!(output.kind(), SlotKind::Latent);
}

#[test]
fn input_slot_def_serde_round_trip_preserves_kind() {
    let slot = InputSlotDef::new("value", SlotKind::String)
        .dynamic(true)
        .required(true)
        .with_default_value(ParamValue::String("prompt".to_owned()));

    let json = serde_json::to_value(&slot).unwrap();
    assert_eq!(json["kind"], "String");

    let back: InputSlotDef = serde_json::from_value(json).unwrap();
    assert_eq!(back, slot);
}
