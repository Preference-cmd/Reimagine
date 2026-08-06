//! Node catalog DTOs.

use reimagine_core::model::{InputSlotDef, OutputSlotDef, SlotKind};
use serde::{Deserialize, Serialize};

/// `GET /nodes` response. This is a host adapter projection of the
/// app-host catalog surface, not an independent node catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCatalogResponse {
    pub nodes: Vec<NodeDefDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeDefDto {
    #[serde(rename = "type")]
    pub type_id: String,
    pub display_name: String,
    pub category: String,
    pub inputs: Vec<SocketSpecDto>,
    pub outputs: Vec<SocketSpecDto>,
    pub parameters: Vec<ParamSpecDto>,
}

impl From<reimagine_core::model::NodeDef> for NodeDefDto {
    fn from(value: reimagine_core::model::NodeDef) -> Self {
        let mut inputs = Vec::new();
        let mut parameters = Vec::new();
        for slot in value.input_slots() {
            if slot.is_dynamic() {
                inputs.push(SocketSpecDto::from(slot));
            } else {
                parameters.push(ParamSpecDto::from(slot));
            }
        }
        Self {
            type_id: value.type_id().to_string(),
            display_name: value.display_name().to_string(),
            category: value.category().to_string(),
            inputs,
            outputs: value
                .output_slots()
                .iter()
                .map(SocketSpecDto::from)
                .collect(),
            parameters,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocketSpecDto {
    pub id: String,
    pub kind: String,
    pub label: String,
}

impl From<&InputSlotDef> for SocketSpecDto {
    fn from(value: &InputSlotDef) -> Self {
        let id = value.id().to_string();
        Self {
            id: id.clone(),
            kind: slot_kind_label(value.kind()),
            label: slot_label(value.ui().label(), &id),
        }
    }
}

impl From<&OutputSlotDef> for SocketSpecDto {
    fn from(value: &OutputSlotDef) -> Self {
        let id = value.id().to_string();
        Self {
            id: id.clone(),
            kind: slot_kind_label(value.kind()),
            label: id,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamSpecDto {
    pub id: String,
    pub label: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    /// Select options from the slot's `options` constraint (comma-separated
    /// constraint value), when the node declares them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<Vec<String>>,
    /// Inclusive numeric bounds from the slot's `min`/`max`/`step`
    /// constraints, when the node declares them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<f64>,
}

impl From<&InputSlotDef> for ParamSpecDto {
    fn from(value: &InputSlotDef) -> Self {
        let id = value.id().to_string();
        Self {
            id: id.clone(),
            label: slot_label(value.ui().label(), &id),
            kind: slot_kind_label(value.kind()),
            default: value
                .default_value()
                .and_then(|v| serde_json::to_value(v).ok()),
            options: constraint_value(value, "options").map(|raw| {
                raw.split(',')
                    .map(str::trim)
                    .filter(|option| !option.is_empty())
                    .map(str::to_owned)
                    .collect()
            }),
            min: parse_constraint_number(value, "min"),
            max: parse_constraint_number(value, "max"),
            step: parse_constraint_number(value, "step"),
        }
    }
}

/// Read the value of a named slot constraint, if present.
fn constraint_value<'a>(slot: &'a InputSlotDef, name: &str) -> Option<&'a str> {
    slot.constraints()
        .iter()
        .find(|constraint| constraint.name() == name)
        .map(|constraint| constraint.value())
}

/// Parse a numeric slot constraint (`min`/`max`/`step`). Unparseable
/// values are treated as absent rather than failing the whole DTO.
fn parse_constraint_number(slot: &InputSlotDef, name: &str) -> Option<f64> {
    constraint_value(slot, name).and_then(|raw| raw.parse::<f64>().ok())
}

fn slot_label(ui_label: Option<&str>, fallback: &str) -> String {
    ui_label.unwrap_or(fallback).to_string()
}

fn slot_kind_label(kind: SlotKind) -> String {
    match kind {
        SlotKind::String => "string",
        SlotKind::Text => "text",
        SlotKind::Integer => "int",
        SlotKind::Float => "float",
        SlotKind::Bool => "bool",
        SlotKind::Seed => "int",
        SlotKind::Select => "select",
        SlotKind::Path => "string",
        SlotKind::ModelRef => "model_ref",
        SlotKind::Model => "model",
        SlotKind::Clip => "clip",
        SlotKind::Vae => "vae",
        SlotKind::Latent => "latent",
        SlotKind::Conditioning => "conditioning",
        SlotKind::Image => "image",
        SlotKind::Artifact => "artifact",
        SlotKind::Null => "null",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_core::model::{InputSlotDef, NodeDef, NodeTypeId, ParamValue, SlotConstraint};

    #[test]
    fn param_spec_dto_carries_options_and_numeric_bounds() {
        let def = NodeDef::new(NodeTypeId::new("test.node"), "Test", "Test")
            .with_input_slot(
                InputSlotDef::new("sampler", SlotKind::Select)
                    .required(true)
                    .with_constraint(SlotConstraint::new(
                        "options",
                        "euler,euler a,dpm++ 2M, ddim",
                    )),
            )
            .with_input_slot(
                InputSlotDef::new("cfg", SlotKind::Float)
                    .required(true)
                    .with_constraint(SlotConstraint::new("min", "1"))
                    .with_constraint(SlotConstraint::new("max", "20"))
                    .with_constraint(SlotConstraint::new("step", "0.1")),
            );
        let dto = NodeDefDto::from(def);

        let sampler = dto
            .parameters
            .iter()
            .find(|param| param.id == "sampler")
            .expect("sampler param");
        assert_eq!(sampler.kind, "select");
        assert_eq!(
            sampler.options.as_deref(),
            Some(
                &["euler", "euler a", "dpm++ 2M", "ddim"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<String>>()[..],
            ),
            "options must be split on commas and trimmed"
        );

        let cfg = dto
            .parameters
            .iter()
            .find(|param| param.id == "cfg")
            .expect("cfg param");
        assert_eq!(cfg.min, Some(1.0));
        assert_eq!(cfg.max, Some(20.0));
        assert_eq!(cfg.step, Some(0.1));

        let json = serde_json::to_value(&dto).expect("DTO serializes");
        let sampler_json = &json["parameters"][0];
        assert_eq!(sampler_json["options"][1], "euler a");
        assert_eq!(json["parameters"][1]["min"], 1.0);
    }

    #[test]
    fn param_spec_dto_omits_missing_constraints() {
        let def = NodeDef::new(NodeTypeId::new("test.plain"), "Plain", "Test").with_input_slot(
            InputSlotDef::new("value", SlotKind::String)
                .with_default_value(ParamValue::String("hello".to_owned())),
        );
        let plain = NodeDefDto::from(def).parameters.into_iter().next().unwrap();
        assert_eq!(plain.id, "value");
        assert_eq!(plain.options, None);
        assert_eq!(plain.min, None);
        assert_eq!(plain.max, None);
        assert_eq!(plain.step, None);

        let json = serde_json::to_value(&plain).expect("DTO serializes");
        assert!(
            json.get("options").is_none(),
            "absent options must not serialize"
        );
        assert!(json.get("min").is_none(), "absent min must not serialize");
    }
}
