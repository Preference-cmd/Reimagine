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
        }
    }
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
