use super::ids::SlotId;
use super::values::ParamValue;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum SlotKind {
    String,
    Text,
    Integer,
    Float,
    Bool,
    Seed,
    Select,
    Path,
    ModelRef,
    Model,
    Clip,
    Vae,
    Latent,
    Conditioning,
    Image,
    Artifact,
    Null,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SlotType {
    Primitive,
    Tensor,
    ModelHandle,
    Conditioning,
    Artifact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SlotEditConstraint {
    None,
    Range,
    Select,
    Multiline,
    FilePath,
    Optional,
}

impl SlotKind {
    pub fn slot_type(self) -> SlotType {
        match self {
            SlotKind::String
            | SlotKind::Text
            | SlotKind::Integer
            | SlotKind::Float
            | SlotKind::Bool
            | SlotKind::Seed
            | SlotKind::Select
            | SlotKind::Path
            | SlotKind::Null => SlotType::Primitive,
            SlotKind::Latent => SlotType::Tensor,
            SlotKind::ModelRef | SlotKind::Model | SlotKind::Clip | SlotKind::Vae => {
                SlotType::ModelHandle
            }
            SlotKind::Conditioning => SlotType::Conditioning,
            SlotKind::Image | SlotKind::Artifact => SlotType::Artifact,
        }
    }

    pub fn edit_constraint(self) -> Option<SlotEditConstraint> {
        match self {
            SlotKind::Text => Some(SlotEditConstraint::Multiline),
            SlotKind::Select => Some(SlotEditConstraint::Select),
            SlotKind::Path => Some(SlotEditConstraint::FilePath),
            _ => None,
        }
    }
}

impl From<(SlotType, SlotEditConstraint)> for SlotKind {
    fn from((slot_type, constraint): (SlotType, SlotEditConstraint)) -> Self {
        match slot_type {
            SlotType::Primitive => match constraint {
                SlotEditConstraint::Multiline => SlotKind::Text,
                SlotEditConstraint::Select => SlotKind::Select,
                SlotEditConstraint::FilePath => SlotKind::Path,
                SlotEditConstraint::Range => SlotKind::Integer,
                SlotEditConstraint::None | SlotEditConstraint::Optional => SlotKind::String,
            },
            SlotType::Tensor => SlotKind::Latent,
            SlotType::ModelHandle => SlotKind::Model,
            SlotType::Conditioning => SlotKind::Conditioning,
            SlotType::Artifact => SlotKind::Artifact,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SlotConstraint {
    name: String,
    value: String,
}

impl SlotConstraint {
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn value(&self) -> &str {
        self.value.as_str()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct SlotUi {
    label: Option<String>,
    description: Option<String>,
}

impl SlotUi {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InputSlotDef {
    id: SlotId,
    kind: SlotKind,
    dynamic: bool,
    required: bool,
    default_value: Option<ParamValue>,
    constraints: Vec<SlotConstraint>,
    ui: SlotUi,
}

impl InputSlotDef {
    pub fn new(id: impl Into<SlotId>, kind: SlotKind) -> Self {
        Self {
            id: id.into(),
            kind,
            dynamic: false,
            required: false,
            default_value: None,
            constraints: Vec::new(),
            ui: SlotUi::default(),
        }
    }

    pub fn dynamic(mut self, dynamic: bool) -> Self {
        self.dynamic = dynamic;
        self
    }

    pub fn required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }

    pub fn with_default_value(mut self, value: ParamValue) -> Self {
        self.default_value = Some(value);
        self
    }

    pub fn with_constraint(mut self, constraint: SlotConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    pub fn with_ui(mut self, ui: SlotUi) -> Self {
        self.ui = ui;
        self
    }

    pub fn id(&self) -> &SlotId {
        &self.id
    }

    pub fn kind(&self) -> SlotKind {
        self.kind
    }

    pub fn is_dynamic(&self) -> bool {
        self.dynamic
    }

    pub fn is_required(&self) -> bool {
        self.required
    }

    pub fn default_value(&self) -> Option<&ParamValue> {
        self.default_value.as_ref()
    }

    pub fn constraints(&self) -> &[SlotConstraint] {
        &self.constraints
    }

    pub fn ui(&self) -> &SlotUi {
        &self.ui
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OutputSlotDef {
    id: SlotId,
    kind: SlotKind,
    required: bool,
}

impl OutputSlotDef {
    pub fn new(id: impl Into<SlotId>, kind: SlotKind) -> Self {
        Self {
            id: id.into(),
            kind,
            required: false,
        }
    }

    pub fn required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }

    pub fn id(&self) -> &SlotId {
        &self.id
    }

    pub fn kind(&self) -> SlotKind {
        self.kind
    }

    pub fn is_required(&self) -> bool {
        self.required
    }
}
