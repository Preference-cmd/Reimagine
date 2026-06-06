use super::values::NodeValue;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParamKind {
    Int,
    Float,
    String,
    Select,
    Bool,
    Text,
    Image,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParamDef {
    pub id: String,
    pub label: String,
    pub kind: ParamKind,
    pub default: Option<NodeValue>,
    pub options: Vec<String>,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl ParamDef {
    pub fn new(id: impl Into<String>, label: impl Into<String>, kind: ParamKind) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            kind,
            default: None,
            options: Vec::new(),
            min: None,
            max: None,
        }
    }

    pub fn with_default(mut self, default: NodeValue) -> Self {
        self.default = Some(default);
        self
    }

    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.min = Some(min);
        self.max = Some(max);
        self
    }

    pub fn with_options(mut self, options: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.options = options.into_iter().map(Into::into).collect();
        self
    }
}
