use super::params::ParamDef;
use super::sockets::SocketDef;

#[derive(Debug, Clone, PartialEq)]
pub struct NodeDef {
    pub type_id: String,
    pub display_name: String,
    pub category: String,
    pub inputs: Vec<SocketDef>,
    pub outputs: Vec<SocketDef>,
    pub parameters: Vec<ParamDef>,
}

impl NodeDef {
    pub fn new(
        type_id: impl Into<String>,
        display_name: impl Into<String>,
        category: impl Into<String>,
    ) -> Self {
        Self {
            type_id: type_id.into(),
            display_name: display_name.into(),
            category: category.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            parameters: Vec::new(),
        }
    }

    pub fn with_input(mut self, input: SocketDef) -> Self {
        self.inputs.push(input);
        self
    }

    pub fn with_output(mut self, output: SocketDef) -> Self {
        self.outputs.push(output);
        self
    }

    pub fn with_parameter(mut self, parameter: ParamDef) -> Self {
        self.parameters.push(parameter);
        self
    }
}
