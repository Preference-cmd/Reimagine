use std::collections::BTreeMap;

use crate::model::NodeId;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Position {
    x: f64,
    y: f64,
}

impl Position {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn x(&self) -> f64 {
        self.x
    }

    pub fn y(&self) -> f64 {
        self.y
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Viewport {
    x: f64,
    y: f64,
    zoom: f64,
}

impl Viewport {
    pub fn new(x: f64, y: f64, zoom: f64) -> Self {
        Self { x, y, zoom }
    }

    pub fn x(&self) -> f64 {
        self.x
    }

    pub fn y(&self) -> f64 {
        self.y
    }

    pub fn zoom(&self) -> f64 {
        self.zoom
    }
}

#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct WorkflowLayout {
    nodes: BTreeMap<NodeId, Position>,
    #[serde(skip_serializing_if = "Option::is_none")]
    viewport: Option<Viewport>,
}

impl WorkflowLayout {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_node(mut self, node_id: NodeId, position: Position) -> Self {
        self.nodes.insert(node_id, position);
        self
    }

    pub fn with_viewport(mut self, viewport: Viewport) -> Self {
        self.viewport = Some(viewport);
        self
    }

    pub fn nodes(&self) -> &BTreeMap<NodeId, Position> {
        &self.nodes
    }

    pub fn viewport(&self) -> Option<&Viewport> {
        self.viewport.as_ref()
    }

    pub(crate) fn set_node_position(&mut self, node_id: NodeId, position: Position) {
        self.nodes.insert(node_id, position);
    }

    pub(crate) fn remove_node(&mut self, node_id: &NodeId) -> Option<Position> {
        self.nodes.remove(node_id)
    }
}
