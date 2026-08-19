use reimagine_core::model::{NodeId, ProjectId, WorkflowId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanvasContext {
    pub project_id: ProjectId,
    #[serde(default)]
    pub focus: CanvasFocus,
    #[serde(default)]
    pub open_workflows: Vec<WorkflowId>,
    #[serde(default)]
    pub selection: CanvasSelection,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CanvasFocus {
    #[default]
    Board,
    WorkflowFrame(WorkflowId),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CanvasSelection {
    #[default]
    None,
    Node {
        workflow_id: WorkflowId,
        node_id: NodeId,
    },
    Workflow {
        workflow_id: WorkflowId,
    },
}

/// Converts ephemeral UI canvas state into a clearly delimited model hint.
pub struct ProjectContextAssembler;

impl ProjectContextAssembler {
    pub fn assemble(context: &CanvasContext) -> String {
        let focus = match &context.focus {
            CanvasFocus::Board => "board".to_owned(),
            CanvasFocus::WorkflowFrame(id) => format!("workflow_frame ({id})"),
        };
        let workflows = if context.open_workflows.is_empty() {
            "none".to_owned()
        } else {
            context
                .open_workflows
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        };
        let selection = match &context.selection {
            CanvasSelection::None => "none (the user has not selected a target; require an explicit workflow_id or target id)".to_owned(),
            CanvasSelection::Node { workflow_id, node_id } => format!("node {node_id} in workflow {workflow_id}"),
            CanvasSelection::Workflow { workflow_id } => format!("workflow {workflow_id}"),
        };
        format!(
            "Canvas context:
project_id: {}
focus: {focus}
open_workflows: {workflows}
selection: {selection}
---",
            context.project_id
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn serde_defaults_and_focus() {
        let value = serde_json::json!({"project_id":"p"});
        let context: CanvasContext = serde_json::from_value(value).unwrap();
        assert_eq!(context.focus, CanvasFocus::Board);
        assert_eq!(context.selection, CanvasSelection::None);
        let wf = WorkflowId::new("wf");
        let context = CanvasContext {
            project_id: ProjectId::new("p"),
            focus: CanvasFocus::WorkflowFrame(wf.clone()),
            open_workflows: vec![wf],
            selection: CanvasSelection::None,
        };
        let round: CanvasContext =
            serde_json::from_value(serde_json::to_value(&context).unwrap()).unwrap();
        assert_eq!(round, context);
    }
    #[test]
    fn assembler_does_not_fabricate_selection() {
        let context = CanvasContext {
            project_id: ProjectId::new("p"),
            focus: CanvasFocus::Board,
            open_workflows: vec![],
            selection: CanvasSelection::None,
        };
        let text = ProjectContextAssembler::assemble(&context);
        assert!(text.contains("selection: none"));
        assert!(!text.contains("node "));
    }
}
