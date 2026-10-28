//! Dynamic workflow documents: declarative step graphs and static validation.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use crate::dynamic_workflow::binding::{Binding, BindingError, collect_bindings};
use crate::dynamic_workflow::run::RunTarget;
use crate::model::{DiagnosticId, DynamicWorkflowId, ProjectId};

/// A saved dynamic workflow document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DynamicWorkflowDocument {
    pub id: DynamicWorkflowId,
    pub project_id: ProjectId,
    pub name: String,
    pub version: u64,
    pub workflow: WorkflowSpec,
}

impl DynamicWorkflowDocument {
    pub fn new(
        id: DynamicWorkflowId,
        project_id: ProjectId,
        name: impl Into<String>,
        workflow: WorkflowSpec,
    ) -> Self {
        Self {
            id,
            project_id,
            name: name.into(),
            version: 1,
            workflow,
        }
    }
}

/// The declarative graph of one dynamic workflow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkflowSpec {
    /// Named input schema fragments. A fragment with `"format": "asset"` is
    /// the input a collection target fills with its slot's asset.
    #[serde(default)]
    pub inputs: BTreeMap<String, Value>,
    /// JSON Schema every run parameter value must satisfy.
    #[serde(default = "default_params_schema")]
    pub params: Value,
    pub steps: Vec<WorkflowStep>,
    pub output: Value,
}

/// One operation call in a dynamic workflow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkflowStep {
    pub id: String,
    pub op: String,
    pub input: Value,
}

fn default_params_schema() -> Value {
    serde_json::json!({ "type": "object" })
}

fn invalid(code: &str, subject: &str, message: impl Into<String>) -> Diagnostic {
    let subject = if subject.is_empty() {
        "workflow"
    } else {
        subject
    };
    Diagnostic::new(
        DiagnosticId::new(format!("dynamic-workflow:{code}:{subject}")),
        DiagnosticCode::new(format!("DYNAMIC_WORKFLOW/{code}")),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("dynamic_workflow"),
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new("dynamic_workflow")).with_id(subject),
    )
}

fn binding_error(error: &BindingError) -> (&'static str, String) {
    let code = match error {
        BindingError::UnknownRoot(_) | BindingError::Unresolved(_) => "UNKNOWN_REFERENCE",
        _ => "INVALID_BINDING",
    };
    (code, error.to_string())
}

impl WorkflowSpec {
    /// Validate the document's shape, its references and its step graph.
    ///
    /// Returns every error found, not just the first.
    pub fn validate(&self) -> Result<(), Vec<Diagnostic>> {
        let mut errors = Vec::new();

        for (name, schema) in &self.inputs {
            if name.is_empty() || name.contains('.') {
                errors.push(invalid(
                    "INVALID_INPUT_NAME",
                    name,
                    format!("input name `{name}` must be non-empty and contain no `.`"),
                ));
            }
            if !schema.is_object() {
                errors.push(invalid(
                    "INVALID_INPUT_SCHEMA",
                    name,
                    format!("input `{name}` must be a JSON Schema object"),
                ));
            }
        }

        match &self.params {
            Value::Object(_) => {
                if let Some(kind) = self.params.get("type").and_then(Value::as_str)
                    && kind != "object"
                {
                    errors.push(invalid(
                        "INVALID_PARAMS_SCHEMA",
                        "params",
                        format!("params schema type must be `object`, got `{kind}`"),
                    ));
                }
            }
            _ => errors.push(invalid(
                "INVALID_PARAMS_SCHEMA",
                "params",
                "params must be an object JSON Schema",
            )),
        }

        let mut seen = BTreeSet::new();
        for step in &self.steps {
            if step.id.is_empty() || step.id.contains('.') {
                errors.push(invalid(
                    "INVALID_STEP_ID",
                    &step.id,
                    format!("step id `{}` must be non-empty and contain no `.`", step.id),
                ));
            }
            if !seen.insert(step.id.as_str()) {
                errors.push(invalid(
                    "DUPLICATE_STEP_ID",
                    &step.id,
                    format!("step id `{}` is declared more than once", step.id),
                ));
            }
            if step.op.is_empty() {
                errors.push(invalid(
                    "INVALID_STEP_OP",
                    &step.id,
                    format!("step `{}` has an empty op", step.id),
                ));
            }

            match collect_bindings(&step.input) {
                Ok(bindings) => {
                    for binding in bindings {
                        if let Some(error) = self.check_binding(&binding) {
                            errors.push(error);
                        }
                    }
                }
                Err(error) => {
                    let (code, message) = binding_error(&error);
                    errors.push(invalid(code, &step.id, message));
                }
            }
        }

        match Binding::parse(&self.output) {
            Ok(binding) => {
                if let Some(error) = self.check_binding(&binding) {
                    errors.push(error);
                }
            }
            Err(error) => {
                let (code, message) = binding_error(&error);
                errors.push(invalid(code, "output", message));
            }
        }

        if let Err(order_errors) = self.ordered_steps_inner() {
            // Binding syntax errors were already reported above; only the
            // ordering pass adds a cycle diagnostic.
            errors.extend(
                order_errors
                    .into_iter()
                    .filter(|error| error.code().as_str() == "DYNAMIC_WORKFLOW/CYCLIC_STEPS"),
            );
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn check_binding(&self, binding: &Binding) -> Option<Diagnostic> {
        match binding.root() {
            "inputs" => {
                if self.inputs.contains_key(binding.name()) {
                    None
                } else {
                    Some(invalid(
                        "UNKNOWN_REFERENCE",
                        binding.name(),
                        format!(
                            "binding `{binding}` references undeclared input `{}`",
                            binding.name()
                        ),
                    ))
                }
            }
            "params" => {
                let declared = self
                    .params
                    .get("properties")
                    .and_then(Value::as_object)
                    .map(|properties| properties.contains_key(binding.name()));
                match declared {
                    Some(false) => Some(invalid(
                        "UNKNOWN_REFERENCE",
                        binding.name(),
                        format!(
                            "binding `{binding}` references undeclared param `{}`",
                            binding.name()
                        ),
                    )),
                    _ => None,
                }
            }
            "steps" => {
                if self.steps.iter().any(|step| step.id == binding.name()) {
                    None
                } else {
                    Some(invalid(
                        "UNKNOWN_REFERENCE",
                        binding.name(),
                        format!(
                            "binding `{binding}` references unknown step `{}`",
                            binding.name()
                        ),
                    ))
                }
            }
            _ => Some(invalid(
                "UNKNOWN_REFERENCE",
                binding.from(),
                format!("binding `{binding}` has an unknown root"),
            )),
        }
    }

    /// Steps in execution order: dependencies first, document order within a
    /// dependency layer.
    pub fn ordered_steps(&self) -> Result<Vec<&WorkflowStep>, Vec<Diagnostic>> {
        self.ordered_steps_inner()
    }

    fn ordered_steps_inner(&self) -> Result<Vec<&WorkflowStep>, Vec<Diagnostic>> {
        let mut dependencies: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        for step in &self.steps {
            let mut deps = BTreeSet::new();
            let bindings = collect_bindings(&step.input).map_err(|error| {
                let (code, message) = binding_error(&error);
                vec![invalid(code, &step.id, message)]
            })?;
            for binding in bindings {
                if binding.root() == "steps"
                    && let Some(dependency) = self
                        .steps
                        .iter()
                        .find(|candidate| candidate.id == binding.name())
                {
                    deps.insert(dependency.id.as_str());
                }
            }
            dependencies.insert(step.id.as_str(), deps);
        }

        let mut pending: Vec<&WorkflowStep> = self.steps.iter().collect();
        let mut emitted: BTreeSet<&str> = BTreeSet::new();
        let mut ordered = Vec::with_capacity(self.steps.len());
        while !pending.is_empty() {
            // Freeze readiness per layer so a step that becomes ready because
            // of this pass waits for the next one: document order wins within
            // a dependency layer.
            let ready: Vec<&WorkflowStep> = pending
                .iter()
                .copied()
                .filter(|step| {
                    dependencies
                        .get(step.id.as_str())
                        .is_some_and(|deps| deps.iter().all(|dep| emitted.contains(dep)))
                })
                .collect();
            if ready.is_empty() {
                return Err(vec![invalid(
                    "CYCLIC_STEPS",
                    "steps",
                    "step references are cyclic",
                )]);
            }
            let ready_ids: BTreeSet<&str> = ready.iter().map(|step| step.id.as_str()).collect();
            for step in &ready {
                emitted.insert(step.id.as_str());
                ordered.push(*step);
            }
            pending.retain(|step| !ready_ids.contains(step.id.as_str()));
        }
        Ok(ordered)
    }

    /// Names of inputs whose schema fragment is an asset (`"format": "asset"`).
    pub fn asset_inputs(&self) -> Vec<&str> {
        self.inputs
            .iter()
            .filter(|(_, schema)| schema.get("format").and_then(Value::as_str) == Some("asset"))
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Check that the input shape matches the target kind.
    pub fn validate_target(&self, target: &RunTarget) -> Result<(), Vec<Diagnostic>> {
        let asset_inputs = self.asset_inputs();
        match target {
            RunTarget::Collection { .. } if asset_inputs.len() != 1 => Err(vec![invalid(
                "ASSET_INPUT_COUNT",
                "target",
                format!(
                    "a collection target requires exactly one asset input, found {} ({})",
                    asset_inputs.len(),
                    asset_inputs.join(", ")
                ),
            )]),
            RunTarget::Groups { .. } if !asset_inputs.is_empty() => Err(vec![invalid(
                "ASSET_INPUT_COUNT",
                "target",
                format!(
                    "a parameter-group target requires no asset inputs, found {} ({})",
                    asset_inputs.len(),
                    asset_inputs.join(", ")
                ),
            )]),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn workflow(steps: Value, output: Value) -> WorkflowSpec {
        serde_json::from_value(json!({
            "inputs": { "doc": { "type": "string", "format": "asset" } },
            "params": { "type": "object", "properties": { "author": { "type": "string" } } },
            "steps": steps,
            "output": output
        }))
        .expect("workflow shape")
    }

    fn copy_step(id: &str, doc: Value) -> Value {
        json!({ "id": id, "op": "test.copy", "input": { "doc": doc } })
    }

    #[test]
    fn valid_workflow_orders_dependencies_and_keeps_layer_order() {
        let spec = workflow(
            json!([
                copy_step("front", json!({ "$from": "steps.terms.doc" })),
                copy_step("norm", json!({ "$from": "inputs.doc" })),
                copy_step("terms", json!({ "$from": "steps.norm.doc" })),
                copy_step("side", json!({ "$from": "inputs.doc" }))
            ]),
            json!({ "$from": "steps.front.doc" }),
        );
        spec.validate().expect("valid workflow");
        let order: Vec<&str> = spec
            .ordered_steps()
            .unwrap()
            .iter()
            .map(|step| step.id.as_str())
            .collect();
        assert_eq!(order, vec!["norm", "side", "terms", "front"]);
    }

    #[test]
    fn duplicate_step_ids_are_rejected() {
        let spec = workflow(
            json!([
                copy_step("norm", json!({ "$from": "inputs.doc" })),
                copy_step("norm", json!({ "$from": "inputs.doc" }))
            ]),
            json!({ "$from": "steps.norm.doc" }),
        );
        let errors = spec.validate().expect_err("duplicate id");
        assert!(
            errors
                .iter()
                .any(|d| d.code().as_str() == "DYNAMIC_WORKFLOW/DUPLICATE_STEP_ID")
        );
    }

    #[test]
    fn unknown_input_param_and_step_references_are_rejected() {
        let spec = workflow(
            json!([
                copy_step("norm", json!({ "$from": "inputs.missing" })),
                copy_step("terms", json!({ "$from": "params.missing" })),
                copy_step("front", json!({ "$from": "steps.nope.doc" }))
            ]),
            json!({ "$from": "steps.front.doc" }),
        );
        let errors = spec.validate().expect_err("unknown references");
        let codes: Vec<&str> = errors.iter().map(|d| d.code().as_str()).collect();
        assert!(
            codes
                .iter()
                .filter(|code| **code == "DYNAMIC_WORKFLOW/UNKNOWN_REFERENCE")
                .count()
                >= 3,
            "got {codes:?}"
        );
    }

    #[test]
    fn cyclic_steps_are_rejected() {
        let spec = workflow(
            json!([
                copy_step("a", json!({ "$from": "steps.b.doc" })),
                copy_step("b", json!({ "$from": "steps.a.doc" }))
            ]),
            json!({ "$from": "steps.a.doc" }),
        );
        let errors = spec.validate().expect_err("cycle");
        assert!(
            errors
                .iter()
                .any(|d| d.code().as_str() == "DYNAMIC_WORKFLOW/CYCLIC_STEPS")
        );
    }

    #[test]
    fn output_is_required_and_must_be_a_binding() {
        let mut spec = workflow(json!([]), json!({ "$from": "inputs.doc" }));
        spec.output = json!({ "not": "a binding" });
        let errors = spec.validate().expect_err("literal output");
        assert!(
            errors
                .iter()
                .any(|d| d.code().as_str() == "DYNAMIC_WORKFLOW/INVALID_BINDING")
        );

        spec.output = Value::Null;
        assert!(spec.validate().is_err());
    }

    #[test]
    fn binding_objects_only_accept_a_single_from_key() {
        let spec = workflow(
            json!([{
                "id": "norm",
                "op": "test.copy",
                "input": { "doc": { "$from": "inputs.doc", "extra": true } }
            }]),
            json!({ "$from": "inputs.doc" }),
        );
        let errors = spec.validate().expect_err("multi-key binding");
        assert!(
            errors
                .iter()
                .any(|d| d.code().as_str() == "DYNAMIC_WORKFLOW/INVALID_BINDING")
        );
    }

    #[test]
    fn target_shape_depends_on_asset_inputs() {
        let spec = workflow(json!([]), json!({ "$from": "inputs.doc" }));
        assert_eq!(spec.asset_inputs(), vec!["doc"]);

        let collection = RunTarget::Collection {
            collection_id: crate::model::CollectionId::new("col-1"),
            slots: None,
        };
        spec.validate_target(&collection).expect("one asset input");

        let groups = RunTarget::Groups {
            groups: vec![json!({ "author": "Ada" })],
            name: "Batch".to_owned(),
        };
        let errors = spec.validate_target(&groups).expect_err("asset input");
        assert_eq!(
            errors[0].code().as_str(),
            "DYNAMIC_WORKFLOW/ASSET_INPUT_COUNT"
        );

        let no_assets: WorkflowSpec = serde_json::from_value(json!({
            "inputs": {},
            "params": { "type": "object" },
            "steps": [],
            "output": { "$from": "params.value" }
        }))
        .unwrap();
        no_assets.validate_target(&groups).expect("no asset input");
        let errors = no_assets
            .validate_target(&collection)
            .expect_err("needs one asset input");
        assert_eq!(
            errors[0].code().as_str(),
            "DYNAMIC_WORKFLOW/ASSET_INPUT_COUNT"
        );
    }
}
