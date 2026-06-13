//! Shared response validation for inference-backed executors.
//!
//! Every inference-backed executor calls [`validate_response`] after
//! receiving the backend response. The validator checks:
//!
//! - every returned `slot_id` is declared by the node's output slots;
//! - required output slots are present;
//! - returned values match the expected slot kind;
//! - duplicate `slot_id` values are rejected;
//! - no undeclared extra outputs are present (unless a dynamic
//!   output policy is enabled).

use reimagine_core::model::{ParamValue, SlotId, SlotKind};
use reimagine_runtime::{NodeExecutorError, RuntimeValue};

use crate::response::InferenceResponse;

/// Expected output slot descriptor for validation.
#[derive(Debug, Clone)]
pub struct ExpectedOutputSlot {
    pub slot_id: SlotId,
    pub kind: SlotKind,
    pub required: bool,
}

impl ExpectedOutputSlot {
    pub fn required(slot_id: impl Into<SlotId>, kind: SlotKind) -> Self {
        Self {
            slot_id: slot_id.into(),
            kind,
            required: true,
        }
    }

    #[allow(dead_code)]
    pub fn optional(slot_id: impl Into<SlotId>, kind: SlotKind) -> Self {
        Self {
            slot_id: slot_id.into(),
            kind,
            required: false,
        }
    }
}

/// Validate a backend response against the expected output slots.
///
/// Returns `Ok(node_outputs)` on success, where `node_outputs` is
/// the response converted into `NodeExecutionOutputs` (Vec of
/// `(SlotId, Arc<RuntimeValue>)` pairs).
///
/// Returns `Err(NodeExecutorError)` when validation fails:
/// - duplicate slot id
/// - undeclared extra slot
/// - missing required slot
/// - value kind mismatch
pub fn validate_response(
    response: &InferenceResponse,
    expected: &[ExpectedOutputSlot],
    allow_extra: bool,
) -> Result<Vec<(SlotId, std::sync::Arc<reimagine_runtime::RuntimeValue>)>, NodeExecutorError> {
    use std::collections::HashSet;

    let outputs = response.outputs();

    // Check for duplicate slot ids.
    let mut seen = HashSet::new();
    for output in outputs {
        if !seen.insert(output.slot_id()) {
            return Err(NodeExecutorError::Failed {
                message: format!("duplicate output slot `{}`", output.slot_id()),
            });
        }
    }

    // Check for undeclared extra slots.
    if !allow_extra {
        for output in outputs {
            if !expected
                .iter()
                .any(|expected| expected.slot_id == *output.slot_id())
            {
                return Err(NodeExecutorError::Failed {
                    message: format!("undeclared output slot `{}`", output.slot_id()),
                });
            }
        }
    }

    // Check kind compatibility for declared outputs.
    for output in outputs {
        if let Some(expected_slot) = expected
            .iter()
            .find(|expected| expected.slot_id == *output.slot_id())
        {
            if !runtime_value_matches_slot_kind(output.value(), expected_slot.kind) {
                return Err(NodeExecutorError::Failed {
                    message: format!(
                        "output slot `{}` expected {:?}, got {}",
                        output.slot_id(),
                        expected_slot.kind,
                        runtime_value_kind_name(output.value())
                    ),
                });
            }
        }
    }

    // Check that all required slots are present.
    for expected_slot in expected {
        if expected_slot.required
            && !outputs
                .iter()
                .any(|o| *o.slot_id() == expected_slot.slot_id)
        {
            return Err(NodeExecutorError::Failed {
                message: format!("missing required output slot `{}`", expected_slot.slot_id),
            });
        }
    }

    Ok(response
        .outputs()
        .iter()
        .map(|o| (o.slot_id().clone(), o.value().clone()))
        .collect())
}

fn runtime_value_matches_slot_kind(value: &RuntimeValue, slot_kind: SlotKind) -> bool {
    match (value, slot_kind) {
        (RuntimeValue::Param(param), kind) => param_value_matches_slot_kind(param, kind),
        (RuntimeValue::Model(_), SlotKind::Model) => true,
        (RuntimeValue::Clip(_), SlotKind::Clip) => true,
        (RuntimeValue::Vae(_), SlotKind::Vae) => true,
        (RuntimeValue::Latent(_), SlotKind::Latent) => true,
        (RuntimeValue::Conditioning(_), SlotKind::Conditioning) => true,
        (RuntimeValue::Image(_), SlotKind::Image) => true,
        (RuntimeValue::Artifact(_), SlotKind::Artifact) => true,
        (RuntimeValue::Null, SlotKind::Null) => true,
        _ => false,
    }
}

fn param_value_matches_slot_kind(value: &ParamValue, slot_kind: SlotKind) -> bool {
    matches!(
        (value, slot_kind),
        (ParamValue::String(_), SlotKind::String)
            | (ParamValue::Text(_), SlotKind::Text)
            | (ParamValue::Integer(_), SlotKind::Integer)
            | (ParamValue::Float(_), SlotKind::Float)
            | (ParamValue::Bool(_), SlotKind::Bool)
            | (ParamValue::Seed(_), SlotKind::Seed)
            | (ParamValue::Select(_), SlotKind::Select)
            | (ParamValue::Path(_), SlotKind::Path)
            | (ParamValue::ModelRef(_), SlotKind::ModelRef)
            | (ParamValue::Null, SlotKind::Null)
    )
}

fn runtime_value_kind_name(value: &RuntimeValue) -> &'static str {
    match value {
        RuntimeValue::Param(_) => "param",
        RuntimeValue::Model(_) => "model",
        RuntimeValue::Clip(_) => "clip",
        RuntimeValue::Vae(_) => "vae",
        RuntimeValue::Latent(_) => "latent",
        RuntimeValue::Conditioning(_) => "conditioning",
        RuntimeValue::Image(_) => "image",
        RuntimeValue::Artifact(_) => "artifact",
        RuntimeValue::Null => "null",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_runtime::RuntimeValue;
    use std::sync::Arc;

    use crate::response::InferenceOutput;

    fn make_response(outputs: Vec<(&str, &str)>) -> InferenceResponse {
        InferenceResponse::new(
            outputs
                .into_iter()
                .map(|(slot, val)| {
                    InferenceOutput::new(
                        SlotId::new(slot),
                        Arc::new(RuntimeValue::Param(
                            reimagine_core::model::ParamValue::String(val.to_string()),
                        )),
                    )
                })
                .collect(),
        )
    }

    #[test]
    fn valid_multi_output_passes() {
        let resp = make_response(vec![("model", "m"), ("clip", "c"), ("vae", "v")]);
        let expected = vec![
            ExpectedOutputSlot::required("model", SlotKind::String),
            ExpectedOutputSlot::required("clip", SlotKind::String),
            ExpectedOutputSlot::required("vae", SlotKind::String),
        ];
        let result = validate_response(&resp, &expected, false);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3);
    }

    #[test]
    fn duplicate_slot_rejected() {
        let resp = make_response(vec![("model", "m"), ("model", "m2")]);
        let expected = vec![ExpectedOutputSlot::required("model", SlotKind::String)];
        let result = validate_response(&resp, &expected, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("duplicate"));
    }

    #[test]
    fn undeclared_slot_rejected() {
        let resp = make_response(vec![("model", "m"), ("extra", "x")]);
        let expected = vec![ExpectedOutputSlot::required("model", SlotKind::String)];
        let result = validate_response(&resp, &expected, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("undeclared"));
    }

    #[test]
    fn missing_required_slot_rejected() {
        let resp = make_response(vec![("model", "m")]);
        let expected = vec![
            ExpectedOutputSlot::required("model", SlotKind::String),
            ExpectedOutputSlot::required("clip", SlotKind::String),
        ];
        let result = validate_response(&resp, &expected, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("missing"));
    }

    #[test]
    fn allow_extra_permits_undeclared_slots() {
        let resp = make_response(vec![("model", "m"), ("extra", "x")]);
        let expected = vec![ExpectedOutputSlot::required("model", SlotKind::String)];
        let result = validate_response(&resp, &expected, true);
        assert!(result.is_ok());
    }

    #[test]
    fn value_kind_mismatch_rejected() {
        let resp = make_response(vec![("model", "m")]);
        let expected = vec![ExpectedOutputSlot::required("model", SlotKind::Model)];
        let result = validate_response(&resp, &expected, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expected"));
    }
}
