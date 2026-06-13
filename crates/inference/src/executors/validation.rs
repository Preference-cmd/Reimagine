//! Shared response validation for inference-backed executors.
//!
//! Every inference-backed executor calls [`validate_response`] after
//! receiving the backend response. The validator checks:
//!
//! - every returned `slot_id` is declared by the node's output slots;
//! - required output slots are present;
//! - duplicate `slot_id` values are rejected;
//! - no undeclared extra outputs are present (unless a dynamic
//!   output policy is enabled).

use reimagine_core::model::SlotId;
use reimagine_runtime::NodeExecutorError;

use crate::response::InferenceResponse;

/// Expected output slot descriptor for validation.
#[derive(Debug, Clone)]
pub struct ExpectedOutputSlot {
    pub slot_id: SlotId,
    pub required: bool,
}

impl ExpectedOutputSlot {
    pub fn required(slot_id: impl Into<SlotId>) -> Self {
        Self {
            slot_id: slot_id.into(),
            required: true,
        }
    }

    #[allow(dead_code)]
    pub fn optional(slot_id: impl Into<SlotId>) -> Self {
        Self {
            slot_id: slot_id.into(),
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
            if !expected.iter().any(|e| e.slot_id == *output.slot_id()) {
                return Err(NodeExecutorError::Failed {
                    message: format!("undeclared output slot `{}`", output.slot_id()),
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
            ExpectedOutputSlot::required("model"),
            ExpectedOutputSlot::required("clip"),
            ExpectedOutputSlot::required("vae"),
        ];
        let result = validate_response(&resp, &expected, false);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3);
    }

    #[test]
    fn duplicate_slot_rejected() {
        let resp = make_response(vec![("model", "m"), ("model", "m2")]);
        let expected = vec![ExpectedOutputSlot::required("model")];
        let result = validate_response(&resp, &expected, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("duplicate"));
    }

    #[test]
    fn undeclared_slot_rejected() {
        let resp = make_response(vec![("model", "m"), ("extra", "x")]);
        let expected = vec![ExpectedOutputSlot::required("model")];
        let result = validate_response(&resp, &expected, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("undeclared"));
    }

    #[test]
    fn missing_required_slot_rejected() {
        let resp = make_response(vec![("model", "m")]);
        let expected = vec![
            ExpectedOutputSlot::required("model"),
            ExpectedOutputSlot::required("clip"),
        ];
        let result = validate_response(&resp, &expected, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("missing"));
    }

    #[test]
    fn allow_extra_permits_undeclared_slots() {
        let resp = make_response(vec![("model", "m"), ("extra", "x")]);
        let expected = vec![ExpectedOutputSlot::required("model")];
        let result = validate_response(&resp, &expected, true);
        assert!(result.is_ok());
    }
}
