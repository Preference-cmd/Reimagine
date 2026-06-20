//! Shared helpers for concrete built-in node executors.

use std::sync::Arc;

use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::model::{ParamValue, SlotId};
use reimagine_inference_core::{
    ExecutionConditioning, ExecutionOutput, ExecutionValue, RuntimeClipHandle, RuntimeImage,
    RuntimeLatent, RuntimeModelHandle, RuntimeVaeHandle,
};

use crate::executor::{NodeExecutionContext, NodeExecutorError};

pub fn required_input(
    context: &NodeExecutionContext,
    slot: &str,
) -> Result<Arc<ExecutionValue>, NodeExecutorError> {
    context
        .inputs()
        .get(&SlotId::new(slot))
        .cloned()
        .ok_or_else(|| NodeExecutorError::MissingInput {
            slot_id: slot.to_string(),
        })
}

pub fn required_model_input(
    context: &NodeExecutionContext,
    slot: &str,
    capability: &str,
) -> Result<RuntimeModelHandle, NodeExecutorError> {
    match required_input(context, slot)?.as_ref() {
        ExecutionValue::Model(handle) => Ok(handle.clone()),
        _ => Err(NodeExecutorError::Failed {
            message: format!("{capability} `{slot}` input must be a Model handle"),
        }),
    }
}

pub fn required_clip_input(
    context: &NodeExecutionContext,
    slot: &str,
    capability: &str,
) -> Result<RuntimeClipHandle, NodeExecutorError> {
    match required_input(context, slot)?.as_ref() {
        ExecutionValue::Clip(handle) => Ok(handle.clone()),
        _ => Err(NodeExecutorError::Failed {
            message: format!("{capability} `{slot}` input must be a Clip handle"),
        }),
    }
}

pub fn required_vae_input(
    context: &NodeExecutionContext,
    slot: &str,
    capability: &str,
) -> Result<RuntimeVaeHandle, NodeExecutorError> {
    match required_input(context, slot)?.as_ref() {
        ExecutionValue::Vae(handle) => Ok(handle.clone()),
        _ => Err(NodeExecutorError::Failed {
            message: format!("{capability} `{slot}` input must be a Vae handle"),
        }),
    }
}

pub fn required_latent_input(
    context: &NodeExecutionContext,
    slot: &str,
    capability: &str,
) -> Result<RuntimeLatent, NodeExecutorError> {
    match required_input(context, slot)?.as_ref() {
        ExecutionValue::Latent(handle) => Ok(handle.clone()),
        _ => Err(NodeExecutorError::Failed {
            message: format!("{capability} `{slot}` input must be a Latent handle"),
        }),
    }
}

pub fn required_image_input(
    context: &NodeExecutionContext,
    slot: &str,
    capability: &str,
) -> Result<RuntimeImage, NodeExecutorError> {
    match required_input(context, slot)?.as_ref() {
        ExecutionValue::Image(handle) => Ok(handle.clone()),
        _ => Err(NodeExecutorError::Failed {
            message: format!("{capability} `{slot}` input must be an Image handle"),
        }),
    }
}

pub fn required_conditioning_input(
    context: &NodeExecutionContext,
    slot: &str,
    capability: &str,
) -> Result<ExecutionConditioning, NodeExecutorError> {
    match required_input(context, slot)?.as_ref() {
        ExecutionValue::Conditioning(handle) => Ok(handle.clone()),
        _ => Err(NodeExecutorError::Failed {
            message: format!("{capability} `{slot}` input must be a Conditioning handle"),
        }),
    }
}

pub fn required_u32_param(
    context: &NodeExecutionContext,
    slot: &str,
) -> Result<u32, NodeExecutorError> {
    match context.params().get(&SlotId::new(slot)) {
        Some(ParamValue::Integer(v)) => u32::try_from(*v).map_err(|_| NodeExecutorError::Failed {
            message: format!("param `{slot}` must fit in u32, got {v}"),
        }),
        Some(other) => Err(NodeExecutorError::Failed {
            message: format!(
                "param `{slot}` must be an integer, got {}",
                param_kind_name(other)
            ),
        }),
        None => Err(NodeExecutorError::MissingInput {
            slot_id: slot.to_string(),
        }),
    }
}

pub fn required_i64_param(
    context: &NodeExecutionContext,
    slot: &str,
) -> Result<i64, NodeExecutorError> {
    match context.params().get(&SlotId::new(slot)) {
        Some(ParamValue::Integer(v)) => Ok(*v),
        Some(other) => Err(NodeExecutorError::Failed {
            message: format!(
                "param `{slot}` must be an integer, got {}",
                param_kind_name(other)
            ),
        }),
        None => Err(NodeExecutorError::MissingInput {
            slot_id: slot.to_string(),
        }),
    }
}

pub fn required_f32_param(
    context: &NodeExecutionContext,
    slot: &str,
) -> Result<f32, NodeExecutorError> {
    match context.params().get(&SlotId::new(slot)) {
        Some(ParamValue::Float(v)) => Ok(*v as f32),
        Some(other) => Err(NodeExecutorError::Failed {
            message: format!(
                "param `{slot}` must be a float, got {}",
                param_kind_name(other)
            ),
        }),
        None => Err(NodeExecutorError::MissingInput {
            slot_id: slot.to_string(),
        }),
    }
}

pub fn required_seed_param(
    context: &NodeExecutionContext,
    slot: &str,
) -> Result<u64, NodeExecutorError> {
    match context.params().get(&SlotId::new(slot)) {
        Some(ParamValue::Seed(v)) => Ok(*v),
        Some(other) => Err(NodeExecutorError::Failed {
            message: format!(
                "param `{slot}` must be a seed, got {}",
                param_kind_name(other)
            ),
        }),
        None => Err(NodeExecutorError::MissingInput {
            slot_id: slot.to_string(),
        }),
    }
}

pub fn optional_string_param(context: &NodeExecutionContext, slot: &str) -> Option<String> {
    match context.params().get(&SlotId::new(slot)) {
        Some(ParamValue::String(s)) => Some(s.clone()),
        _ => None,
    }
}

pub fn optional_correlation_id(context: &NodeExecutionContext) -> Option<CorrelationId> {
    context.correlation_id().cloned()
}

pub fn run_output(slot: &str, value: ExecutionValue) -> ExecutionOutput {
    ExecutionOutput::run_scoped(SlotId::new(slot), Arc::new(value))
}

pub fn workspace_output(slot: &str, value: ExecutionValue) -> ExecutionOutput {
    ExecutionOutput::workspace_scoped(SlotId::new(slot), Arc::new(value))
}

pub fn param_kind_name(value: &ParamValue) -> &'static str {
    match value {
        ParamValue::String(_) => "string",
        ParamValue::Text(_) => "text",
        ParamValue::Integer(_) => "integer",
        ParamValue::Float(_) => "float",
        ParamValue::Bool(_) => "bool",
        ParamValue::Seed(_) => "seed",
        ParamValue::Select(_) => "select",
        ParamValue::Path(_) => "path",
        ParamValue::ModelRef(_) => "model_ref",
        ParamValue::Null => "null",
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reimagine_core::event::Timestamp;
    use reimagine_core::model::{
        ModelId, ParamValue, RunId, SlotId, TensorDType, TensorShape, WorkflowId, WorkflowVersion,
    };
    use reimagine_inference_core::{
        BackendKind, BackendPayloadKey, BackendTensorHandle, ExecutionValue, RuntimeClipHandle,
    };

    use super::{required_clip_input, required_u32_param, run_output};
    use crate::executor::{NodeExecutionContext, NodeExecutorError};
    use crate::node_context::{NodeInputs, NodeParams};
    use crate::testing::{NoopArtifactPublisher, NoopNodeCancellation};

    fn test_context(inputs: NodeInputs, params: NodeParams) -> NodeExecutionContext {
        NodeExecutionContext::new(
            RunId::new("run"),
            WorkflowId::new("workflow"),
            WorkflowVersion::from(1_u64),
            None,
            "node-1".into(),
            "builtin.test".into(),
            inputs,
            params,
            Arc::new(NoopArtifactPublisher::new()),
            Arc::new(NoopNodeCancellation::new()),
            Timestamp::new("2026-06-21T00:00:00Z"),
        )
    }

    #[test]
    fn required_clip_input_reports_expected_handle_kind() {
        let mut inputs = NodeInputs::new();
        let tensor = BackendTensorHandle::new(
            BackendKind::new("fake"),
            BackendPayloadKey::new("latent-1"),
            TensorDType::F32,
            TensorShape::new(vec![1, 4, 8, 8]),
            "cpu",
        );
        inputs.insert(
            SlotId::new("clip"),
            Arc::new(ExecutionValue::Latent(
                reimagine_inference_core::RuntimeLatent::new(tensor, 64, 64, 1, 4),
            )),
        );

        let err = required_clip_input(
            &test_context(inputs, NodeParams::new()),
            "clip",
            "text.encode",
        )
        .expect_err("clip extraction should fail");

        assert_eq!(
            err,
            NodeExecutorError::Failed {
                message: "text.encode `clip` input must be a Clip handle".to_string(),
            }
        );
    }

    #[test]
    fn required_u32_param_rejects_negative_values() {
        let mut params = NodeParams::new();
        params.insert(SlotId::new("width"), ParamValue::Integer(-1));

        let err = required_u32_param(&test_context(NodeInputs::new(), params), "width")
            .expect_err("negative widths should fail");

        assert_eq!(
            err,
            NodeExecutorError::Failed {
                message: "param `width` must fit in u32, got -1".to_string(),
            }
        );
    }

    #[test]
    fn run_output_marks_value_run_scoped() {
        let value = ExecutionValue::Clip(RuntimeClipHandle::new(
            ModelId::new("sdxl-base-1.0"),
            BackendKind::new("fake"),
            "clip-1",
        ));

        let output = run_output("conditioning", value);

        assert_eq!(output.slot_id(), &SlotId::new("conditioning"));
        assert_eq!(
            output.retention(),
            reimagine_inference_core::ExecutionValueRetention::RunScoped
        );
    }
}
