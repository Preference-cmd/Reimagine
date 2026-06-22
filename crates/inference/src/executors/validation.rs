//! Slot-shaped response validation and output mapping helpers.
//!
//! Typed backend responses already guarantee the payload kind. These
//! helpers centralize the node slot and retention contract that each
//! built-in executor must preserve when mapping a typed response back
//! into runtime-facing execution outputs.

use crate::{
    CreateEmptyLatentResponse, DiffusionSampleResponse, ExecutionOutput, ExecutionValue,
    LatentDecodeResponse, LoadBundleResponse, TextEncodeResponse,
};

use super::common::{run_output, workspace_output};

pub fn load_bundle_outputs(response: &LoadBundleResponse) -> Vec<ExecutionOutput> {
    vec![
        workspace_output("model", ExecutionValue::Model(response.model().clone())),
        workspace_output("clip", ExecutionValue::Clip(response.clip().clone())),
        workspace_output("vae", ExecutionValue::Vae(response.vae().clone())),
    ]
}

pub fn conditioning_output(response: &TextEncodeResponse) -> ExecutionOutput {
    run_output(
        "conditioning",
        ExecutionValue::Conditioning(response.conditioning().clone()),
    )
}

pub fn latent_output(response: &CreateEmptyLatentResponse) -> ExecutionOutput {
    run_output("latent", ExecutionValue::Latent(response.latent().clone()))
}

pub fn sampled_latent_output(response: &DiffusionSampleResponse) -> ExecutionOutput {
    run_output("latent", ExecutionValue::Latent(response.latent().clone()))
}

pub fn image_output(response: &LatentDecodeResponse) -> ExecutionOutput {
    run_output("image", ExecutionValue::Image(response.image().clone()))
}

#[cfg(test)]
mod tests {
    use crate::{
        Backend, CreateEmptyLatentResponse, DiffusionSampleResponse, ExecutionValueRetention,
        LoadBundleResponse, RuntimeClipHandle, RuntimeLatent, RuntimeModelHandle, RuntimeVaeHandle,
    };
    use reimagine_core::model::{ModelId, ModelRole, SlotId};

    use super::{latent_output, load_bundle_outputs, sampled_latent_output};

    fn latent() -> RuntimeLatent {
        crate::RuntimeLatent::new(
            crate::BackendTensorHandle::new(
                Backend::new("fake"),
                crate::BackendPayloadKey::new("latent-1"),
                reimagine_core::model::TensorDType::F32,
                reimagine_core::model::TensorShape::new(vec![1, 4, 8, 8]),
                "cpu",
            ),
            64,
            64,
            1,
            4,
        )
    }

    #[test]
    fn load_bundle_outputs_preserve_workspace_retention_and_slots() {
        let response = LoadBundleResponse::new(
            RuntimeModelHandle::new(
                ModelId::new("sdxl-base-1.0"),
                ModelRole::DiffusionModel,
                Backend::new("fake"),
                "model-1",
            ),
            RuntimeClipHandle::new(
                ModelId::new("sdxl-base-1.0"),
                Backend::new("fake"),
                "clip-1",
            ),
            RuntimeVaeHandle::new(ModelId::new("sdxl-base-1.0"), Backend::new("fake"), "vae-1"),
        );

        let outputs = load_bundle_outputs(&response);

        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].slot_id(), &SlotId::new("model"));
        assert_eq!(outputs[1].slot_id(), &SlotId::new("clip"));
        assert_eq!(outputs[2].slot_id(), &SlotId::new("vae"));
        assert!(
            outputs
                .iter()
                .all(|output| output.retention() == ExecutionValueRetention::WorkspaceScoped)
        );
    }

    #[test]
    fn latent_output_uses_run_scope_for_empty_latent() {
        let output = latent_output(&CreateEmptyLatentResponse::new(latent()));
        assert_eq!(output.slot_id(), &SlotId::new("latent"));
        assert_eq!(output.retention(), ExecutionValueRetention::RunScoped);
    }

    #[test]
    fn sampled_latent_output_uses_run_scope_for_ksampler() {
        let output = sampled_latent_output(&DiffusionSampleResponse::new(latent()));
        assert_eq!(output.slot_id(), &SlotId::new("latent"));
        assert_eq!(output.retention(), ExecutionValueRetention::RunScoped);
    }
}
