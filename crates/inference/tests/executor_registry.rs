//! `NodeExecutorRegistry` capability index tests (BE-14).
//!
//! Exercises the public registry API: index built during
//! registration, capability queries, and the capability union over
//! registered node types.

use std::sync::Arc;

use reimagine_core::model::NodeTypeId;
use reimagine_inference::{
    InferenceCapability, NodeExecutionOutputs, NodeExecutor, NodeExecutorError,
    NodeExecutorRegistry, NodeExecutionContext,
};

struct DummyExecutor;

#[async_trait::async_trait]
impl NodeExecutor for DummyExecutor {
    async fn execute(
        &self,
        _context: NodeExecutionContext,
    ) -> Result<NodeExecutionOutputs, NodeExecutorError> {
        Ok(Vec::new())
    }
}

struct CapableExecutor(&'static [InferenceCapability]);

#[async_trait::async_trait]
impl NodeExecutor for CapableExecutor {
    async fn execute(
        &self,
        _context: NodeExecutionContext,
    ) -> Result<NodeExecutionOutputs, NodeExecutorError> {
        Ok(Vec::new())
    }

    fn required_capabilities(&self) -> &'static [InferenceCapability] {
        self.0
    }
}

#[test]
fn builtin_registration_builds_capability_index() {
    let mut registry = NodeExecutorRegistry::default();
    for (type_id, capabilities) in [
        ("builtin.checkpoint_loader", &[InferenceCapability::LoadBundle][..]),
        ("builtin.load_image", &[InferenceCapability::ImageImport][..]),
        ("builtin.clip_text_encode", &[InferenceCapability::TextEncode][..]),
        ("builtin.empty_latent_image", &[InferenceCapability::CreateEmptyLatent][..]),
        ("builtin.vae_encode", &[InferenceCapability::LatentEncode][..]),
        ("builtin.ksampler", &[InferenceCapability::DiffusionSample][..]),
        ("builtin.vae_decode", &[InferenceCapability::LatentDecode][..]),
        ("builtin.save_image", &[InferenceCapability::ImageSave][..]),
        ("builtin.preview_image", &[InferenceCapability::ImagePreview][..]),
        ("builtin.string", &[][..]),
    ] {
        registry
            .register_with_capabilities(type_id, Arc::new(DummyExecutor), capabilities)
            .expect("register");
    }

    assert_eq!(registry.len(), 10);
    assert!(registry.has_capability(InferenceCapability::DiffusionSample));
    assert!(registry.has_capability(InferenceCapability::TextEncode));
    assert!(registry.has_capability(InferenceCapability::LoadBundle));
    assert_eq!(
        registry.query_by_capability(InferenceCapability::LoadBundle),
        vec![NodeTypeId::new("builtin.checkpoint_loader")]
    );
    assert_eq!(
        registry.query_by_capability(InferenceCapability::DiffusionSample),
        vec![NodeTypeId::new("builtin.ksampler")]
    );
    assert_eq!(
        registry.query_by_capability(InferenceCapability::ImageImport),
        vec![NodeTypeId::new("builtin.load_image")]
    );
}

#[test]
fn register_uses_executor_declared_capabilities() {
    let mut registry = NodeExecutorRegistry::default();
    registry
        .register(
            "builtin.ksampler",
            Arc::new(CapableExecutor(&[InferenceCapability::DiffusionSample])),
        )
        .expect("register");

    assert!(registry.has_capability(InferenceCapability::DiffusionSample));
    assert!(!registry.has_capability(InferenceCapability::TextEncode));
}

#[test]
fn register_with_capabilities_indexes_explicit_capabilities() {
    let mut registry = NodeExecutorRegistry::default();
    registry
        .register_with_capabilities(
            "builtin.clip_text_encode",
            Arc::new(DummyExecutor),
            &[InferenceCapability::TextEncode],
        )
        .expect("register");

    assert_eq!(
        registry.query_by_capability(InferenceCapability::TextEncode),
        vec![NodeTypeId::new("builtin.clip_text_encode")]
    );
}

#[test]
fn capability_union_spans_all_registered_types() {
    let mut registry = NodeExecutorRegistry::default();
    registry
        .register_with_capabilities(
            "builtin.ksampler",
            Arc::new(DummyExecutor),
            &[InferenceCapability::DiffusionSample],
        )
        .expect("register");
    registry
        .register_with_capabilities(
            "builtin.vae_decode",
            Arc::new(DummyExecutor),
            &[InferenceCapability::LatentDecode],
        )
        .expect("register");
    registry
        .register("builtin.string", Arc::new(DummyExecutor))
        .expect("register");

    assert_eq!(
        registry.capability_union(),
        vec![
            InferenceCapability::DiffusionSample,
            InferenceCapability::LatentDecode,
        ]
    );
}

#[test]
fn duplicate_registration_does_not_pollute_capability_index() {
    let mut registry = NodeExecutorRegistry::default();
    registry
        .register_with_capabilities(
            "builtin.ksampler",
            Arc::new(DummyExecutor),
            &[InferenceCapability::DiffusionSample],
        )
        .expect("register");
    let err = registry
        .register_with_capabilities(
            "builtin.ksampler",
            Arc::new(DummyExecutor),
            &[InferenceCapability::TextEncode],
        )
        .expect_err("duplicate registration must fail");

    assert!(err.to_string().contains("builtin.ksampler"));
    assert_eq!(
        registry.query_by_capability(InferenceCapability::DiffusionSample),
        vec![NodeTypeId::new("builtin.ksampler")]
    );
    assert!(!registry.has_capability(InferenceCapability::TextEncode));
}

#[test]
fn clone_for_runner_preserves_capability_index() {
    let mut registry = NodeExecutorRegistry::default();
    registry
        .register_with_capabilities(
            "builtin.ksampler",
            Arc::new(DummyExecutor),
            &[InferenceCapability::DiffusionSample],
        )
        .expect("register");

    let runner = registry.clone_for_runner();
    assert!(runner.has_capability(InferenceCapability::DiffusionSample));
    assert_eq!(
        runner.query_by_capability(InferenceCapability::DiffusionSample),
        vec![NodeTypeId::new("builtin.ksampler")]
    );
}
