//! Stable identifiers for inference operations.
//!
//! An [`InferenceOperationId`] is a backend-neutral, model-family-neutral
//! string identifier. The operation protocol is the same for SDXL, Flux,
//! or future model families; only the backend adapter changes.

/// Stable, backend-neutral identifier for an inference operation.
///
/// V1 uses a dot-separated `"domain.action"` naming convention.
/// New operations should be added as named constants rather than
/// raw string literals so the crate's public API is self-documenting.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct InferenceOperationId(String);

impl InferenceOperationId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for InferenceOperationId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for InferenceOperationId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for InferenceOperationId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

// ── V1 operation ids ───────────────────────────────────────────────

/// Load a model checkpoint bundle (model + CLIP + VAE handles).
pub const OP_MODEL_LOAD_BUNDLE: &str = "model.load_bundle";
/// Encode text through a text encoder (e.g. CLIP).
pub const OP_TEXT_ENCODE: &str = "text.encode";
/// Create an empty latent tensor.
pub const OP_LATENT_CREATE_EMPTY: &str = "latent.create_empty";
/// Run a diffusion sampling step (e.g. K-sampler).
pub const OP_DIFFUSION_SAMPLE: &str = "diffusion.sample";
/// Decode a latent tensor into pixel space.
pub const OP_LATENT_DECODE: &str = "latent.decode";
/// Save an image to disk.
pub const OP_IMAGE_SAVE: &str = "image.save";
/// Produce a preview image (not persisted).
pub const OP_IMAGE_PREVIEW: &str = "image.preview";

/// All V1 operation ids, in a fixed order.
pub const ALL_V1_OPERATIONS: &[&str] = &[
    OP_MODEL_LOAD_BUNDLE,
    OP_TEXT_ENCODE,
    OP_LATENT_CREATE_EMPTY,
    OP_DIFFUSION_SAMPLE,
    OP_LATENT_DECODE,
    OP_IMAGE_SAVE,
    OP_IMAGE_PREVIEW,
];
