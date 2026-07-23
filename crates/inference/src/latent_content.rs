//! Runtime content semantics for a latent payload.
//!
//! [`LatentContent`] describes **what a latent payload represents at
//! runtime**, independent of where it came from. It is the answer to
//! "is this payload actually a real starting latent, or just a
//! txt2img placeholder?".
//!
//! [`LatentContent`] is **not provenance**. It does not record source
//! paths, node ids, model ids, history, or actor information. Those
//! concerns belong on the workflow node execution context and the
//! run event stream, not on the value type.
//!
//! [`LatentSpaceMetadata`](crate::LatentSpaceMetadata) answers "which
//! latent space is this payload compatible with". [`LatentContent`]
//! answers "is this payload a real latent, or empty geometry?". The
//! two are deliberately separate vocabularies: a real SDXL sampled
//! latent is `Sampled` content on `stable_diffusion/sdxl/base` latent
//! space, while a txt2img geometry latent is `EmptyGeometry` content
//! on the same latent space.
//!
//! V1 vocabulary:
//!
//! - [`LatentContent::EmptyGeometry`] — produced by
//!   `latent.create_empty` and consumed only by `diffusion.sample` as
//!   initial geometry. It is **not** a real starting latent for
//!   partial denoise and **must not** be decoded.
//! - [`LatentContent::EncodedImage`] — produced by `latent.encode`.
//!   It is a real latent payload derived from an image through a VAE
//!   encoder.
//! - [`LatentContent::Sampled`] — produced by `diffusion.sample`. It
//!   is a real sampled latent payload.
//! - [`LatentContent::Imported`] — reserved for future latent
//!   loader/import capabilities. It is a real latent payload sourced
//!   from outside the current graph (e.g. an in-memory latent loaded
//!   from a previously saved file).

use serde::{Deserialize, Serialize};

/// Runtime content semantics for a [`crate::RuntimeLatent`] payload.
///
/// See the [module documentation](self) for the full vocabulary and
/// the distinction between content semantics and provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatentContent {
    /// Empty geometry produced by `latent.create_empty`. Carries the
    /// batch/shape/dtype metadata but **no** actual sampled tensor.
    EmptyGeometry,
    /// Latent payload produced by `latent.encode` from an image.
    EncodedImage,
    /// Latent payload produced by `diffusion.sample`.
    Sampled,
    /// Latent payload imported from outside the current graph.
    /// Reserved vocabulary; V1 has no latent loader capability yet.
    Imported,
}

impl LatentContent {
    /// Lowercase, dot-separated label used in diagnostics.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::EmptyGeometry => "empty_geometry",
            Self::EncodedImage => "encoded_image",
            Self::Sampled => "sampled",
            Self::Imported => "imported",
        }
    }
}

impl std::fmt::Display for LatentContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Errors raised when a latent's [`LatentContent`] does not match the
/// capability being invoked.
///
/// V1 surfaces these as precise
/// `InferenceError::InvalidRequest` messages; the runtime does not
/// need to inspect the variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LatentContentError {
    /// The capability does not accept this kind of latent content.
    UnsupportedForCapability {
        capability: &'static str,
        actual: LatentContent,
    },
}

impl std::fmt::Display for LatentContentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedForCapability { capability, actual } => write!(
                f,
                "capability `{capability}` does not accept latent content `{actual}`"
            ),
        }
    }
}

impl std::error::Error for LatentContentError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_are_stable_strings() {
        assert_eq!(LatentContent::EmptyGeometry.as_str(), "empty_geometry");
        assert_eq!(LatentContent::EncodedImage.as_str(), "encoded_image");
        assert_eq!(LatentContent::Sampled.as_str(), "sampled");
        assert_eq!(LatentContent::Imported.as_str(), "imported");
    }

    #[test]
    fn display_matches_label() {
        assert_eq!(LatentContent::EmptyGeometry.to_string(), "empty_geometry");
        assert_eq!(LatentContent::Sampled.to_string(), "sampled");
    }

    #[test]
    fn serde_round_trip_uses_snake_case() {
        let json = serde_json::to_string(&LatentContent::EmptyGeometry).unwrap();
        assert_eq!(json, "\"empty_geometry\"");
        let parsed: LatentContent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, LatentContent::EmptyGeometry);
    }

    #[test]
    fn unsupported_for_capability_message_names_capability_and_actual() {
        let err = LatentContentError::UnsupportedForCapability {
            capability: "latent.decode",
            actual: LatentContent::EmptyGeometry,
        };
        let msg = err.to_string();
        assert!(msg.contains("latent.decode"), "{msg}");
        assert!(msg.contains("empty_geometry"), "{msg}");
    }
}
