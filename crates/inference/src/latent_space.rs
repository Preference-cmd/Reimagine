//! Model-neutral latent-space metadata vocabulary.
//!
//! Every latent value flowing through the inference layer carries a
//! [`LatentSpaceMetadata`] record so backends and the runtime can
//! validate compatibility without inferring it from "4 channels" or
//! image dimensions. The vocabulary is owned by `reimagine-inference`
//! and is deliberately **not** an enum tied to a specific model
//! family (SDXL, SD3, Flux, ...). Built-in spaces are exposed as
//! helper constants so backends and tests do not have to hand-roll
//! the canonical record.
//!
//! The metadata is intentionally complete on the value: validation
//! at [`RuntimeLatent`](crate::RuntimeLatent) boundaries does not
//! need a global registry lookup, only the value's own metadata and
//! the loaded model's expected latent space.

use reimagine_core::model::TensorDType;
use serde::{Deserialize, Serialize};

/// Errors raised when validating latent-space metadata at the
/// inference request boundary.
///
/// V1 surfaces these as precise
/// `InferenceError::InvalidRequest` messages; the runtime does not
/// need to inspect the variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LatentSpaceError {
    /// Latent pixel dimensions disagreed with the latent space's
    /// spatial scale factor.
    ScaleMismatch {
        axis: &'static str,
        value: u32,
        scale: u32,
    },
    /// Latent pixel dimension was zero or otherwise invalid for the
    /// latent space.
    InvalidDimensions {
        axis: &'static str,
        value: u32,
        reason: &'static str,
    },
    /// The backend does not implement support for a latent space
    /// that is otherwise well-formed.
    Unsupported { id: String, backend: String },
}

impl std::fmt::Display for LatentSpaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ScaleMismatch { axis, value, scale } => write!(
                f,
                "latent {axis}={value} is not divisible by latent-space spatial_scale_factor={scale}"
            ),
            Self::InvalidDimensions {
                axis,
                value,
                reason,
            } => {
                write!(f, "latent {axis}={value} invalid: {reason}")
            }
            Self::Unsupported { id, backend } => write!(
                f,
                "latent space `{id}` is not supported by backend `{backend}`"
            ),
        }
    }
}

impl std::error::Error for LatentSpaceError {}

/// Stable, model-neutral identifier for a latent space.
///
/// This is an open string newtype rather than an enum so future
/// backends and plugin-owned spaces can carry their own ids without
/// modifying the inference crate.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LatentSpaceId(String);

impl LatentSpaceId {
    /// Build a new latent-space id.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Borrow the id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for LatentSpaceId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for LatentSpaceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Tensor layout descriptor carried on latent-space metadata.
///
/// V1 only knows `Nchw`; future latent spaces (e.g. SD3/Flux with
/// packed 2×2 patches) can introduce a new variant or a backend-
/// specific `Other(String)` tag without breaking existing callers.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum TensorLayout {
    /// Standard `[batch, channels, height, width]` layout.
    #[default]
    Nchw,
    /// Backend-defined layout tag that V1 callers do not need to
    /// interpret; future work may promote frequent cases to variants.
    Other(String),
}

impl TensorLayout {
    /// Lowercase string form used in diagnostics and assertions.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Nchw => "nchw",
            Self::Other(s) => s.as_str(),
        }
    }
}

impl std::fmt::Display for TensorLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Complete latent-space metadata carried on every latent value.
///
/// The struct is intentionally complete (id, channels, scale, dtype,
/// layout) so validation at request boundaries can run without a
/// global registry lookup.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LatentSpaceMetadata {
    id: LatentSpaceId,
    channels: u32,
    spatial_scale_factor: u32,
    dtype: TensorDType,
    layout: TensorLayout,
}

impl LatentSpaceMetadata {
    /// Construct a complete metadata record. Prefer
    /// [`sdxl_base`](Self::sdxl_base) or
    /// [`stable_diffusion_sdxl_base`](crate::latent_space::stable_diffusion_sdxl_base)
    /// for the V1 built-in; this constructor is for tests, future
    /// latent spaces, and explicit metadata reconstruction.
    pub fn new(
        id: LatentSpaceId,
        channels: u32,
        spatial_scale_factor: u32,
        dtype: TensorDType,
        layout: TensorLayout,
    ) -> Self {
        Self {
            id,
            channels,
            spatial_scale_factor,
            dtype,
            layout,
        }
    }

    /// Latent-space identifier.
    pub fn id(&self) -> &LatentSpaceId {
        &self.id
    }

    /// Latent channel count (e.g. 4 for SDXL base).
    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// Pixel-to-latent spatial scale factor (e.g. 8 for SDXL base).
    pub fn spatial_scale_factor(&self) -> u32 {
        self.spatial_scale_factor
    }

    /// Expected latent tensor dtype.
    pub fn dtype(&self) -> TensorDType {
        self.dtype
    }

    /// Latent tensor layout.
    pub fn layout(&self) -> TensorLayout {
        self.layout.clone()
    }

    /// V1 built-in: SDXL base latent space.
    ///
    /// Records: id `stable_diffusion/sdxl/base`, 4 channels, 8x
    /// pixel-to-latent scale, f32 dtype, nchw layout.
    pub fn sdxl_base() -> Self {
        stable_diffusion_sdxl_base()
    }

    /// Two metadata records describe the same latent space when their
    /// id, channels, scale factor, dtype, and layout all agree.
    pub fn is_compatible(&self, other: &Self) -> bool {
        self == other
    }
}

/// V1 built-in: SDXL base latent space.
///
/// This is the canonical record every V1 workflow uses. Candle V1
/// only supports this latent space; future spaces (SD3, Flux,
/// custom) will add new helpers.
pub fn stable_diffusion_sdxl_base() -> LatentSpaceMetadata {
    LatentSpaceMetadata::new(
        LatentSpaceId::new("stable_diffusion/sdxl/base"),
        4,
        8,
        TensorDType::F32,
        TensorLayout::Nchw,
    )
}

/// Validate that `width` and `height` are divisible by the latent
/// space's spatial scale factor.
///
/// Returns a [`LatentSpaceError`] suitable for the
/// `latent.create_empty` request boundary when the request's latent
/// space disagrees with the requested pixel dimensions.
pub fn validate_pixel_dimensions_against(
    width: u32,
    height: u32,
    metadata: &LatentSpaceMetadata,
) -> Result<(), LatentSpaceError> {
    let scale = metadata.spatial_scale_factor();
    if width == 0 {
        return Err(LatentSpaceError::InvalidDimensions {
            axis: "width",
            value: width,
            reason: "latent width must be positive",
        });
    }
    if height == 0 {
        return Err(LatentSpaceError::InvalidDimensions {
            axis: "height",
            value: height,
            reason: "latent height must be positive",
        });
    }
    if scale == 0 {
        return Err(LatentSpaceError::InvalidDimensions {
            axis: "spatial_scale_factor",
            value: scale,
            reason: "latent spatial_scale_factor must be positive",
        });
    }
    if !width.is_multiple_of(scale) {
        return Err(LatentSpaceError::ScaleMismatch {
            axis: "width",
            value: width,
            scale,
        });
    }
    if !height.is_multiple_of(scale) {
        return Err(LatentSpaceError::ScaleMismatch {
            axis: "height",
            value: height,
            scale,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdxl_base_records_match_documented_shape() {
        let m = stable_diffusion_sdxl_base();
        assert_eq!(m.id().as_str(), "stable_diffusion/sdxl/base");
        assert_eq!(m.channels(), 4);
        assert_eq!(m.spatial_scale_factor(), 8);
        assert_eq!(m.dtype(), TensorDType::F32);
        assert_eq!(m.layout(), TensorLayout::Nchw);
    }

    #[test]
    fn sdxl_base_helper_matches_constant() {
        assert_eq!(
            LatentSpaceMetadata::sdxl_base(),
            stable_diffusion_sdxl_base()
        );
    }

    #[test]
    fn metadata_equality_compares_all_fields() {
        let a = stable_diffusion_sdxl_base();
        let b = stable_diffusion_sdxl_base();
        assert!(a.is_compatible(&b));

        let c = LatentSpaceMetadata::new(
            LatentSpaceId::new("stable_diffusion/sdxl/refiner"),
            4,
            8,
            TensorDType::F32,
            TensorLayout::Nchw,
        );
        assert!(!a.is_compatible(&c));
    }

    #[test]
    fn metadata_equality_is_id_strict() {
        let a = stable_diffusion_sdxl_base();
        let b = LatentSpaceMetadata::new(
            LatentSpaceId::new("stable_diffusion/sdxl/base/v2"),
            4,
            8,
            TensorDType::F32,
            TensorLayout::Nchw,
        );
        assert_ne!(a, b);
        assert!(!a.is_compatible(&b));
    }

    #[test]
    fn layout_default_is_nchw() {
        assert_eq!(TensorLayout::default(), TensorLayout::Nchw);
        assert_eq!(TensorLayout::Nchw.as_str(), "nchw");
    }

    #[test]
    fn layout_other_round_trips_str() {
        assert_eq!(
            TensorLayout::Other("packed_2x2".to_string()).as_str(),
            "packed_2x2"
        );
    }

    #[test]
    fn latent_space_id_display_and_as_ref() {
        let id = LatentSpaceId::new("stable_diffusion/sdxl/base");
        assert_eq!(id.as_str(), "stable_diffusion/sdxl/base");
        assert_eq!(id.as_ref(), "stable_diffusion/sdxl/base");
        assert_eq!(format!("{id}"), "stable_diffusion/sdxl/base");
    }

    #[test]
    fn validate_pixel_dimensions_accepts_sdxl_base_scale() {
        let m = stable_diffusion_sdxl_base();
        assert!(validate_pixel_dimensions_against(64, 64, &m).is_ok());
        assert!(validate_pixel_dimensions_against(1024, 1024, &m).is_ok());
    }

    #[test]
    fn validate_pixel_dimensions_rejects_non_multiple() {
        let m = stable_diffusion_sdxl_base();
        let err = validate_pixel_dimensions_against(63, 64, &m).unwrap_err();
        match err {
            LatentSpaceError::ScaleMismatch { axis, value, scale } => {
                assert_eq!(axis, "width");
                assert_eq!(value, 63);
                assert_eq!(scale, 8);
            }
            other => panic!("expected ScaleMismatch, got {other:?}"),
        }
    }

    #[test]
    fn validate_pixel_dimensions_rejects_zero() {
        let m = stable_diffusion_sdxl_base();
        let err = validate_pixel_dimensions_against(0, 64, &m).unwrap_err();
        assert!(matches!(err, LatentSpaceError::InvalidDimensions { .. }));
    }

    #[test]
    fn validate_pixel_dimensions_rejects_zero_scale() {
        let m = LatentSpaceMetadata::new(
            LatentSpaceId::new("invalid/zero-scale"),
            4,
            0,
            TensorDType::F32,
            TensorLayout::Nchw,
        );
        let err = validate_pixel_dimensions_against(64, 64, &m).unwrap_err();
        assert!(matches!(
            err,
            LatentSpaceError::InvalidDimensions {
                axis: "spatial_scale_factor",
                value: 0,
                ..
            }
        ));
    }
}
