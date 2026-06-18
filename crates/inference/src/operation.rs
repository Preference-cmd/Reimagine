//! V1 operation id constants.
//!
//! Re-exports the V1 constants from `reimagine-inference-core` so
//! existing call sites under `reimagine_inference::operation::OP_X`
//! keep working during the V1 transition window. The canonical
//! home of the V1 enum and the constants is
//! `reimagine_inference_core::request`; new code should prefer
//! that path.

pub use reimagine_inference_core::{
    ALL_V1_OPERATIONS, OP_DIFFUSION_SAMPLE, OP_IMAGE_PREVIEW, OP_IMAGE_SAVE,
    OP_LATENT_CREATE_EMPTY, OP_LATENT_DECODE, OP_MODEL_LOAD_BUNDLE, OP_TEXT_ENCODE,
};
