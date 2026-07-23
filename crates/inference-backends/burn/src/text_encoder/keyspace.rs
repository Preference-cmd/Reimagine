//! Deterministic Burn-native key builders for text-encoder tensor
//! families. Each family groups related tensors (e.g. all weights/bias
//! pairs for a single MLP block) and provides a method that emits the
//! concrete key(s) for a given block index.

use std::fmt;

/// Backend-private tensor family vocabulary for executable text
/// encoder components. Each variant maps to a set of Burn-native
/// target keys produced by the matching [`TextEncoderKeyspace`]
/// method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextEncoderTensorFamily {
    TokenEmbedding,
    PositionEmbedding,
    FinalLayerNorm,
    TextProjection,
    AttentionInProjection,
    AttentionOutProjection,
    LayerNorm1,
    LayerNorm2,
    MlpFc1,
    MlpFc2,
}

impl TextEncoderTensorFamily {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TokenEmbedding => "token_embedding",
            Self::PositionEmbedding => "position_embedding",
            Self::FinalLayerNorm => "final_layer_norm",
            Self::TextProjection => "text_projection",
            Self::AttentionInProjection => "attn.in_proj",
            Self::AttentionOutProjection => "attn.out_proj",
            Self::LayerNorm1 => "ln_1",
            Self::LayerNorm2 => "ln_2",
            Self::MlpFc1 => "mlp.fc1",
            Self::MlpFc2 => "mlp.fc2",
        }
    }
}

impl fmt::Display for TextEncoderTensorFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Deterministic key builder for one text-encoder component. The
/// caller supplies a [`ClipTextEncoderProfile`] to stamp out all
/// keys for that component.
///
/// ```ignore
/// let keys = TextEncoderKeyspace::new(&profile);
/// let in_proj_weight = keys.attn_in_proj_weight(3); // layer 3
/// ```
pub struct TextEncoderKeyspace<'a> {
    profile: &'a super::clip::ClipTextEncoderProfile,
}

impl<'a> TextEncoderKeyspace<'a> {
    pub fn new(profile: &'a super::clip::ClipTextEncoderProfile) -> Self {
        Self { profile }
    }

    // ── Non-block keys ──────────────────────────────────────────

    pub fn token_embedding(&self) -> String {
        self.profile.token_embedding_key()
    }

    pub fn position_embedding(&self) -> String {
        self.profile.position_embedding_key()
    }

    pub fn final_layer_norm_weight(&self) -> String {
        self.profile.final_layer_norm_weight_key()
    }

    pub fn final_layer_norm_bias(&self) -> String {
        self.profile.final_layer_norm_bias_key()
    }

    pub fn text_projection_weight(&self) -> Option<String> {
        self.profile.text_projection_weight_key()
    }

    pub fn text_projection_bias(&self) -> Option<String> {
        self.profile.text_projection_bias_key()
    }

    // ── Per-block keys ───────────────────────────────────────────

    pub fn attn_in_proj_weight(&self, layer: u32) -> String {
        self.profile.attn_in_proj_weight_key(layer)
    }

    pub fn attn_in_proj_bias(&self, layer: u32) -> String {
        self.profile.attn_in_proj_bias_key(layer)
    }

    pub fn attn_out_proj_weight(&self, layer: u32) -> String {
        self.profile.attn_out_proj_weight_key(layer)
    }

    pub fn attn_out_proj_bias(&self, layer: u32) -> String {
        self.profile.attn_out_proj_bias_key(layer)
    }

    pub fn ln_1_weight(&self, layer: u32) -> String {
        self.profile.ln_1_weight_key(layer)
    }

    pub fn ln_1_bias(&self, layer: u32) -> String {
        self.profile.ln_1_bias_key(layer)
    }

    pub fn ln_2_weight(&self, layer: u32) -> String {
        self.profile.ln_2_weight_key(layer)
    }

    pub fn ln_2_bias(&self, layer: u32) -> String {
        self.profile.ln_2_bias_key(layer)
    }

    pub fn mlp_fc1_weight(&self, layer: u32) -> String {
        self.profile.mlp_fc1_weight_key(layer)
    }

    pub fn mlp_fc1_bias(&self, layer: u32) -> String {
        self.profile.mlp_fc1_bias_key(layer)
    }

    pub fn mlp_fc2_weight(&self, layer: u32) -> String {
        self.profile.mlp_fc2_weight_key(layer)
    }

    pub fn mlp_fc2_bias(&self, layer: u32) -> String {
        self.profile.mlp_fc2_bias_key(layer)
    }
}
