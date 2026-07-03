//! Burn-private CLIP / OpenCLIP profile vocabulary.
//!
//! A [`ClipTextEncoderProfile`] is a data record that describes the shape of
//! one executable text-encoder component. It is used by the spec generator to
//! enumerate the full set of required tensors for the module, and by the
//! SDXL profile to select between CLIP-L and OpenCLIP-G.
//!
//! V1 only ships two profiles: SDXL's primary (CLIP-L) and secondary
//! (OpenCLIP-G). The shape constants follow the public SDXL reference
//! configuration:
//!
//! | Profile | Layers | Width | Heads | Inner width | Projection |
//! |---------|--------|-------|-------|-------------|------------|
//! | CLIP-L (ViT-L/14)   | 12 | 768  | 12 | 3072 | none  |
//! | OpenCLIP-G (bigG/14) | 32 | 1280 | 20 | 5120 | `text_projection` (768→1280) |
//!
//! All constants are in Burn-native naming, with the `model.text_encoder{,_2}.*`
//! prefix attached at the call site so this module is fully backend-private
//! and reusable.

use std::fmt;

/// V1 SDXL text encoder variants. The discriminator matches the
/// `BurnSdxlComponentRole::TextEncoder` / `TextEncoder2` split, but the
/// underlying profile is per-component (CLIP-L vs OpenCLIP-G).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClipTextEncoderVariant {
    /// ViT-L/14 CLIP text encoder used as the primary text encoder.
    ClipL,
    /// ViT-bigG/14 OpenCLIP text encoder used as the secondary SDXL
    /// text encoder. Produces both hidden and pooled outputs.
    OpenClipG,
}

impl ClipTextEncoderVariant {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ClipL => "clip_l",
            Self::OpenClipG => "open_clip_g",
        }
    }
}

impl fmt::Display for ClipTextEncoderVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Number of attention heads per CLIP profile. The Burn module forward
/// path does not need heads as a structural field, but spec generation
/// uses it to label the attention sub-modules consistently.
const CLIP_L_HEADS: u32 = 12;
const OPEN_CLIP_G_HEADS: u32 = 20;

/// Data record describing the executable shape of one text-encoder
/// component. `target_prefix` is the component-local Burn key prefix
/// (e.g. `"model.text_encoder"`) — the SDXL facade populates it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipTextEncoderProfile {
    pub variant: ClipTextEncoderVariant,
    pub target_prefix: String,
    pub num_layers: u32,
    pub width: u32,
    pub heads: u32,
    pub inner_width: u32,
    pub vocab_size: u32,
    pub sequence_length: u32,
    /// Whether this profile produces a separate pooled output tensor
    /// (OpenCLIP-G does; CLIP-L does not).
    pub produces_pooled_output: bool,
}

impl ClipTextEncoderProfile {
    /// Build a CLIP-L profile with no target prefix. Callers
    /// typically chain `with_target_prefix` immediately.
    pub const fn clip_l() -> Self {
        Self {
            variant: ClipTextEncoderVariant::ClipL,
            target_prefix: String::new(),
            num_layers: 12,
            width: 768,
            heads: CLIP_L_HEADS,
            inner_width: 3072,
            vocab_size: 49408,
            sequence_length: 77,
            produces_pooled_output: false,
        }
    }

    /// Build an OpenCLIP-G profile with no target prefix.
    pub const fn open_clip_g() -> Self {
        Self {
            variant: ClipTextEncoderVariant::OpenClipG,
            target_prefix: String::new(),
            num_layers: 32,
            width: 1280,
            heads: OPEN_CLIP_G_HEADS,
            inner_width: 5120,
            vocab_size: 49408,
            sequence_length: 77,
            produces_pooled_output: true,
        }
    }

    /// Consume the profile and return one with the target prefix
    /// set. Used by the SDXL facade.
    pub fn with_target_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.target_prefix = prefix.into();
        self
    }

    /// Convenience constructor for the SDXL primary profile bound
    /// to the canonical `model.text_encoder` prefix.
    pub fn sdxl_clip_l() -> Self {
        Self::clip_l().with_target_prefix("model.text_encoder")
    }

    /// Convenience constructor for the SDXL secondary profile
    /// bound to the canonical `model.text_encoder_2` prefix.
    pub fn sdxl_open_clip_g() -> Self {
        Self::open_clip_g().with_target_prefix("model.text_encoder_2")
    }

    pub fn block_prefix(&self, layer: u32) -> String {
        format!("{}.transformer.resblocks.{}", self.target_prefix, layer)
    }

    pub fn attn_in_proj_weight_key(&self, layer: u32) -> String {
        format!("{}.attn.in_proj_weight", self.block_prefix(layer))
    }

    pub fn attn_in_proj_bias_key(&self, layer: u32) -> String {
        format!("{}.attn.in_proj_bias", self.block_prefix(layer))
    }

    pub fn attn_out_proj_weight_key(&self, layer: u32) -> String {
        format!("{}.attn.out_proj.weight", self.block_prefix(layer))
    }

    pub fn attn_out_proj_bias_key(&self, layer: u32) -> String {
        format!("{}.attn.out_proj.bias", self.block_prefix(layer))
    }

    pub fn ln_1_weight_key(&self, layer: u32) -> String {
        format!("{}.ln_1.weight", self.block_prefix(layer))
    }

    pub fn ln_1_bias_key(&self, layer: u32) -> String {
        format!("{}.ln_1.bias", self.block_prefix(layer))
    }

    pub fn ln_2_weight_key(&self, layer: u32) -> String {
        format!("{}.ln_2.weight", self.block_prefix(layer))
    }

    pub fn ln_2_bias_key(&self, layer: u32) -> String {
        format!("{}.ln_2.bias", self.block_prefix(layer))
    }

    pub fn mlp_fc1_weight_key(&self, layer: u32) -> String {
        format!("{}.mlp.fc1.weight", self.block_prefix(layer))
    }

    pub fn mlp_fc1_bias_key(&self, layer: u32) -> String {
        format!("{}.mlp.fc1.bias", self.block_prefix(layer))
    }

    pub fn mlp_fc2_weight_key(&self, layer: u32) -> String {
        format!("{}.mlp.fc2.weight", self.block_prefix(layer))
    }

    pub fn mlp_fc2_bias_key(&self, layer: u32) -> String {
        format!("{}.mlp.fc2.bias", self.block_prefix(layer))
    }

    pub fn token_embedding_key(&self) -> String {
        format!("{}.token_embedding.weight", self.target_prefix)
    }

    pub fn position_embedding_key(&self) -> String {
        format!("{}.position_embedding.weight", self.target_prefix)
    }

    pub fn final_layer_norm_weight_key(&self) -> String {
        format!("{}.final_layer_norm.gamma", self.target_prefix)
    }

    pub fn final_layer_norm_bias_key(&self) -> String {
        format!("{}.final_layer_norm.beta", self.target_prefix)
    }

    /// OpenCLIP-G only: the text-projection matrix that maps the
    /// final hidden state into the pooled embedding space. CLIP-L
    /// does not need this because the pooled output is taken from
    /// the EOS hidden state directly.
    pub fn text_projection_weight_key(&self) -> Option<String> {
        if self.produces_pooled_output {
            Some(format!("{}.text_projection.weight", self.target_prefix))
        } else {
            None
        }
    }

    pub fn text_projection_bias_key(&self) -> Option<String> {
        if self.produces_pooled_output {
            Some(format!("{}.text_projection.bias", self.target_prefix))
        } else {
            None
        }
    }
}
