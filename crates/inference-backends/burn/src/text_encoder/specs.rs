//! Required tensor spec generation from a [`ClipTextEncoderProfile`].
//!
//! The burner generates an owned spec collection that covers every
//! tensor the Burn `#[derive(Module)]` loader will try to deserialize
//! — embeddings, norms, attention projections, and MLP blocks — so
//! the burn/03 contract validation can reject missing keys before
//! loading.

use std::collections::BTreeSet;

use super::clip::ClipTextEncoderProfile;
use super::keyspace::TextEncoderKeyspace;
use crate::models::stable_diffusion::sdxl::BurnTensorShapeSpec;

/// A single required tensor spec with an owned key. Unlike the
/// `&'static str` variant in burn/03's [`BurnTensorSpec`], this type
/// owns the key so generated transformer-block specs are possible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedTensorSpec {
    pub key: String,
    pub shape: BurnTensorShapeSpec,
    pub required: bool,
    pub notes: String,
}

/// An owned collection of required tensor specs for a text-encoder
/// component. Produced by [`text_encoder_spec_set`].
#[derive(Debug, Clone)]
pub struct TextEncoderSpecSet {
    pub specs: Vec<OwnedTensorSpec>,
}

impl TextEncoderSpecSet {
    /// Check whether every key in `available` matches a required
    /// spec. Returns the list of missing required keys and the list
    /// of unexpected (unmatched) keys.
    pub fn classify(&self, available: &BTreeSet<String>) -> (Vec<String>, Vec<String>) {
        let required_keys: BTreeSet<&str> = self
            .specs
            .iter()
            .filter(|s| s.required)
            .map(|s| s.key.as_str())
            .collect();
        let available_keys: BTreeSet<&str> = available.iter().map(|s| s.as_str()).collect();

        let missing: Vec<String> = required_keys
            .difference(&available_keys)
            .map(|s| s.to_string())
            .collect();
        let unexpected: Vec<String> = available_keys
            .difference(&required_keys)
            .map(|s| s.to_string())
            .collect();
        (missing, unexpected)
    }

    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    pub fn len(&self) -> usize {
        self.specs.len()
    }
}

/// Build a [`TextEncoderSpecSet`] from a profile. Iterates all
/// transformer layers and appends a `BurnTensorShapeSpec::Rank(n)`
/// for each tensor.
pub fn text_encoder_spec_set(profile: &ClipTextEncoderProfile) -> TextEncoderSpecSet {
    let keys = TextEncoderKeyspace::new(profile);
    let mut specs = Vec::new();

    // Token + position embeddings (always 2D).
    push_spec(
        &mut specs,
        keys.token_embedding(),
        2,
        true,
        format!("{} token embedding weight", profile.variant),
    );
    push_spec(
        &mut specs,
        keys.position_embedding(),
        2,
        true,
        format!("{} position embedding weight", profile.variant),
    );

    // Final layer norm (1D weight + 1D bias).
    push_spec(
        &mut specs,
        keys.final_layer_norm_weight(),
        1,
        true,
        format!("{} final layer norm gamma", profile.variant),
    );
    push_spec(
        &mut specs,
        keys.final_layer_norm_bias(),
        1,
        true,
        format!("{} final layer norm beta", profile.variant),
    );

    // Optional text projection (OpenCLIP-G only).
    if let Some(w) = keys.text_projection_weight() {
        push_spec(
            &mut specs,
            w,
            2,
            true,
            "OpenCLIP-G text projection weight".into(),
        );
    }
    if let Some(b) = keys.text_projection_bias() {
        push_spec(
            &mut specs,
            b,
            1,
            profile.produces_pooled_output,
            "OpenCLIP-G text projection bias".into(),
        );
    }

    // Per-layer block specs.
    for layer in 0..profile.num_layers {
        push_spec(
            &mut specs,
            keys.attn_in_proj_weight(layer),
            2,
            true,
            format!("layer {layer} attn.in_proj weight"),
        );
        push_spec(
            &mut specs,
            keys.attn_in_proj_bias(layer),
            1,
            true,
            format!("layer {layer} attn.in_proj bias"),
        );
        push_spec(
            &mut specs,
            keys.attn_out_proj_weight(layer),
            2,
            true,
            format!("layer {layer} attn.out_proj weight"),
        );
        push_spec(
            &mut specs,
            keys.attn_out_proj_bias(layer),
            1,
            true,
            format!("layer {layer} attn.out_proj bias"),
        );
        push_spec(
            &mut specs,
            keys.ln_1_weight(layer),
            1,
            true,
            format!("layer {layer} ln_1 weight"),
        );
        push_spec(
            &mut specs,
            keys.ln_1_bias(layer),
            1,
            true,
            format!("layer {layer} ln_1 bias"),
        );
        push_spec(
            &mut specs,
            keys.ln_2_weight(layer),
            1,
            true,
            format!("layer {layer} ln_2 weight"),
        );
        push_spec(
            &mut specs,
            keys.ln_2_bias(layer),
            1,
            true,
            format!("layer {layer} ln_2 bias"),
        );
        push_spec(
            &mut specs,
            keys.mlp_fc1_weight(layer),
            2,
            true,
            format!("layer {layer} mlp.fc1 weight"),
        );
        push_spec(
            &mut specs,
            keys.mlp_fc1_bias(layer),
            1,
            true,
            format!("layer {layer} mlp.fc1 bias"),
        );
        push_spec(
            &mut specs,
            keys.mlp_fc2_weight(layer),
            2,
            true,
            format!("layer {layer} mlp.fc2 weight"),
        );
        push_spec(
            &mut specs,
            keys.mlp_fc2_bias(layer),
            1,
            true,
            format!("layer {layer} mlp.fc2 bias"),
        );
    }

    TextEncoderSpecSet { specs }
}

fn push_spec(
    specs: &mut Vec<OwnedTensorSpec>,
    key: String,
    rank: usize,
    required: bool,
    notes: String,
) {
    specs.push(OwnedTensorSpec {
        key,
        shape: BurnTensorShapeSpec::Rank(rank),
        required,
        notes,
    });
}

pub struct TextEncoderSpecSetBuilder;

impl TextEncoderSpecSetBuilder {
    /// Generate specs for an SDXL primary CLIP-L component.
    pub fn sdxl_clip_l() -> TextEncoderSpecSet {
        text_encoder_spec_set(&ClipTextEncoderProfile::sdxl_clip_l())
    }

    /// Generate specs for an SDXL secondary OpenCLIP-G component.
    pub fn sdxl_open_clip_g() -> TextEncoderSpecSet {
        text_encoder_spec_set(&ClipTextEncoderProfile::sdxl_open_clip_g())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_l_specs_cover_all_layers() {
        let profile = ClipTextEncoderProfile::sdxl_clip_l();
        let specs = text_encoder_spec_set(&profile);
        // 4 static specs (token_emb, pos_emb, fn_weight, fn_bias)
        // + 12 layers × 12 per-layer specs = 144
        assert_eq!(specs.len(), 4 + 12 * 12);
    }

    #[test]
    fn open_clip_g_specs_include_text_projection() {
        let profile = ClipTextEncoderProfile::sdxl_open_clip_g();
        let specs = text_encoder_spec_set(&profile);
        // 6 static specs (same 4 + text_projection weight+bias)
        // + 32 layers × 12 per-layer specs = 384
        assert_eq!(specs.len(), 6 + 32 * 12);
    }

    #[test]
    fn classify_reports_missing_required_key() {
        let profile = ClipTextEncoderProfile::sdxl_clip_l();
        let set = text_encoder_spec_set(&profile);
        let mut available = BTreeSet::new();
        // Add all keys that are present
        for spec in &set.specs {
            available.insert(spec.key.clone());
        }
        // Remove one required key
        let removed = available.iter().next().unwrap().clone();
        available.remove(&removed);

        let (missing, _unexpected) = set.classify(&available);
        assert!(
            missing.contains(&removed),
            "missing should contain {removed}"
        );
    }

    #[test]
    fn classify_reports_unexpected_keys() {
        let profile = ClipTextEncoderProfile::sdxl_clip_l();
        let set = text_encoder_spec_set(&profile);
        let mut available = BTreeSet::new();
        for spec in &set.specs {
            available.insert(spec.key.clone());
        }
        available.insert("unexpected.key".into());

        let (_missing, unexpected) = set.classify(&available);
        assert!(unexpected.iter().any(|k| k == "unexpected.key"));
    }
}
