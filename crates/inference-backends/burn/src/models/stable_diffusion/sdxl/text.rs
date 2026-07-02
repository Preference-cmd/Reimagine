//! Burn-private SDXL text encoder resource facade.
//!
//! This module owns the backend-private text encoder resource wiring
//! for the Burn SDXL backend. The 07a slice only resolves bundled
//! tokenizer resources and produces deterministic 77-token prompt
//! encodings. The actual Burn CLIP-L / CLIP-G forward passes, the
//! `TextEncode` capability, and executable component contracts are
//! staged in the 08a / 08b / 08d / 08f issues.
//!
//! The Burn backend crate must not advertise `TextEncode` until 08f
//! lands in the same branch.

use crate::error::BurnBackendError;

use super::tokenizer::{
    BurnSdxlTokenizedPrompt, BurnSdxlTokenizedPromptPair, BurnSdxlTokenizer,
    BurnSdxlTokenizerResources, MAX_SEQUENCE_LENGTH,
};

/// Resolve SDXL tokenizer resources for the given backend config and
/// return a configured [`BurnSdxlTokenizer`].
///
/// This is the only public entry point on the text facade for the
/// 07a slice. It does not run inference, does not touch the model
/// cache, and does not advertise `TextEncode`. Later slices will
/// introduce the executable CLIP component contract and replace this
/// loader with a bundle-owned text encoder graph.
pub fn load_sdxl_tokenizer(
    config: &crate::config::BurnBackendConfig,
) -> Result<BurnSdxlTokenizer, BurnBackendError> {
    let resources = BurnSdxlTokenizerResources::from_config(config)?;
    Ok(BurnSdxlTokenizer::from_resources(resources)?)
}

/// Bundle of a [`BurnSdxlTokenizer`] and the [`MAX_SEQUENCE_LENGTH`]
/// context constant, exposed for the 08a preflight / store boundary.
///
/// The bundle stays private to the Burn backend; it is not exposed as
/// an `ExecutionValue`, a model-manager descriptor, or a runtime
/// handle.
#[derive(Debug)]
pub struct BurnSdxlTextEncoderResources {
    tokenizer: BurnSdxlTokenizer,
}

impl BurnSdxlTextEncoderResources {
    pub fn load(config: &crate::config::BurnBackendConfig) -> Result<Self, BurnBackendError> {
        Ok(Self {
            tokenizer: load_sdxl_tokenizer(config)?,
        })
    }

    pub fn tokenizer(&self) -> &BurnSdxlTokenizer {
        &self.tokenizer
    }

    pub fn sequence_length(&self) -> usize {
        MAX_SEQUENCE_LENGTH
    }

    pub fn tokenize(&self, text: &str) -> Result<BurnSdxlTokenizedPrompt, BurnBackendError> {
        Ok(self.tokenizer.tokenize(text)?)
    }

    pub fn tokenize_pair(
        &self,
        text: &str,
    ) -> Result<BurnSdxlTokenizedPromptPair, BurnBackendError> {
        Ok(self.tokenizer.tokenize_pair(text)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::stable_diffusion::sdxl::BurnTokenizerError;

    #[test]
    fn loads_bundled_tokenizer_for_default_config() {
        let config = crate::config::BurnBackendConfig::new("/models", "/output");
        let resources = BurnSdxlTextEncoderResources::load(&config).expect("resources");
        assert_eq!(resources.sequence_length(), MAX_SEQUENCE_LENGTH);
        let prompt = resources.tokenize("hello world").expect("tokenize primary");
        assert_eq!(prompt.token_ids.len(), MAX_SEQUENCE_LENGTH);
        let pair = resources
            .tokenize_pair("hello world")
            .expect("tokenize pair");
        assert_eq!(pair.clip_l.token_ids.len(), MAX_SEQUENCE_LENGTH);
        assert_eq!(pair.clip_g.token_ids.len(), MAX_SEQUENCE_LENGTH);
    }

    #[test]
    fn respects_configured_tokenizer_root() {
        // Use a temp dir that does not contain the bundled layout —
        // loading should fail, exercising the config-driven root.
        let dir = std::env::temp_dir().join(format!(
            "reimagine-burn-text-empty-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let config = crate::config::BurnBackendConfig::new("/models", "/output")
            .with_tokenizer_root(dir.clone());
        let err = BurnSdxlTextEncoderResources::load(&config).unwrap_err();
        match err {
            BurnBackendError::Tokenizer(BurnTokenizerError::Missing { path }) => {
                assert!(path.contains("tokenizer.json"));
            }
            other => panic!("expected Tokenizer Missing, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
