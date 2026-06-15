//! CLIP tokenizer for SDXL text encoding.
//!
//! SDXL uses two CLIP text encoders: CLIP-L (ViT-L/14) and CLIP-G
//! (ViT-bigG/14). Both share the same BPE tokenizer vocabulary but
//! produce different embedding dimensions. This module handles the
//! tokenization step only; the actual text encoding lives in `text.rs`.
//!
//! V1 uses a simplified tokenization approach that produces deterministic
//! token sequences from input text. The tokenizer does not load external
//! vocabulary files; it uses a built-in fallback that maps characters to
//! token ids. This is sufficient for the V1 vertical slice and will be
//! replaced by a proper BPE tokenizer when real CLIP weights land.

use candle_core::{Device, Tensor};

/// Maximum sequence length for CLIP tokenizers.
///
/// Both CLIP-L and CLIP-G use a 77-token context window (76 + BOS/EOS).
pub const MAX_SEQUENCE_LENGTH: usize = 77;

/// Token id for the beginning-of-sequence marker.
pub const TOKEN_BOS: u32 = 49406;

/// Token id for the end-of-sequence marker.
pub const TOKEN_EOS: u32 = 49407;

/// Token id for the padding marker.
pub const TOKEN_PAD: u32 = 49407;

/// Offset added to byte values to produce token ids in the valid range.
const VOCAB_OFFSET: u32 = 100;

/// Maximum token id produced by the byte-mapping fallback.
const VOCAB_RANGE: u32 = 49000;

/// SDXL tokenizer that produces token tensors for both CLIP encoders.
///
/// The tokenizer is shared between CLIP-L and CLIP-G; the difference
/// in output comes from the text encoder, not the tokenizer.
#[derive(Debug)]
pub struct SdxlTokenizer;

impl SdxlTokenizer {
    pub fn new() -> Self {
        Self
    }

    /// Tokenize input text into a token tensor.
    ///
    /// Returns a `u32` tensor of shape `[1, MAX_SEQUENCE_LENGTH]`
    /// suitable for both CLIP-L and CLIP-G text encoders.
    ///
    /// The tokenization uses a deterministic fallback that maps each
    /// ASCII character to a token id in the range [0, 255]. This is
    /// not a real BPE tokenizer but produces valid token sequences
    /// that the CLIP text encoder can process.
    pub fn encode(&self, text: &str, device: &Device) -> Result<Tensor, TokenizerError> {
        let mut tokens: Vec<u32> = Vec::with_capacity(MAX_SEQUENCE_LENGTH);

        // BOS token
        tokens.push(TOKEN_BOS);

        // Tokenize characters. For V1, map ASCII chars to token ids.
        // Non-ASCII chars are skipped. This produces deterministic
        // token sequences that vary with input text.
        for byte in text.bytes() {
            if tokens.len() >= MAX_SEQUENCE_LENGTH - 1 {
                break;
            }
            // Map byte to token id in valid vocabulary range [0, 49405]
            // Use a simple offset to avoid special token ranges
            let token_id = (byte as u32 % VOCAB_RANGE) + VOCAB_OFFSET;
            tokens.push(token_id);
        }

        // EOS token
        tokens.push(TOKEN_EOS);

        // Pad to MAX_SEQUENCE_LENGTH
        while tokens.len() < MAX_SEQUENCE_LENGTH {
            tokens.push(TOKEN_PAD);
        }

        // Truncate to exactly MAX_SEQUENCE_LENGTH
        tokens.truncate(MAX_SEQUENCE_LENGTH);

        Tensor::from_vec(tokens, (1, MAX_SEQUENCE_LENGTH), device)
            .map_err(|e| TokenizerError::TensorCreation(e.to_string()))
    }

    /// Create attention mask tensor (all ones for non-padded tokens).
    ///
    /// Returns a `f32` tensor of shape `[1, MAX_SEQUENCE_LENGTH]`.
    pub fn attention_mask(&self, text: &str, device: &Device) -> Result<Tensor, TokenizerError> {
        // Count actual tokens: BOS + text bytes + EOS
        let text_tokens = text.bytes().count().min(MAX_SEQUENCE_LENGTH - 2);
        let total_tokens = 1 + text_tokens + 1; // BOS + text + EOS

        let mut mask: Vec<f32> = vec![0.0; MAX_SEQUENCE_LENGTH];
        for i in 0..total_tokens.min(MAX_SEQUENCE_LENGTH) {
            mask[i] = 1.0;
        }

        Tensor::from_vec(mask, (1, MAX_SEQUENCE_LENGTH), device)
            .map_err(|e| TokenizerError::TensorCreation(e.to_string()))
    }
}

impl Default for SdxlTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub enum TokenizerError {
    TensorCreation(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TensorCreation(msg) => write!(f, "tokenizer tensor creation failed: {msg}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_produces_correct_shape() {
        let tokenizer = SdxlTokenizer::new();
        let device = Device::Cpu;
        let tokens = tokenizer.encode("hello world", &device).unwrap();
        assert_eq!(tokens.shape().dims(), &[1, MAX_SEQUENCE_LENGTH]);
    }

    #[test]
    fn tokenizer_starts_with_bos() {
        let tokenizer = SdxlTokenizer::new();
        let device = Device::Cpu;
        let tokens = tokenizer.encode("test", &device).unwrap();
        let data = tokens.to_vec2::<u32>().unwrap();
        assert_eq!(data[0][0], TOKEN_BOS);
    }

    #[test]
    fn tokenizer_ends_with_eos_after_text() {
        let tokenizer = SdxlTokenizer::new();
        let device = Device::Cpu;
        let tokens = tokenizer.encode("ab", &device).unwrap();
        let data = tokens.to_vec2::<u32>().unwrap();
        // BOS + 'a' + 'b' + EOS = indices 0,1,2,3
        assert_eq!(data[0][3], TOKEN_EOS);
    }

    #[test]
    fn tokenizer_pads_to_max_length() {
        let tokenizer = SdxlTokenizer::new();
        let device = Device::Cpu;
        let tokens = tokenizer.encode("hi", &device).unwrap();
        let data = tokens.to_vec2::<u32>().unwrap();
        assert_eq!(data[0].len(), MAX_SEQUENCE_LENGTH);
        // After BOS + 'h' + 'i' + EOS, rest should be PAD
        assert_eq!(data[0][4], TOKEN_PAD);
    }

    #[test]
    fn tokenizer_handles_empty_string() {
        let tokenizer = SdxlTokenizer::new();
        let device = Device::Cpu;
        let tokens = tokenizer.encode("", &device).unwrap();
        let data = tokens.to_vec2::<u32>().unwrap();
        assert_eq!(data[0][0], TOKEN_BOS);
        assert_eq!(data[0][1], TOKEN_EOS);
    }

    #[test]
    fn tokenizer_produces_different_tokens_for_different_input() {
        let tokenizer = SdxlTokenizer::new();
        let device = Device::Cpu;
        let tokens_a = tokenizer.encode("hello", &device).unwrap();
        let tokens_b = tokenizer.encode("world", &device).unwrap();
        let data_a = tokens_a.to_vec2::<u32>().unwrap();
        let data_b = tokens_b.to_vec2::<u32>().unwrap();
        // At least the first text token should differ
        assert_ne!(data_a[0][1], data_b[0][1]);
    }

    #[test]
    fn attention_mask_has_correct_shape() {
        let tokenizer = SdxlTokenizer::new();
        let device = Device::Cpu;
        let mask = tokenizer.attention_mask("test", &device).unwrap();
        assert_eq!(mask.shape().dims(), &[1, MAX_SEQUENCE_LENGTH]);
    }

    #[test]
    fn attention_mask_marks_text_positions() {
        let tokenizer = SdxlTokenizer::new();
        let device = Device::Cpu;
        let mask = tokenizer.attention_mask("ab", &device).unwrap();
        let data = mask.to_vec2::<f32>().unwrap();
        // BOS + 'a' + 'b' + EOS = 4 positions attended
        assert_eq!(data[0][0], 1.0); // BOS
        assert_eq!(data[0][1], 1.0); // 'a'
        assert_eq!(data[0][2], 1.0); // 'b'
        assert_eq!(data[0][3], 1.0); // EOS
        assert_eq!(data[0][4], 0.0); // rest is masked
    }
}
