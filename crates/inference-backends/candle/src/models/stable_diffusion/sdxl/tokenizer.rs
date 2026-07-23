//! SDXL CLIP tokenizer backed by the `tokenizers` crate.
//!
//! SDXL uses two CLIP text encoders: CLIP-L (ViT-L/14) and CLIP-G
//! (ViT-bigG/14). Both share the same BPE tokenizer vocabulary but
//! produce different embedding dimensions. This module handles only
//! the tokenization step; the actual text encoding lives in `text.rs`.
//!
//! The tokenizer outputs raw `Vec<u32>` / `Vec<f32>` buffers.
//! Tensor construction happens in `text.rs`.

use std::path::{Path, PathBuf};

use reimagine_core::model::ModelRole;
use reimagine_inference::ResolvedInferenceModelSourceSet;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Context length for CLIP tokenizers (77 = 76 + BOS).
pub const MAX_SEQUENCE_LENGTH: usize = 77;

/// Token id for the beginning-of-sequence marker.
pub const TOKEN_BOS: u32 = 49406;

/// Token id for the end-of-sequence marker.
pub const TOKEN_EOS: u32 = 49407;

/// Token id for the padding marker (same as EOS per CLIP convention).
pub const TOKEN_PAD: u32 = 49407;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum TokenizerError {
    /// Tokenizer resource not found at the expected path.
    Missing { path: String },
    /// Tokenizer resource found but could not be loaded (malformed JSON,
    /// incompatible format, I/O error, etc.).
    LoadFailed { path: String, reason: String },
    /// The tokenizers library rejected the input during encoding.
    TokenizationFailed { reason: String },
    /// The model family is not supported by the bundled fallback
    /// (no default tokenizer asset available).
    UnsupportedModelFamily { series: String, variant: String },
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing { path } => write!(f, "tokenizer resource not found: {path}"),
            Self::LoadFailed { path, reason } => {
                write!(f, "failed to load tokenizer at {path}: {reason}")
            }
            Self::TokenizationFailed { reason } => write!(f, "tokenization failed: {reason}"),
            Self::UnsupportedModelFamily { series, variant } => {
                write!(
                    f,
                    "no bundled tokenizer for model family {series}/{variant}"
                )
            }
        }
    }
}

impl std::error::Error for TokenizerError {}

// ---------------------------------------------------------------------------
// Resource resolver
// ---------------------------------------------------------------------------

/// Backend-private resolver for SDXL tokenizer resources.
///
/// Resolution order:
/// 1. Explicit source-set entry with [`ModelRole::TextEncoder`] and
///    tokenizer metadata.
/// 2. Sidecar files next to the checkpoint.
/// 3. Bundled default under `assets/tokenizers/stable_diffusion/sdxl/`.
pub struct SdxlTokenizerResources;

impl SdxlTokenizerResources {
    /// Resolve the primary tokenizer path for CLIP-L.
    pub fn resolve_tokenizer_path(
        source_set: &ResolvedInferenceModelSourceSet,
        source_path: &Path,
    ) -> Result<PathBuf, TokenizerError> {
        // 1. Explicit source-set entry with TextEncoder role.
        for source in source_set.sources() {
            if source.role() == ModelRole::TextEncoder
                && let Some(meta) = source.metadata().and_then(primary_tokenizer_metadata_path)
            {
                // Explicit entry exists — must resolve or error.
                let p = Path::new(meta);
                if p.is_file() {
                    return Ok(p.to_path_buf());
                }
                let inner = p.join("tokenizer.json");
                if inner.exists() {
                    return Ok(inner);
                }
                return Err(TokenizerError::Missing {
                    path: meta.to_string(),
                });
            }
        }

        // 2. Sidecar files next to the checkpoint.
        let parent = source_path.parent().unwrap_or(Path::new("."));
        if let Some(p) = Self::try_sidecar_tokenizer(parent) {
            return Ok(p);
        }

        // 3. Bundled default.
        let bundled = Self::bundled_tokenizer_dir().join("tokenizer/tokenizer.json");
        if bundled.exists() {
            return Ok(bundled);
        }

        Err(TokenizerError::Missing {
            path: source_path.display().to_string(),
        })
    }

    /// Resolve the secondary tokenizer path for CLIP-G.
    pub fn resolve_tokenizer_2_path(
        source_set: &ResolvedInferenceModelSourceSet,
        source_path: &Path,
    ) -> Result<PathBuf, TokenizerError> {
        // 1. Explicit source-set entry with TextEncoder role (separate
        //    metadata convention: "tokenizer_2" key).
        for source in source_set.sources() {
            if source.role() == ModelRole::TextEncoder
                && let Some(meta) = source
                    .metadata()
                    .and_then(secondary_tokenizer_metadata_path)
                    .or_else(|| source.metadata().and_then(primary_tokenizer_metadata_path))
            {
                // Explicit entry exists — must resolve or error.
                let meta_path = Path::new(meta);
                let candidate = if meta_path.is_dir() {
                    meta_path.join("tokenizer_2/tokenizer.json")
                } else {
                    meta_path.to_path_buf()
                };
                if candidate.exists() {
                    return Ok(candidate);
                }
                return Err(TokenizerError::Missing {
                    path: meta.to_string(),
                });
            }
        }

        // 2. Sidecar files next to the checkpoint (tokenizer_2 directory).
        let parent = source_path.parent().unwrap_or(Path::new("."));
        let sidecar = parent.join("tokenizer_2/tokenizer.json");
        if sidecar.exists() {
            return Ok(sidecar);
        }

        // Also check the regular sidecar tokenizer (CLIP-G may share it).
        if let Some(p) = Self::try_sidecar_tokenizer(parent) {
            return Ok(p);
        }

        // 3. Bundled default.
        let bundled = Self::bundled_tokenizer_dir().join("tokenizer_2/tokenizer.json");
        if bundled.exists() {
            return Ok(bundled);
        }

        Err(TokenizerError::Missing {
            path: source_path.display().to_string(),
        })
    }

    /// Returns the path to the bundled tokenizer_2 asset.
    #[cfg(test)]
    pub fn bundled_tokenizer_2_path() -> PathBuf {
        Self::bundled_tokenizer_dir().join("tokenizer_2/tokenizer.json")
    }

    // -- internal helpers ---------------------------------------------------

    fn bundled_tokenizer_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .join("assets/tokenizers/stable_diffusion/sdxl")
    }

    /// Scan the checkpoint directory for known sidecar layouts.
    fn try_sidecar_tokenizer(parent: &Path) -> Option<PathBuf> {
        // tokenizer.json (single file)
        let f = parent.join("tokenizer.json");
        if f.exists() {
            return Some(f);
        }
        // tokenizer/tokenizer.json (directory layout)
        let d = parent.join("tokenizer/tokenizer.json");
        if d.exists() {
            return Some(d);
        }
        // vocab.json + merges.txt (two-file BPE) — return the vocab path;
        // the tokenizer loader handles the two-file case.
        let vocab = parent.join("vocab.json");
        let merges = parent.join("merges.txt");
        if vocab.exists() && merges.exists() {
            return Some(vocab);
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Tokenized prompt output
// ---------------------------------------------------------------------------

/// Output of tokenization, suitable for CLIP text encoder consumption.
#[derive(Debug, Clone)]
pub struct SdxlTokenizedPrompt {
    /// Token ids, length [`MAX_SEQUENCE_LENGTH`].
    pub token_ids: Vec<u32>,
    /// Attention mask, length [`MAX_SEQUENCE_LENGTH`] (1.0 for attended,
    /// 0.0 for padded).
    pub attention_mask: Vec<f32>,
}

/// Tokenization output for SDXL's two CLIP text encoders.
#[derive(Debug, Clone)]
pub struct SdxlTokenizedPromptPair {
    pub clip_l: SdxlTokenizedPrompt,
    pub clip_g: SdxlTokenizedPrompt,
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// A real SDXL tokenizer backed by the `tokenizers` crate.
///
/// Holds two independent [`tokenizers::Tokenizer`] instances:
/// - `tokenizer`    — used by CLIP-L
/// - `tokenizer_2`  — used by CLIP-G (same vocabulary file)
#[derive(Debug)]
pub struct SdxlTokenizer {
    tokenizer: tokenizers::Tokenizer,
    tokenizer_2: tokenizers::Tokenizer,
}

impl SdxlTokenizer {
    /// Create a tokenizer using the bundled default assets (SDXL).
    pub fn from_bundled() -> Result<Self, TokenizerError> {
        Self::from_bundled_for_family("sdxl")
    }

    /// Create a tokenizer from bundled assets for the given model family.
    ///
    /// Currently only `"sdxl"` is supported. Any other family returns
    /// [`TokenizerError::UnsupportedModelFamily`].
    pub fn from_bundled_for_family(family: &str) -> Result<Self, TokenizerError> {
        match family.to_lowercase().as_str() {
            "sdxl" => {
                let dir = SdxlTokenizerResources::bundled_tokenizer_dir();
                let p = dir.join("tokenizer/tokenizer.json");
                let p2 = dir.join("tokenizer_2/tokenizer.json");
                Self::from_paths(&p, &p2)
            }
            other => Err(TokenizerError::UnsupportedModelFamily {
                series: other.to_string(),
                variant: "unknown".to_string(),
            }),
        }
    }

    /// Create a tokenizer by loading two tokenizer files from the given
    /// paths.
    ///
    /// Each path may be:
    /// - A `tokenizer.json` file → loaded directly.
    /// - A directory containing `tokenizer.json` → loaded from inside.
    /// - A `vocab.json` file with an adjacent `merges.txt` → BPE model
    ///   constructed from the pair.
    pub fn from_paths(
        tokenizer_path: &Path,
        tokenizer_2_path: &Path,
    ) -> Result<Self, TokenizerError> {
        let tok = load_one(tokenizer_path)?;
        let tok2 = load_one(tokenizer_2_path)?;
        Ok(Self {
            tokenizer: tok,
            tokenizer_2: tok2,
        })
    }

    /// Create a tokenizer using the resolve-then-load pipeline.
    pub fn from_source(
        source_set: &ResolvedInferenceModelSourceSet,
        source_path: &Path,
    ) -> Result<Self, TokenizerError> {
        let p = SdxlTokenizerResources::resolve_tokenizer_path(source_set, source_path)?;
        let p2 = SdxlTokenizerResources::resolve_tokenizer_2_path(source_set, source_path)?;
        Self::from_paths(&p, &p2)
    }

    /// Tokenize text using the primary CLIP-L tokenizer.
    ///
    /// Returns an [`SdxlTokenizedPrompt`] with token ids and attention
    /// mask, both exactly [`MAX_SEQUENCE_LENGTH`] elements long.
    ///
    /// The sequence layout is:
    /// ```text
    /// [BOS] [token_1] ... [token_n] [EOS] [PAD] ... [PAD]
    /// ```
    pub fn tokenize(&self, text: &str) -> Result<SdxlTokenizedPrompt, TokenizerError> {
        encode_one(&self.tokenizer, text)
    }

    /// Tokenize text for both SDXL CLIP encoders.
    pub fn tokenize_pair(&self, text: &str) -> Result<SdxlTokenizedPromptPair, TokenizerError> {
        Ok(SdxlTokenizedPromptPair {
            clip_l: encode_one(&self.tokenizer, text)?,
            clip_g: encode_one(&self.tokenizer_2, text)?,
        })
    }

    // -- internal helpers ---------------------------------------------------

    fn configure(tok: &mut tokenizers::Tokenizer) -> Result<(), TokenizerError> {
        // Tell the tokenizer to pad to exactly MAX_SEQUENCE_LENGTH tokens
        // whenever the encoded sequence is shorter.
        let padding = tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(MAX_SEQUENCE_LENGTH),
            direction: tokenizers::PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: TOKEN_PAD,
            pad_type_id: 0,
            pad_token: String::new(),
        };
        tok.with_padding(Some(padding));

        // Truncate from the right when the encoded sequence exceeds
        // the context window.
        let truncation = tokenizers::TruncationParams {
            max_length: MAX_SEQUENCE_LENGTH,
            direction: tokenizers::TruncationDirection::Right,
            ..Default::default()
        };
        tok.with_truncation(Some(truncation))
            .map_err(|e| TokenizerError::LoadFailed {
                path: String::new(),
                reason: format!("failed to configure truncation: {e}"),
            })?;

        Ok(())
    }
}

fn encode_one(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
) -> Result<SdxlTokenizedPrompt, TokenizerError> {
    let encoding =
        tokenizer
            .encode(text, true)
            .map_err(|e| TokenizerError::TokenizationFailed {
                reason: e.to_string(),
            })?;

    let mut ids: Vec<u32> = encoding.get_ids().to_vec();
    let mask_raw: Vec<u32> = encoding.get_attention_mask().to_vec();
    if ids.first().copied() != Some(TOKEN_BOS) {
        return Err(TokenizerError::TokenizationFailed {
            reason: format!("encoded SDXL prompt did not start with BOS token {TOKEN_BOS}"),
        });
    }
    let attended_len = mask_raw.iter().take_while(|&&v| v != 0).count();
    if !ids
        .iter()
        .take(attended_len)
        .any(|&token_id| token_id == TOKEN_EOS)
    {
        return Err(TokenizerError::TokenizationFailed {
            reason: format!("encoded SDXL prompt did not contain EOS token {TOKEN_EOS}"),
        });
    }

    // Safety net: ensure buffers are exactly MAX_SEQUENCE_LENGTH.
    ids.resize(MAX_SEQUENCE_LENGTH, TOKEN_PAD);
    let attention_mask: Vec<f32> = mask_raw
        .iter()
        .map(|&v| v as f32)
        .chain(std::iter::repeat(0.0))
        .take(MAX_SEQUENCE_LENGTH)
        .collect();

    Ok(SdxlTokenizedPrompt {
        token_ids: ids,
        attention_mask,
    })
}

fn primary_tokenizer_metadata_path(metadata: &str) -> Option<&str> {
    tokenizer_metadata_path(metadata, &["tokenizer", "tokenizer_path", "tokenizer_dir"])
}

fn secondary_tokenizer_metadata_path(metadata: &str) -> Option<&str> {
    tokenizer_metadata_path(
        metadata,
        &["tokenizer_2", "tokenizer_2_path", "tokenizer_2_dir"],
    )
}

fn tokenizer_metadata_path<'a>(metadata: &'a str, accepted_keys: &[&str]) -> Option<&'a str> {
    let trimmed = metadata.trim();
    if trimmed.is_empty() {
        return None;
    }
    if !trimmed.contains('=') {
        return accepted_keys.contains(&"tokenizer").then_some(trimmed);
    }
    trimmed
        .split(';')
        .flat_map(|part| part.split(','))
        .filter_map(|part| part.trim().split_once('='))
        .find_map(|(key, value)| accepted_keys.contains(&key.trim()).then_some(value.trim()))
}

// ---------------------------------------------------------------------------
// Loading helpers
// ---------------------------------------------------------------------------

/// Load a single tokenizer from `path`, dispatching on the on-disk layout.
fn load_one(path: &Path) -> Result<tokenizers::Tokenizer, TokenizerError> {
    let path_str = path.display().to_string();

    // File: tokenizer.json
    if path.is_file() && path.extension().is_some_and(|e| e == "json") {
        // Check if this is actually a vocab.json used with merges.txt
        if path.file_name().is_some_and(|n| n == "vocab.json") {
            let merges = path.with_file_name("merges.txt");
            if merges.exists() {
                return load_bpe(path, &merges);
            }
        }
        // Regular tokenizer.json
        let mut tok =
            tokenizers::Tokenizer::from_file(path).map_err(|e| TokenizerError::LoadFailed {
                path: path_str.clone(),
                reason: e.to_string(),
            })?;
        SdxlTokenizer::configure(&mut tok)?;
        return Ok(tok);
    }

    // Directory: look for tokenizer.json inside.
    if path.is_dir() {
        let inner = path.join("tokenizer.json");
        if inner.exists() {
            return load_one(&inner);
        }
    }

    // Fallback: check if this is a directory containing vocab+merges.
    let parent = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(Path::new("."))
    };
    let vocab = parent.join("vocab.json");
    let merges = parent.join("merges.txt");
    if vocab.exists() && merges.exists() {
        return load_bpe(&vocab, &merges);
    }

    Err(TokenizerError::Missing { path: path_str })
}

/// Build a BPE tokenizer from separate `vocab.json` + `merges.txt` files.
fn load_bpe(
    vocab_path: &Path,
    merges_path: &Path,
) -> Result<tokenizers::Tokenizer, TokenizerError> {
    let vocab_str = vocab_path.display().to_string();
    let merges_str = merges_path.display().to_string();
    let bpe = tokenizers::models::bpe::BPE::from_file(&vocab_str, &merges_str)
        .build()
        .map_err(|e| TokenizerError::LoadFailed {
            path: vocab_str.clone(),
            reason: e.to_string(),
        })?;
    let mut tok = tokenizers::Tokenizer::new(bpe);
    SdxlTokenizer::configure(&mut tok)?;
    Ok(tok)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_dir() -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let counter = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let process = std::process::id();
        std::env::temp_dir().join(format!(
            "reimagine-sdxl-tokenizer-{process}-{nonce}-{counter}"
        ))
    }

    fn load_tokenizer() -> SdxlTokenizer {
        SdxlTokenizer::from_bundled().expect("bundled SDXL tokenizer should be loadable")
    }

    fn copy_bundled_tokenizer_fixture(dir: &Path) -> (PathBuf, PathBuf) {
        let bundled_dir = SdxlTokenizerResources::bundled_tokenizer_dir();
        let tokenizer_dir = dir.join("tokenizer");
        let tokenizer_2_dir = dir.join("tokenizer_2");
        fs::create_dir_all(&tokenizer_dir).unwrap();
        fs::create_dir_all(&tokenizer_2_dir).unwrap();
        let tokenizer_path = tokenizer_dir.join("tokenizer.json");
        let tokenizer_2_path = tokenizer_2_dir.join("tokenizer.json");
        fs::copy(
            bundled_dir.join("tokenizer/tokenizer.json"),
            &tokenizer_path,
        )
        .unwrap();
        fs::copy(
            bundled_dir.join("tokenizer_2/tokenizer.json"),
            &tokenizer_2_path,
        )
        .unwrap();
        (tokenizer_path, tokenizer_2_path)
    }

    #[test]
    fn bundled_tokenizer_loads() {
        let _tok = load_tokenizer();
    }

    #[test]
    fn from_paths_loads_local_tokenizer_fixture() {
        let dir = unique_temp_dir();
        let (tokenizer_path, tokenizer_2_path) = copy_bundled_tokenizer_fixture(&dir);
        let tok = SdxlTokenizer::from_paths(&tokenizer_path, &tokenizer_2_path)
            .expect("local tokenizer fixture should load");
        let prompt = tok.tokenize("local fixture").unwrap();
        assert_eq!(prompt.token_ids.len(), MAX_SEQUENCE_LENGTH);
        assert_eq!(prompt.attention_mask.len(), MAX_SEQUENCE_LENGTH);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn tokenize_returns_correct_length() {
        let tok = load_tokenizer();
        let prompt = tok.tokenize("hello world").unwrap();
        assert_eq!(prompt.token_ids.len(), MAX_SEQUENCE_LENGTH);
        assert_eq!(prompt.attention_mask.len(), MAX_SEQUENCE_LENGTH);
    }

    #[test]
    fn tokenize_starts_with_bos() {
        let tok = load_tokenizer();
        let prompt = tok.tokenize("test").unwrap();
        assert_eq!(prompt.token_ids[0], TOKEN_BOS);
    }

    #[test]
    fn tokenize_attention_mask_is_f32() {
        let tok = load_tokenizer();
        let prompt = tok.tokenize("hello").unwrap();
        assert!(prompt.attention_mask.iter().all(|&v| v == 0.0 || v == 1.0));
    }

    #[test]
    fn tokenize_handles_empty_string() {
        let tok = load_tokenizer();
        let prompt = tok.tokenize("").unwrap();
        assert_eq!(prompt.token_ids[0], TOKEN_BOS);
        // BOS + EOS should both be attended
        assert_eq!(prompt.attention_mask[0], 1.0);
        assert_eq!(prompt.attention_mask[1], 1.0);
        // Everything after EOS should be 0.0
        assert_eq!(prompt.attention_mask[2], 0.0);
    }

    #[test]
    fn tokenize_produces_different_ids_for_different_input() {
        let tok = load_tokenizer();
        let a = tok.tokenize("hello").unwrap();
        let b = tok.tokenize("world").unwrap();
        assert_ne!(a.token_ids, b.token_ids);
    }

    #[test]
    fn tokenize_safety_net_pads_to_max_length() {
        let tok = load_tokenizer();
        let prompt = tok.tokenize("a").unwrap();
        assert_eq!(prompt.token_ids.len(), MAX_SEQUENCE_LENGTH);
        // After BOS + 'a' token + EOS, rest should be PAD
        assert_eq!(prompt.token_ids[3], TOKEN_PAD);
    }

    #[test]
    fn tokenize_safety_net_truncates_to_max_length() {
        let tok = load_tokenizer();
        let long = "word ".repeat(200);
        let prompt = tok.tokenize(&long).unwrap();
        assert_eq!(prompt.token_ids.len(), MAX_SEQUENCE_LENGTH);
    }

    #[test]
    fn from_paths_with_invalid_file() {
        let err = SdxlTokenizer::from_paths(
            Path::new("/nonexistent/tokenizer.json"),
            Path::new("/nonexistent/tokenizer_2.json"),
        )
        .unwrap_err();
        assert!(
            matches!(err, TokenizerError::Missing { .. })
                | matches!(err, TokenizerError::LoadFailed { .. })
        );
    }

    #[test]
    fn from_paths_reports_malformed_tokenizer_file() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let malformed = dir.join("tokenizer.json");
        fs::write(&malformed, b"not-json").unwrap();
        let err = SdxlTokenizer::from_paths(&malformed, &malformed).unwrap_err();
        match err {
            TokenizerError::LoadFailed { path, reason } => {
                assert!(path.ends_with("tokenizer.json"));
                assert!(!reason.is_empty());
            }
            other => panic!("expected LoadFailed, got {other:?}"),
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn tokenizer_error_display() {
        let err = TokenizerError::Missing {
            path: "/some/path".into(),
        };
        assert!(err.to_string().contains("/some/path"));

        let err = TokenizerError::LoadFailed {
            path: "/bad.json".into(),
            reason: "syntax error".into(),
        };
        assert!(err.to_string().contains("/bad.json"));
        assert!(err.to_string().contains("syntax error"));

        let err = TokenizerError::UnsupportedModelFamily {
            series: "flux".into(),
            variant: "dev".into(),
        };
        assert!(err.to_string().contains("flux"));
        assert!(err.to_string().contains("dev"));
    }

    #[test]
    fn bundled_tokenizer_2_path_points_to_json_file() {
        let p = SdxlTokenizerResources::bundled_tokenizer_2_path();
        assert!(
            p.ends_with("tokenizer_2/tokenizer.json"),
            "unexpected bundled path: {}",
            p.display()
        );
    }

    #[test]
    fn explicit_metadata_resolves_primary_and_secondary_tokenizer_paths_separately() {
        let dir = unique_temp_dir();
        let checkpoint = dir.join("model.safetensors");
        fs::create_dir_all(&dir).unwrap();
        fs::write(&checkpoint, b"placeholder").unwrap();
        let (tokenizer_path, tokenizer_2_path) = copy_bundled_tokenizer_fixture(&dir);
        let source_set = ResolvedInferenceModelSourceSet::new(
            reimagine_inference::ResolvedInferenceModelSource::new(
                reimagine_inference::ModelSourceKind::CheckpointBundle,
                ModelRole::CheckpointBundle,
                checkpoint.clone(),
                reimagine_inference::ModelFormat::SafeTensors,
            ),
        )
        .with_source(
            reimagine_inference::ResolvedInferenceModelSource::new(
                reimagine_inference::ModelSourceKind::SplitComponent,
                ModelRole::TextEncoder,
                dir.join("clip.safetensors"),
                reimagine_inference::ModelFormat::SafeTensors,
            )
            .with_metadata(format!(
                "tokenizer_path={};tokenizer_2_path={}",
                tokenizer_path.display(),
                tokenizer_2_path.display()
            )),
        );

        let primary =
            SdxlTokenizerResources::resolve_tokenizer_path(&source_set, &checkpoint).unwrap();
        let secondary =
            SdxlTokenizerResources::resolve_tokenizer_2_path(&source_set, &checkpoint).unwrap();

        assert_eq!(primary, tokenizer_path);
        assert_eq!(secondary, tokenizer_2_path);
        let _ = fs::remove_dir_all(&dir);
    }
}
