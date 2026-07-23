//! SDXL bundled CLIP tokenizer resources for the Burn backend.
//!
//! SDXL uses two CLIP text encoders: CLIP-L (ViT-L/14) and CLIP-G
//! (ViT-bigG/14). Both share the same BPE tokenizer vocabulary but
//! produce different embedding dimensions. This module owns the
//! backend-private tokenizer resources and deterministic prompt
//! tokenization helpers.
//!
//! Resolution order implemented in `BurnSdxlTokenizerResources::resolve`:
//!
//! 1. explicit per-source metadata (`tokenizer*` for the primary
//!    text encoder, `tokenizer_2*` for the secondary text encoder).
//!    Invalid explicit paths fail without falling back.
//! 2. sidecar tokenizer files next to the loaded component source
//!    (the text encoder / text_encoder_2 source paths supplied by
//!    burn/05). Missing sidecars fall back to bundled defaults.
//! 3. bundled workspace assets under
//!    `assets/tokenizers/stable_diffusion/sdxl/{tokenizer,tokenizer_2}/tokenizer.json`.
//!
//! Primary and secondary metadata keys are parsed by separate
//! resolvers; the primary resolver never accepts `tokenizer_2*` keys
//! and vice versa. A regression test enforces this split.
//!
//! Tokenizer resources are private to the Burn backend. They are not
//! model-manager roles, runtime execution values, or workflow JSON.

use std::path::{Path, PathBuf};

use reimagine_inference::ResolvedInferenceModelSourceSet;

use crate::config::BurnBackendConfig;
use crate::error::BurnBackendError;

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

/// Relative path to the bundled primary tokenizer asset, rooted at the
/// workspace `assets/tokenizers/stable_diffusion/sdxl` directory.
pub const PRIMARY_TOKENIZER_ASSET: &str = "tokenizer/tokenizer.json";

/// Relative path to the bundled secondary tokenizer asset, rooted at
/// the workspace `assets/tokenizers/stable_diffusion/sdxl` directory.
pub const SECONDARY_TOKENIZER_ASSET: &str = "tokenizer_2/tokenizer.json";

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BurnTokenizerError {
    /// Tokenizer resource not found at the expected path.
    Missing { path: String },
    /// Tokenizer resource found but could not be loaded (malformed JSON,
    /// incompatible format, I/O error, etc.).
    LoadFailed { path: String, reason: String },
    /// The `tokenizers` library rejected an internal configuration step
    /// (padding, truncation, model construction) rather than an input
    /// encoding step.
    Configure { reason: String },
    /// The `tokenizers` library rejected the input during encoding.
    TokenizationFailed { reason: String },
    /// Explicit metadata was provided but the resolver could not turn
    /// it into a loadable file (missing file, missing inner
    /// `tokenizer.json` for a directory override, etc.). Bundled
    /// fallback is intentionally not attempted.
    ExplicitOverrideInvalid {
        role: BurnTokenizerRole,
        raw: String,
    },
}

impl std::fmt::Display for BurnTokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing { path } => write!(f, "Burn tokenizer resource not found: {path}"),
            Self::LoadFailed { path, reason } => {
                write!(f, "failed to load Burn tokenizer at {path}: {reason}")
            }
            Self::Configure { reason } => {
                write!(f, "failed to configure Burn tokenizer: {reason}")
            }
            Self::TokenizationFailed { reason } => {
                write!(f, "Burn tokenization failed: {reason}")
            }
            Self::ExplicitOverrideInvalid { role, raw } => write!(
                f,
                "Burn {} tokenizer metadata override `{raw}` is invalid; bundled fallback is disabled",
                role.as_str()
            ),
        }
    }
}

impl std::error::Error for BurnTokenizerError {}

// ---------------------------------------------------------------------------
// Tokenized prompt output
// ---------------------------------------------------------------------------

/// Output of tokenization, suitable for CLIP text encoder consumption.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnSdxlTokenizedPrompt {
    /// Token ids, length [`MAX_SEQUENCE_LENGTH`].
    pub token_ids: Vec<u32>,
    /// Attention mask, length [`MAX_SEQUENCE_LENGTH`] (1 for attended,
    /// 0 for padded).
    pub attention_mask: Vec<u32>,
}

/// Tokenization output for SDXL's two CLIP text encoders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnSdxlTokenizedPromptPair {
    pub clip_l: BurnSdxlTokenizedPrompt,
    pub clip_g: BurnSdxlTokenizedPrompt,
}

// ---------------------------------------------------------------------------
// Resource resolver
// ---------------------------------------------------------------------------

/// Resolved paths to bundled SDXL tokenizer resources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnSdxlTokenizerResources {
    primary: PathBuf,
    secondary: PathBuf,
    root: PathBuf,
}

impl BurnSdxlTokenizerResources {
    /// Build resources from the workspace's bundled default asset root.
    pub fn bundled() -> Result<Self, BurnTokenizerError> {
        Self::for_root(&bundled_tokenizer_root())
    }

    /// Build resources from the tokenizer root configured on a
    /// [`BurnBackendConfig`]. When the config does not override the
    /// root, the bundled workspace assets are used.
    pub fn from_config(config: &BurnBackendConfig) -> Result<Self, BurnBackendError> {
        let root = config
            .tokenizer_root()
            .cloned()
            .unwrap_or_else(bundled_tokenizer_root);
        Ok(Self::for_root(&root)?)
    }

    /// Build resources from an explicit asset root.
    ///
    /// Both primary and secondary tokenizer files must exist under the
    /// root, otherwise the call fails. This is the test seam — tests
    /// can construct a temp directory populated with the bundled
    /// fixtures and pass it in without touching the workspace default.
    pub fn for_root(root: &Path) -> Result<Self, BurnTokenizerError> {
        let primary = root.join(PRIMARY_TOKENIZER_ASSET);
        let secondary = root.join(SECONDARY_TOKENIZER_ASSET);

        if !primary.is_file() {
            return Err(BurnTokenizerError::Missing {
                path: primary.display().to_string(),
            });
        }
        if !secondary.is_file() {
            return Err(BurnTokenizerError::Missing {
                path: secondary.display().to_string(),
            });
        }

        Ok(Self {
            primary,
            secondary,
            root: root.to_path_buf(),
        })
    }

    /// Build resources from explicit primary and secondary paths.
    ///
    /// This is the second test seam: tests can point at any two
    /// tokenizer files without setting up a directory layout. The
    /// reported `root()` is derived from the primary file's parent
    /// directory, which matches the typical case where both files
    /// share a common directory.
    pub fn from_paths(primary: PathBuf, secondary: PathBuf) -> Self {
        let root = primary
            .parent()
            .map(|parent| parent.to_path_buf())
            .unwrap_or_default();
        Self {
            primary,
            secondary,
            root,
        }
    }

    pub fn primary_path(&self) -> &Path {
        &self.primary
    }

    pub fn secondary_path(&self) -> &Path {
        &self.secondary
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

/// Which SDXL text encoder a tokenizer resolves for.
///
/// `Primary` corresponds to `text_encoder` (CLIP-L); `Secondary` to
/// `text_encoder_2` (CLIP-G). The two roles are resolved by
/// independent key sets and independent sidecar searches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BurnTokenizerRole {
    Primary,
    Secondary,
}

impl BurnTokenizerRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Primary => "primary",
            Self::Secondary => "secondary",
        }
    }

    fn bundled_relative(self) -> &'static str {
        match self {
            Self::Primary => PRIMARY_TOKENIZER_ASSET,
            Self::Secondary => SECONDARY_TOKENIZER_ASSET,
        }
    }
}

/// Bundle of source paths that scope Burn SDXL tokenizer sidecar
/// search to a single loaded model bundle.
///
/// Built from a `ResolvedInferenceModelSourceSet` after burn/05
/// validation has classified the components, so the sidecar search
/// inspects only the loaded `text_encoder` / `text_encoder_2` source
/// paths — not the entire workspace.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BurnSdxlTokenizerContext {
    primary_source: Option<PathBuf>,
    secondary_source: Option<PathBuf>,
}

impl BurnSdxlTokenizerContext {
    pub fn empty() -> Self {
        Self::default()
    }

    /// Build a context by finding the `text_encoder` and
    /// `text_encoder_2` sources inside a resolved bundle. The
    /// component role is read from the resolver projection
    /// (`component=text_encoder` / `component=text_encoder_2`).
    pub fn from_source_set(source_set: &ResolvedInferenceModelSourceSet) -> Self {
        let mut context = Self::default();
        for source in source_set.sources() {
            let Some(meta) = source.metadata() else {
                continue;
            };
            match parse_projection_value(meta, "component") {
                Some("text_encoder") => {
                    context.primary_source = Some(source.path().clone());
                }
                Some("text_encoder_2") => {
                    context.secondary_source = Some(source.path().clone());
                }
                _ => {}
            }
        }
        context
    }

    pub fn with_primary_source(mut self, path: PathBuf) -> Self {
        self.primary_source = Some(path);
        self
    }

    pub fn with_secondary_source(mut self, path: PathBuf) -> Self {
        self.secondary_source = Some(path);
        self
    }

    pub fn primary_source(&self) -> Option<&Path> {
        self.primary_source.as_deref()
    }

    pub fn secondary_source(&self) -> Option<&Path> {
        self.secondary_source.as_deref()
    }
}

impl BurnSdxlTokenizerResources {
    /// Resolve SDXL tokenizer resources through the burn/07b
    /// 3-tier order:
    ///
    /// 1. explicit per-source metadata (`tokenizer*` for the primary
    ///    text encoder, `tokenizer_2*` for the secondary);
    /// 2. sidecar files next to the loaded component source;
    /// 3. bundled defaults under the workspace
    ///    `assets/tokenizers/stable_diffusion/sdxl` directory
    ///    (overridable through `BurnBackendConfig::tokenizer_root`).
    ///
    /// `primary_metadata` and `secondary_metadata` should be the raw
    /// resolver metadata strings attached to the corresponding
    /// text-encoder sources; they are parsed by independent key sets
    /// so the primary resolver cannot pick up `tokenizer_2*` keys and
    /// vice versa.
    ///
    /// Invalid explicit overrides fail without falling back to
    /// bundled defaults. Missing sidecars do fall back.
    pub fn resolve(
        config: &BurnBackendConfig,
        context: &BurnSdxlTokenizerContext,
        primary_metadata: Option<&str>,
        secondary_metadata: Option<&str>,
    ) -> Result<Self, BurnBackendError> {
        let asset_root = config
            .tokenizer_root()
            .cloned()
            .unwrap_or_else(bundled_tokenizer_root);

        let primary = resolve_one(
            BurnTokenizerRole::Primary,
            &asset_root,
            context.primary_source(),
            primary_metadata,
        )?;
        let secondary = resolve_one(
            BurnTokenizerRole::Secondary,
            &asset_root,
            context.secondary_source(),
            secondary_metadata,
        )?;

        Ok(Self {
            primary,
            secondary,
            root: asset_root,
        })
    }
}

fn resolve_one(
    role: BurnTokenizerRole,
    asset_root: &Path,
    source_path: Option<&Path>,
    metadata: Option<&str>,
) -> Result<PathBuf, BurnTokenizerError> {
    // 1. Explicit metadata override — must resolve or fail without
    //    bundled fallback.
    if let Some(meta) = metadata
        && let Some(raw) = role.extract_metadata_path(meta)
    {
        return resolve_explicit_override(role, raw);
    }

    // 2. Sidecar search next to the loaded component source.
    if let Some(source) = source_path
        && let Some(sidecar) = find_sidecar(role, source)
    {
        return Ok(sidecar);
    }

    // 3. Bundled default under the configured / workspace asset root.
    let bundled = asset_root.join(role.bundled_relative());
    if bundled.is_file() {
        return Ok(bundled);
    }
    Err(BurnTokenizerError::Missing {
        path: bundled.display().to_string(),
    })
}

fn resolve_explicit_override(
    role: BurnTokenizerRole,
    raw: &str,
) -> Result<PathBuf, BurnTokenizerError> {
    let candidate = Path::new(raw);
    if candidate.is_file() {
        return Ok(candidate.to_path_buf());
    }
    if candidate.is_dir() {
        let inner = candidate.join(role.bundled_relative());
        if inner.is_file() {
            return Ok(inner);
        }
        return Err(BurnTokenizerError::ExplicitOverrideInvalid {
            role,
            raw: raw.to_owned(),
        });
    }
    Err(BurnTokenizerError::ExplicitOverrideInvalid {
        role,
        raw: raw.to_owned(),
    })
}

fn find_sidecar(role: BurnTokenizerRole, source: &Path) -> Option<PathBuf> {
    let parent = source.parent()?;
    match role {
        BurnTokenizerRole::Primary => find_primary_sidecar(parent),
        BurnTokenizerRole::Secondary => find_secondary_sidecar(parent),
    }
}

fn find_primary_sidecar(parent: &Path) -> Option<PathBuf> {
    // 1. tokenizer.json (single file)
    let f = parent.join("tokenizer.json");
    if f.is_file() {
        return Some(f);
    }
    // 2. tokenizer/tokenizer.json (directory layout)
    let d = parent.join("tokenizer/tokenizer.json");
    if d.is_file() {
        return Some(d);
    }
    // 3. vocab.json + merges.txt (two-file BPE)
    if let Some(p) = find_bpe_pair(parent) {
        return Some(p);
    }
    None
}

fn find_secondary_sidecar(parent: &Path) -> Option<PathBuf> {
    // 1. tokenizer_2/tokenizer.json (SDXL secondary directory layout)
    let d = parent.join("tokenizer_2/tokenizer.json");
    if d.is_file() {
        return Some(d);
    }
    // 2. tokenizer_2/{vocab.json,merges.txt} (SDXL secondary BPE pair)
    if let Some(p) = find_bpe_pair(&parent.join("tokenizer_2")) {
        return Some(p);
    }
    // 3. Fall back to the primary sidecar layout (CLIP-G may share
    //    the primary tokenizer vocabulary file).
    find_primary_sidecar(parent)
}

fn find_bpe_pair(parent: &Path) -> Option<PathBuf> {
    let vocab = parent.join("vocab.json");
    let merges = parent.join("merges.txt");
    if vocab.is_file() && merges.is_file() {
        return Some(vocab);
    }
    None
}

impl BurnTokenizerRole {
    /// Extract the first matching tokenizer override for this role
    /// from a resolver metadata string.
    ///
    /// The primary and secondary resolvers use disjoint key sets, so
    /// `tokenizer_2*` keys can never reach the primary path and
    /// `tokenizer*` keys can never reach the secondary path. The
    /// regression test `primary_resolver_never_accepts_secondary_keys`
    /// / `secondary_resolver_never_accepts_primary_keys` enforce this.
    pub fn extract_metadata_path(self, metadata: &str) -> Option<&str> {
        let keys: &[&str] = match self {
            Self::Primary => &["tokenizer", "tokenizer_path", "tokenizer_dir"],
            Self::Secondary => &["tokenizer_2", "tokenizer_2_path", "tokenizer_2_dir"],
        };
        extract_tokenizer_path(metadata, keys)
    }
}

fn extract_tokenizer_path<'a>(metadata: &'a str, accepted_keys: &[&str]) -> Option<&'a str> {
    let trimmed = metadata.trim();
    if trimmed.is_empty() {
        return None;
    }
    if !trimmed.contains('=') {
        // Bare path convention: only accepted when the role's first
        // accepted key is the bare-path alias.
        return accepted_keys.contains(&"tokenizer").then_some(trimmed);
    }
    trimmed
        .split(';')
        .flat_map(|part| part.split(','))
        .filter_map(|part| part.trim().split_once('='))
        .find_map(|(key, value)| (accepted_keys.contains(&key.trim())).then_some(value.trim()))
}

fn parse_projection_value<'a>(metadata: &'a str, key: &str) -> Option<&'a str> {
    metadata
        .split(';')
        .flat_map(|part| part.split(','))
        .filter_map(|part| part.trim().split_once('='))
        .find_map(|(k, v)| (k.trim() == key).then_some(v.trim()))
}

/// Workspace-bundled SDXL tokenizer asset root.
///
/// Computed from `CARGO_MANIFEST_DIR` at build time so library code
/// never depends on the process current working directory.
fn bundled_tokenizer_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join("assets/tokenizers/stable_diffusion/sdxl")
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// Burn-private SDXL tokenizer backed by the workspace `tokenizers`
/// crate. Holds two independent [`tokenizers::Tokenizer`] instances:
/// - `primary`   — used by the primary CLIP-L text encoder
/// - `secondary` — used by the secondary CLIP-G text encoder
#[derive(Debug)]
pub struct BurnSdxlTokenizer {
    primary: tokenizers::Tokenizer,
    secondary: tokenizers::Tokenizer,
    resources: BurnSdxlTokenizerResources,
}

impl BurnSdxlTokenizer {
    /// Load the bundled SDXL tokenizers from the workspace default.
    pub fn from_bundled() -> Result<Self, BurnTokenizerError> {
        Self::from_resources(BurnSdxlTokenizerResources::bundled()?)
    }

    /// Load SDXL tokenizers from an explicit asset root.
    pub fn from_root(root: &Path) -> Result<Self, BurnTokenizerError> {
        Self::from_resources(BurnSdxlTokenizerResources::for_root(root)?)
    }

    /// Load SDXL tokenizers using the configured
    /// [`BurnBackendConfig::tokenizer_root`].
    pub fn from_config(config: &BurnBackendConfig) -> Result<Self, BurnBackendError> {
        let resources = BurnSdxlTokenizerResources::from_config(config)?;
        Ok(Self::from_resources(resources)?)
    }

    /// Load SDXL tokenizers from pre-resolved resources.
    pub fn from_resources(
        resources: BurnSdxlTokenizerResources,
    ) -> Result<Self, BurnTokenizerError> {
        let primary = load_one(resources.primary_path())?;
        let secondary = load_one(resources.secondary_path())?;
        Ok(Self {
            primary,
            secondary,
            resources,
        })
    }

    /// Load SDXL tokenizers from two explicit files. Used as a
    /// per-path test seam and to support future sidecar/metadata
    /// resolution in 07b.
    pub fn from_paths(
        primary_path: &Path,
        secondary_path: &Path,
    ) -> Result<Self, BurnTokenizerError> {
        let primary = load_one(primary_path)?;
        let secondary = load_one(secondary_path)?;
        Ok(Self {
            primary,
            secondary,
            resources: BurnSdxlTokenizerResources::from_paths(
                primary_path.to_path_buf(),
                secondary_path.to_path_buf(),
            ),
        })
    }

    pub fn resources(&self) -> &BurnSdxlTokenizerResources {
        &self.resources
    }

    /// Tokenize text using the primary (CLIP-L) tokenizer.
    ///
    /// Returns a [`BurnSdxlTokenizedPrompt`] with token ids and
    /// attention mask, both exactly [`MAX_SEQUENCE_LENGTH`] elements
    /// long.
    pub fn tokenize(&self, text: &str) -> Result<BurnSdxlTokenizedPrompt, BurnTokenizerError> {
        encode_one(&self.primary, text)
    }

    /// Tokenize text for both SDXL CLIP encoders.
    pub fn tokenize_pair(
        &self,
        text: &str,
    ) -> Result<BurnSdxlTokenizedPromptPair, BurnTokenizerError> {
        Ok(BurnSdxlTokenizedPromptPair {
            clip_l: encode_one(&self.primary, text)?,
            clip_g: encode_one(&self.secondary, text)?,
        })
    }

    // -- internal helpers ---------------------------------------------------

    fn configure(tok: &mut tokenizers::Tokenizer) -> Result<(), BurnTokenizerError> {
        let padding = tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(MAX_SEQUENCE_LENGTH),
            direction: tokenizers::PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: TOKEN_PAD,
            pad_type_id: 0,
            pad_token: String::new(),
        };
        tok.with_padding(Some(padding));

        let truncation = tokenizers::TruncationParams {
            max_length: MAX_SEQUENCE_LENGTH,
            direction: tokenizers::TruncationDirection::Right,
            ..Default::default()
        };
        tok.with_truncation(Some(truncation))
            .map_err(|e| BurnTokenizerError::Configure {
                reason: format!("truncation: {e}"),
            })?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Encoding / loading helpers
// ---------------------------------------------------------------------------

fn encode_one(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
) -> Result<BurnSdxlTokenizedPrompt, BurnTokenizerError> {
    let encoding =
        tokenizer
            .encode(text, true)
            .map_err(|e| BurnTokenizerError::TokenizationFailed {
                reason: e.to_string(),
            })?;

    let ids: Vec<u32> = encoding.get_ids().to_vec();
    let mask: Vec<u32> = encoding.get_attention_mask().to_vec();

    if ids.first().copied() != Some(TOKEN_BOS) {
        return Err(BurnTokenizerError::TokenizationFailed {
            reason: format!("encoded Burn SDXL prompt did not start with BOS token {TOKEN_BOS}"),
        });
    }
    let attended_len = mask.iter().take_while(|&&v| v != 0).count();
    if !ids
        .iter()
        .take(attended_len)
        .any(|&token_id| token_id == TOKEN_EOS)
    {
        return Err(BurnTokenizerError::TokenizationFailed {
            reason: format!("encoded Burn SDXL prompt did not contain EOS token {TOKEN_EOS}"),
        });
    }

    // Safety net: ensure buffers are exactly MAX_SEQUENCE_LENGTH.
    let mut ids = ids;
    ids.resize(MAX_SEQUENCE_LENGTH, TOKEN_PAD);
    let mut attention_mask = mask;
    attention_mask.resize(MAX_SEQUENCE_LENGTH, 0);

    Ok(BurnSdxlTokenizedPrompt {
        token_ids: ids,
        attention_mask,
    })
}

fn load_one(path: &Path) -> Result<tokenizers::Tokenizer, BurnTokenizerError> {
    let path_str = path.display().to_string();

    // Directory: look for `tokenizer.json` inside, or fall through
    // to a `vocab.json` + `merges.txt` BPE pair rooted at the dir.
    if path.is_dir() {
        let inner = path.join("tokenizer.json");
        if inner.is_file() {
            return load_one(&inner);
        }
        if let Some(p) = find_bpe_pair(path) {
            return load_bpe(&p, &p.with_file_name("merges.txt"));
        }
        return Err(BurnTokenizerError::Missing {
            path: path_str.clone(),
        });
    }

    if !path.is_file() {
        return Err(BurnTokenizerError::Missing {
            path: path_str.clone(),
        });
    }

    // File: `vocab.json` next to a `merges.txt` builds a BPE pair.
    if path.file_name().is_some_and(|name| name == "vocab.json") {
        let merges = path.with_file_name("merges.txt");
        if merges.is_file() {
            return load_bpe(path, &merges);
        }
    }

    // File: regular `tokenizer.json` (or any compatible JSON).
    let mut tok =
        tokenizers::Tokenizer::from_file(path).map_err(|e| BurnTokenizerError::LoadFailed {
            path: path_str.clone(),
            reason: e.to_string(),
        })?;
    BurnSdxlTokenizer::configure(&mut tok)?;
    Ok(tok)
}

fn load_bpe(
    vocab_path: &Path,
    merges_path: &Path,
) -> Result<tokenizers::Tokenizer, BurnTokenizerError> {
    let vocab_str = vocab_path.display().to_string();
    let merges_str = merges_path.display().to_string();
    let bpe = tokenizers::models::bpe::BPE::from_file(&vocab_str, &merges_str)
        .build()
        .map_err(|e| BurnTokenizerError::LoadFailed {
            path: vocab_str.clone(),
            reason: e.to_string(),
        })?;
    let mut tok = tokenizers::Tokenizer::new(bpe);
    BurnSdxlTokenizer::configure(&mut tok)?;
    Ok(tok)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
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
            "reimagine-burn-sdxl-tokenizer-{process}-{nonce}-{counter}"
        ))
    }

    fn bundled_dir() -> PathBuf {
        bundled_tokenizer_root()
    }

    fn copy_bundled_tokenizer_fixture(dir: &Path) -> (PathBuf, PathBuf) {
        let tokenizer_dir = dir.join("tokenizer");
        let tokenizer_2_dir = dir.join("tokenizer_2");
        fs::create_dir_all(&tokenizer_dir).unwrap();
        fs::create_dir_all(&tokenizer_2_dir).unwrap();
        let primary = tokenizer_dir.join("tokenizer.json");
        let secondary = tokenizer_2_dir.join("tokenizer.json");
        fs::copy(bundled_dir().join(PRIMARY_TOKENIZER_ASSET), &primary).unwrap();
        fs::copy(bundled_dir().join(SECONDARY_TOKENIZER_ASSET), &secondary).unwrap();
        (primary, secondary)
    }

    #[test]
    fn bundled_tokenizer_loads() {
        let tok = BurnSdxlTokenizer::from_bundled().expect("bundled SDXL tokenizer should load");
        let _ = tok.tokenize("hello world").expect("tokenize");
    }

    #[test]
    fn resources_bundled_points_to_workspace_assets() {
        let resources = BurnSdxlTokenizerResources::bundled().expect("bundled resources");
        assert!(resources.primary_path().ends_with(PRIMARY_TOKENIZER_ASSET));
        assert!(
            resources
                .secondary_path()
                .ends_with(SECONDARY_TOKENIZER_ASSET)
        );
        assert_eq!(resources.root(), bundled_dir());
    }

    #[test]
    fn resources_for_root_uses_explicit_root() {
        let dir = unique_temp_dir();
        let (primary, secondary) = copy_bundled_tokenizer_fixture(&dir);

        let resources = BurnSdxlTokenizerResources::for_root(&dir).expect("resources");
        assert_eq!(resources.primary_path(), &primary);
        assert_eq!(resources.secondary_path(), &secondary);
        assert_eq!(resources.root(), dir);
    }

    #[test]
    fn resources_for_root_rejects_missing_primary() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        // Only the secondary side is populated.
        let tokenizer_2_dir = dir.join("tokenizer_2");
        fs::create_dir_all(&tokenizer_2_dir).unwrap();
        fs::copy(
            bundled_dir().join(SECONDARY_TOKENIZER_ASSET),
            tokenizer_2_dir.join("tokenizer.json"),
        )
        .unwrap();

        let err = BurnSdxlTokenizerResources::for_root(&dir).unwrap_err();
        match err {
            BurnTokenizerError::Missing { path } => {
                assert!(path.ends_with(PRIMARY_TOKENIZER_ASSET));
            }
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    #[test]
    fn resources_for_root_rejects_missing_secondary() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let tokenizer_dir = dir.join("tokenizer");
        fs::create_dir_all(&tokenizer_dir).unwrap();
        fs::copy(
            bundled_dir().join(PRIMARY_TOKENIZER_ASSET),
            tokenizer_dir.join("tokenizer.json"),
        )
        .unwrap();

        let err = BurnSdxlTokenizerResources::for_root(&dir).unwrap_err();
        match err {
            BurnTokenizerError::Missing { path } => {
                assert!(path.ends_with(SECONDARY_TOKENIZER_ASSET));
            }
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    #[test]
    fn from_config_uses_bundled_when_unset() {
        let config = BurnBackendConfig::new("/models", "/output");
        let resources = BurnSdxlTokenizerResources::from_config(&config).expect("resources");
        assert_eq!(resources.root(), bundled_dir());
    }

    #[test]
    fn from_config_uses_explicit_root() {
        let dir = unique_temp_dir();
        let _ = copy_bundled_tokenizer_fixture(&dir);

        let config = BurnBackendConfig::new("/models", "/output").with_tokenizer_root(dir.clone());
        let resources = BurnSdxlTokenizerResources::from_config(&config).expect("resources");
        assert_eq!(resources.root(), dir);
    }

    #[test]
    fn from_paths_loads_explicit_tokenizer_files() {
        let dir = unique_temp_dir();
        let (primary, secondary) = copy_bundled_tokenizer_fixture(&dir);

        let tok = BurnSdxlTokenizer::from_paths(&primary, &secondary).expect("tokenizer");
        let prompt = tok.tokenize("explicit paths").unwrap();
        assert_eq!(prompt.token_ids.len(), MAX_SEQUENCE_LENGTH);
        assert_eq!(prompt.attention_mask.len(), MAX_SEQUENCE_LENGTH);
    }

    #[test]
    fn from_paths_reports_malformed_tokenizer_file() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let malformed = dir.join("tokenizer.json");
        fs::write(&malformed, b"not-json").unwrap();
        let err = BurnSdxlTokenizer::from_paths(&malformed, &malformed).unwrap_err();
        match err {
            BurnTokenizerError::LoadFailed { path, reason } => {
                assert!(path.ends_with("tokenizer.json"));
                assert!(!reason.is_empty());
            }
            other => panic!("expected LoadFailed, got {other:?}"),
        }
    }

    #[test]
    fn from_paths_with_missing_file() {
        let err = BurnSdxlTokenizer::from_paths(
            Path::new("/nonexistent/primary.json"),
            Path::new("/nonexistent/secondary.json"),
        )
        .unwrap_err();
        // A missing file is reported as `Missing` by `load_one`
        // (a missing resource is not a malformed file). Both error
        // variants are acceptable evidence that the load failed.
        assert!(
            matches!(err, BurnTokenizerError::Missing { .. })
                | matches!(err, BurnTokenizerError::LoadFailed { .. })
        );
    }

    #[test]
    fn from_root_loads_via_test_seam() {
        let dir = unique_temp_dir();
        let _ = copy_bundled_tokenizer_fixture(&dir);
        let tok = BurnSdxlTokenizer::from_root(&dir).expect("tokenizer from root");
        let prompt = tok.tokenize("root seam").unwrap();
        assert_eq!(prompt.token_ids[0], TOKEN_BOS);
    }

    #[test]
    fn tokenize_returns_correct_length() {
        let tok = BurnSdxlTokenizer::from_bundled().expect("tokenizer");
        let prompt = tok.tokenize("hello world").unwrap();
        assert_eq!(prompt.token_ids.len(), MAX_SEQUENCE_LENGTH);
        assert_eq!(prompt.attention_mask.len(), MAX_SEQUENCE_LENGTH);
    }

    #[test]
    fn tokenize_starts_with_bos() {
        let tok = BurnSdxlTokenizer::from_bundled().expect("tokenizer");
        let prompt = tok.tokenize("test").unwrap();
        assert_eq!(prompt.token_ids[0], TOKEN_BOS);
    }

    #[test]
    fn tokenize_attention_mask_is_binary() {
        let tok = BurnSdxlTokenizer::from_bundled().expect("tokenizer");
        let prompt = tok.tokenize("hello").unwrap();
        assert!(prompt.attention_mask.iter().all(|&v| v == 0 || v == 1));
    }

    #[test]
    fn tokenize_handles_empty_string() {
        let tok = BurnSdxlTokenizer::from_bundled().expect("tokenizer");
        let prompt = tok.tokenize("").unwrap();
        assert_eq!(prompt.token_ids[0], TOKEN_BOS);
        // BOS + EOS should both be attended.
        assert_eq!(prompt.attention_mask[0], 1);
        assert_eq!(prompt.attention_mask[1], 1);
        assert_eq!(prompt.attention_mask[2], 0);
    }

    #[test]
    fn tokenize_produces_different_ids_for_different_input() {
        let tok = BurnSdxlTokenizer::from_bundled().expect("tokenizer");
        let a = tok.tokenize("hello").unwrap();
        let b = tok.tokenize("world").unwrap();
        assert_ne!(a.token_ids, b.token_ids);
    }

    #[test]
    fn tokenize_safety_net_pads_to_max_length() {
        let tok = BurnSdxlTokenizer::from_bundled().expect("tokenizer");
        let prompt = tok.tokenize("a").unwrap();
        assert_eq!(prompt.token_ids.len(), MAX_SEQUENCE_LENGTH);
        // After BOS + 'a' + EOS, the rest should be PAD.
        assert_eq!(prompt.token_ids[3], TOKEN_PAD);
    }

    #[test]
    fn tokenize_safety_net_truncates_to_max_length() {
        let tok = BurnSdxlTokenizer::from_bundled().expect("tokenizer");
        let long = "word ".repeat(200);
        let prompt = tok.tokenize(&long).unwrap();
        assert_eq!(prompt.token_ids.len(), MAX_SEQUENCE_LENGTH);
    }

    #[test]
    fn tokenize_pair_uses_independent_tokenizers() {
        let tok = BurnSdxlTokenizer::from_bundled().expect("tokenizer");
        let pair = tok.tokenize_pair("hello world").unwrap();

        // Both encoders pad to the same fixed length.
        assert_eq!(pair.clip_l.token_ids.len(), MAX_SEQUENCE_LENGTH);
        assert_eq!(pair.clip_g.token_ids.len(), MAX_SEQUENCE_LENGTH);

        // BOS is present in both sequences.
        assert_eq!(pair.clip_l.token_ids[0], TOKEN_BOS);
        assert_eq!(pair.clip_g.token_ids[0], TOKEN_BOS);

        // The bundled primary and secondary tokenizers use the same
        // BPE vocabulary, so token ids are identical — the split is
        // structural (separate instances, separate future module
        // contracts) and must not be silently collapsed into one.
        assert_eq!(pair.clip_l.token_ids, pair.clip_g.token_ids);
    }

    #[test]
    fn tokenize_pair_returns_independent_buffers() {
        let tok = BurnSdxlTokenizer::from_bundled().expect("tokenizer");
        let pair = tok.tokenize_pair("hello").unwrap();
        assert_eq!(
            pair.clip_l.token_ids,
            tok.tokenize("hello").unwrap().token_ids
        );
        assert_eq!(
            pair.clip_g.token_ids,
            tok.tokenize("hello").unwrap().token_ids
        );
    }

    #[test]
    fn bos_eos_pad_are_validated_for_both_assets() {
        let dir = unique_temp_dir();
        let (primary, secondary) = copy_bundled_tokenizer_fixture(&dir);
        let tok = BurnSdxlTokenizer::from_paths(&primary, &secondary).expect("tokenizer");

        for text in ["", "a", "the quick brown fox", " "] {
            let primary_prompt = tok.tokenize(text).unwrap();
            let pair = tok.tokenize_pair(text).unwrap();

            // Primary matches the standalone call.
            assert_eq!(primary_prompt.token_ids, pair.clip_l.token_ids);

            // Both buffers start with BOS, are padded to MAX_SEQUENCE_LENGTH,
            // and the attended prefix contains EOS.
            for prompt in [&pair.clip_l, &pair.clip_g] {
                assert_eq!(prompt.token_ids[0], TOKEN_BOS);
                let attended_len = prompt
                    .attention_mask
                    .iter()
                    .take_while(|&&v| v != 0)
                    .count();
                assert!(
                    attended_len <= MAX_SEQUENCE_LENGTH,
                    "attended length {attended_len} exceeds MAX_SEQUENCE_LENGTH"
                );
                assert!(
                    prompt
                        .token_ids
                        .iter()
                        .take(attended_len)
                        .any(|&id| id == TOKEN_EOS),
                    "attended prefix must contain EOS for text `{text}`"
                );
                // Padding tail should be PAD.
                for &id in &prompt.token_ids[attended_len..] {
                    assert_eq!(id, TOKEN_PAD, "padded tail must be PAD");
                }
            }
        }
    }

    #[test]
    fn tokenizer_error_display_includes_context() {
        let err = BurnTokenizerError::Missing {
            path: "/some/path".into(),
        };
        assert!(err.to_string().contains("/some/path"));

        let err = BurnTokenizerError::LoadFailed {
            path: "/bad.json".into(),
            reason: "syntax error".into(),
        };
        assert!(err.to_string().contains("/bad.json"));
        assert!(err.to_string().contains("syntax error"));

        let err = BurnTokenizerError::Configure {
            reason: "truncation: bad".into(),
        };
        assert!(err.to_string().contains("configure"));
        assert!(err.to_string().contains("truncation"));

        let err = BurnTokenizerError::TokenizationFailed {
            reason: "reject".into(),
        };
        assert!(err.to_string().contains("reject"));
    }

    #[test]
    fn from_paths_derives_root_from_primary_parent() {
        let dir = unique_temp_dir();
        let (primary, secondary) = copy_bundled_tokenizer_fixture(&dir);
        let resources = BurnSdxlTokenizerResources::from_paths(primary.clone(), secondary);
        assert_eq!(resources.root(), primary.parent().unwrap());
    }

    #[test]
    fn backend_error_wraps_tokenizer_error() {
        let burn_err: BurnBackendError = BurnTokenizerError::Missing {
            path: "/missing.json".into(),
        }
        .into();
        match burn_err {
            BurnBackendError::Tokenizer(BurnTokenizerError::Missing { path }) => {
                assert_eq!(path, "/missing.json");
            }
            other => panic!("expected Tokenizer variant, got {other:?}"),
        }
    }

    #[test]
    fn bundled_tokenizer_root_is_workspace_relative_and_not_cwd() {
        let root = bundled_tokenizer_root();
        // The bundled root must be a non-empty absolute(ish) path that
        // does not start with a single component (CWD-relative). The
        // value is anchored on CARGO_MANIFEST_DIR, so it always points
        // at the workspace tree.
        let root_str = root.to_string_lossy();
        assert!(
            root_str.contains("assets/tokenizers/stable_diffusion/sdxl"),
            "bundled root should point at workspace assets, got `{root_str}`"
        );
        assert!(root.is_absolute() || root_str.starts_with(".."));
    }

    // -----------------------------------------------------------------
    // burn/07b — explicit metadata + sidecar resolution tests
    // -----------------------------------------------------------------

    fn sdxl_temp_dir() -> PathBuf {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn copy_sidecar_tokenizer_files(source_dir: &Path) {
        let bundled = bundled_dir();
        fs::copy(
            bundled.join(PRIMARY_TOKENIZER_ASSET),
            source_dir.join("tokenizer.json"),
        )
        .unwrap();
        fs::create_dir_all(source_dir.join("tokenizer_2")).unwrap();
        fs::copy(
            bundled.join(SECONDARY_TOKENIZER_ASSET),
            source_dir.join("tokenizer_2/tokenizer.json"),
        )
        .unwrap();
    }

    fn primary_resolver_metadata(path: &Path) -> String {
        format!(
            "component=text_encoder;backend=burn;tokenizer_path={}",
            path.display()
        )
    }

    fn secondary_resolver_metadata(path: &Path) -> String {
        format!(
            "component=text_encoder_2;backend=burn;tokenizer_2_path={}",
            path.display()
        )
    }

    fn make_component_source(
        dir: &Path,
        _component_role: &str,
        path_in_dir: &str,
        metadata: String,
    ) -> reimagine_inference::ResolvedInferenceModelSource {
        use reimagine_core::model::ModelRole;
        use reimagine_inference::{ModelFormat, ModelSourceKind, ResolvedInferenceModelSource};
        let path = dir.join(path_in_dir);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&path, b"placeholder").unwrap();
        ResolvedInferenceModelSource::new(
            ModelSourceKind::SplitComponent,
            ModelRole::TextEncoder,
            path,
            ModelFormat::SafeTensors,
        )
        .with_metadata(metadata)
    }

    fn sample_source_set(dir: &Path) -> ResolvedInferenceModelSourceSet {
        use reimagine_core::model::ModelRole;
        use reimagine_inference::{ModelFormat, ModelSourceKind, ResolvedInferenceModelSource};
        let mut sources = Vec::new();
        for (role, path) in [
            ("diffusion", "diffusion/model.safetensors"),
            ("vae", "vae/model.safetensors"),
            ("text_encoder", "text_encoder/model.safetensors"),
            ("text_encoder_2", "text_encoder_2/model.safetensors"),
        ] {
            let p = dir.join(path);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(&p, b"placeholder").unwrap();
            let model_role = match role {
                "diffusion" => ModelRole::DiffusionModel,
                "vae" => ModelRole::Vae,
                _ => ModelRole::TextEncoder,
            };
            let metadata = format!("component={role};backend=burn");
            sources.push(
                ResolvedInferenceModelSource::new(
                    ModelSourceKind::SplitComponent,
                    model_role,
                    p,
                    ModelFormat::SafeTensors,
                )
                .with_metadata(metadata),
            );
        }
        ResolvedInferenceModelSourceSet::from_sources(sources)
    }

    #[test]
    fn role_metadata_keys_are_disjoint() {
        let primary = BurnTokenizerRole::Primary
            .extract_metadata_path("tokenizer=/a;tokenizer_2=/b;tokenizer_2_path=/c");
        let secondary = BurnTokenizerRole::Secondary
            .extract_metadata_path("tokenizer=/a;tokenizer_2=/b;tokenizer_2_path=/c");
        assert_eq!(primary, Some("/a"));
        assert_eq!(secondary, Some("/b"));
    }

    #[test]
    fn primary_resolver_never_accepts_secondary_keys() {
        // Even when only secondary keys are present, the primary
        // resolver must return None so resolution falls through to
        // sidecar / bundled instead of silently using the secondary
        // override.
        let primary = BurnTokenizerRole::Primary.extract_metadata_path(
            "tokenizer_2_path=/some/secondary.json;tokenizer_2_dir=/elsewhere",
        );
        assert_eq!(primary, None);
    }

    #[test]
    fn secondary_resolver_never_accepts_primary_keys() {
        let secondary = BurnTokenizerRole::Secondary
            .extract_metadata_path("tokenizer=/some/primary.json;tokenizer_dir=/elsewhere");
        assert_eq!(secondary, None);
    }

    #[test]
    fn empty_metadata_returns_none() {
        assert_eq!(BurnTokenizerRole::Primary.extract_metadata_path(""), None);
        assert_eq!(
            BurnTokenizerRole::Primary.extract_metadata_path("   "),
            None
        );
        assert_eq!(BurnTokenizerRole::Secondary.extract_metadata_path(""), None);
    }

    #[test]
    fn bare_path_accepted_only_by_primary_tokenizer_alias() {
        assert_eq!(
            BurnTokenizerRole::Primary.extract_metadata_path("/bare/path.json"),
            Some("/bare/path.json")
        );
        // Secondary does not accept the bare-path alias; an explicit
        // key is required.
        assert_eq!(
            BurnTokenizerRole::Secondary.extract_metadata_path("/bare/path.json"),
            None
        );
    }

    #[test]
    fn comma_separator_is_accepted_in_addition_to_semicolon() {
        let meta = "tokenizer_path=/a.json,backend=burn";
        assert_eq!(
            BurnTokenizerRole::Primary.extract_metadata_path(meta),
            Some("/a.json")
        );
        let meta = "tokenizer_2_path=/b.json,backend=burn";
        assert_eq!(
            BurnTokenizerRole::Secondary.extract_metadata_path(meta),
            Some("/b.json")
        );
    }

    #[test]
    fn context_from_source_set_finds_text_encoder_sources() {
        let dir = sdxl_temp_dir();
        let source_set = sample_source_set(&dir);
        let context = BurnSdxlTokenizerContext::from_source_set(&source_set);

        assert_eq!(
            context.primary_source(),
            Some(dir.join("text_encoder/model.safetensors").as_path())
        );
        assert_eq!(
            context.secondary_source(),
            Some(dir.join("text_encoder_2/model.safetensors").as_path())
        );
    }

    #[test]
    fn context_from_source_set_ignores_non_text_encoder_components() {
        let dir = sdxl_temp_dir();
        // Source set with only diffusion + vae — the tokenizer
        // context must come back empty.
        let diff = make_component_source(
            &dir,
            "diffusion",
            "diffusion.safetensors",
            "component=diffusion;backend=burn".to_owned(),
        );
        let vae = make_component_source(
            &dir,
            "vae",
            "vae.safetensors",
            "component=vae;backend=burn".to_owned(),
        );
        let source_set = ResolvedInferenceModelSourceSet::from_sources(vec![diff, vae]);
        let context = BurnSdxlTokenizerContext::from_source_set(&source_set);
        assert!(context.primary_source().is_none());
        assert!(context.secondary_source().is_none());
    }

    #[test]
    fn context_from_source_set_skips_sources_without_metadata() {
        use reimagine_core::model::ModelRole;
        use reimagine_inference::{ModelFormat, ModelSourceKind, ResolvedInferenceModelSource};

        let dir = sdxl_temp_dir();
        let p = dir.join("clip.safetensors");
        fs::write(&p, b"placeholder").unwrap();
        // Source with no metadata at all must not panic and must not
        // be promoted to a tokenizer sidecar anchor.
        let unnamed = ResolvedInferenceModelSource::new(
            ModelSourceKind::SplitComponent,
            ModelRole::TextEncoder,
            p,
            ModelFormat::SafeTensors,
        );
        let source_set = ResolvedInferenceModelSourceSet::from_sources(vec![unnamed]);
        let context = BurnSdxlTokenizerContext::from_source_set(&source_set);
        assert!(context.primary_source().is_none());
        assert!(context.secondary_source().is_none());
    }

    #[test]
    fn resolve_uses_explicit_primary_metadata_without_fallback() {
        let dir = sdxl_temp_dir();
        let (primary, _secondary) = copy_bundled_tokenizer_fixture(&dir);

        // Build a context that points at empty sources so the
        // resolver cannot use sidecar fallback.
        let context = BurnSdxlTokenizerContext::empty();
        let meta = primary_resolver_metadata(&primary);

        let resources = BurnSdxlTokenizerResources::resolve(
            &BurnBackendConfig::new("/models", "/output"),
            &context,
            Some(&meta),
            None,
        )
        .expect("resolve with primary metadata");
        assert_eq!(resources.primary_path(), primary);
        // Secondary is not overridden → bundled fallback.
        assert!(
            resources
                .secondary_path()
                .ends_with(SECONDARY_TOKENIZER_ASSET)
        );
    }

    #[test]
    fn resolve_uses_explicit_secondary_metadata_without_fallback() {
        let dir = sdxl_temp_dir();
        let (_primary, secondary) = copy_bundled_tokenizer_fixture(&dir);

        let context = BurnSdxlTokenizerContext::empty();
        let meta = secondary_resolver_metadata(&secondary);

        let resources = BurnSdxlTokenizerResources::resolve(
            &BurnBackendConfig::new("/models", "/output"),
            &context,
            None,
            Some(&meta),
        )
        .expect("resolve with secondary metadata");
        assert_eq!(resources.secondary_path(), secondary);
        // Primary is not overridden → bundled fallback.
        assert!(resources.primary_path().ends_with(PRIMARY_TOKENIZER_ASSET));
    }

    #[test]
    fn resolve_uses_both_explicit_metadata_when_supplied() {
        let dir = sdxl_temp_dir();
        let (primary, secondary) = copy_bundled_tokenizer_fixture(&dir);

        let context = BurnSdxlTokenizerContext::empty();
        let primary_meta = primary_resolver_metadata(&primary);
        let secondary_meta = secondary_resolver_metadata(&secondary);

        let resources = BurnSdxlTokenizerResources::resolve(
            &BurnBackendConfig::new("/models", "/output"),
            &context,
            Some(&primary_meta),
            Some(&secondary_meta),
        )
        .expect("resolve with both metadata");
        assert_eq!(resources.primary_path(), primary);
        assert_eq!(resources.secondary_path(), secondary);
    }

    #[test]
    fn resolve_rejects_invalid_explicit_primary_path_without_fallback() {
        // The resolver must NOT silently fall back to bundled when
        // explicit metadata points at a missing file.
        let bogus = PathBuf::from("/nonexistent/primary-tokenizer.json");
        let context = BurnSdxlTokenizerContext::empty();
        let meta = primary_resolver_metadata(&bogus);

        let err = BurnSdxlTokenizerResources::resolve(
            &BurnBackendConfig::new("/models", "/output"),
            &context,
            Some(&meta),
            None,
        )
        .expect_err("invalid explicit path must fail");
        match err {
            BurnBackendError::Tokenizer(BurnTokenizerError::ExplicitOverrideInvalid {
                role,
                raw,
            }) => {
                assert_eq!(role, BurnTokenizerRole::Primary);
                assert_eq!(raw, bogus.display().to_string());
            }
            other => panic!("expected ExplicitOverrideInvalid, got {other:?}"),
        }
    }

    #[test]
    fn resolve_rejects_invalid_explicit_secondary_path_without_fallback() {
        let bogus = PathBuf::from("/nonexistent/secondary-tokenizer.json");
        let context = BurnSdxlTokenizerContext::empty();
        let meta = secondary_resolver_metadata(&bogus);

        let err = BurnSdxlTokenizerResources::resolve(
            &BurnBackendConfig::new("/models", "/output"),
            &context,
            None,
            Some(&meta),
        )
        .expect_err("invalid explicit secondary path must fail");
        match err {
            BurnBackendError::Tokenizer(BurnTokenizerError::ExplicitOverrideInvalid {
                role,
                raw,
            }) => {
                assert_eq!(role, BurnTokenizerRole::Secondary);
                assert_eq!(raw, bogus.display().to_string());
            }
            other => panic!("expected ExplicitOverrideInvalid, got {other:?}"),
        }
    }

    #[test]
    fn resolve_rejects_directory_metadata_override_missing_inner_tokenizer() {
        // Directory override: the dir exists but the expected
        // `tokenizer.json` is missing. Bundled fallback must NOT
        // silently kick in. The error is reported as
        // `ExplicitOverrideInvalid` so callers can distinguish a
        // user-supplied but malformed override from a missing
        // bundled default.
        let dir = sdxl_temp_dir();
        let empty_dir = dir.join("empty-primary-dir");
        fs::create_dir_all(&empty_dir).unwrap();
        let meta = format!("tokenizer_path={}", empty_dir.display());

        let context = BurnSdxlTokenizerContext::empty();
        let err = BurnSdxlTokenizerResources::resolve(
            &BurnBackendConfig::new("/models", "/output"),
            &context,
            Some(&meta),
            None,
        )
        .expect_err("directory override missing inner file must fail");
        match err {
            BurnBackendError::Tokenizer(BurnTokenizerError::ExplicitOverrideInvalid {
                role,
                raw,
            }) => {
                assert_eq!(role, BurnTokenizerRole::Primary);
                assert_eq!(raw, empty_dir.display().to_string());
            }
            other => panic!("expected ExplicitOverrideInvalid, got {other:?}"),
        }
    }

    #[test]
    fn resolve_falls_back_to_sidecar_when_metadata_absent() {
        // Sidecar layout: `tokenizer.json` and `tokenizer_2/tokenizer.json`
        // next to a fictional text encoder source.
        let dir = sdxl_temp_dir();
        let text_encoder_dir = dir.join("text_encoder");
        fs::create_dir_all(&text_encoder_dir).unwrap();
        let primary_sidecar = text_encoder_dir.join("tokenizer.json");
        let secondary_sidecar = text_encoder_dir.join("tokenizer_2/tokenizer.json");
        copy_sidecar_tokenizer_files(&text_encoder_dir);

        let source_path = text_encoder_dir.join("model.safetensors");
        fs::write(&source_path, b"placeholder").unwrap();
        // Single text encoder source — primary/secondary both share
        // the same parent directory, matching the common SDXL layout
        // where sidecar tokenizers sit beside either text encoder.
        let context = BurnSdxlTokenizerContext::empty()
            .with_primary_source(source_path.clone())
            .with_secondary_source(source_path);

        let resources = BurnSdxlTokenizerResources::resolve(
            &BurnBackendConfig::new("/models", "/output"),
            &context,
            None,
            None,
        )
        .expect("resolve falls back to sidecar");
        assert_eq!(resources.primary_path(), primary_sidecar);
        assert_eq!(resources.secondary_path(), secondary_sidecar);
    }

    #[test]
    fn resolve_falls_back_to_bundled_when_sidecar_missing() {
        // No sidecar, no metadata: bundled default must be returned
        // for both sides.
        let dir = sdxl_temp_dir();
        // Source paths point at a directory that does NOT contain
        // any tokenizer sidecars.
        let orphan_source = dir.join("clip.safetensors");
        fs::write(&orphan_source, b"placeholder").unwrap();
        let context = BurnSdxlTokenizerContext::empty()
            .with_primary_source(orphan_source.clone())
            .with_secondary_source(orphan_source);

        let resources = BurnSdxlTokenizerResources::resolve(
            &BurnBackendConfig::new("/models", "/output"),
            &context,
            None,
            None,
        )
        .expect("resolve falls back to bundled");
        assert!(resources.primary_path().ends_with(PRIMARY_TOKENIZER_ASSET));
        assert!(
            resources
                .secondary_path()
                .ends_with(SECONDARY_TOKENIZER_ASSET)
        );
    }

    #[test]
    fn resolve_secondary_sidecar_falls_back_to_primary_layout() {
        // Secondary has its own layout (tokenizer_2/tokenizer.json)
        // missing, but the primary layout (tokenizer.json) is
        // present — secondary must inherit it (CLIP-G may share the
        // primary vocabulary file).
        let dir = sdxl_temp_dir();
        let clip_dir = dir.join("clip");
        fs::create_dir_all(&clip_dir).unwrap();
        fs::copy(
            bundled_dir().join(PRIMARY_TOKENIZER_ASSET),
            clip_dir.join("tokenizer.json"),
        )
        .unwrap();
        // Deliberately do NOT add a tokenizer_2 subdir.

        let source_path = clip_dir.join("model.safetensors");
        fs::write(&source_path, b"placeholder").unwrap();
        let context = BurnSdxlTokenizerContext::empty()
            .with_primary_source(source_path.clone())
            .with_secondary_source(source_path);

        let resources = BurnSdxlTokenizerResources::resolve(
            &BurnBackendConfig::new("/models", "/output"),
            &context,
            None,
            None,
        )
        .expect("secondary sidecar falls back to primary layout");
        assert_eq!(resources.secondary_path(), clip_dir.join("tokenizer.json"));
    }

    #[test]
    fn resolve_uses_configured_tokenizer_root_for_bundled_fallback() {
        // Custom asset root + no metadata + no sidecar → bundled
        // files must come from the configured root, not the
        // workspace default.
        let dir = sdxl_temp_dir();
        let _ = copy_bundled_tokenizer_fixture(&dir);
        let config = BurnBackendConfig::new("/models", "/output").with_tokenizer_root(dir.clone());
        let context = BurnSdxlTokenizerContext::empty();

        let resources = BurnSdxlTokenizerResources::resolve(&config, &context, None, None)
            .expect("resolve with custom root");
        assert_eq!(resources.root(), dir);
        assert!(resources.primary_path().starts_with(&dir));
        assert!(resources.secondary_path().starts_with(&dir));
    }

    #[test]
    fn resolve_extracts_metadata_from_source_set_via_context() {
        let dir = sdxl_temp_dir();
        let (primary, secondary) = copy_bundled_tokenizer_fixture(&dir);
        let text_encoder_dir = dir.join("text_encoder");
        fs::create_dir_all(&text_encoder_dir).unwrap();
        let text_encoder_2_dir = dir.join("text_encoder_2");
        fs::create_dir_all(&text_encoder_2_dir).unwrap();

        let primary_source = text_encoder_dir.join("model.safetensors");
        let secondary_source = text_encoder_2_dir.join("model.safetensors");
        fs::write(&primary_source, b"placeholder").unwrap();
        fs::write(&secondary_source, b"placeholder").unwrap();

        let primary_source_meta = primary_resolver_metadata(&primary);
        let secondary_source_meta = secondary_resolver_metadata(&secondary);

        use reimagine_core::model::ModelRole;
        use reimagine_inference::{ModelFormat, ModelSourceKind, ResolvedInferenceModelSource};
        let primary_entry = ResolvedInferenceModelSource::new(
            ModelSourceKind::SplitComponent,
            ModelRole::TextEncoder,
            primary_source.clone(),
            ModelFormat::SafeTensors,
        )
        .with_metadata(primary_source_meta.clone());
        let secondary_entry = ResolvedInferenceModelSource::new(
            ModelSourceKind::SplitComponent,
            ModelRole::TextEncoder,
            secondary_source.clone(),
            ModelFormat::SafeTensors,
        )
        .with_metadata(secondary_source_meta.clone());

        let source_set = ResolvedInferenceModelSourceSet::from_sources(vec![
            make_component_source(
                &dir,
                "diffusion",
                "diffusion/model.safetensors",
                "component=diffusion;backend=burn".to_owned(),
            ),
            make_component_source(
                &dir,
                "vae",
                "vae/model.safetensors",
                "component=vae;backend=burn".to_owned(),
            ),
            primary_entry,
            secondary_entry,
        ]);

        // The downstream caller is expected to pair each context
        // source with its source-set metadata. Simulate that by
        // looking up the metadata for the source path the context
        // already discovered.
        let context = BurnSdxlTokenizerContext::from_source_set(&source_set);
        let primary_meta = context
            .primary_source()
            .and_then(|path| {
                source_set
                    .sources()
                    .iter()
                    .find(|s| s.path() == path)
                    .and_then(|s| s.metadata())
            })
            .map(str::to_owned);
        let secondary_meta = context
            .secondary_source()
            .and_then(|path| {
                source_set
                    .sources()
                    .iter()
                    .find(|s| s.path() == path)
                    .and_then(|s| s.metadata())
            })
            .map(str::to_owned);

        let resources = BurnSdxlTokenizerResources::resolve(
            &BurnBackendConfig::new("/models", "/output"),
            &context,
            primary_meta.as_deref(),
            secondary_meta.as_deref(),
        )
        .expect("resolve via source-set context");
        assert_eq!(resources.primary_path(), primary);
        assert_eq!(resources.secondary_path(), secondary);
    }

    #[test]
    fn load_bpe_pair_loads_via_explicit_paths() {
        // `load_bpe` is invoked by `load_one` whenever a `vocab.json`
        // file sits next to a `merges.txt` file. Construct a minimal
        // self-consistent BPE pair (vocab containing every token
        // referenced in the merges) and confirm the loader accepts
        // it. The exact tokenization output is not exercised — only
        // that the loader wires up the BPE model and the workspace
        // `tokenizers` crate accepts the construction.
        let dir = sdxl_temp_dir();
        let vocab = dir.join("vocab.json");
        let merges = dir.join("merges.txt");
        // Minimal BPE vocab: a single entry mapping "a" -> 0. The
        // `tokenizers` crate accepts vocab.json shaped as a flat
        // `string -> id` map.
        fs::write(&vocab, br#"{"a":0,"b":1,"ab":2}"#).unwrap();
        // A merges.txt file must contain at least one valid merge
        // line — even an empty merges file is rejected by the
        // tokenizers crate. Provide a single merge line so the
        // file is well-formed and self-consistent with the vocab
        // above.
        fs::write(&merges, "#version: 0.2\na b\n").unwrap();

        let tok = BurnSdxlTokenizer::from_paths(&vocab, &vocab).expect("BPE pair must load");
        let _ = tok.resources();
    }

    #[test]
    fn load_vocab_without_merges_falls_back_to_regular_load() {
        // A `vocab.json` file with no adjacent `merges.txt` is
        // treated as a regular `tokenizer.json` (some packages ship
        // a `tokenizer.json` under a different name). The bundled
        // primary `tokenizer.json` is a valid `tokenizers` JSON
        // document so it should load cleanly.
        let dir = sdxl_temp_dir();
        let vocab = dir.join("vocab.json");
        fs::copy(bundled_dir().join(PRIMARY_TOKENIZER_ASSET), &vocab).unwrap();
        // No merges.txt next to it. load_one should fall through
        // to the regular `Tokenizer::from_file` path and succeed.
        let tok = BurnSdxlTokenizer::from_paths(&vocab, &vocab)
            .expect("vocab.json without merges must load as a regular tokenizer");
        let _ = tok.resources();
    }

    #[test]
    fn sidecar_search_handles_dir_layout_tokenizer_directory() {
        // Directory layout: `<parent>/tokenizer/tokenizer.json`.
        let dir = sdxl_temp_dir();
        let clip_dir = dir.join("clip");
        fs::create_dir_all(clip_dir.join("tokenizer")).unwrap();
        fs::copy(
            bundled_dir().join(PRIMARY_TOKENIZER_ASSET),
            clip_dir.join("tokenizer/tokenizer.json"),
        )
        .unwrap();
        let source = clip_dir.join("model.safetensors");
        fs::write(&source, b"placeholder").unwrap();
        let context = BurnSdxlTokenizerContext::empty().with_primary_source(source);

        let resources = BurnSdxlTokenizerResources::resolve(
            &BurnBackendConfig::new("/models", "/output"),
            &context,
            None,
            None,
        )
        .expect("sidecar directory layout must resolve");
        assert_eq!(
            resources.primary_path(),
            clip_dir.join("tokenizer/tokenizer.json")
        );
    }

    #[test]
    fn role_reports_stable_as_str() {
        assert_eq!(BurnTokenizerRole::Primary.as_str(), "primary");
        assert_eq!(BurnTokenizerRole::Secondary.as_str(), "secondary");
    }
}
