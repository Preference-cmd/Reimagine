use serde::{Deserialize, Serialize};

pub const DEFAULT_EXCLUDE_PATTERNS: &[&str] = &[
    ".git/**",
    "target/**",
    "node_modules/**",
    "**/.git/**",
    "**/target/**",
    "**/node_modules/**",
    "**/.cache/**",
    "**/cache/**",
    "**/build/**",
    "**/dist/**",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScanConfig {
    recursive: bool,
    ignore_hidden: bool,
    include_patterns: Vec<String>,
    exclude_patterns: Vec<String>,
    supported_extensions: Vec<String>,
}

impl ScanConfig {
    pub fn new(recursive: bool, ignore_hidden: bool) -> Self {
        Self {
            recursive,
            ignore_hidden,
            include_patterns: Vec::new(),
            exclude_patterns: DEFAULT_EXCLUDE_PATTERNS
                .iter()
                .map(|pattern| (*pattern).to_owned())
                .collect(),
            supported_extensions: vec!["safetensors".to_owned()],
        }
    }

    pub fn with_recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }

    pub fn with_ignore_hidden(mut self, ignore_hidden: bool) -> Self {
        self.ignore_hidden = ignore_hidden;
        self
    }

    pub fn with_include_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.include_patterns.push(pattern.into());
        self
    }

    pub fn with_exclude_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.exclude_patterns.push(pattern.into());
        self
    }

    pub fn with_supported_extension(mut self, extension: impl Into<String>) -> Self {
        self.supported_extensions
            .push(normalize_extension(&extension.into()));
        self
    }

    pub fn recursive(&self) -> bool {
        self.recursive
    }

    pub fn ignore_hidden(&self) -> bool {
        self.ignore_hidden
    }

    pub fn include_patterns(&self) -> &[String] {
        &self.include_patterns
    }

    pub fn exclude_patterns(&self) -> &[String] {
        &self.exclude_patterns
    }

    pub fn supported_extensions(&self) -> &[String] {
        &self.supported_extensions
    }
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self::new(true, true)
    }
}

fn normalize_extension(extension: &str) -> String {
    extension.trim_start_matches('.').to_ascii_lowercase()
}
