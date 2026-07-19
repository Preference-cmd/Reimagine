use std::fmt;

#[derive(Debug)]
pub enum PackageError {
    /// Maximum entry count exceeded.
    EntryCountLimit { max: usize, actual: usize },
    /// Maximum expanded size exceeded.
    ExpandedSizeLimit { max: u64, actual: u64 },
    /// Archive entry path is invalid (absolute, empty, etc.).
    InvalidPath { entry: String },
    /// Archive entry path attempts path traversal (e.g. `..`).
    PathTraversal { entry: String },
    /// Symlinks are not allowed in worker packages.
    SymlinkRejected { entry: String },
    /// Hardlinks are not allowed in worker packages.
    HardlinkRejected { entry: String },
    /// Duplicate path in archive.
    DuplicatePath { entry: String },
    /// Unexpected file type in archive.
    UnexpectedFileType { entry: String, kind: String },
    /// Required `package.json` manifest is missing from archive.
    ManifestMissing,
    /// The package manifest identity does not match expectations.
    ManifestMismatch {
        field: String,
        expected: String,
        actual: String,
    },
    /// Filesystem I/O error during extraction.
    Extraction { message: String },
    /// Hash mismatch for an extracted entry.
    HashMismatch { entry: String },
    /// Package build error (serialization, archive creation, etc.).
    Build { message: String },
}

impl fmt::Display for PackageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EntryCountLimit { max, actual } => {
                write!(f, "archive entry count {actual} exceeds limit {max}")
            }
            Self::ExpandedSizeLimit { max, actual } => {
                write!(f, "archive expanded size {actual} exceeds limit {max}")
            }
            Self::InvalidPath { entry } => {
                write!(f, "invalid archive entry path: `{entry}`")
            }
            Self::PathTraversal { entry } => {
                write!(f, "archive entry path traverses outside staging: `{entry}`")
            }
            Self::SymlinkRejected { entry } => {
                write!(f, "symlink not allowed in worker package: `{entry}`")
            }
            Self::HardlinkRejected { entry } => {
                write!(f, "hardlink not allowed in worker package: `{entry}`")
            }
            Self::DuplicatePath { entry } => {
                write!(f, "duplicate archive entry: `{entry}`")
            }
            Self::UnexpectedFileType { entry, kind } => {
                write!(f, "unexpected {kind} entry in archive: `{entry}`")
            }
            Self::ManifestMissing => write!(f, "archive missing required package.json manifest"),
            Self::ManifestMismatch {
                field,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "manifest field `{field}` mismatch: expected `{expected}`, got `{actual}`"
                )
            }
            Self::Extraction { message } => write!(f, "extraction error: {message}"),
            Self::HashMismatch { entry } => {
                write!(f, "hash mismatch for extracted entry: `{entry}`")
            }
            Self::Build { message } => write!(f, "package build error: {message}"),
        }
    }
}

impl std::error::Error for PackageError {}

impl From<std::io::Error> for PackageError {
    fn from(error: std::io::Error) -> Self {
        Self::Extraction {
            message: error.to_string(),
        }
    }
}

pub type PackageResult<T> = Result<T, PackageError>;
