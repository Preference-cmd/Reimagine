use std::fmt;
use std::path::PathBuf;

use crate::catalog::CatalogError;
use crate::inventory::InventoryError;
use crate::package::PackageError;

#[derive(Debug)]
pub enum InstallError {
    /// TUF catalog operation failed.
    Catalog(CatalogError),
    /// Package extraction failed.
    Package(PackageError),
    /// Staging directory already exists.
    StagingExists { path: PathBuf },
    /// Failed to clean up a staging directory.
    StagingCleanup { path: PathBuf, message: String },
    /// Self-check timed out.
    SelfCheckTimeout,
    /// Self-check failed due to identity mismatch.
    SelfCheckIdentityMismatch {
        field: String,
        expected: String,
        actual: String,
    },
    /// Self-check failed for other reasons.
    SelfCheckFailed { message: String },
    /// Atomic promotion failed.
    PromoteFailed {
        from: PathBuf,
        to: PathBuf,
        message: String,
    },
    /// Inventory operation failed.
    Inventory(InventoryError),
    /// Journal file is corrupt.
    JournalCorrupt { path: PathBuf },
    /// Another installer process holds the lock.
    LockContended,
    /// I/O error.
    Io { path: PathBuf, message: String },
}

impl fmt::Display for InstallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Catalog(e) => write!(f, "catalog error: {e}"),
            Self::Package(e) => write!(f, "package error: {e}"),
            Self::StagingExists { path } => {
                write!(
                    f,
                    "staging directory already exists at `{}`",
                    path.display()
                )
            }
            Self::StagingCleanup { path, message } => {
                write!(
                    f,
                    "failed to clean up staging `{}`: {message}",
                    path.display()
                )
            }
            Self::SelfCheckTimeout => write!(f, "worker self-check timed out"),
            Self::SelfCheckIdentityMismatch {
                field,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "self-check identity `{field}` mismatch: expected `{expected}`, got `{actual}`"
                )
            }
            Self::SelfCheckFailed { message } => write!(f, "self-check failed: {message}"),
            Self::PromoteFailed { from, to, message } => {
                write!(
                    f,
                    "promote from `{}` to `{}` failed: {message}",
                    from.display(),
                    to.display()
                )
            }
            Self::Inventory(e) => write!(f, "inventory error: {e}"),
            Self::JournalCorrupt { path } => {
                write!(f, "journal file is corrupt at `{}`", path.display())
            }
            Self::LockContended => write!(f, "another installer process holds the lock"),
            Self::Io { path, message } => write!(f, "I/O error at `{}`: {message}", path.display()),
        }
    }
}

impl std::error::Error for InstallError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Catalog(e) => Some(e),
            Self::Package(e) => Some(e),
            Self::Inventory(e) => Some(e),
            _ => None,
        }
    }
}

impl From<CatalogError> for InstallError {
    fn from(e: CatalogError) -> Self {
        Self::Catalog(e)
    }
}

impl From<PackageError> for InstallError {
    fn from(e: PackageError) -> Self {
        Self::Package(e)
    }
}

impl From<InventoryError> for InstallError {
    fn from(e: InventoryError) -> Self {
        Self::Inventory(e)
    }
}

impl From<std::io::Error> for InstallError {
    fn from(e: std::io::Error) -> Self {
        Self::Io {
            path: std::path::PathBuf::new(),
            message: e.to_string(),
        }
    }
}

pub type InstallResult<T> = Result<T, InstallError>;
