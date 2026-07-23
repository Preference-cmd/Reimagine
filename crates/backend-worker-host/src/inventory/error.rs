use std::fmt;
use std::path::PathBuf;

#[derive(Debug)]
pub enum InventoryError {
    /// Filesystem I/O error.
    Io { path: PathBuf, message: String },
    /// JSON serialization/deserialization error.
    Json {
        path: Option<PathBuf>,
        message: String,
    },
    /// Inventory index or record file is corrupt.
    Corrupt { path: PathBuf },
    /// Installation not found in inventory.
    NotFound { installation_id: String },
    /// Concurrent modification detected (write-write conflict).
    ConcurrentModification,
}

impl fmt::Display for InventoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, message } => {
                write!(f, "I/O error at `{}`: {message}", path.display())
            }
            Self::Json {
                path: Some(p),
                message,
            } => {
                write!(f, "JSON error at `{}`: {message}", p.display())
            }
            Self::Json {
                path: None,
                message,
            } => write!(f, "JSON error: {message}"),
            Self::Corrupt { path } => {
                write!(f, "corrupt inventory at `{}`", path.display())
            }
            Self::NotFound { installation_id } => {
                write!(f, "installation `{installation_id}` not found in inventory")
            }
            Self::ConcurrentModification => {
                write!(f, "concurrent inventory modification detected")
            }
        }
    }
}

impl std::error::Error for InventoryError {}

impl From<std::io::Error> for InventoryError {
    fn from(error: std::io::Error) -> Self {
        Self::Io {
            path: std::path::PathBuf::new(),
            message: error.to_string(),
        }
    }
}

pub type InventoryResult<T> = Result<T, InventoryError>;
